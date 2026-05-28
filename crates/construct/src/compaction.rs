//! Compaction log — append-only record of compaction operations.
//!
//! Each operation is a tagged JSON object appended to `compaction.jsonl`.
//! The context builder reads these to transform the raw transcript into
//! the live message list sent to the API.
//!
//! # Public API
//! - [`CompactionLog`] — write handle for the compaction log
//! - [`CompactionOp`] — a single compaction operation (for reading)
//!
//! # Dependencies
//! `serde`, `serde_json`

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::transcript::Turn;

/// A compaction operation stored in the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CompactionOp {
    /// S1: duplicate file reads removed from context.
    FileDedup {
        stage: String,
        targetIds: Vec<String>,
        afterTurn: String,
        ts: u64,
    },
    /// S1: long tool results truncated to head+tail.
    MiddleOut {
        stage: String,
        targetIds: Vec<String>,
        threshold: usize,
        afterTurn: String,
        ts: u64,
    },
    /// S2: a single exchange block summarized by LLM.
    BlockCompact {
        stage: String,
        blockId: String,
        summary: String,
        sourceIds: Vec<String>,
        afterTurn: String,
        ts: u64,
    },
    /// S3: a topic span (multiple blocks) summarized by LLM.
    TopicCompact {
        stage: String,
        topicLabel: String,
        summary: String,
        sourceBlockIds: Vec<String>,
        afterTurn: String,
        ts: u64,
    },
    /// S4: S3 summaries merged into a handoff briefing.
    FullCompact {
        stage: String,
        summary: String,
        sourceIds: Vec<String>,
        afterTurn: String,
        ts: u64,
    },
}

impl CompactionOp {
    /// The turn ID at which this operation was applied.
    pub fn afterTurn(&self) -> &str {
        match self {
            Self::FileDedup { afterTurn, .. }
            | Self::MiddleOut { afterTurn, .. }
            | Self::BlockCompact { afterTurn, .. }
            | Self::TopicCompact { afterTurn, .. }
            | Self::FullCompact { afterTurn, .. } => afterTurn,
        }
    }
}

/// Build a map of blockId → summary char count from BlockCompact entries.
///
/// Used by S2/S3 zone calculations to account for post-compaction sizes.
pub fn compactedBlockSizes(ops: &[CompactionOp]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for op in ops {
        if let CompactionOp::BlockCompact {
            blockId, summary, ..
        } = op
        {
            map.insert(blockId.clone(), summary.len());
        }
    }
    map
}

/// Collect block IDs that are superseded by S3/S4 compaction.
///
/// These blocks are replaced by a single summary in the live context
/// and should not count toward zone budgets.
pub fn supersededBlocks(ops: &[CompactionOp]) -> HashSet<String> {
    let mut superseded = HashSet::new();
    for op in ops {
        match op {
            CompactionOp::TopicCompact { sourceBlockIds, .. } => {
                for bid in sourceBlockIds {
                    superseded.insert(bid.clone());
                }
            }
            CompactionOp::FullCompact { sourceIds, .. } => {
                for bid in sourceIds {
                    superseded.insert(bid.clone());
                }
            }
            _ => {}
        }
    }
    superseded
}

/// Determine which block IDs fall in the oldest `fraction` of the
/// effective context size.
///
/// Walks the active branch turns, groups by block, and uses the compacted
/// summary size for already-S2'd blocks (instead of the raw content size).
/// Blocks in `superseded` (covered by S3/S4) are skipped entirely — they
/// don't exist as individual blocks in the live context and should not
/// consume zone budget.
///
/// Args:
///     activeTurns: Turns on the active branch (from `walkBranchTurns`).
///     compactedSizes: blockId → summary char count (from `compactedBlockSizes`).
///     superseded: Block IDs replaced by S3/S4 (from `supersededBlocks`).
///     fraction: Zone fraction (0.60 for S2, 0.30 for S3).
pub fn zoneBlocks(
    activeTurns: &[Turn],
    compactedSizes: &HashMap<String, usize>,
    superseded: &HashSet<String>,
    fraction: f64,
) -> HashSet<String> {
    // Group turns by block and compute raw char size per block.
    let mut blocks: Vec<(String, usize)> = Vec::new();
    let mut currentBlockId = String::new();
    let mut currentRawSize: usize = 0;

    for turn in activeTurns {
        if turn.blockId != currentBlockId {
            if !currentBlockId.is_empty() && !superseded.contains(&currentBlockId) {
                let effective = compactedSizes
                    .get(&currentBlockId)
                    .copied()
                    .unwrap_or(currentRawSize);
                blocks.push((currentBlockId.clone(), effective));
            }
            currentBlockId = turn.blockId.clone();
            currentRawSize = 0;
        }
        currentRawSize += turn.content.len();
    }
    // Flush last block.
    if !currentBlockId.is_empty() && !superseded.contains(&currentBlockId) {
        let effective = compactedSizes
            .get(&currentBlockId)
            .copied()
            .unwrap_or(currentRawSize);
        blocks.push((currentBlockId, effective));
    }

    let total: usize = blocks.iter().map(|(_, s)| s).sum();
    let target = (total as f64 * fraction) as usize;

    let mut zone = HashSet::new();
    let mut cumulative: usize = 0;
    for (blockId, size) in &blocks {
        if cumulative >= target {
            break;
        }
        zone.insert(blockId.clone());
        cumulative += size;
    }

    zone
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Write handle for the compaction log.
pub struct CompactionLog {
    sessionDir: PathBuf,
    writer: BufWriter<fs::File>,
}

impl CompactionLog {
    /// Open or create the compaction log in a session directory.
    pub fn open(sessionDir: &Path) -> Result<Self> {
        let path = sessionDir.join("compaction.jsonl");
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open compaction log: {}", path.display()))?;
        Ok(Self {
            sessionDir: sessionDir.to_path_buf(),
            writer: BufWriter::new(file),
        })
    }

    /// Append an operation to the log and flush.
    fn writeOp(&mut self, op: &CompactionOp) -> Result<()> {
        let line = serde_json::to_string(op)?;
        writeln!(self.writer, "{line}")?;
        self.writer.flush()?;
        Ok(())
    }

    /// S1: record that duplicate file reads were removed.
    pub fn recordFileDedup(&mut self, targetIds: Vec<String>, afterTurn: &str) -> Result<()> {
        self.writeOp(&CompactionOp::FileDedup {
            stage: "s1".into(),
            targetIds,
            afterTurn: afterTurn.into(),
            ts: now(),
        })
    }

    /// S1: record that long tool results were middle-out truncated.
    pub fn recordMiddleOut(
        &mut self,
        targetIds: Vec<String>,
        afterTurn: &str,
        threshold: usize,
    ) -> Result<()> {
        self.writeOp(&CompactionOp::MiddleOut {
            stage: "s1".into(),
            targetIds,
            threshold,
            afterTurn: afterTurn.into(),
            ts: now(),
        })
    }

    /// S2: record that an exchange block was LLM-summarized.
    pub fn recordBlockCompact(
        &mut self,
        blockId: &str,
        summary: &str,
        sourceIds: Vec<String>,
        afterTurn: &str,
    ) -> Result<()> {
        self.writeOp(&CompactionOp::BlockCompact {
            stage: "s2".into(),
            blockId: blockId.into(),
            summary: summary.into(),
            sourceIds,
            afterTurn: afterTurn.into(),
            ts: now(),
        })
    }

    /// S3: record that a topic span was LLM-summarized.
    pub fn recordTopicCompact(
        &mut self,
        topicLabel: &str,
        summary: &str,
        blockIds: Vec<String>,
        afterTurn: &str,
    ) -> Result<()> {
        self.writeOp(&CompactionOp::TopicCompact {
            stage: "s3".into(),
            topicLabel: topicLabel.into(),
            summary: summary.into(),
            sourceBlockIds: blockIds,
            afterTurn: afterTurn.into(),
            ts: now(),
        })
    }

    /// S4: record that S3 summaries were merged into a handoff briefing.
    pub fn recordFullCompact(
        &mut self,
        summary: &str,
        compactedIds: Vec<String>,
        afterTurn: &str,
    ) -> Result<()> {
        self.writeOp(&CompactionOp::FullCompact {
            stage: "s4".into(),
            summary: summary.into(),
            sourceIds: compactedIds,
            afterTurn: afterTurn.into(),
            ts: now(),
        })
    }

    /// Load all operations from the compaction log.
    pub fn loadAll(&self) -> Result<Vec<CompactionOp>> {
        let path = self.sessionDir.join("compaction.jsonl");
        if !path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("read compaction log: {}", path.display()))?;

        let mut ops = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let op: CompactionOp =
                serde_json::from_str(line).with_context(|| "parse compaction log entry")?;
            ops.push(op);
        }
        Ok(ops)
    }
}
