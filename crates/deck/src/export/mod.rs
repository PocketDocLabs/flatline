//! Export a session transcript as SFT training data.
//!
//! Emits a pretty-printed JSON array of examples in OpenAI fine-tuning
//! message shape. One example per stable-prefix segment — a segment is the
//! run of assistant turns whose request snapshots share the same system
//! prompt, tools, and the previous snapshot's messages as a prefix. A new
//! segment starts whenever any of those break (system/tools edit, MCP load,
//! compaction rewrite, etc.).
//!
//! To convert to JSONL for OpenAI's fine-tuning API: `jq -c '.[]' out.json`.
//!
//! Reasoning traces are emitted as a non-standard `reasoning` field at the
//! example level, aligned to the assistant turns in `messages`.
//!
//! # Public API
//! - [`ExportArgs`] — CLI-resolved arguments
//! - [`run`] — execute an export
//!
//! # Dependencies
//! `construct::snapshot`, `construct::transcript`, `serde_json`

pub mod openai;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use construct::snapshot::RequestSnapshot;
use construct::transcript::{self, SessionMeta, Transcript, Turn, TurnRole, TurnStatus};

/// CLI-resolved export arguments.
pub struct ExportArgs {
    /// Single session to export. Mutually exclusive with `all`.
    pub sessionId: Option<String>,
    /// Export every session that has snapshots.
    pub all: bool,
    /// When `all` is set, only include sessions whose `meta.projectDir`
    /// matches this directory. Ignored otherwise.
    pub project: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub noReasoning: bool,
    pub minMessages: usize,
    pub dryRun: bool,
    /// Include cancelled assistant turns as training targets. By default
    /// they're skipped — partial/interrupted content isn't good SFT data.
    pub includeCancelled: bool,
}

/// Execute the export. Exits process on fatal errors.
pub fn run(args: ExportArgs) -> Result<()> {
    let examples = if args.all {
        collectAll(&args)?
    } else {
        let sid = args
            .sessionId
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("provide a sessionId or pass --all"))?;
        collectOne(sid, &args)?
    };

    let count = examples.len();

    if !args.dryRun {
        let rendered = serde_json::to_string_pretty(&examples)?;
        if let Some(path) = &args.output {
            fs::write(path, &rendered)
                .with_context(|| format!("write output file {}", path.display()))?;
        } else {
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(rendered.as_bytes())?;
            handle.write_all(b"\n")?;
        }
    }

    eprintln!("total: {count} examples");

    if count == 0 {
        std::process::exit(2);
    }
    Ok(())
}

/// Build examples for a single session. Exits with code 2 if the session
/// has no snapshots (predates the feature) — matches the previous behavior.
fn collectOne(sessionId: &str, args: &ExportArgs) -> Result<Vec<openai::Example>> {
    let sessionDir = transcript::sessionsDir().join(sessionId);
    if !sessionDir.exists() {
        anyhow::bail!("session not found: {}", sessionDir.display());
    }
    let snapshotsDir = sessionDir.join("snapshots");
    if !snapshotsDir.exists() {
        eprintln!(
            "skipped: no snapshots (session {sessionId} predates the snapshot feature)"
        );
        std::process::exit(2);
    }
    let (examples, stats) = buildExamplesForSession(sessionId, &sessionDir, args)?;
    eprintln!(
        "{sessionId}: {} asst turns \u{2192} {} examples ({} skipped)",
        stats.asstTurns, stats.emitted, stats.skipped
    );
    Ok(examples)
}

/// Build examples across all sessions with snapshots, merged into one array.
fn collectAll(args: &ExportArgs) -> Result<Vec<openai::Example>> {
    let dir = transcript::sessionsDir();
    if !dir.exists() {
        eprintln!("no sessions directory at {}", dir.display());
        return Ok(Vec::new());
    }

    let projectFilter = args
        .project
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());

    let mut all: Vec<openai::Example> = Vec::new();
    let mut sessions = 0usize;
    let mut withoutSnapshots = 0usize;
    let mut filtered = 0usize;
    let mut failed = 0usize;

    for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let sessionDir = entry.path();
        if !sessionDir.is_dir() {
            continue;
        }
        let sessionId = match sessionDir.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };

        if !sessionDir.join("snapshots").exists() {
            withoutSnapshots += 1;
            continue;
        }

        // Optional project filter via meta.json.
        if let Some(ref target) = projectFilter {
            let meta = match Transcript::loadMeta(&sessionDir) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if &meta.projectDir != target {
                filtered += 1;
                continue;
            }
        }

        sessions += 1;
        match buildExamplesForSession(&sessionId, &sessionDir, args) {
            Ok((examples, stats)) => {
                eprintln!(
                    "{sessionId}: {} asst turns \u{2192} {} examples ({} skipped)",
                    stats.asstTurns, stats.emitted, stats.skipped
                );
                all.extend(examples);
            }
            Err(e) => {
                failed += 1;
                eprintln!("{sessionId}: failed ({e})");
            }
        }
    }

    eprintln!(
        "sessions processed: {sessions} | no-snapshots skipped: {withoutSnapshots} \
         | project-filtered: {filtered} | failed: {failed}"
    );
    Ok(all)
}

/// Stats for a single session's export.
struct SessionStats {
    asstTurns: usize,
    emitted: usize,
    skipped: usize,
}

/// Core: for one session directory already verified to have snapshots,
/// produce the examples and a stat summary.
fn buildExamplesForSession(
    sessionId: &str,
    sessionDir: &Path,
    args: &ExportArgs,
) -> Result<(Vec<openai::Example>, SessionStats)> {
    let snapshotsDir = sessionDir.join("snapshots");
    let meta = Transcript::loadMeta(sessionDir)
        .with_context(|| format!("load meta.json for {sessionId}"))?;

    let allTurns = {
        let t = Transcript::open(sessionId)
            .with_context(|| format!("open transcript for {sessionId}"))?;
        t.loadAll()?
    };

    let headTurnId = meta
        .headTurn
        .clone()
        .or_else(|| allTurns.last().map(|t| t.id.clone()))
        .ok_or_else(|| anyhow::anyhow!("session {sessionId} has no turns"))?;
    let branch = transcript::walkBranchTurns(&allTurns, &headTurnId);

    let index = loadSnapshotIndex(&snapshotsDir)
        .with_context(|| format!("load snapshot index for {sessionId}"))?;
    let segments = buildSegments(&branch, &index, args.includeCancelled);

    let mut emitted: Vec<openai::Example> = Vec::new();
    let mut skipped = 0usize;

    for (segIdx, seg) in segments.iter().enumerate() {
        match openai::buildExample(
            sessionId,
            segIdx,
            seg,
            &branch,
            &snapshotsDir,
            args.noReasoning,
            args.minMessages,
        )? {
            Some(example) => emitted.push(example),
            None => skipped += 1,
        }
    }

    let stats = SessionStats {
        asstTurns: branch
            .iter()
            .filter(|t| matches!(t.role, TurnRole::Assistant))
            .count(),
        emitted: emitted.len(),
        skipped,
    };
    Ok((emitted, stats))
}

/// A segment of the head chain whose assistant turns share the same anchor
/// snapshot (same system, same tools, and the anchor's messages as a prefix).
pub(crate) struct Segment {
    /// Index into `branch` of the first assistant turn (the anchor).
    pub anchorIdx: usize,
    /// Index into `branch` of the last assistant turn in this segment.
    pub lastAsstIdx: usize,
    /// The anchor's snapshot (system prompt, tools, prefix messages).
    pub anchor: RequestSnapshot,
    /// The last assistant turn's snapshot (has the full segment messages as prefix).
    pub last: RequestSnapshot,
}

use std::collections::HashMap;

pub(crate) fn loadSnapshotIndex(snapshotsDir: &Path) -> Result<HashMap<String, RequestSnapshot>> {
    let path = snapshotsDir.join("index.jsonl");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut map = HashMap::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: IndexEntry = serde_json::from_str(line)
            .with_context(|| format!("parse index entry: {line}"))?;
        map.insert(entry.hash, entry.snapshot);
    }
    Ok(map)
}

#[derive(serde::Deserialize)]
struct IndexEntry {
    hash: String,
    snapshot: RequestSnapshot,
}

/// Partition the branch into stable-prefix segments.
///
/// A new segment starts whenever the current assistant turn's snapshot
/// differs from the anchor in system prompt hash, tools hash, OR does not
/// extend the anchor's messages as a prefix.
///
/// When `includeCancelled` is false (default), assistant turns whose
/// `status` is not `Completed` are excluded from segment construction —
/// their content is partial/interrupted and shouldn't become a training
/// target. They still exist in the history referenced by later turns'
/// snapshots, which is faithful to what the model saw.
pub(crate) fn buildSegments(
    branch: &[Turn],
    index: &HashMap<String, RequestSnapshot>,
    includeCancelled: bool,
) -> Vec<Segment> {
    let asstIdxs: Vec<usize> = branch
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            matches!(t.role, TurnRole::Assistant)
                && t.snapshotHash.is_some()
                && (includeCancelled || matches!(t.status, TurnStatus::Completed))
        })
        .map(|(i, _)| i)
        .collect();

    let mut segs = Vec::new();
    let mut cur: Option<Segment> = None;

    for &idx in &asstIdxs {
        let hash = branch[idx].snapshotHash.as_deref().unwrap();
        let snap = match index.get(hash) {
            Some(s) => s.clone(),
            None => continue, // dangling reference; skip
        };

        let startNew = match &cur {
            None => true,
            Some(c) => {
                c.anchor.systemPromptHash != snap.systemPromptHash
                    || c.anchor.toolsHash != snap.toolsHash
                    || !snap.messages.starts_with(&c.anchor.messages)
            }
        };

        if startNew {
            if let Some(finished) = cur.take() {
                segs.push(finished);
            }
            cur = Some(Segment {
                anchorIdx: idx,
                lastAsstIdx: idx,
                anchor: snap.clone(),
                last: snap,
            });
        } else if let Some(ref mut c) = cur {
            c.lastAsstIdx = idx;
            c.last = snap;
        }
    }

    if let Some(finished) = cur.take() {
        segs.push(finished);
    }

    segs
}

#[allow(dead_code)]
fn describeMeta(meta: &SessionMeta) -> String {
    meta.sessionId.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummySnap(sys: &str, tools: &str, msgs: &[&str]) -> RequestSnapshot {
        RequestSnapshot {
            v: 1,
            model: "m".into(),
            provider: "openrouter".into(),
            baseUrl: "x".into(),
            providerOrder: vec![],
            maxTokens: None,
            reasoning: None,
            systemPromptHash: Some(sys.into()),
            toolsHash: Some(tools.into()),
            toolsCount: 0,
            messages: msgs.iter().map(|s| (*s).into()).collect(),
            temperature: None,
            topP: None,
            seed: None,
            ts: 0,
        }
    }

    fn asstTurn(id: &str, block: &str, snap: &str) -> Turn {
        Turn {
            id: id.into(),
            blockId: block.into(),
            topicId: String::new(),
            role: TurnRole::Assistant,
            content: String::new(),
            ts: 0,
            parentId: None,
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: Some(snap.into()),
            status: construct::transcript::TurnStatus::Completed,
        }
    }

    #[test]
    fn stablePrefixPacksIntoOneSegment() {
        let mut index = HashMap::new();
        index.insert("s1".into(), dummySnap("sp", "tl", &["u1"]));
        index.insert("s2".into(), dummySnap("sp", "tl", &["u1", "a1", "u2"]));
        index.insert("s3".into(), dummySnap("sp", "tl", &["u1", "a1", "u2", "a2", "u3"]));

        let branch = vec![
            asstTurn("t1", "b1", "s1"),
            asstTurn("t2", "b2", "s2"),
            asstTurn("t3", "b3", "s3"),
        ];

        let segs = buildSegments(&branch, &index, false);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].anchorIdx, 0);
        assert_eq!(segs[0].lastAsstIdx, 2);
    }

    #[test]
    fn toolsChangeStartsNewSegment() {
        let mut index = HashMap::new();
        index.insert("s1".into(), dummySnap("sp", "tlA", &["u1"]));
        index.insert("s2".into(), dummySnap("sp", "tlA", &["u1", "a1", "u2"]));
        index.insert("s3".into(), dummySnap("sp", "tlB", &["u1", "a1", "u2", "a2", "u3"]));

        let branch = vec![
            asstTurn("t1", "b1", "s1"),
            asstTurn("t2", "b2", "s2"),
            asstTurn("t3", "b3", "s3"),
        ];

        let segs = buildSegments(&branch, &index, false);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].lastAsstIdx, 1);
        assert_eq!(segs[1].anchorIdx, 2);
    }

    #[test]
    fn systemChangeStartsNewSegment() {
        let mut index = HashMap::new();
        index.insert("s1".into(), dummySnap("spA", "tl", &["u1"]));
        index.insert("s2".into(), dummySnap("spB", "tl", &["u1", "a1", "u2"]));

        let branch = vec![
            asstTurn("t1", "b1", "s1"),
            asstTurn("t2", "b2", "s2"),
        ];

        let segs = buildSegments(&branch, &index, false);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn compactionRewriteStartsNewSegment() {
        // s2.messages doesn't start with s1.messages because the earliest
        // user message got rewritten into a summary blob.
        let mut index = HashMap::new();
        index.insert("s1".into(), dummySnap("sp", "tl", &["u1", "a1", "u2", "a2", "u3"]));
        index.insert("s2".into(), dummySnap("sp", "tl", &["summary", "u3", "a3", "u4"]));

        let branch = vec![
            asstTurn("t1", "b1", "s1"),
            asstTurn("t2", "b2", "s2"),
        ];

        let segs = buildSegments(&branch, &index, false);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn cancelledTurnsExcludedByDefault() {
        let mut index = HashMap::new();
        index.insert("s1".into(), dummySnap("sp", "tl", &["u1"]));
        index.insert("s2".into(), dummySnap("sp", "tl", &["u1", "a1", "u2"]));

        // Second turn was cancelled mid-stream.
        let mut t2 = asstTurn("t2", "b2", "s2");
        t2.status = construct::transcript::TurnStatus::Cancelled;
        let branch = vec![asstTurn("t1", "b1", "s1"), t2];

        let segs = buildSegments(&branch, &index, false);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].lastAsstIdx, 0, "cancelled turn should not extend segment");

        let segs = buildSegments(&branch, &index, true);
        assert_eq!(segs[0].lastAsstIdx, 1, "with --include-cancelled, segment extends");
    }

    #[test]
    fn missingSnapshotIsSkipped() {
        let mut index = HashMap::new();
        index.insert("s1".into(), dummySnap("sp", "tl", &["u1"]));
        // s2 not inserted; branch references it but index doesn't have it.

        let branch = vec![
            asstTurn("t1", "b1", "s1"),
            asstTurn("t2", "b2", "s2"),
        ];

        let segs = buildSegments(&branch, &index, false);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].lastAsstIdx, 0);
    }
}
