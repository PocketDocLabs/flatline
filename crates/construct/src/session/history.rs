use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use tokio::sync::mpsc;

use super::Session;
use crate::control::LogEvent;
use crate::message::Message;
use crate::transcript::{self, SessionMeta, Transcript};

impl Session {
    /// Current topic label (for title bar display on resume).
    pub fn currentTopicLabel(&self) -> &str {
        self.topicTracker.currentLabel()
    }

    /// Load all transcript turns for this session.
    pub fn loadTranscript(&self) -> Result<Vec<crate::transcript::Turn>> {
        self.transcript.loadAll()
    }

    /// Load turns on the active branch by walking the parent-child chain.
    fn loadBranchTurns(&self) -> Result<Vec<crate::transcript::Turn>> {
        let headId = match &self.headTurnId {
            Some(id) => id.clone(),
            None => return Ok(Vec::new()),
        };
        let allTurns = self.transcript.loadAll()?;
        let turnMap: HashMap<&str, &crate::transcript::Turn> =
            allTurns.iter().map(|t| (t.id.as_str(), t)).collect();

        let mut chain = Vec::new();
        let mut current: Option<&str> = Some(&headId);
        while let Some(id) = current {
            if let Some(turn) = turnMap.get(id) {
                if !matches!(turn.role, crate::transcript::TurnRole::System) {
                    chain.push((*turn).clone());
                }
                current = turn.parentId.as_deref();
            } else {
                break;
            }
        }
        chain.reverse();
        Ok(chain)
    }

    /// Rebuild the topic tracker from the active branch turns.
    ///
    /// Called after rewind or fork-switch so topic state reflects only the
    /// active branch. Labels are sourced from the union of the live tracker
    /// and the on-disk `meta.topics`.
    pub(super) fn rebuildTopicTracker(&mut self) {
        let branchTurns = self.loadBranchTurns().unwrap_or_default();

        let mut labelSources: Vec<crate::topic::TopicInfo> = self.topicTracker.topics().to_vec();
        if let Ok(meta) = Transcript::loadMeta(self.transcript.sessionDir()) {
            for t in meta.topics {
                if !labelSources.iter().any(|x| x.topicId == t.topicId) {
                    labelSources.push(t);
                }
            }
        }

        let rebuilt = crate::topic::rebuildTopicInfos(&branchTurns, &labelSources);
        self.topicTracker.restoreState(rebuilt);
        self.transcript
            .setTopicId(self.topicTracker.currentTopicId());
    }

    /// Load turns for display — extends past the current head through any
    /// un-branched continuation. Once the user sends a new message (creating
    /// a second child at the head), this collapses to match `loadBranchTurns`.
    pub fn loadDisplayTurns(&self) -> Result<Vec<crate::transcript::Turn>> {
        let tipId = match self.findChainTip() {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };
        let allTurns = self.transcript.loadAll()?;
        let turnMap: HashMap<&str, &crate::transcript::Turn> =
            allTurns.iter().map(|t| (t.id.as_str(), t)).collect();

        let mut chain = Vec::new();
        let mut current: Option<&str> = Some(&tipId);
        while let Some(id) = current {
            if let Some(turn) = turnMap.get(id) {
                if !matches!(turn.role, crate::transcript::TurnRole::System) {
                    chain.push((*turn).clone());
                }
                current = turn.parentId.as_deref();
            } else {
                break;
            }
        }
        chain.reverse();
        Ok(chain)
    }

    /// Walk forward from the current head through single-child continuations.
    /// Returns the tip of the un-branched chain, or the head itself if it has
    /// 0 or 2+ children.
    fn findChainTip(&self) -> Option<String> {
        let headId = self.headTurnId.as_ref()?;
        let allTurns = self.transcript.loadAll().ok()?;

        let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
        for turn in &allTurns {
            if let Some(ref pid) = turn.parentId {
                children.entry(pid.as_str()).or_default().push(&turn.id);
            }
        }

        let mut current = headId.as_str();
        loop {
            match children.get(current) {
                Some(kids) if kids.len() == 1 => current = kids[0],
                _ => break,
            }
        }

        Some(current.to_string())
    }

    /// Derive compaction markers from the compaction log.
    ///
    /// Returns `(stage, blockIdx)` pairs for each stage that has replaced
    /// whole blocks. The block index is relative to the transcript's block
    /// sequence (0 = first block).
    pub fn compactionMarkers(&self) -> Vec<(String, usize)> {
        let ops = match self.compactionLog.loadAll() {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        let mut markers: Vec<(String, usize)> = Vec::new();
        let mut hasS2 = false;
        let mut hasS3 = false;
        let mut hasS4 = false;

        for op in &ops {
            match op {
                crate::compaction::CompactionOp::BlockCompact { .. } => hasS2 = true,
                crate::compaction::CompactionOp::TopicCompact { .. } => hasS3 = true,
                crate::compaction::CompactionOp::FullCompact { .. } => hasS4 = true,
                _ => {}
            }
        }

        if hasS4 {
            markers.push(("S4".into(), 0));
        } else if hasS3 {
            markers.push(("S3".into(), 0));
        } else if hasS2 {
            markers.push(("S2".into(), 0));
        }

        markers
    }

    /// List available sessions, optionally filtered by project directory.
    pub fn listSessions(projectDir: Option<&Path>) -> Result<Vec<SessionMeta>> {
        transcript::listSessions(projectDir.and_then(|p| p.to_str()))
    }

    /// Persist session metadata to disk. Called after each user message
    /// so that `/resume` can discover and list sessions.
    pub(super) fn updateMeta(&self) {
        let existingMeta = Transcript::loadMeta(self.transcript.sessionDir()).ok();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let projectDir = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let meta = SessionMeta {
            sessionId: self.transcript.sessionId.clone(),
            projectDir,
            createdAt: existingMeta.as_ref().map(|m| m.createdAt).unwrap_or(now),
            updatedAt: now,
            name: existingMeta.as_ref().and_then(|m| m.name.clone()),
            topicLabels: self.topicTracker.topicLabels(),
            topics: self.topicTracker.topics().to_vec(),
            headTurn: self.headTurnId.clone(),
            forks: existingMeta.map(|m| m.forks).unwrap_or_default(),
            totalCost: self.costTracker.sessionCost(),
        };
        if let Err(e) = self.transcript.writeMeta(&meta) {
            tracing::warn!("meta write failed: {e}");
        }
    }

    /// Build context state for the /context display.
    pub fn buildContextState(&self) -> crate::context::ContextState {
        let input = crate::context::BuildStateInput {
            contextWindow: self.config.heavy.contextWindow,
            compactionLog: &self.compactionLog,
            reportedTokens: self.compactionTracker.lastTokens(),
            transcript: &self.transcript,
            headTurnId: self.headTurnId.as_deref().unwrap_or(""),
        };
        crate::context::buildState(&input)
    }

    /// Format the list of saved sessions as a text listing (for `/resume` without id).
    pub fn listSessionsText(&self) -> String {
        match transcript::listSessions(None) {
            Ok(sessions) => {
                if sessions.is_empty() {
                    return "No saved sessions found.".to_string();
                }
                let mut output = String::from("Available sessions:\n\n");
                for (i, meta) in sessions.iter().take(20).enumerate() {
                    let name = meta.name.as_deref().unwrap_or("unnamed");
                    let topics = if meta.topicLabels.is_empty() {
                        String::new()
                    } else {
                        format!(" \u{2014} {}", meta.topicLabels.join(", "))
                    };
                    output.push_str(&format!(
                        "{}. {} [{}]{}\n   {}\n\n",
                        i + 1,
                        meta.sessionId,
                        name,
                        topics,
                        meta.projectDir,
                    ));
                }
                output
            }
            Err(e) => format!("Failed to list sessions: {e}"),
        }
    }

    /// Rewind conversation to a prior turn.
    ///
    /// If the user has sent messages on the current branch, the current
    /// state is saved as a fork before rewinding.
    pub async fn rewind(
        &mut self,
        targetTurnId: &str,
        saveFork: bool,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        let mut meta = match Transcript::loadMeta(self.transcript.sessionDir()) {
            Ok(m) => m,
            Err(e) => return format!("Failed to load session metadata: {e}"),
        };

        if saveFork {
            self.maybeSaveFork(&mut meta);
        }

        self.headTurnId = Some(targetTurnId.to_string());
        meta.headTurn = self.headTurnId.clone();

        if let Err(e) = self.transcript.writeMeta(&meta) {
            return format!("Failed to save rewind: {e}");
        }

        if let Ok(allTurns) = self.transcript.loadAll()
            && let Some(turn) = allTurns.iter().find(|t| t.id == targetTurnId)
        {
            self.transcript.setHead(targetTurnId, &turn.blockId);
        }

        match crate::context::reconstruct(&self.transcript, &self.compactionLog, targetTurnId) {
            Ok(h) => {
                self.history = vec![Message::System {
                    content: self.systemPrompt.clone(),
                }];
                self.history.extend(h);
            }
            Err(e) => return format!("Failed to reconstruct history after rewind: {e}"),
        }

        self.filesRead.clear();
        self.compactionTracker.clearExhaustion();

        if let Some(handle) = self.pendingTopicEval.take() {
            handle.abort();
        }
        self.pendingTopicBlockId = None;
        self.rebuildTopicTracker();

        let branchTurns = self.loadBranchTurns().unwrap_or_default();
        let markers = self.compactionMarkers();
        let _ = logTx
            .send(LogEvent::Rewound {
                targetTurnId: targetTurnId.to_string(),
            })
            .await;
        let _ = logTx
            .send(LogEvent::SessionRestored {
                turns: branchTurns,
                markers,
            })
            .await;

        let label = self.topicTracker.currentLabel();
        if !label.is_empty() {
            let _ = logTx
                .send(LogEvent::TopicChanged {
                    label: label.to_string(),
                })
                .await;
        }

        format!("Rewound to {targetTurnId}")
    }

    /// Switch to a previously saved fork.
    pub async fn switchFork(&mut self, forkId: &str, logTx: &mpsc::Sender<LogEvent>) -> String {
        let mut meta = match Transcript::loadMeta(self.transcript.sessionDir()) {
            Ok(m) => m,
            Err(e) => return format!("Failed to load session metadata: {e}"),
        };

        let forkIdx = match meta.forks.iter().position(|f| f.id == forkId) {
            Some(i) => i,
            None => return format!("Fork {forkId} not found."),
        };

        self.maybeSaveFork(&mut meta);

        let fork = meta.forks.remove(forkIdx);
        self.headTurnId = Some(fork.headTurn.clone());
        meta.headTurn = self.headTurnId.clone();

        if let Err(e) = self.transcript.writeMeta(&meta) {
            return format!("Failed to save fork switch: {e}");
        }

        if let Ok(allTurns) = self.transcript.loadAll()
            && let Some(turn) = allTurns.iter().find(|t| t.id == fork.headTurn)
        {
            self.transcript.setHead(&fork.headTurn, &turn.blockId);
        }

        match crate::context::reconstruct(&self.transcript, &self.compactionLog, &fork.headTurn) {
            Ok(h) => {
                self.history = vec![Message::System {
                    content: self.systemPrompt.clone(),
                }];
                self.history.extend(h);
            }
            Err(e) => return format!("Failed to reconstruct after fork switch: {e}"),
        }

        self.filesRead.clear();
        self.compactionTracker.clearExhaustion();

        if let Some(handle) = self.pendingTopicEval.take() {
            handle.abort();
        }
        self.pendingTopicBlockId = None;
        self.rebuildTopicTracker();

        let branchTurns = self.loadBranchTurns().unwrap_or_default();
        let markers = self.compactionMarkers();
        let _ = logTx
            .send(LogEvent::Rewound {
                targetTurnId: fork.headTurn,
            })
            .await;
        let _ = logTx
            .send(LogEvent::SessionRestored {
                turns: branchTurns,
                markers,
            })
            .await;

        let label = self.topicTracker.currentLabel();
        if !label.is_empty() {
            let _ = logTx
                .send(LogEvent::TopicChanged {
                    label: label.to_string(),
                })
                .await;
        }

        format!("Switched to fork: {}", fork.label)
    }

    /// Save the current branch as a fork if the user sent messages.
    fn maybeSaveFork(&self, meta: &mut SessionMeta) {
        let branchTurns = match self.loadBranchTurns() {
            Ok(t) => t,
            Err(_) => return,
        };

        let hasUserTurns = branchTurns
            .iter()
            .any(|t| matches!(t.role, crate::transcript::TurnRole::User));

        if !hasUserTurns {
            return;
        }

        let currentHead = match &self.headTurnId {
            Some(id) => id.clone(),
            None => return,
        };

        let label = branchTurns
            .iter()
            .find(|t| matches!(t.role, crate::transcript::TurnRole::User))
            .map(|t| {
                let first = t
                    .content
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("");
                let trimmed = first.trim();
                if trimmed.len() > 60 {
                    format!("{}\u{2026}", &trimmed[..trimmed.floor_char_boundary(59)])
                } else {
                    trimmed.to_string()
                }
            })
            .unwrap_or_else(|| "unnamed fork".to_string());

        let forkId = crate::transcript::randomHexId("fork");
        meta.forks.push(crate::transcript::Fork {
            id: forkId,
            label,
            headTurn: currentHead,
            createdAt: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
    }

    /// List available forks.
    pub fn listForks(&self) -> Vec<crate::transcript::Fork> {
        Transcript::loadMeta(self.transcript.sessionDir())
            .map(|m| m.forks)
            .unwrap_or_default()
    }

    /// Format forks for inline display.
    pub fn formatForksListing(&self) -> String {
        let forks = self.listForks();
        if forks.is_empty() {
            return "No saved forks.".to_string();
        }

        let mut out = format!("**Saved forks** ({})\n", forks.len());
        for fork in &forks {
            let age = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .saturating_sub(fork.createdAt);
            let agoStr = if age < 60 {
                "just now".to_string()
            } else if age < 3600 {
                format!("{}m ago", age / 60)
            } else if age < 86400 {
                format!("{}h ago", age / 3600)
            } else {
                format!("{}d ago", age / 86400)
            };
            out.push_str(&format!(
                "  `{}` \u{2014} {} ({})\n",
                fork.id, fork.label, agoStr
            ));
        }
        out.push_str("\nSwitch with `/forks <id>`");
        out
    }
}
