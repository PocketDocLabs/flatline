//! Transcript storage — append-only JSONL record of every conversation turn.
//!
//! The transcript is the permanent source of truth. All derived state
//! (compaction summaries, live context) is reconstructable from the
//! transcript + compaction log.
//!
//! Turns form a parent-child tree: each turn points to its predecessor
//! via `parentId`. Branching (rewind) creates new chains from shared
//! ancestors. Reconstruction walks the tree from a head turn backward.
//!
//! # Public API
//! - [`Transcript`] — session transcript handle
//! - [`SessionMeta`] — session metadata for discovery/resume
//! - [`TurnRole`] — role tag for transcript entries
//! - [`newSessionId`] — generate a fresh session ID
//! - [`sessionsDir`] — path to the sessions directory
//! - [`listSessions`] — enumerate available sessions
//!
//! # Dependencies
//! `serde`, `serde_json`, `dirs`

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Role of a turn in the transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    User,
    Assistant,
    ToolCall,
    ToolResult,
    System,
}

/// A single turn in the transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Turn {
    pub id: String,
    pub blockId: String,
    #[serde(default)]
    pub topicId: String,
    pub role: TurnRole,
    pub content: String,
    pub ts: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parentId: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toolCallId: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning: Option<String>,
    /// Image attachments persisted for session resume.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub attachments: Option<Vec<TurnAttachment>>,
}

/// An image attachment stored in the transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnAttachment {
    pub mimeType: String,
    /// Base64-encoded image data.
    pub data: String,
}

/// A saved conversation fork — a branch you can switch back to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fork {
    /// Unique fork ID.
    pub id: String,
    /// Human-readable label (first user message on this branch, truncated).
    pub label: String,
    /// Head turn ID of this branch.
    pub headTurn: String,
    /// Unix timestamp.
    pub createdAt: u64,
}

/// Session metadata — persisted as meta.json alongside the transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMeta {
    pub sessionId: String,
    pub projectDir: String,
    pub createdAt: u64,
    pub updatedAt: u64,
    pub name: Option<String>,
    pub topicLabels: Vec<String>,
    /// Full topic metadata (startBlock, blockCount). Backward-compatible:
    /// old sessions without this field deserialize as empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub topics: Vec<crate::topic::TopicInfo>,
    #[serde(default)]
    pub headTurn: Option<String>,
    #[serde(default)]
    pub forks: Vec<Fork>,
}

/// Handle to an open session transcript.
pub struct Transcript {
    pub sessionId: String,
    sessionDir: PathBuf,
    writer: BufWriter<fs::File>,
    /// Most recently written turn ID (for within-block parent chaining).
    lastTurnId: Option<String>,
    currentBlockId: String,
    currentTopicId: String,
}

impl Transcript {
    /// Create a new transcript in a fresh session directory.
    pub fn create(sessionId: &str) -> Result<Self> {
        let dir = sessionsDir().join(sessionId);
        fs::create_dir_all(&dir)
            .with_context(|| format!("create session dir: {}", dir.display()))?;

        let path = dir.join("transcript.jsonl");
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open transcript: {}", path.display()))?;

        Ok(Self {
            sessionId: sessionId.to_string(),
            sessionDir: dir,
            writer: BufWriter::new(file),
            lastTurnId: None,
            currentBlockId: String::new(),
            currentTopicId: String::new(),
        })
    }

    /// Create a transcript at an explicit directory (for tests).
    pub fn createAt(dir: &Path, sessionId: &str) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create session dir: {}", dir.display()))?;

        let path = dir.join("transcript.jsonl");
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open transcript: {}", path.display()))?;

        Ok(Self {
            sessionId: sessionId.to_string(),
            sessionDir: dir.to_path_buf(),
            writer: BufWriter::new(file),
            lastTurnId: None,
            currentBlockId: String::new(),
            currentTopicId: String::new(),
        })
    }

    /// Open an existing transcript for append.
    pub fn open(sessionId: &str) -> Result<Self> {
        let dir = sessionsDir().join(sessionId);
        let path = dir.join("transcript.jsonl");
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open transcript for append: {}", path.display()))?;

        // Scan to find the last turn's ID and block.
        let existingContent = fs::read_to_string(&path).unwrap_or_default();
        let mut lastTurnId: Option<String> = None;
        let mut lastBlockId = String::new();
        let mut lastTopicId = String::new();

        for line in existingContent.lines() {
            if let Ok(turn) = serde_json::from_str::<Turn>(line) {
                lastTurnId = Some(turn.id.clone());
                lastBlockId = turn.blockId.clone();
                lastTopicId = turn.topicId.clone();
            }
        }

        Ok(Self {
            sessionId: sessionId.to_string(),
            sessionDir: dir,
            writer: BufWriter::new(file),
            lastTurnId,
            currentBlockId: lastBlockId,
            currentTopicId: lastTopicId,
        })
    }

    /// Path to the session directory.
    pub fn sessionDir(&self) -> &Path {
        &self.sessionDir
    }

    /// Current block ID.
    pub fn currentBlock(&self) -> &str {
        &self.currentBlockId
    }

    /// The turn ID of the most recently written turn.
    pub fn lastTurnId(&self) -> Option<String> {
        self.lastTurnId.clone()
    }

    /// Override the active head after loading meta (for branching on resume).
    pub fn setHead(&mut self, turnId: &str, blockId: &str) {
        self.lastTurnId = Some(turnId.to_string());
        self.currentBlockId = blockId.to_string();
    }

    /// Set the topic ID for subsequent turns.
    pub fn setTopicId(&mut self, topicId: &str) {
        self.currentTopicId = topicId.to_string();
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn writeTurn(&mut self, turn: &Turn) -> Result<String> {
        let line = serde_json::to_string(turn)?;
        writeln!(self.writer, "{line}")?;
        self.writer.flush()?;
        let id = turn.id.clone();
        self.lastTurnId = Some(id.clone());
        Ok(id)
    }

    /// Record a user message. Starts a new block.
    ///
    /// `parentId` is the turn to branch from. `None` for the first message
    /// in a session; `Some(headTurnId)` for continuing or branching.
    pub fn recordUser(
        &mut self,
        content: &str,
        parentId: Option<&str>,
        attachments: Option<Vec<TurnAttachment>>,
    ) -> Result<String> {
        let blockId = randomHexId("b");
        self.currentBlockId = blockId.clone();
        let turn = Turn {
            id: randomHexId("t"),
            blockId,
            topicId: self.currentTopicId.clone(),
            role: TurnRole::User,
            content: content.to_string(),
            ts: Self::now(),
            parentId: parentId.map(|s| s.to_string()),
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments,
        };
        self.writeTurn(&turn)
    }

    /// Record an assistant response (in the current block).
    pub fn recordAssistant(&mut self, content: &str, reasoning: Option<&str>) -> Result<String> {
        let turn = Turn {
            id: randomHexId("t"),
            blockId: self.currentBlockId.clone(),
            topicId: self.currentTopicId.clone(),
            role: TurnRole::Assistant,
            content: content.to_string(),
            ts: Self::now(),
            parentId: self.lastTurnId.clone(),
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: reasoning.map(|s| s.to_string()),
            attachments: None,
        };
        self.writeTurn(&turn)
    }

    /// Record a tool call.
    pub fn recordToolCall(
        &mut self,
        callId: &str,
        toolName: &str,
        args: &serde_json::Value,
    ) -> Result<String> {
        let turn = Turn {
            id: randomHexId("t"),
            blockId: self.currentBlockId.clone(),
            topicId: self.currentTopicId.clone(),
            role: TurnRole::ToolCall,
            content: String::new(),
            ts: Self::now(),
            parentId: self.lastTurnId.clone(),
            tool: Some(toolName.to_string()),
            args: Some(args.clone()),
            toolCallId: Some(callId.to_string()),
            reasoning: None,
            attachments: None,
        };
        self.writeTurn(&turn)
    }

    /// Record a tool result.
    pub fn recordToolResult(
        &mut self,
        callId: &str,
        content: &str,
        attachments: Option<Vec<TurnAttachment>>,
    ) -> Result<String> {
        let turn = Turn {
            id: randomHexId("t"),
            blockId: self.currentBlockId.clone(),
            topicId: self.currentTopicId.clone(),
            role: TurnRole::ToolResult,
            content: content.to_string(),
            ts: Self::now(),
            parentId: self.lastTurnId.clone(),
            tool: None,
            args: None,
            toolCallId: Some(callId.to_string()),
            reasoning: None,
            attachments,
        };
        self.writeTurn(&turn)
    }

    /// Load all turns from the transcript file.
    pub fn loadAll(&self) -> Result<Vec<Turn>> {
        let path = self.sessionDir.join("transcript.jsonl");
        let content = fs::read_to_string(&path)
            .with_context(|| format!("read transcript: {}", path.display()))?;

        let mut turns = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let turn: Turn = serde_json::from_str(line)
                .with_context(|| "parse transcript line")?;
            turns.push(turn);
        }
        Ok(turns)
    }

    /// Write session metadata to meta.json.
    pub fn writeMeta(&self, meta: &SessionMeta) -> Result<()> {
        let path = self.sessionDir.join("meta.json");
        let content = serde_json::to_string_pretty(meta)?;
        fs::write(&path, content)
            .with_context(|| format!("write meta: {}", path.display()))?;
        Ok(())
    }

    /// Load session metadata from a session directory.
    pub fn loadMeta(sessionDir: &Path) -> Result<SessionMeta> {
        let path = sessionDir.join("meta.json");
        let content = fs::read_to_string(&path)
            .with_context(|| format!("read meta: {}", path.display()))?;
        let meta: SessionMeta = serde_json::from_str(&content)?;
        Ok(meta)
    }
}

/// Generate a random hex ID with the given prefix (e.g. "t" → "t_a3f8b2c1").
pub fn randomHexId(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32;
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mixed = ts ^ seq ^ (std::process::id());
    format!("{prefix}_{mixed:08x}")
}

/// Generate a new random session ID.
pub fn newSessionId() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let rand: u16 = (ts as u16) ^ (std::process::id() as u16);
    format!("ses_{ts:x}_{rand:04x}")
}

/// Path to the sessions directory.
pub fn sessionsDir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("flatline")
        .join("sessions")
}

/// List available sessions, optionally filtered to a project directory.
pub fn listSessions(projectDir: Option<&str>) -> Result<Vec<SessionMeta>> {
    let dir = sessionsDir();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let metaPath = entry.path().join("meta.json");
        if !metaPath.exists() {
            continue;
        }
        match Transcript::loadMeta(&entry.path()) {
            Ok(meta) => {
                if let Some(filter) = projectDir {
                    if meta.projectDir != filter {
                        continue;
                    }
                }
                sessions.push(meta);
            }
            Err(_) => continue,
        }
    }

    // Sort by updatedAt descending (most recent first).
    sessions.sort_by(|a, b| b.updatedAt.cmp(&a.updatedAt));
    Ok(sessions)
}

/// Walk the parent chain from headTurnId to root, return turns in chronological order.
///
/// Used by S2/S3 compaction to operate only on the active branch after rewinds.
pub fn walkBranchTurns(allTurns: &[Turn], headTurnId: &str) -> Vec<Turn> {
    let turnMap: std::collections::HashMap<&str, &Turn> =
        allTurns.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut chain = Vec::new();
    let mut current: Option<&str> = Some(headTurnId);
    while let Some(id) = current {
        if let Some(turn) = turnMap.get(id) {
            chain.push((*turn).clone());
            current = turn.parentId.as_deref();
        } else {
            break;
        }
    }
    chain.reverse();
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turnAttachmentRoundTrip() {
        let turn = Turn {
            id: "t_test".into(),
            blockId: "b_test".into(),
            topicId: String::new(),
            role: TurnRole::User,
            content: "look at this image".into(),
            ts: 1000,
            parentId: None,
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments: Some(vec![
                TurnAttachment {
                    mimeType: "image/png".into(),
                    data: "iVBORw0KGgo=".into(),
                },
            ]),
        };

        let json = serde_json::to_string(&turn).unwrap();
        let restored: Turn = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.content, "look at this image");
        let atts = restored.attachments.unwrap();
        assert_eq!(atts.len(), 1);
        assert_eq!(atts[0].mimeType, "image/png");
        assert_eq!(atts[0].data, "iVBORw0KGgo=");
    }

    #[test]
    fn turnWithoutAttachmentsBackwardCompatible() {
        // Simulate old transcript line with no attachments field.
        let json = r#"{"id":"t_1","blockId":"b_1","topicId":"","role":"user","content":"hello","ts":1000}"#;
        let turn: Turn = serde_json::from_str(json).unwrap();
        assert!(turn.attachments.is_none());
        assert_eq!(turn.content, "hello");
    }

    /// Create a transcript in a temp directory for testing.
    fn tempTranscript(dir: &std::path::Path) -> Transcript {
        let path = dir.join("transcript.jsonl");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        Transcript {
            sessionId: "test".into(),
            sessionDir: dir.to_path_buf(),
            writer: std::io::BufWriter::new(file),
            lastTurnId: None,
            currentBlockId: String::new(),
            currentTopicId: String::new(),
        }
    }

    #[test]
    fn recordUserPersistsAttachments() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut transcript = tempTranscript(dir.path());

        let atts = vec![TurnAttachment {
            mimeType: "image/jpeg".into(),
            data: "/9j/4AAQ".into(),
        }];
        transcript.recordUser("check this", None, Some(atts)).unwrap();

        let turns = transcript.loadAll().unwrap();
        assert_eq!(turns.len(), 1);
        let atts = turns[0].attachments.as_ref().unwrap();
        assert_eq!(atts.len(), 1);
        assert_eq!(atts[0].mimeType, "image/jpeg");
        assert_eq!(atts[0].data, "/9j/4AAQ");
    }

    #[test]
    fn recordToolResultPersistsAttachments() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut transcript = tempTranscript(dir.path());

        // Need a user message first to start a block.
        transcript.recordUser("start", None, None).unwrap();

        let atts = vec![TurnAttachment {
            mimeType: "image/png".into(),
            data: "iVBOR=".into(),
        }];
        transcript.recordToolResult("call_1", "[screenshot.png]", Some(atts)).unwrap();

        let turns = transcript.loadAll().unwrap();
        assert_eq!(turns.len(), 2);
        let toolTurn = &turns[1];
        assert_eq!(toolTurn.content, "[screenshot.png]");
        let atts = toolTurn.attachments.as_ref().unwrap();
        assert_eq!(atts.len(), 1);
        assert_eq!(atts[0].mimeType, "image/png");
    }
}
