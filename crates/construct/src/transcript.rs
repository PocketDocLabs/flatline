//! Transcript storage — append-only SQLite record of every conversation turn.
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
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::Connection;
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
    /// A wake-injected synthetic user message. Stored separately from
    /// `User` so resume can render it as a notice instead of a real user
    /// turn. The model still receives it as user-shaped content via
    /// `context::reconstruct`.
    Wake,
}

/// Outcome state for an assistant turn. Other roles always use `Completed`.
///
/// Used at export time to filter training targets — cancelled or errored
/// assistant responses shouldn't become SFT labels, since their content is
/// partial or diverged from what the user wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum TurnStatus {
    /// Stream finished normally. Default for turns missing the field.
    #[default]
    Completed,
    /// User interrupted mid-stream. Content may be partial.
    Cancelled,
    /// Transport-level error cut the stream short (broken pipe, network
    /// drop, mid-stream provider error). Content may be partial; a later
    /// turn may want to resume from where this one was cut off.
    Errored,
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
    /// Tool-call display metadata persisted for session resume. This is
    /// semantic state, not ephemeral UI state like collapse/copy/scroll.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub toolMeta: Option<ToolCallMeta>,
    /// USD cost of this turn (assistant turns only).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cost: Option<f64>,
    /// Prompt tokens reported by the API (assistant turns only).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub promptTokens: Option<usize>,
    /// Completion tokens reported by the API (assistant turns only).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub completionTokens: Option<usize>,
    /// Model identifier used to generate this turn (assistant turns only).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model: Option<String>,
    /// Provider-reported finish reason, e.g. "stop", "tool_calls", "length".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub finishReason: Option<String>,
    /// Content-addressed hash of the request snapshot that produced this turn.
    /// See `snapshot::RequestSnapshot`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub snapshotHash: Option<String>,
    /// Outcome state. Omitted on disk when equal to the default (`Completed`).
    #[serde(default, skip_serializing_if = "isCompleted")]
    pub status: TurnStatus,
}

fn isCompleted(s: &TurnStatus) -> bool {
    matches!(s, TurnStatus::Completed)
}

/// Metadata attached to an assistant turn.
#[derive(Debug, Clone, Default)]
pub struct AssistantMeta<'a> {
    pub reasoning: Option<&'a str>,
    pub cost: Option<f64>,
    pub promptTokens: Option<usize>,
    pub completionTokens: Option<usize>,
    pub model: Option<&'a str>,
    pub finishReason: Option<&'a str>,
    pub snapshotHash: Option<&'a str>,
    pub status: TurnStatus,
}

/// Persisted display-relevant metadata for a tool call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallMeta {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub diff: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub outcome: Option<ToolCallOutcome>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub review: Option<crate::control::AutoReviewReport>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub startedAtMs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub completedAtMs: Option<u64>,
}

/// Persisted permission/execution outcome for a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallOutcome {
    Approved,
    Denied,
    Aborted,
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
    /// Total USD cost accumulated in this session.
    #[serde(default)]
    pub totalCost: f64,
}

/// Handle to an open session transcript.
pub struct Transcript {
    pub sessionId: String,
    sessionDir: PathBuf,
    conn: Arc<Mutex<Connection>>,
    /// Most recently written turn ID (for within-block parent chaining).
    lastTurnId: Option<String>,
    currentBlockId: String,
    currentTopicId: String,
}

impl Transcript {
    /// Create a new transcript in a fresh session directory.
    pub fn create(sessionId: &str) -> Result<Self> {
        let dir = sessionsDir().join(sessionId);
        let conn = crate::storage::openSessionDb(&dir)?;

        Ok(Self {
            sessionId: sessionId.to_string(),
            sessionDir: dir,
            conn: Arc::new(Mutex::new(conn)),
            lastTurnId: None,
            currentBlockId: String::new(),
            currentTopicId: String::new(),
        })
    }

    /// Create a transcript at an explicit directory (for tests).
    pub fn createAt(dir: &Path, sessionId: &str) -> Result<Self> {
        let conn = crate::storage::openSessionDb(dir)?;

        Ok(Self {
            sessionId: sessionId.to_string(),
            sessionDir: dir.to_path_buf(),
            conn: Arc::new(Mutex::new(conn)),
            lastTurnId: None,
            currentBlockId: String::new(),
            currentTopicId: String::new(),
        })
    }

    /// Open an existing transcript for append.
    pub fn open(sessionId: &str) -> Result<Self> {
        let dir = sessionsDir().join(sessionId);
        let conn = crate::storage::openSessionDb(&dir)?;

        // Scan to find the last turn's ID and block.
        let mut lastTurnId: Option<String> = None;
        let mut lastBlockId = String::new();
        let mut lastTopicId = String::new();

        for turn in crate::storage::loadTurns(&conn)? {
            lastTurnId = Some(turn.id.clone());
            lastBlockId = turn.blockId.clone();
            lastTopicId = turn.topicId.clone();
        }

        Ok(Self {
            sessionId: sessionId.to_string(),
            sessionDir: dir,
            conn: Arc::new(Mutex::new(conn)),
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
        crate::storage::insertTurn(&self.conn.lock().unwrap(), turn)?;
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
            toolMeta: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
        };
        self.writeTurn(&turn)
    }

    /// Record a wake-injected synthetic user message. Starts a new block
    /// (wakes are conversational boundaries — they begin a fresh turn).
    /// The content is the formatted `<wakes>…</wakes>` envelope; the model
    /// sees it as user-shaped via `context::reconstruct`.
    pub fn recordWake(&mut self, content: &str, parentId: Option<&str>) -> Result<String> {
        let blockId = randomHexId("b");
        self.currentBlockId = blockId.clone();
        let turn = Turn {
            id: randomHexId("t"),
            blockId,
            topicId: self.currentTopicId.clone(),
            role: TurnRole::Wake,
            content: content.to_string(),
            ts: Self::now(),
            parentId: parentId.map(|s| s.to_string()),
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments: None,
            toolMeta: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
        };
        self.writeTurn(&turn)
    }

    /// Record an assistant response (in the current block).
    pub fn recordAssistant(&mut self, content: &str, meta: AssistantMeta<'_>) -> Result<String> {
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
            reasoning: meta.reasoning.map(|s| s.to_string()),
            attachments: None,
            toolMeta: None,
            cost: meta.cost,
            promptTokens: meta.promptTokens,
            completionTokens: meta.completionTokens,
            model: meta.model.map(|s| s.to_string()),
            finishReason: meta.finishReason.map(|s| s.to_string()),
            snapshotHash: meta.snapshotHash.map(|s| s.to_string()),
            status: meta.status,
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
        tracing::debug!(%callId, tool = %toolName, "recording tool call");
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
            toolMeta: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
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
        tracing::debug!(%callId, contentLen = content.len(), "recording tool result");
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
            toolMeta: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
        };
        self.writeTurn(&turn)
    }

    /// Load all turns from the transcript file.
    pub fn loadAll(&self) -> Result<Vec<Turn>> {
        crate::storage::loadTurns(&self.conn.lock().unwrap())
    }

    /// Update persisted semantic metadata for a recorded tool call.
    pub fn updateToolCallMeta(
        &mut self,
        callId: &str,
        f: impl FnOnce(&mut ToolCallMeta),
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let mut turns = crate::storage::loadTurns(&conn)?;
        let Some(turn) = turns.iter_mut().find(|turn| {
            matches!(turn.role, TurnRole::ToolCall) && turn.toolCallId.as_deref() == Some(callId)
        }) else {
            return Ok(());
        };
        let meta = turn.toolMeta.get_or_insert_with(ToolCallMeta::default);
        f(meta);
        crate::storage::updateTurn(&conn, turn)
    }

    /// Write session metadata to SQLite.
    pub fn writeMeta(&self, meta: &SessionMeta) -> Result<()> {
        crate::storage::upsertMeta(&self.conn.lock().unwrap(), meta)
    }

    /// Load session metadata from a session directory.
    pub fn loadMeta(sessionDir: &Path) -> Result<SessionMeta> {
        let conn = crate::storage::openSessionDb(sessionDir)?;
        crate::storage::loadMeta(&conn)?.context("missing session metadata")
    }

    /// Export the SQLite transcript as legacy JSONL text for debugging.
    pub fn exportJsonl(&self) -> Result<String> {
        let mut out = String::new();
        for turn in self.loadAll()? {
            out.push_str(&serde_json::to_string(&turn)?);
            out.push('\n');
        }
        Ok(out)
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
///
/// Overridable via the `FLATLINE_SESSIONS_DIR` env var — used by integration
/// tests and advanced workflows that need to isolate sessions from the normal
/// user-data location.
pub fn sessionsDir() -> PathBuf {
    if let Ok(explicit) = std::env::var("FLATLINE_SESSIONS_DIR")
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
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
        let dbPath = crate::storage::dbPath(&entry.path());
        let metaPath = entry.path().join("meta.json");
        if !dbPath.exists() && !metaPath.exists() {
            continue;
        }
        match Transcript::loadMeta(&entry.path()) {
            Ok(meta) => {
                if let Some(filter) = projectDir
                    && meta.projectDir != filter
                {
                    continue;
                }
                sessions.push(meta);
            }
            Err(_) => continue,
        }
    }

    // Sort by updatedAt descending (most recent first).
    sessions.sort_by_key(|session| std::cmp::Reverse(session.updatedAt));
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
            attachments: Some(vec![TurnAttachment {
                mimeType: "image/png".into(),
                data: "iVBORw0KGgo=".into(),
            }]),
            toolMeta: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
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
        assert!(turn.toolMeta.is_none());
        assert_eq!(turn.content, "hello");
    }

    /// Create a transcript in a temp directory for testing.
    fn tempTranscript(dir: &std::path::Path) -> Transcript {
        Transcript::createAt(dir, "test").unwrap()
    }

    #[test]
    fn recordUserPersistsAttachments() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut transcript = tempTranscript(dir.path());

        let atts = vec![TurnAttachment {
            mimeType: "image/jpeg".into(),
            data: "/9j/4AAQ".into(),
        }];
        transcript
            .recordUser("check this", None, Some(atts))
            .unwrap();

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
        transcript
            .recordToolResult("call_1", "[screenshot.png]", Some(atts))
            .unwrap();

        let turns = transcript.loadAll().unwrap();
        assert_eq!(turns.len(), 2);
        let toolTurn = &turns[1];
        assert_eq!(toolTurn.content, "[screenshot.png]");
        let atts = toolTurn.attachments.as_ref().unwrap();
        assert_eq!(atts.len(), 1);
        assert_eq!(atts[0].mimeType, "image/png");
    }

    #[test]
    fn updateToolCallMetaPersistsReviewerOutcomeAndTiming() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut transcript = tempTranscript(dir.path());

        transcript.recordUser("start", None, None).unwrap();
        transcript
            .recordToolCall(
                "call_1",
                "shell",
                &serde_json::json!({"command": "echo hi"}),
            )
            .unwrap();
        transcript
            .updateToolCallMeta("call_1", |meta| {
                meta.summary = Some("Run: echo hi".into());
                meta.diff = Some("--- before\n+++ after\n@@\n-old\n+new".into());
                meta.outcome = Some(ToolCallOutcome::Approved);
                meta.startedAtMs = Some(100);
                meta.completedAtMs = Some(250);
                meta.review = Some(crate::control::AutoReviewReport {
                    decision: "allow".into(),
                    raiseToUser: "none".into(),
                    risk: "low".into(),
                    authorization: "inline".into(),
                    reason: "safe".into(),
                    messageToAgent: String::new(),
                });
            })
            .unwrap();

        let turns = transcript.loadAll().unwrap();
        let meta = turns[1].toolMeta.as_ref().expect("tool meta");
        assert_eq!(meta.summary.as_deref(), Some("Run: echo hi"));
        assert_eq!(
            meta.diff.as_deref(),
            Some("--- before\n+++ after\n@@\n-old\n+new")
        );
        assert_eq!(meta.outcome, Some(ToolCallOutcome::Approved));
        assert_eq!(meta.startedAtMs, Some(100));
        assert_eq!(meta.completedAtMs, Some(250));
        assert_eq!(meta.review.as_ref().unwrap().reason, "safe");
    }
}
