#![allow(non_snake_case)]

//! SQLite-backed per-session storage.
//!
//! The public transcript / compaction / snapshot APIs stay small, but their
//! source of truth is `session.sqlite` in the session directory. Legacy JSONL
//! files are imported once when a database is first opened.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, Row, params};

use crate::compaction::CompactionOp;
use crate::snapshot::RequestSnapshot;
use crate::transcript::{SessionMeta, Turn};

pub const DB_FILE: &str = "session.sqlite";

pub(crate) fn dbPath(sessionDir: &Path) -> std::path::PathBuf {
    sessionDir.join(DB_FILE)
}

pub fn ensureSessionDb(sessionDir: &Path) -> Result<()> {
    openSessionDb(sessionDir).map(drop)
}

pub(crate) fn openSessionDb(sessionDir: &Path) -> Result<Connection> {
    sqliteBlocking(|| {
        fs::create_dir_all(sessionDir)
            .with_context(|| format!("create session dir: {}", sessionDir.display()))?;
        let mut conn = Connection::open(dbPath(sessionDir))
            .with_context(|| format!("open sqlite session db in {}", sessionDir.display()))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        initSchema(&conn)?;
        importLegacyIfNeeded(&mut conn, sessionDir)?;
        Ok(conn)
    })
}

fn sqliteBlocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    if let Ok(handle) = tokio::runtime::Handle::try_current()
        && matches!(
            handle.runtime_flavor(),
            tokio::runtime::RuntimeFlavor::MultiThread
        )
    {
        return tokio::task::block_in_place(f);
    }
    f()
}

fn initSchema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '1');

        CREATE TABLE IF NOT EXISTS session_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            session_id TEXT NOT NULL,
            project_dir TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            name TEXT,
            head_turn TEXT,
            total_cost REAL NOT NULL DEFAULT 0,
            meta_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS turns (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            block_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            role TEXT NOT NULL,
            ts INTEGER NOT NULL,
            parent_id TEXT,
            tool TEXT,
            tool_call_id TEXT,
            snapshot_hash TEXT,
            turn_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_turns_parent ON turns(parent_id);
        CREATE INDEX IF NOT EXISTS idx_turns_block ON turns(block_id);
        CREATE INDEX IF NOT EXISTS idx_turns_topic ON turns(topic_id);
        CREATE INDEX IF NOT EXISTS idx_turns_role ON turns(role);
        CREATE INDEX IF NOT EXISTS idx_turns_snapshot ON turns(snapshot_hash);

        CREATE TABLE IF NOT EXISTS compaction_ops (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            after_turn TEXT NOT NULL,
            ts INTEGER NOT NULL,
            op_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_compaction_after_turn ON compaction_ops(after_turn);

        CREATE TABLE IF NOT EXISTS snapshots (
            hash TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS snapshot_blobs (
            hash TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            bytes BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snapshot_blobs_ns ON snapshot_blobs(namespace);

        CREATE TABLE IF NOT EXISTS terminal_runs (
            run_id TEXT PRIMARY KEY,
            terminal_name TEXT NOT NULL,
            command TEXT NOT NULL,
            purpose TEXT NOT NULL,
            impact TEXT NOT NULL,
            ephemeral INTEGER NOT NULL,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            status TEXT NOT NULL,
            exit_code INTEGER,
            line_count INTEGER NOT NULL DEFAULT 0,
            replay_blob BLOB NOT NULL DEFAULT X''
        );
        CREATE INDEX IF NOT EXISTS idx_terminal_runs_started ON terminal_runs(started_at);
        CREATE INDEX IF NOT EXISTS idx_terminal_runs_impact_status ON terminal_runs(impact, status);
        "#,
    )?;
    Ok(())
}

fn tableEmpty(conn: &Connection, table: &str) -> Result<bool> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let count: i64 = conn.query_row(&sql, [], |r| r.get(0))?;
    Ok(count == 0)
}

fn importLegacyIfNeeded(conn: &mut Connection, sessionDir: &Path) -> Result<()> {
    let tx = conn.transaction()?;

    if tableEmpty(&tx, "session_meta")? {
        let path = sessionDir.join("meta.json");
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("read legacy meta: {}", path.display()))?;
            let meta: SessionMeta = serde_json::from_str(&content)
                .with_context(|| format!("parse legacy meta: {}", path.display()))?;
            upsertMetaTx(&tx, &meta)?;
        }
    }

    if tableEmpty(&tx, "turns")? {
        let path = sessionDir.join("transcript.jsonl");
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("read legacy transcript: {}", path.display()))?;
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let turn: Turn = serde_json::from_str(line).with_context(|| {
                    format!("parse legacy transcript line in {}", path.display())
                })?;
                insertTurnTx(&tx, &turn)?;
            }
        }
    }

    if tableEmpty(&tx, "compaction_ops")? {
        let path = sessionDir.join("compaction.jsonl");
        if path.exists() {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("read legacy compaction log: {}", path.display()))?;
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let op: CompactionOp = serde_json::from_str(line).with_context(|| {
                    format!("parse legacy compaction line in {}", path.display())
                })?;
                insertCompactionTx(&tx, &op)?;
            }
        }
    }

    if tableEmpty(&tx, "snapshots")? {
        importLegacySnapshots(&tx, sessionDir)?;
    }

    tx.commit()?;
    Ok(())
}

fn importLegacySnapshots(tx: &rusqlite::Transaction<'_>, sessionDir: &Path) -> Result<()> {
    let snapshots = sessionDir.join("snapshots");
    let indexPath = snapshots.join("index.jsonl");
    if indexPath.exists() {
        let content = fs::read_to_string(&indexPath)
            .with_context(|| format!("read legacy snapshots index: {}", indexPath.display()))?;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line).with_context(|| {
                format!(
                    "parse legacy snapshot index line in {}",
                    indexPath.display()
                )
            })?;
            let hash = value
                .get("hash")
                .and_then(|v| v.as_str())
                .context("legacy snapshot index entry missing hash")?;
            let snapValue = value
                .get("snapshot")
                .cloned()
                .context("legacy snapshot index entry missing snapshot")?;
            let snap: RequestSnapshot = serde_json::from_value(snapValue)?;
            insertSnapshotTx(tx, hash, &snap)?;
        }
    }

    for (ns, dirName, ext) in [
        ("system_prompt", "sp", "txt"),
        ("tools", "tl", "json"),
        ("message", "ms", "json"),
    ] {
        let dir = snapshots.join("blobs").join(dirName);
        if !dir.exists() {
            continue;
        }
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(ext) {
                continue;
            }
            let Some(hash) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let bytes = fs::read(&path)
                .with_context(|| format!("read legacy snapshot blob: {}", path.display()))?;
            tx.execute(
                "INSERT OR IGNORE INTO snapshot_blobs (hash, namespace, bytes) VALUES (?1, ?2, ?3)",
                params![hash, ns, bytes],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn insertTurn(conn: &Connection, turn: &Turn) -> Result<()> {
    sqliteBlocking(|| insertTurnTx(conn, turn))
}

pub(crate) fn updateTurn(conn: &Connection, turn: &Turn) -> Result<()> {
    sqliteBlocking(|| updateTurnTx(conn, turn))
}

fn insertTurnTx(conn: &Connection, turn: &Turn) -> Result<()> {
    let role = serde_json::to_value(&turn.role)?
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let json = serde_json::to_string(turn)?;
    conn.execute(
        "INSERT OR REPLACE INTO turns
         (id, block_id, topic_id, role, ts, parent_id, tool, tool_call_id, snapshot_hash, turn_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            turn.id,
            turn.blockId,
            turn.topicId,
            role,
            turn.ts as i64,
            turn.parentId,
            turn.tool,
            turn.toolCallId,
            turn.snapshotHash,
            json,
        ],
    )?;
    Ok(())
}

fn updateTurnTx(conn: &Connection, turn: &Turn) -> Result<()> {
    let role = serde_json::to_value(&turn.role)?
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let json = serde_json::to_string(turn)?;
    conn.execute(
        "UPDATE turns
         SET block_id = ?2,
             topic_id = ?3,
             role = ?4,
             ts = ?5,
             parent_id = ?6,
             tool = ?7,
             tool_call_id = ?8,
             snapshot_hash = ?9,
             turn_json = ?10
         WHERE id = ?1",
        params![
            turn.id,
            turn.blockId,
            turn.topicId,
            role,
            turn.ts as i64,
            turn.parentId,
            turn.tool,
            turn.toolCallId,
            turn.snapshotHash,
            json,
        ],
    )?;
    Ok(())
}

pub(crate) fn loadTurns(conn: &Connection) -> Result<Vec<Turn>> {
    sqliteBlocking(|| {
        let mut stmt = conn.prepare("SELECT turn_json FROM turns ORDER BY seq ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    })
}

pub(crate) fn upsertMeta(conn: &Connection, meta: &SessionMeta) -> Result<()> {
    sqliteBlocking(|| upsertMetaTx(conn, meta))
}

fn upsertMetaTx(conn: &Connection, meta: &SessionMeta) -> Result<()> {
    let json = serde_json::to_string_pretty(meta)?;
    conn.execute(
        "INSERT INTO session_meta
         (id, session_id, project_dir, created_at, updated_at, name, head_turn, total_cost, meta_json)
         VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(id) DO UPDATE SET
           session_id=excluded.session_id,
           project_dir=excluded.project_dir,
           created_at=excluded.created_at,
           updated_at=excluded.updated_at,
           name=excluded.name,
           head_turn=excluded.head_turn,
           total_cost=excluded.total_cost,
           meta_json=excluded.meta_json",
        params![
            meta.sessionId,
            meta.projectDir,
            meta.createdAt as i64,
            meta.updatedAt as i64,
            meta.name,
            meta.headTurn,
            meta.totalCost,
            json,
        ],
    )?;
    Ok(())
}

pub(crate) fn loadMeta(conn: &Connection) -> Result<Option<SessionMeta>> {
    sqliteBlocking(|| {
        let json: Option<String> = conn
            .query_row("SELECT meta_json FROM session_meta WHERE id = 1", [], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(match json {
            Some(j) => Some(serde_json::from_str(&j)?),
            None => None,
        })
    })
}

pub(crate) fn insertCompaction(conn: &Connection, op: &CompactionOp) -> Result<()> {
    sqliteBlocking(|| insertCompactionTx(conn, op))
}

fn insertCompactionTx(conn: &Connection, op: &CompactionOp) -> Result<()> {
    let value = serde_json::to_value(op)?;
    let action = value
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let ts = value.get("ts").and_then(|v| v.as_u64()).unwrap_or(0);
    conn.execute(
        "INSERT INTO compaction_ops (action, after_turn, ts, op_json)
         VALUES (?1, ?2, ?3, ?4)",
        params![
            action,
            op.afterTurn(),
            ts as i64,
            serde_json::to_string(op)?
        ],
    )?;
    Ok(())
}

pub(crate) fn loadCompaction(conn: &Connection) -> Result<Vec<CompactionOp>> {
    sqliteBlocking(|| {
        let mut stmt = conn.prepare("SELECT op_json FROM compaction_ops ORDER BY seq ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    })
}

pub(crate) fn insertSnapshot(conn: &Connection, hash: &str, snap: &RequestSnapshot) -> Result<()> {
    sqliteBlocking(|| insertSnapshotTx(conn, hash, snap))
}

fn insertSnapshotTx(conn: &Connection, hash: &str, snap: &RequestSnapshot) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO snapshots (hash, ts, snapshot_json) VALUES (?1, ?2, ?3)",
        params![hash, snap.ts as i64, serde_json::to_string(snap)?],
    )?;
    Ok(())
}

pub(crate) fn snapshotHashes(conn: &Connection) -> Result<std::collections::HashSet<String>> {
    sqliteBlocking(|| {
        let mut stmt = conn.prepare("SELECT hash FROM snapshots")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = std::collections::HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
    })
}

pub(crate) fn putSnapshotBlob(
    conn: &Connection,
    namespace: &str,
    hash: &str,
    bytes: &[u8],
) -> Result<()> {
    sqliteBlocking(|| {
        conn.execute(
            "INSERT OR IGNORE INTO snapshot_blobs (hash, namespace, bytes) VALUES (?1, ?2, ?3)",
            params![hash, namespace, bytes],
        )?;
        Ok(())
    })
}

pub fn loadSnapshotIndexForSession(sessionDir: &Path) -> Result<HashMap<String, RequestSnapshot>> {
    let conn = openSessionDb(sessionDir)?;
    sqliteBlocking(|| {
        let mut stmt = conn.prepare("SELECT hash, snapshot_json FROM snapshots")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut out = HashMap::new();
        for row in rows {
            let (hash, json) = row?;
            out.insert(hash, serde_json::from_str(&json)?);
        }
        Ok(out)
    })
}

pub fn snapshotBlobForSession(
    sessionDir: &Path,
    namespace: &str,
    hash: &str,
) -> Result<Option<Vec<u8>>> {
    let conn = openSessionDb(sessionDir)?;
    sqliteBlocking(|| {
        let bytes = conn
            .query_row(
                "SELECT bytes FROM snapshot_blobs WHERE namespace = ?1 AND hash = ?2",
                params![namespace, hash],
                |r| r.get(0),
            )
            .optional()?;
        Ok(bytes)
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TerminalRunFieldParseError {
    field: &'static str,
    value: String,
}

impl TerminalRunFieldParseError {
    fn new(field: &'static str, value: &str) -> Self {
        Self {
            field,
            value: value.to_string(),
        }
    }
}

impl fmt::Display for TerminalRunFieldParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown terminal run {} `{}`", self.field, self.value)
    }
}

impl std::error::Error for TerminalRunFieldParseError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalRunImpact {
    Read,
    MinorMod,
    MajorMod,
    Delete,
}

impl TerminalRunImpact {
    pub(crate) fn asStorageName(self) -> &'static str {
        match self {
            TerminalRunImpact::Read => "read",
            TerminalRunImpact::MinorMod => "minorMod",
            TerminalRunImpact::MajorMod => "majorMod",
            TerminalRunImpact::Delete => "delete",
        }
    }

    pub(crate) fn fromStorageName(
        value: &str,
    ) -> std::result::Result<Self, TerminalRunFieldParseError> {
        match value {
            "read" => Ok(TerminalRunImpact::Read),
            "minorMod" => Ok(TerminalRunImpact::MinorMod),
            "majorMod" => Ok(TerminalRunImpact::MajorMod),
            "delete" => Ok(TerminalRunImpact::Delete),
            other => Err(TerminalRunFieldParseError::new("impact", other)),
        }
    }
}

impl fmt::Display for TerminalRunImpact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.asStorageName())
    }
}

impl From<&crate::tool::ShellImpact> for TerminalRunImpact {
    fn from(impact: &crate::tool::ShellImpact) -> Self {
        match impact {
            crate::tool::ShellImpact::Read => TerminalRunImpact::Read,
            crate::tool::ShellImpact::MinorMod => TerminalRunImpact::MinorMod,
            crate::tool::ShellImpact::MajorMod => TerminalRunImpact::MajorMod,
            crate::tool::ShellImpact::Delete => TerminalRunImpact::Delete,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalRunStatus {
    Running,
    Completed,
    Failed,
    TimedOut,
    Rejected,
}

impl TerminalRunStatus {
    pub(crate) fn asStorageName(self) -> &'static str {
        match self {
            TerminalRunStatus::Running => "running",
            TerminalRunStatus::Completed => "completed",
            TerminalRunStatus::Failed => "failed",
            TerminalRunStatus::TimedOut => "timed_out",
            TerminalRunStatus::Rejected => "rejected",
        }
    }

    pub(crate) fn fromStorageName(
        value: &str,
    ) -> std::result::Result<Self, TerminalRunFieldParseError> {
        match value {
            "running" => Ok(TerminalRunStatus::Running),
            "completed" => Ok(TerminalRunStatus::Completed),
            "failed" => Ok(TerminalRunStatus::Failed),
            "timed_out" => Ok(TerminalRunStatus::TimedOut),
            "rejected" => Ok(TerminalRunStatus::Rejected),
            other => Err(TerminalRunFieldParseError::new("status", other)),
        }
    }
}

impl fmt::Display for TerminalRunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.asStorageName())
    }
}

#[derive(Debug, Clone)]
pub struct TerminalRunRecord {
    pub runId: String,
    pub terminalName: String,
    pub command: String,
    pub purpose: String,
    pub impact: TerminalRunImpact,
    pub ephemeral: bool,
    pub startedAt: u64,
    pub endedAt: Option<u64>,
    pub status: TerminalRunStatus,
    pub exitCode: Option<i32>,
    pub lineCount: usize,
    pub replayBlob: Vec<u8>,
}

pub(crate) fn upsertTerminalRun(conn: &Connection, run: &TerminalRunRecord) -> Result<()> {
    sqliteBlocking(|| {
        conn.execute(
            "INSERT INTO terminal_runs
         (run_id, terminal_name, command, purpose, impact, ephemeral, started_at, ended_at,
          status, exit_code, line_count, replay_blob)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT(run_id) DO UPDATE SET
           terminal_name=excluded.terminal_name,
           command=excluded.command,
           purpose=excluded.purpose,
           impact=excluded.impact,
           ephemeral=excluded.ephemeral,
           started_at=excluded.started_at,
           ended_at=excluded.ended_at,
           status=excluded.status,
           exit_code=excluded.exit_code,
           line_count=excluded.line_count,
           replay_blob=excluded.replay_blob",
            params![
                run.runId,
                run.terminalName,
                run.command,
                run.purpose,
                run.impact.asStorageName(),
                if run.ephemeral { 1 } else { 0 },
                run.startedAt as i64,
                run.endedAt.map(|v| v as i64),
                run.status.asStorageName(),
                run.exitCode,
                run.lineCount as i64,
                run.replayBlob,
            ],
        )?;
        Ok(())
    })
}

pub(crate) fn listTerminalRuns(conn: &Connection) -> Result<Vec<TerminalRunRecord>> {
    sqliteBlocking(|| {
        let mut stmt = conn.prepare(
            "SELECT run_id, terminal_name, command, purpose, impact, ephemeral, started_at,
                ended_at, status, exit_code, line_count, replay_blob
         FROM terminal_runs ORDER BY started_at DESC",
        )?;
        let rows = stmt.query_map([], terminalRunRecordFromRow)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    })
}

pub(crate) fn getTerminalRun(conn: &Connection, runId: &str) -> Result<Option<TerminalRunRecord>> {
    sqliteBlocking(|| {
        conn.query_row(
            "SELECT run_id, terminal_name, command, purpose, impact, ephemeral, started_at,
                ended_at, status, exit_code, line_count, replay_blob
             FROM terminal_runs WHERE run_id = ?1",
            [runId],
            terminalRunRecordFromRow,
        )
        .optional()
        .map_err(Into::into)
    })
}

fn terminalRunRecordFromRow(r: &Row<'_>) -> rusqlite::Result<TerminalRunRecord> {
    let impactRaw: String = r.get(4)?;
    let statusRaw: String = r.get(8)?;
    Ok(TerminalRunRecord {
        runId: r.get(0)?,
        terminalName: r.get(1)?,
        command: r.get(2)?,
        purpose: r.get(3)?,
        impact: TerminalRunImpact::fromStorageName(&impactRaw)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, Type::Text, Box::new(e)))?,
        ephemeral: r.get::<_, i64>(5)? != 0,
        startedAt: r.get::<_, i64>(6)? as u64,
        endedAt: r.get::<_, Option<i64>>(7)?.map(|v| v as u64),
        status: TerminalRunStatus::fromStorageName(&statusRaw)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(8, Type::Text, Box::new(e)))?,
        exitCode: r.get(9)?,
        lineCount: r.get::<_, i64>(10)? as usize,
        replayBlob: r.get(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminalRunRoundTripsReplayBlob() {
        let dir = tempfile::TempDir::new().unwrap();
        let conn = openSessionDb(dir.path()).unwrap();
        let run = TerminalRunRecord {
            runId: "run_1".into(),
            terminalName: "build".into(),
            command: "cargo test".into(),
            purpose: "run tests".into(),
            impact: TerminalRunImpact::Read,
            ephemeral: true,
            startedAt: 10,
            endedAt: Some(20),
            status: TerminalRunStatus::Completed,
            exitCode: Some(0),
            lineCount: 2,
            replayBlob: b"\x1b[32mok\x1b[0m\n".to_vec(),
        };
        upsertTerminalRun(&conn, &run).unwrap();

        let loaded = getTerminalRun(&conn, "run_1").unwrap().unwrap();
        assert_eq!(loaded.terminalName, "build");
        assert_eq!(loaded.impact, TerminalRunImpact::Read);
        assert!(loaded.ephemeral);
        assert_eq!(loaded.status, TerminalRunStatus::Completed);
        assert_eq!(loaded.exitCode, Some(0));
        assert_eq!(loaded.replayBlob, b"\x1b[32mok\x1b[0m\n");

        let list = listTerminalRuns(&conn).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].runId, "run_1");
    }

    #[test]
    fn terminalRunRejectsUnknownStatus() {
        let dir = tempfile::TempDir::new().unwrap();
        let conn = openSessionDb(dir.path()).unwrap();
        conn.execute(
            "INSERT INTO terminal_runs
             (run_id, terminal_name, command, purpose, impact, ephemeral, started_at,
              status, line_count, replay_blob)
             VALUES ('run_bad', 'main', 'cmd', 'purpose', 'read', 0, 10,
              'paused_maybe', 0, X'')",
            [],
        )
        .unwrap();

        let err = listTerminalRuns(&conn).unwrap_err();
        assert!(format!("{err:?}").contains("unknown terminal run status"));
    }

    #[test]
    fn sqliteSnapshotIndexAndBlobLoadBySessionDir() {
        let dir = tempfile::TempDir::new().unwrap();
        let conn = openSessionDb(dir.path()).unwrap();
        let snap = RequestSnapshot {
            v: 1,
            model: "m".into(),
            provider: "openrouter".into(),
            baseUrl: "https://example".into(),
            providerOrder: vec![],
            maxTokens: None,
            reasoning: None,
            systemPromptHash: Some("sp_hash".into()),
            toolsHash: None,
            toolsCount: 0,
            messages: vec!["msg_hash".into()],
            temperature: None,
            topP: None,
            seed: None,
            ts: 42,
        };
        insertSnapshot(&conn, "snap_hash", &snap).unwrap();
        putSnapshotBlob(
            &conn,
            "message",
            "msg_hash",
            br#"{"role":"user","content":"hi"}"#,
        )
        .unwrap();

        let index = loadSnapshotIndexForSession(dir.path()).unwrap();
        assert_eq!(index["snap_hash"].messages, vec!["msg_hash"]);
        let blob = snapshotBlobForSession(dir.path(), "message", "msg_hash")
            .unwrap()
            .unwrap();
        assert_eq!(blob, br#"{"role":"user","content":"hi"}"#);
    }
}
