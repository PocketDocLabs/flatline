//! Per-turn request snapshots — content-addressed capture of the exact
//! prompt / tools / config each assistant turn was produced from.
//!
//! The transcript records what the model *said*; the snapshot records what
//! the model *saw*. Together they let a transcript be replayed as SFT
//! training data without ambiguity.
//!
//! Bulky pieces (system prompt text, tool schemas, individual messages) are
//! stored as SHA-1-addressed blobs on disk so identical inputs across turns
//! collapse to one file. The snapshot itself is also content-addressed and
//! written append-only to `index.jsonl`.
//!
//! # Public API
//! - [`RequestSnapshot`] — the per-turn record
//! - [`SnapshotStore`] — on-disk store with dedup
//! - [`BlobNs`] — blob namespace tag
//! - [`captureSnapshot`] — assemble + persist a snapshot for the current turn
//! - [`canonicalJson`] — deterministic JSON serialization (sorted keys)
//! - [`sha1Hex`] — 40-char hex SHA-1 of arbitrary bytes
//!
//! # Dependencies
//! `serde`, `serde_json`, `sha1_smol`

use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;
use crate::message::{Message, ReasoningConfig, ToolDef};

/// 40-char SHA-1 hex digest.
pub type BlobHash = String;

/// Blob namespace — maps to a subdirectory under `snapshots/blobs/`.
#[derive(Debug, Clone, Copy)]
pub enum BlobNs {
    SystemPrompt,
    Tools,
    Message,
}

impl BlobNs {
    fn dirName(self) -> &'static str {
        match self {
            BlobNs::SystemPrompt => "sp",
            BlobNs::Tools => "tl",
            BlobNs::Message => "ms",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            BlobNs::SystemPrompt => "txt",
            BlobNs::Tools => "json",
            BlobNs::Message => "json",
        }
    }
}

/// A per-turn record of exactly what was sent to the provider on the
/// request that produced the next assistant turn.
///
/// Field naming mirrors the wire body in `api.rs:81-115` so a reader can
/// map the snapshot back to what the API saw.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestSnapshot {
    /// Schema version. Bump on incompatible changes.
    pub v: u16,

    pub model: String,
    pub provider: String,
    pub baseUrl: String,

    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub providerOrder: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub maxTokens: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning: Option<ReasoningConfig>,

    /// SHA-1 of the rendered system prompt UTF-8 bytes.
    /// `None` when no system message was present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub systemPromptHash: Option<BlobHash>,

    /// SHA-1 of the canonical JSON serialization of `Vec<ToolDef>`.
    /// `None` when no tools were sent.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub toolsHash: Option<BlobHash>,

    /// Count of tools — cheap denormalization for scanning without dereffing.
    #[serde(default)]
    pub toolsCount: u16,

    /// Ordered list of message-blob hashes forming `history` at request time,
    /// excluding the leading system message (captured separately above).
    #[serde(default)]
    pub messages: Vec<BlobHash>,

    // Sampling params — currently always `None`; captured for future-proofing.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub topP: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seed: Option<u64>,

    /// Unix seconds when the snapshot was captured.
    pub ts: u64,
}

/// One line in `snapshots/index.jsonl`.
#[derive(Debug, Serialize, Deserialize)]
struct IndexEntry {
    hash: BlobHash,
    snapshot: RequestSnapshot,
}

/// On-disk snapshot store for a session.
pub struct SnapshotStore {
    dir: PathBuf,
    seenSnapshots: HashSet<BlobHash>,
    indexWriter: BufWriter<fs::File>,
}

impl SnapshotStore {
    /// Open (or create) the snapshot store for a session directory.
    pub fn open(sessionDir: &Path) -> Result<Self> {
        let dir = sessionDir.join("snapshots");
        let blobsDir = dir.join("blobs");
        for ns in [BlobNs::SystemPrompt, BlobNs::Tools, BlobNs::Message] {
            fs::create_dir_all(blobsDir.join(ns.dirName()))
                .with_context(|| format!("create {}", ns.dirName()))?;
        }

        let indexPath = dir.join("index.jsonl");

        // Scan existing index to seed the dedup set.
        let mut seen = HashSet::new();
        if indexPath.exists() {
            let content = fs::read_to_string(&indexPath)
                .with_context(|| format!("read {}", indexPath.display()))?;
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<IndexEntry>(line) {
                    seen.insert(entry.hash);
                }
            }
        }

        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&indexPath)
            .with_context(|| format!("open {}", indexPath.display()))?;

        Ok(Self {
            dir,
            seenSnapshots: seen,
            indexWriter: BufWriter::new(file),
        })
    }

    /// Write a blob to the given namespace. Idempotent — if the file exists,
    /// returns the hash without rewriting.
    pub fn putBlob(&mut self, ns: BlobNs, bytes: &[u8]) -> Result<BlobHash> {
        let hash = sha1Hex(bytes);
        let path = self
            .dir
            .join("blobs")
            .join(ns.dirName())
            .join(format!("{hash}.{}", ns.extension()));
        if !path.exists() {
            fs::write(&path, bytes).with_context(|| format!("write {}", path.display()))?;
        }
        Ok(hash)
    }

    /// Canonicalize a `Message`, write it as a blob, return the hash.
    pub fn putMessage(&mut self, msg: &Message) -> Result<BlobHash> {
        let value = serde_json::to_value(msg).context("serialize message")?;
        let canonical = canonicalJson(&value);
        self.putBlob(BlobNs::Message, canonical.as_bytes())
    }

    /// Write the snapshot to `index.jsonl`, deduplicated by its canonical hash.
    /// Returns the snapshot's hash.
    pub fn record(&mut self, snap: &RequestSnapshot) -> Result<BlobHash> {
        let snapValue = serde_json::to_value(snap).context("serialize snapshot")?;
        let canonical = canonicalJson(&snapValue);
        let hash = sha1Hex(canonical.as_bytes());

        if self.seenSnapshots.insert(hash.clone()) {
            let entry = IndexEntry {
                hash: hash.clone(),
                snapshot: snap.clone(),
            };
            let line = serde_json::to_string(&entry).context("serialize index entry")?;
            writeln!(self.indexWriter, "{line}").context("write index line")?;
            self.indexWriter.flush().context("flush index")?;
        }

        Ok(hash)
    }
}

/// Inputs for assembling a snapshot.
pub struct BuildCtx<'a> {
    pub history: &'a [Message],
    pub tools: &'a [ToolDef],
    pub reasoning: Option<&'a ReasoningConfig>,
    pub cfg: &'a ModelConfig,
}

/// Assemble a snapshot from the current session state and persist it.
///
/// Splits `history` into a leading system message (if present) and the
/// remaining turn messages, hashing each piece independently. Writes blobs
/// for system prompt, tools, and each message, then records the snapshot.
///
/// Returns the snapshot hash to store on the resulting assistant `Turn`.
pub fn captureSnapshot(store: &mut SnapshotStore, ctx: BuildCtx<'_>) -> Result<BlobHash> {
    let (systemContent, turnMsgs) = splitSystem(ctx.history);

    let systemPromptHash = if let Some(sys) = systemContent {
        Some(store.putBlob(BlobNs::SystemPrompt, sys.as_bytes())?)
    } else {
        None
    };

    let (toolsHash, toolsCount) = if ctx.tools.is_empty() {
        (None, 0u16)
    } else {
        let value = serde_json::to_value(ctx.tools).context("serialize tools")?;
        let canonical = canonicalJson(&value);
        let hash = store.putBlob(BlobNs::Tools, canonical.as_bytes())?;
        (Some(hash), ctx.tools.len().min(u16::MAX as usize) as u16)
    };

    let mut messageHashes = Vec::with_capacity(turnMsgs.len());
    for msg in turnMsgs {
        messageHashes.push(store.putMessage(msg)?);
    }

    let snap = RequestSnapshot {
        v: 1,
        model: ctx.cfg.model.clone(),
        provider: ctx.cfg.provider.clone(),
        baseUrl: ctx.cfg.baseUrl.clone(),
        providerOrder: ctx.cfg.providerOrder.clone(),
        maxTokens: ctx.cfg.maxTokens,
        reasoning: ctx.reasoning.cloned(),
        systemPromptHash,
        toolsHash,
        toolsCount,
        messages: messageHashes,
        temperature: None,
        topP: None,
        seed: None,
        ts: now(),
    };

    store.record(&snap)
}

/// Extract the leading system message content (if `history[0]` is System),
/// returning it plus the remaining non-leading messages.
fn splitSystem(history: &[Message]) -> (Option<&str>, &[Message]) {
    match history.first() {
        Some(Message::System { content }) => (Some(content.as_str()), &history[1..]),
        _ => (None, history),
    }
}

/// 40-char hex SHA-1 digest of bytes. Mirrors the convention in
/// `mcp::schema::sha1Hex` (which takes `&str`).
pub fn sha1Hex(bytes: &[u8]) -> String {
    let hash = sha1_smol::Sha1::from(bytes).digest();
    hash.bytes().iter().map(|b| format!("{b:02x}")).collect()
}

/// Deterministic JSON serialization — sorted object keys, no whitespace.
///
/// Required for content-addressed hashing: two semantically-identical
/// `serde_json::Value`s with different key insertion order must produce
/// byte-equal output.
pub fn canonicalJson(v: &serde_json::Value) -> String {
    let mut out = String::new();
    writeCanonical(v, &mut out);
    out
}

fn writeCanonical(v: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            // Use serde_json's string escaping so e.g. non-ASCII survives.
            out.push_str(&serde_json::to_string(s).unwrap_or_default());
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                writeCanonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                writeCanonical(&map[*k], out);
            }
            out.push('}');
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Content, FunctionDef};

    #[test]
    fn sha1HexMatchesKnownVector() {
        // SHA-1("abc") = a9993e364706816aba3e25717850c26c9cd0d89d
        assert_eq!(sha1Hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn canonicalJsonSortsKeys() {
        let a = serde_json::json!({"b": 1, "a": 2, "c": [3, {"z": 1, "y": 2}]});
        let b = serde_json::json!({"c": [3, {"y": 2, "z": 1}], "a": 2, "b": 1});
        assert_eq!(canonicalJson(&a), canonicalJson(&b));
    }

    #[test]
    fn canonicalJsonIsStable() {
        let v = serde_json::json!({"model": "foo", "temperature": 0.7, "messages": ["hi"]});
        assert_eq!(
            canonicalJson(&v),
            r#"{"messages":["hi"],"model":"foo","temperature":0.7}"#
        );
    }

    fn tempStore() -> (tempfile::TempDir, SnapshotStore) {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SnapshotStore::open(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn putBlobIsIdempotent() {
        let (_dir, mut store) = tempStore();
        let h1 = store.putBlob(BlobNs::SystemPrompt, b"hello").unwrap();
        let h2 = store.putBlob(BlobNs::SystemPrompt, b"hello").unwrap();
        assert_eq!(h1, h2);
        let path = store.dir.join("blobs/sp").join(format!("{h1}.txt"));
        assert!(path.exists());
        assert_eq!(fs::read(&path).unwrap(), b"hello");
    }

    #[test]
    fn snapshotDedupesOnIndex() {
        let (_dir, mut store) = tempStore();
        let snap = RequestSnapshot {
            v: 1,
            model: "m".into(),
            provider: "openrouter".into(),
            baseUrl: "x".into(),
            providerOrder: vec![],
            maxTokens: None,
            reasoning: None,
            systemPromptHash: None,
            toolsHash: None,
            toolsCount: 0,
            messages: vec![],
            temperature: None,
            topP: None,
            seed: None,
            ts: 42,
        };
        let h1 = store.record(&snap).unwrap();
        let h2 = store.record(&snap).unwrap();
        assert_eq!(h1, h2);
        let idxPath = store.dir.join("index.jsonl");
        let content = fs::read_to_string(idxPath).unwrap();
        assert_eq!(content.lines().count(), 1);
    }

    #[test]
    fn captureSnapshotRoundTripsMessages() {
        let (_dir, mut store) = tempStore();
        let history = vec![
            Message::System {
                content: "you are helpful".into(),
            },
            Message::User {
                content: Content::Text("hi".into()),
            },
            Message::Assistant {
                content: Some("hello".into()),
                tool_calls: None,
                reasoning: None,
            },
        ];
        let tools = vec![ToolDef {
            defType: "function".into(),
            function: FunctionDef {
                name: "ping".into(),
                description: "ping".into(),
                parameters: serde_json::json!({}),
            },
        }];
        let cfg = ModelConfig {
            provider: "openrouter".into(),
            key: "".into(),
            model: "test/model".into(),
            baseUrl: "https://example".into(),
            reasoning: None,
            promptThinking: false,
            providerOrder: vec![],
            maxTokens: None,
            contextWindow: 1_000,
            maxContextWindow: Some(1_000),
            supportsAnthropicCache: None,
        };

        let hash = captureSnapshot(
            &mut store,
            BuildCtx {
                history: &history,
                tools: &tools,
                reasoning: None,
                cfg: &cfg,
            },
        )
        .unwrap();

        // Snapshot index has one line with our hash.
        let idx = fs::read_to_string(store.dir.join("index.jsonl")).unwrap();
        let entry: IndexEntry = serde_json::from_str(idx.lines().next().unwrap()).unwrap();
        assert_eq!(entry.hash, hash);
        let snap = entry.snapshot;
        assert_eq!(snap.toolsCount, 1);
        assert!(snap.systemPromptHash.is_some());
        assert!(snap.toolsHash.is_some());
        assert_eq!(snap.messages.len(), 2); // user + assistant (system split off)

        // System prompt blob round-trips.
        let spPath = store
            .dir
            .join("blobs/sp")
            .join(format!("{}.txt", snap.systemPromptHash.as_ref().unwrap()));
        assert_eq!(fs::read_to_string(spPath).unwrap(), "you are helpful");

        // Each message blob deserializes back to a Message.
        for h in &snap.messages {
            let p = store.dir.join("blobs/ms").join(format!("{h}.json"));
            let raw = fs::read_to_string(&p).unwrap();
            let _: Message = serde_json::from_str(&raw).unwrap();
        }
    }
}
