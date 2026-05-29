//! OpenAI fine-tuning JSONL writer.
//!
//! Each emitted line is one SFT example:
//!
//! ```json
//! {
//!   "messages": [ {"role":"system","content":"..."}, ... ],
//!   "tools":    [ {"type":"function", ...}, ... ],
//!   "reasoning": ["...", "..."],    // optional, non-standard; per-asst-turn aligned
//!   "_flatline": { "sessionId": "...", "segmentIndex": N, "snapshotHash": "...", "turnRange": [...] }
//! }
//! ```

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use construct::message::{Content, FunctionCall, Message, ToolCall, ToolDef};
use construct::transcript::{Turn, TurnRole};

use super::Segment;

/// A single emitted SFT example.
#[derive(Serialize)]
pub struct Example {
    pub messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Vec<String>>,
    #[serde(rename = "_flatline")]
    pub flatline: FlatlineMeta,
}

#[derive(Serialize)]
pub struct FlatlineMeta {
    pub sessionId: String,
    pub segmentIndex: usize,
    pub snapshotHash: String,
    pub turnRange: [String; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Build an example from a segment. Returns `None` if the result is below
/// `minMessages`.
pub fn buildExample(
    sessionId: &str,
    segmentIndex: usize,
    segment: &Segment,
    branch: &[Turn],
    sessionDir: &Path,
    noReasoning: bool,
    minMessages: usize,
) -> Result<Option<Example>> {
    // `last.messages` is the history as sent on the request that produced the
    // final assistant turn of the segment — i.e. everything before that asst
    // response. Dereference each message blob and then append the asst turn
    // itself as the training target.
    let mut messages: Vec<serde_json::Value> = Vec::new();

    if let Some(hash) = &segment.anchor.systemPromptHash {
        let text = readSystemPrompt(sessionDir, hash)?;
        messages.push(serde_json::json!({ "role": "system", "content": text }));
    }

    for hash in &segment.last.messages {
        let msg = readMessage(sessionDir, hash)?;
        messages.push(messageToOpenAi(&msg)?);
    }

    // Append the final assistant turn as the training target. Collect any
    // tool calls that follow it in the transcript within the same block so
    // the Message::Assistant mirrors what actually went on the wire.
    let lastAsst = &branch[segment.lastAsstIdx];
    let lastMsg = assembleAssistantTarget(branch, segment.lastAsstIdx)?;
    messages.push(messageToOpenAi(&lastMsg)?);

    if messages.len() < minMessages {
        return Ok(None);
    }

    let tools: Vec<ToolDef> = match &segment.anchor.toolsHash {
        Some(hash) => readTools(sessionDir, hash)?,
        None => Vec::new(),
    };

    // Collect per-assistant-turn reasoning, aligned with the asst turns in
    // `messages`. Only turns within this segment contribute (snapshot covers
    // messages before the anchor but those aren't part of this example's
    // training target). Include an entry per asst turn, empty string if absent.
    let reasoning = if noReasoning {
        None
    } else {
        let mut traces: Vec<String> = Vec::new();
        for t in branch
            .iter()
            .take(segment.lastAsstIdx + 1)
            .skip(segment.anchorIdx)
        {
            if matches!(t.role, TurnRole::Assistant) {
                traces.push(t.reasoning.clone().unwrap_or_default());
            }
        }
        if traces.iter().all(|s| s.is_empty()) {
            None
        } else {
            Some(traces)
        }
    };

    let snapshotHash = lastAsst.snapshotHash.clone().unwrap_or_default();

    Ok(Some(Example {
        messages,
        tools,
        reasoning,
        flatline: FlatlineMeta {
            sessionId: sessionId.to_string(),
            segmentIndex,
            snapshotHash,
            turnRange: [branch[segment.anchorIdx].id.clone(), lastAsst.id.clone()],
            model: Some(segment.last.model.clone()),
        },
    }))
}

/// Build a `Message::Assistant` from an assistant turn and any ToolCall turns
/// that immediately follow it in the same block (those share one API response).
fn assembleAssistantTarget(branch: &[Turn], asstIdx: usize) -> Result<Message> {
    let asst = &branch[asstIdx];
    let blockId = asst.blockId.as_str();

    let mut toolCalls: Vec<ToolCall> = Vec::new();
    for t in branch.iter().skip(asstIdx + 1) {
        if t.blockId != blockId {
            break;
        }
        match t.role {
            TurnRole::ToolCall => {
                let callId = t.toolCallId.clone().unwrap_or_default();
                let name = t.tool.clone().unwrap_or_default();
                let args = t
                    .args
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
                    .unwrap_or_else(|| "{}".into());
                toolCalls.push(ToolCall {
                    id: callId,
                    callType: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: args,
                    },
                });
            }
            // Stop at tool results or further assistant/user/system turns.
            _ => break,
        }
    }

    let content = if asst.content.is_empty() {
        None
    } else {
        Some(asst.content.clone())
    };
    let toolCalls = if toolCalls.is_empty() {
        None
    } else {
        Some(toolCalls)
    };
    let reasoning = asst.reasoning.clone();

    Ok(Message::Assistant {
        content,
        tool_calls: toolCalls,
        reasoning,
    })
}

/// Convert an internal `Message` into an OpenAI-shaped JSON value.
fn messageToOpenAi(msg: &Message) -> Result<serde_json::Value> {
    let value = serde_json::to_value(msg)?;
    Ok(value)
}

fn readSystemPrompt(sessionDir: &Path, hash: &str) -> Result<String> {
    let snapshotsDir = sessionDir.join("snapshots");
    let path = snapshotsDir.join("blobs/sp").join(format!("{hash}.txt"));
    match fs::read_to_string(&path) {
        Ok(s) => Ok(s),
        Err(fsErr) => {
            let bytes =
                construct::storage::snapshotBlobForSession(sessionDir, "system_prompt", hash)?
                    .ok_or_else(|| {
                        anyhow::anyhow!("read {} or sqlite blob: {fsErr}", path.display())
                    })?;
            String::from_utf8(bytes).with_context(|| format!("decode system prompt blob {hash}"))
        }
    }
}

fn readTools(sessionDir: &Path, hash: &str) -> Result<Vec<ToolDef>> {
    let snapshotsDir = sessionDir.join("snapshots");
    let path = snapshotsDir.join("blobs/tl").join(format!("{hash}.json"));
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(fsErr) => {
            let bytes = construct::storage::snapshotBlobForSession(sessionDir, "tools", hash)?
                .ok_or_else(|| {
                    anyhow::anyhow!("read {} or sqlite blob: {fsErr}", path.display())
                })?;
            String::from_utf8(bytes).with_context(|| format!("decode tools blob {hash}"))?
        }
    };
    serde_json::from_str(&raw).with_context(|| format!("parse tools {}", path.display()))
}

fn readMessage(sessionDir: &Path, hash: &str) -> Result<Message> {
    let snapshotsDir = sessionDir.join("snapshots");
    let path = snapshotsDir.join("blobs/ms").join(format!("{hash}.json"));
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(fsErr) => {
            let bytes = construct::storage::snapshotBlobForSession(sessionDir, "message", hash)?
                .ok_or_else(|| {
                    anyhow::anyhow!("read {} or sqlite blob: {fsErr}", path.display())
                })?;
            String::from_utf8(bytes).with_context(|| format!("decode message blob {hash}"))?
        }
    };
    serde_json::from_str(&raw).with_context(|| format!("parse message {}", path.display()))
}

/// Avoid dead-code warnings for helper types used only in tests.
#[allow(dead_code)]
fn _keepContentType(_: Content) {}
