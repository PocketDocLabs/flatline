//! S1 — mechanical tool output pruning (no LLM calls).
//!
//! Fires at 70% of compactLimit. Two operations:
//!
//! 1. **File read dedup**: when the same file has been fully read
//!    multiple times, remove all but the latest read. Only targets
//!    full reads (no offset/limit/anchor).
//!
//! 2. **Middle-out truncation**: long tool results are truncated to
//!    keep the head and tail, with a byte-count marker in between.
//!    Invalidates the edit gate for any readFile results that get
//!    truncated (model must re-read before editing).
//!
//! Returns tool_call_ids for each affected item so the session can
//! record them in the compaction log.
//!
//! # Public API
//! - [`run`] — execute S1 pruning on the live message history
//! - [`S1Result`] — what was pruned
//! - [`DEFAULT_MIDDLE_OUT_THRESHOLD`]
//!
//! # Dependencies
//! `serde_json`

use std::collections::{HashMap, HashSet};

use crate::message::Message;

/// Default middle-out threshold in characters.
/// Content longer than this is truncated to head + tail.
pub const DEFAULT_MIDDLE_OUT_THRESHOLD: usize = 4000;

pub struct S1Result {
    pub didWork: bool,
    /// tool_call_ids of deduped readFile calls.
    pub dedupedCallIds: Vec<String>,
    /// tool_call_ids of tool results that were middle-out truncated.
    pub middleOutCallIds: Vec<String>,
    /// Normalized file paths invalidated by middle-out (edit gate).
    pub invalidatedFiles: Vec<String>,
    /// The threshold used for middle-out (for compaction log recording).
    pub middleOutThreshold: usize,
}

/// Run S1 pruning on the live context history.
///
/// Mutates `history` in place: removes duplicate file read messages
/// and truncates long tool results. Both operations only target the
/// oldest 30% of context by character count (skipping the system message).
///
/// `blockHints` maps tool_call_id → blockId so truncation markers can
/// tell the model which block to fetch via `historyFetch`.
///
/// `alreadyProcessed` contains tool_call_ids from previous compaction log
/// entries. These are skipped to keep S1 idempotent across restarts.
pub fn run(
    history: &mut Vec<Message>,
    middleOutThreshold: usize,
    blockHints: &HashMap<String, String>,
    alreadyProcessed: &HashSet<String>,
) -> S1Result {
    let mut dedupedCallIds = Vec::new();
    let mut middleOutCallIds = Vec::new();
    let mut invalidatedFiles = Vec::new();

    // Calculate the S1 zone: oldest 30% by character count (skip system msg).
    let zoneIndices = calculateZone(history);

    dedupFileReads(history, &mut dedupedCallIds, alreadyProcessed, &zoneIndices);
    truncateLongResults(
        history,
        middleOutThreshold,
        blockHints,
        &mut middleOutCallIds,
        &mut invalidatedFiles,
        alreadyProcessed,
        &zoneIndices,
    );

    let didWork = !dedupedCallIds.is_empty() || !middleOutCallIds.is_empty();

    S1Result {
        didWork,
        dedupedCallIds,
        middleOutCallIds,
        invalidatedFiles,
        middleOutThreshold,
    }
}

/// Remove older full reads of the same file, keeping only the latest.
///
/// Walks all Assistant messages to find readFile tool calls. For each
/// normalized path that appears more than once (full reads only),
/// removes the older reads by stripping the ToolCall from its Assistant
/// message and removing the corresponding Tool result message.
fn dedupFileReads(
    history: &mut Vec<Message>,
    dedupedCallIds: &mut Vec<String>,
    alreadyProcessed: &HashSet<String>,
    zoneIndices: &HashSet<usize>,
) {
    // Map: normalized_path → Vec<(tool_call_id, messageIndex)> in order.
    let mut readsByPath: HashMap<String, Vec<(String, usize)>> = HashMap::new();

    for (i, msg) in history.iter().enumerate() {
        if let Message::Assistant {
            tool_calls: Some(calls),
            ..
        } = msg
        {
            for call in calls {
                if call.function.name != "readFile" {
                    continue;
                }
                let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                else {
                    continue;
                };
                // Only dedup full reads (no offset/limit/anchor).
                if args.get("offset").is_some()
                    || args.get("limit").is_some()
                    || args.get("anchor").is_some()
                {
                    continue;
                }
                if let Some(path) = args["path"].as_str() {
                    let norm = normalizePath(path);
                    readsByPath
                        .entry(norm)
                        .or_default()
                        .push((call.id.clone(), i));
                }
            }
        }
    }

    // Remove older reads, but only if they're in the S1 zone.
    let mut removeCallIds: HashSet<String> = HashSet::new();
    for reads in readsByPath.values() {
        if reads.len() <= 1 {
            continue;
        }
        // Keep the last (latest) read, remove earlier ones that are in-zone.
        for (callId, msgIdx) in &reads[..reads.len() - 1] {
            if alreadyProcessed.contains(callId) {
                continue;
            }
            if !zoneIndices.contains(msgIdx) {
                continue;
            }
            removeCallIds.insert(callId.clone());
            dedupedCallIds.push(callId.clone());
        }
    }

    if removeCallIds.is_empty() {
        return;
    }

    // Remove Tool result messages for deduped calls.
    history.retain(|msg| {
        if let Message::Tool { tool_call_id, .. } = msg {
            !removeCallIds.contains(tool_call_id)
        } else {
            true
        }
    });

    // Remove ToolCalls from Assistant messages.
    for msg in history.iter_mut() {
        if let Message::Assistant {
            tool_calls: Some(calls),
            ..
        } = msg
        {
            calls.retain(|call| !removeCallIds.contains(&call.id));
        }
    }

    // Clean up: set empty tool_calls to None, remove fully empty Assistant messages.
    for msg in history.iter_mut() {
        if let Message::Assistant { tool_calls, .. } = msg {
            if tool_calls.as_ref().map_or(false, |c| c.is_empty()) {
                *tool_calls = None;
            }
        }
    }
    history.retain(|msg| {
        if let Message::Assistant {
            content,
            tool_calls,
            ..
        } = msg
        {
            let hasContent = content.as_ref().map_or(false, |c| !c.is_empty());
            let hasCalls = tool_calls.is_some();
            hasContent || hasCalls
        } else {
            true
        }
    });
}

/// Truncate tool results longer than `threshold` to head + tail.
///
/// For readFile results that get truncated, adds the path to
/// `invalidatedFiles` so the edit gate is invalidated.
fn truncateLongResults(
    history: &mut Vec<Message>,
    threshold: usize,
    blockHints: &HashMap<String, String>,
    middleOutCallIds: &mut Vec<String>,
    invalidatedFiles: &mut Vec<String>,
    alreadyProcessed: &HashSet<String>,
    zoneIndices: &HashSet<usize>,
) {
    // Build tool_call_id → path map for readFile calls (to detect edit gate invalidation).
    let mut readFilePaths: HashMap<String, String> = HashMap::new();
    for msg in history.iter() {
        if let Message::Assistant {
            tool_calls: Some(calls),
            ..
        } = msg
        {
            for call in calls {
                if call.function.name == "readFile" {
                    if let Ok(args) =
                        serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    {
                        if let Some(path) = args["path"].as_str() {
                            readFilePaths.insert(call.id.clone(), normalizePath(path));
                        }
                    }
                }
            }
        }
    }

    // Truncate long Tool results in place — only within the S1 zone.
    // Also strip image blocks from multimodal content (same pass).
    for (i, msg) in history.iter_mut().enumerate() {
        if !zoneIndices.contains(&i) {
            continue;
        }
        if let Message::Tool {
            tool_call_id,
            content,
        } = msg
        {
            // Strip images from multimodal content in the S1 zone.
            if content.hasImages() {
                *content = content.stripImages();
            }

            let textLen = content.charCount();
            if textLen <= threshold {
                continue;
            }
            if alreadyProcessed.contains(tool_call_id.as_str()) {
                continue;
            }
            let blockId = blockHints.get(tool_call_id).map(|s| s.as_str());
            let before = textLen;
            let text = content.textContent();
            let truncated = middleOut(text, threshold, blockId);
            *content = crate::message::Content::text(truncated);
            tracing::debug!(
                tool_call_id = %tool_call_id,
                threshold,
                beforeLen = before,
                afterLen = content.charCount(),
                "S1 middle-out applied"
            );
            middleOutCallIds.push(tool_call_id.clone());

            // If this was a readFile result, invalidate the edit gate.
            if let Some(path) = readFilePaths.get(tool_call_id) {
                if !invalidatedFiles.contains(path) {
                    invalidatedFiles.push(path.clone());
                }
            }
        }

        // Strip images from User messages in the S1 zone too.
        if let Message::User { content } = msg {
            if content.hasImages() {
                *content = content.stripImages();
            }
        }
    }
}

/// Middle-out truncation: keep head and tail, remove the middle.
///
/// When `blockId` is provided, the marker tells the model which block
/// to fetch with `historyFetch` to get the untruncated content.
fn middleOut(text: &str, threshold: usize, blockId: Option<&str>) -> String {
    if text.len() <= threshold {
        return text.to_string();
    }

    let keepEach = threshold / 2;
    let head = &text[..text.floor_char_boundary(keepEach)];
    let tail = &text[text.ceil_char_boundary(text.len() - keepEach)..];
    let removedBytes = text.len() - head.len() - tail.len();

    let hint = match blockId {
        Some(bid) => format!(
            "[... {removedBytes} bytes truncated \u{2014} \
             use historyFetch(\"{bid}\") to retrieve full content ...]"
        ),
        None => format!(
            "[... {removedBytes} bytes truncated \u{2014} \
             use historySearch to find full content ...]"
        ),
    };

    format!("{head}\n\n{hint}\n\n{tail}")
}

/// Calculate the S1 zone: message indices in the oldest 30% by char count.
///
/// Skips index 0 (system message). Walks from oldest to newest,
/// accumulating character counts until 30% of total is reached.
fn calculateZone(history: &[Message]) -> HashSet<usize> {
    let mut zone = HashSet::new();
    if history.len() <= 1 {
        return zone;
    }

    let totalChars: usize = history[1..].iter().map(|m| messageLen(m)).sum();
    if totalChars == 0 {
        return zone;
    }

    let boundary = totalChars * 30 / 100;
    let mut cumulative: usize = 0;

    for i in 1..history.len() {
        if cumulative >= boundary {
            break;
        }
        cumulative += messageLen(&history[i]);
        // Include the message even if it straddles the boundary.
        zone.insert(i);
    }

    zone
}

/// Rough character count for a message.
fn messageLen(msg: &Message) -> usize {
    match msg {
        Message::System { content } => content.len(),
        Message::User { content } => content.charCount(),
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let textLen = content.as_ref().map_or(0, |c| c.len());
            let callsLen = tool_calls.as_ref().map_or(0, |calls| {
                calls
                    .iter()
                    .map(|c| c.function.arguments.len() + c.function.name.len())
                    .sum()
            });
            textLen + callsLen
        }
        Message::Tool { content, .. } => content.charCount(),
    }
}

/// Best-effort path normalization for dedup comparison.
fn normalizePath(path: &str) -> String {
    std::path::Path::new(path)
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string())
}
