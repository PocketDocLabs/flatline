//! Live context — derives the API message list from transcript + compaction log.
//!
//! The transcript is the permanent record; the compaction log records
//! transformations applied to it. This module replays both to produce
//! the `Vec<Message>` sent to the API.
//!
//! Reconstruction walks the parent-child turn tree from a head turn
//! backward to the root, then applies compaction operations to the
//! resulting chain.
//!
//! # Public API
//! - [`reconstruct`] — rebuild history from transcript + compaction log
//! - [`calculateZones`] — determine S2/S3 zone boundaries for compaction
//! - [`buildState`] — `/context` geological layer data (rendered by deck)
//!
//! # Dependencies
//! `crate::compaction`, `crate::transcript`, `crate::message`

use std::collections::{HashMap, HashSet};

use crate::compaction::{CompactionLog, CompactionOp};
use crate::message::{FunctionCall, Message, ToolCall};
use crate::transcript::{Transcript, Turn, TurnRole};

use anyhow::Result;

/// Reconstruct conversation history by walking the parent-child chain
/// from `headTurnId` back to the root.
///
/// Applies compaction operations whose targets are in the active chain,
/// then reassembles into the grouped Message format the API expects.
/// Does NOT include the system prompt — the caller prepends that.
pub fn reconstruct(
    transcript: &Transcript,
    compactionLog: &CompactionLog,
    headTurnId: &str,
    promptThinking: bool,
) -> Result<Vec<Message>> {
    let allTurns = transcript.loadAll()?;
    let chain = walkChain(&allTurns, headTurnId);

    // Build sets for compaction op filtering.
    let activeBlockIds: HashSet<&str> = chain.iter().map(|t| t.blockId.as_str()).collect();

    let allOps = compactionLog.loadAll()?;
    let ops = filterOpsForChain(allOps, &activeBlockIds);

    let transformed = applyOps(&chain, &ops);
    Ok(assembleMessages(&transformed, promptThinking))
}

/// Walk the parent-child chain from `headTurnId` to root.
///
/// Returns turns in chronological order (root first).
fn walkChain<'a>(allTurns: &'a [Turn], headTurnId: &str) -> Vec<&'a Turn> {
    let turnMap: HashMap<&str, &Turn> = allTurns.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut chain: Vec<&Turn> = Vec::new();
    let mut current: Option<&str> = Some(headTurnId);

    while let Some(id) = current {
        if let Some(turn) = turnMap.get(id) {
            // Skip system turns (ephemeral, shouldn't be in transcript but might
            // exist in old sessions).
            if !matches!(turn.role, TurnRole::System) {
                chain.push(turn);
            }
            current = turn.parentId.as_deref();
        } else {
            break;
        }
    }

    chain.reverse();
    chain
}

/// Filter compaction ops to those whose targets are in the active chain.
///
/// - FileDedup/MiddleOut: apply all (toolCallId matching is a natural no-op
///   for turns not in the chain).
/// - BlockCompact: apply if blockId is in active chain.
/// - TopicCompact/FullCompact: apply if ALL source blocks are in active chain
///   (cross-branch summaries are invalid).
fn filterOpsForChain(ops: Vec<CompactionOp>, activeBlockIds: &HashSet<&str>) -> Vec<CompactionOp> {
    ops.into_iter()
        .filter(|op| match op {
            CompactionOp::FileDedup { .. } | CompactionOp::MiddleOut { .. } => true,
            CompactionOp::BlockCompact { blockId, .. } => {
                activeBlockIds.contains(blockId.as_str())
            }
            CompactionOp::TopicCompact { sourceBlockIds, .. } => {
                sourceBlockIds.iter().all(|id| activeBlockIds.contains(id.as_str()))
            }
            CompactionOp::FullCompact { sourceIds, .. } => {
                sourceIds.iter().all(|id| activeBlockIds.contains(id.as_str()))
            }
        })
        .collect()
}

/// What kind of compaction produced this summary.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SummaryKind {
    /// S2 per-block compaction — wraps in `<compressed_content>` as User.
    Block,
    /// S3 topic compaction.
    Topic,
    /// S4 full compaction.
    Full,
}

/// A turn after compaction operations have been applied.
/// Some turns are removed, some have content replaced.
#[allow(dead_code)]
enum TransformedTurn<'a> {
    /// Turn passes through unchanged.
    Original(&'a Turn),
    /// Turn content was replaced (middle-out, or block summary injected).
    Replaced { turn: &'a Turn, newContent: String },
    /// A synthetic summary turn injected by S2/S3/S4.
    Summary {
        blockId: String,
        content: String,
        kind: SummaryKind,
        /// Block IDs covered by this summary (for historyFetch references).
        sourceBlockIds: Vec<String>,
    },
}

impl<'a> TransformedTurn<'a> {
    #[allow(dead_code)]
    fn blockId(&self) -> &str {
        match self {
            Self::Original(t) | Self::Replaced { turn: t, .. } => &t.blockId,
            Self::Summary { blockId, .. } => blockId,
        }
    }
}

/// Apply compaction operations to a turn stream.
///
/// Operations are applied in log order (oldest first). Each operation
/// type transforms the turn list differently:
///
/// - FileDedup: remove turns by tool_call_id (both ToolCall and ToolResult)
/// - MiddleOut: truncate ToolResult content by tool_call_id
/// - BlockCompact: replace assistant/tool turns in a block with summary
///   (user message stays verbatim)
/// - TopicCompact: replace all turns in the source blocks with a single summary
/// - FullCompact: replace all turns in source blocks with briefing
fn applyOps<'a>(turns: &'a [&'a Turn], ops: &[CompactionOp]) -> Vec<TransformedTurn<'a>> {
    // S1: tool_call_ids to remove entirely (FileDedup).
    let mut removedCallIds: HashSet<String> = HashSet::new();
    // S1: tool_call_ids to middle-out truncate (MiddleOut). Maps callId → threshold.
    let mut middleOutCallIds: HashMap<String, usize> = HashMap::new();

    // Track which blocks are fully replaced by a summary.
    // Maps blockId → summary content.
    let mut blockSummaries: HashMap<String, String> = HashMap::new();
    // Blocks whose turns should be removed (replaced by summary).
    let mut summarizedBlocks: HashSet<String> = HashSet::new();
    // For S2: blocks where only assistant/tool turns are replaced (user stays).
    let mut s2SummarizedBlocks: HashSet<String> = HashSet::new();
    // Track which compaction kind produced each summary (for XML wrapping).
    let mut summaryKinds: HashMap<String, SummaryKind> = HashMap::new();
    // Track source block IDs for each summary (for historyFetch references).
    let mut summaryBlockIds: HashMap<String, Vec<String>> = HashMap::new();

    for op in ops {
        match op {
            CompactionOp::FileDedup { targetIds, .. } => {
                for id in targetIds {
                    removedCallIds.insert(id.clone());
                }
            }
            CompactionOp::MiddleOut { targetIds, threshold, .. } => {
                for id in targetIds {
                    middleOutCallIds.insert(id.clone(), *threshold);
                }
            }
            CompactionOp::BlockCompact { blockId, summary, .. } => {
                s2SummarizedBlocks.insert(blockId.clone());
                blockSummaries.insert(blockId.clone(), summary.clone());
                summaryKinds.insert(blockId.clone(), SummaryKind::Block);
                summaryBlockIds.insert(blockId.clone(), vec![blockId.clone()]);
            }
            CompactionOp::TopicCompact { sourceBlockIds, summary, .. } => {
                if let Some(first) = sourceBlockIds.first() {
                    blockSummaries.insert(first.clone(), summary.clone());
                    summaryKinds.insert(first.clone(), SummaryKind::Topic);
                    summaryBlockIds.insert(first.clone(), sourceBlockIds.clone());
                }
                for bid in sourceBlockIds {
                    summarizedBlocks.insert(bid.clone());
                }
            }
            CompactionOp::FullCompact { sourceIds, summary, .. } => {
                // Clear stale S3 summaries for blocks this S4 now covers.
                for bid in sourceIds {
                    blockSummaries.remove(bid.as_str());
                    summaryKinds.remove(bid.as_str());
                    summaryBlockIds.remove(bid.as_str());
                }
                if let Some(first) = sourceIds.first() {
                    blockSummaries.insert(first.clone(), summary.clone());
                    summaryKinds.insert(first.clone(), SummaryKind::Full);
                    summaryBlockIds.insert(first.clone(), sourceIds.clone());
                }
                for bid in sourceIds {
                    summarizedBlocks.insert(bid.clone());
                }
            }
        }
    }

    let mut result: Vec<TransformedTurn> = Vec::new();
    // Track which block summaries we've already emitted.
    let mut emittedSummaries: HashSet<String> = HashSet::new();

    for turn in turns {
        let bid = &turn.blockId;

        // Full block replacement (S3/S4): emit summary once, skip all turns.
        if summarizedBlocks.contains(bid.as_str()) {
            if !emittedSummaries.contains(bid.as_str()) {
                if let Some(summary) = blockSummaries.get(bid.as_str()) {
                    let kind = *summaryKinds.get(bid.as_str()).unwrap_or(&SummaryKind::Topic);
                    let blockIds = summaryBlockIds.get(bid.as_str())
                        .cloned()
                        .unwrap_or_else(|| vec![bid.clone()]);
                    result.push(TransformedTurn::Summary {
                        blockId: bid.clone(),
                        content: summary.clone(),
                        kind,
                        sourceBlockIds: blockIds,
                    });
                    emittedSummaries.insert(bid.clone());
                }
            }
            continue;
        }

        // S2 block compact: user messages stay, assistant/tool turns replaced.
        if s2SummarizedBlocks.contains(bid.as_str()) {
            if matches!(turn.role, TurnRole::User) {
                result.push(TransformedTurn::Original(turn));
            } else if !emittedSummaries.contains(bid.as_str()) {
                if let Some(summary) = blockSummaries.get(bid.as_str()) {
                    result.push(TransformedTurn::Summary {
                        blockId: bid.clone(),
                        content: summary.clone(),
                        kind: SummaryKind::Block,
                        sourceBlockIds: vec![bid.clone()],
                    });
                    emittedSummaries.insert(bid.clone());
                }
            }
            // Skip remaining assistant/tool turns in this block.
            continue;
        }

        // FileDedup: skip turns whose toolCallId was removed.
        if let Some(tcid) = &turn.toolCallId {
            if removedCallIds.contains(tcid.as_str()) {
                continue;
            }
        }

        // MiddleOut: truncate ToolResult content.
        if let Some(tcid) = &turn.toolCallId {
            if let Some(&thresh) = middleOutCallIds.get(tcid.as_str()) {
                if matches!(turn.role, TurnRole::ToolResult) {
                    result.push(TransformedTurn::Replaced {
                        turn,
                        newContent: middleOut(&turn.content, thresh, Some(&turn.blockId)),
                    });
                    continue;
                }
            }
        }

        result.push(TransformedTurn::Original(turn));
    }

    result
}

/// Pending assistant text/reasoning that may be merged with tool calls.
struct PendingAssistant {
    content: Option<String>,
    reasoning: Option<String>,
}

/// Assemble transformed turns into the grouped Message format.
///
/// Tool calls are stored as individual turns in the transcript but the
/// API expects them grouped into a single Assistant message with a
/// `tool_calls` vec, followed by one Tool message per result.
///
/// When an Assistant text turn is immediately followed by ToolCall turns,
/// the text is merged into the tool-call Assistant message rather than
/// emitting two consecutive Assistant messages.
fn assembleMessages(turns: &[TransformedTurn], promptThinking: bool) -> Vec<Message> {
    let mut history: Vec<Message> = Vec::new();
    let mut pendingCalls: Vec<ToolCall> = Vec::new();
    // Assistant content waiting to see if tool calls follow.
    let mut pendingAssistant: Option<PendingAssistant> = None;

    for tt in turns {
        match tt {
            TransformedTurn::Summary { content, blockId, kind, sourceBlockIds } => {
                flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls, promptThinking);
                let wrapped = match kind {
                    SummaryKind::Block => formatBlockSummary(content, blockId),
                    SummaryKind::Topic => formatTopicSummary(content, sourceBlockIds),
                    SummaryKind::Full => formatFullBriefing(content, sourceBlockIds),
                };
                history.push(Message::User { content: wrapped.into() });
            }
            TransformedTurn::Original(turn) | TransformedTurn::Replaced { turn, .. } => {
                let content = match tt {
                    TransformedTurn::Replaced { newContent, .. } => newContent.clone(),
                    _ => turn.content.clone(),
                };

                match turn.role {
                    TurnRole::ToolCall => {
                        // Don't flush pendingAssistant — it will be merged
                        // into the Assistant message when tool calls flush.
                        let callId = turn.toolCallId.clone().unwrap_or_default();
                        let toolName = turn.tool.clone().unwrap_or_default();
                        let argsStr = turn.args.as_ref()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "{}".to_string());

                        pendingCalls.push(ToolCall {
                            id: callId,
                            callType: "function".to_string(),
                            function: FunctionCall {
                                name: toolName,
                                arguments: argsStr,
                            },
                        });
                    }
                    TurnRole::Assistant => {
                        // Flush any prior pending state before holding this one.
                        flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls, promptThinking);

                        let (finalContent, finalReasoning) = if promptThinking {
                            let merged = match (&turn.reasoning, content.is_empty()) {
                                (Some(r), false) => format!("<scratchpad>\n{r}\n</scratchpad>\n{content}"),
                                (Some(r), true) => format!("<scratchpad>\n{r}\n</scratchpad>"),
                                (None, _) => content,
                            };
                            (Some(merged), None)
                        } else {
                            (Some(content), turn.reasoning.clone())
                        };

                        // Hold — don't emit yet. If ToolCall turns follow,
                        // this content will be merged into that message.
                        pendingAssistant = Some(PendingAssistant {
                            content: finalContent,
                            reasoning: finalReasoning,
                        });
                    }
                    TurnRole::User => {
                        flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls, promptThinking);
                        let msgContent = rebuildContent(&content, &turn.attachments);
                        history.push(Message::User { content: msgContent });
                    }
                    TurnRole::ToolResult => {
                        flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls, promptThinking);
                        let msgContent = rebuildContent(&content, &turn.attachments);
                        history.push(Message::Tool {
                            tool_call_id: turn.toolCallId.clone().unwrap_or_default(),
                            content: msgContent,
                        });
                    }
                    TurnRole::System => {}
                }
            }
        }
    }

    flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls, promptThinking);
    history
}

/// Rebuild `Content` from turn text + optional persisted attachments.
///
/// When a turn has TurnAttachments, reconstruct multimodal content with
/// data URIs. Otherwise, return plain text content.
fn rebuildContent(
    text: &str,
    attachments: &Option<Vec<crate::transcript::TurnAttachment>>,
) -> crate::message::Content {
    match attachments {
        Some(atts) if !atts.is_empty() => {
            let imageUris: Vec<String> = atts.iter().map(|a| {
                format!("data:{};base64,{}", a.mimeType, a.data)
            }).collect();
            crate::message::Content::withImages(text, imageUris)
        }
        _ => crate::message::Content::text(text),
    }
}

/// Flush pending assistant content and/or tool calls into history.
///
/// If both are pending, they merge into a single Assistant message
/// (content + tool_calls). If only one is pending, it flushes alone.
fn flushPending(
    history: &mut Vec<Message>,
    pendingAssistant: &mut Option<PendingAssistant>,
    pendingCalls: &mut Vec<ToolCall>,
    _promptThinking: bool,
) {
    if !pendingCalls.is_empty() {
        // Merge any pending assistant content into the tool-call message.
        let (content, reasoning) = match pendingAssistant.take() {
            Some(pa) => (pa.content, pa.reasoning),
            None => (None, None),
        };
        history.push(Message::Assistant {
            content,
            tool_calls: Some(std::mem::take(pendingCalls)),
            reasoning,
        });
    } else if let Some(pa) = pendingAssistant.take() {
        // No tool calls followed — emit as standalone assistant text.
        history.push(Message::Assistant {
            content: pa.content,
            tool_calls: None,
            reasoning: pa.reasoning,
        });
    }
}

// ---------------------------------------------------------------------------
// Zone calculation
// ---------------------------------------------------------------------------

/// Zone boundaries for compaction targeting.
///
/// Indices are into the history `Vec<Message>`.
pub struct Zones {
    /// S3 zone: oldest 30% of context by character count.
    pub s3Zone: Vec<usize>,
    /// S2 zone: next 30% (30%–60%) of context by character count.
    pub s2Zone: Vec<usize>,
}

/// Calculate zone boundaries for the current history.
///
/// Walks messages from oldest to newest, accumulating character counts.
/// The first 30% of total chars forms the S3 zone, the next 30% forms
/// the S2 zone. The remaining 40% is the raw zone (untouched).
///
/// The system message (index 0) is excluded from zones — it's never compacted.
pub fn calculateZones(
    history: &[Message],
    _contextWindow: usize,
    _compactRatio: f64,
) -> Zones {
    if history.len() <= 1 {
        return Zones {
            s3Zone: Vec::new(),
            s2Zone: Vec::new(),
        };
    }

    // Calculate total character count (excluding system message).
    let charCounts: Vec<usize> = history
        .iter()
        .map(|m| messageCharCount(m))
        .collect();

    let totalChars: usize = charCounts[1..].iter().sum();
    if totalChars == 0 {
        return Zones {
            s3Zone: Vec::new(),
            s2Zone: Vec::new(),
        };
    }

    let s3Boundary = totalChars * 30 / 100;
    let s2Boundary = totalChars * 60 / 100;

    let mut s3Zone = Vec::new();
    let mut s2Zone = Vec::new();
    let mut cumulative: usize = 0;

    // Start at index 1 to skip the system message.
    for i in 1..history.len() {
        cumulative += charCounts[i];

        if cumulative <= s3Boundary {
            s3Zone.push(i);
        } else if cumulative <= s2Boundary {
            s2Zone.push(i);
        }
        // Past 60% = raw zone, not included.
    }

    Zones { s3Zone, s2Zone }
}

/// Rough character count for a message (used for zone calculation).
fn messageCharCount(msg: &Message) -> usize {
    match msg {
        Message::System { content } => content.len(),
        Message::User { content } => content.charCount(),
        Message::Assistant { content, tool_calls, .. } => {
            let textLen = content.as_ref().map_or(0, |c| c.len());
            let callsLen = tool_calls.as_ref().map_or(0, |calls| {
                calls.iter().map(|c| c.function.arguments.len() + c.function.name.len()).sum()
            });
            textLen + callsLen
        }
        Message::Tool { content, .. } => content.charCount(),
    }
}

// ---------------------------------------------------------------------------
// /context display
// ---------------------------------------------------------------------------

/// Input parameters for building context state.
pub struct BuildStateInput<'a> {
    pub contextWindow: usize,
    pub compactionLog: &'a CompactionLog,
    pub reportedTokens: usize,
    pub transcript: &'a Transcript,
    pub headTurnId: &'a str,
}

/// S4 briefing layer — the deepest compression.
#[derive(Debug, Clone)]
pub struct S4Layer {
    /// How many topics were merged into the briefing.
    pub topicsMerged: usize,
    /// How many prior S4 briefings were superseded.
    pub priorBriefings: usize,
    /// Total turns covered by this briefing.
    pub turnsCovered: usize,
    pub estimatedTokens: usize,
}

/// S3 topic summary layer.
#[derive(Debug, Clone)]
pub struct S3Layer {
    /// Labels of topics currently held as summaries.
    pub topicLabels: Vec<String>,
    /// Total turns condensed into these topic summaries.
    pub turnsCondensed: usize,
    pub estimatedTokens: usize,
}

/// S2 per-turn summary layer.
#[derive(Debug, Clone)]
pub struct S2Layer {
    /// Total turns condensed into summaries.
    pub turnsCondensed: usize,
    pub estimatedTokens: usize,
}

/// Raw (uncompressed) layer.
#[derive(Debug, Clone)]
pub struct RawLayer {
    pub turns: usize,
    pub estimatedTokens: usize,
}

/// Full context state for /context display.
#[derive(Debug, Clone)]
pub struct ContextState {
    pub estimatedTokens: usize,
    pub reportedTokens: usize,
    pub contextWindow: usize,
    // Layers — only present if that stage has fired.
    pub s4: Option<S4Layer>,
    pub s3: Option<S3Layer>,
    pub s2: Option<S2Layer>,
    pub raw: RawLayer,
}

/// Build context state for /context display.
///
/// Walks the compaction log to determine what lives in each layer,
/// then estimates token counts from the transformed turn stream.
pub fn buildState(input: &BuildStateInput) -> ContextState {
    let ops = input.compactionLog.loadAll().unwrap_or_default();

    if input.headTurnId.is_empty() {
        return ContextState {
            estimatedTokens: 0,
            reportedTokens: input.reportedTokens,
            contextWindow: input.contextWindow,
            s4: None,
            s3: None,
            s2: None,
            raw: RawLayer { turns: 0, estimatedTokens: 0 },
        };
    }

    let allTurns = match input.transcript.loadAll() {
        Ok(t) => t,
        Err(_) => {
            return ContextState {
                estimatedTokens: 0,
                reportedTokens: input.reportedTokens,
                contextWindow: input.contextWindow,
                s4: None,
                s3: None,
                s2: None,
                raw: RawLayer { turns: 0, estimatedTokens: 0 },
            };
        }
    };

    let chain = walkChain(&allTurns, input.headTurnId);
    let activeBlockIds: HashSet<&str> = chain.iter().map(|t| t.blockId.as_str()).collect();
    let filteredOps = filterOpsForChain(ops, &activeBlockIds);
    let transformed = applyOps(&chain, &filteredOps);

    // Count unique blocks in the chain (each block = one user-visible turn).
    let allBlockIds: Vec<&str> = {
        let mut seen = HashSet::new();
        chain.iter()
            .filter(|t| seen.insert(t.blockId.as_str()))
            .map(|t| t.blockId.as_str())
            .collect()
    };

    // Track which blocks are covered by each compaction stage.
    let mut s4Blocks: HashSet<&str> = HashSet::new();
    let mut s3Blocks: HashSet<&str> = HashSet::new();
    let mut s2Blocks: HashSet<&str> = HashSet::new();

    // NOTE: FullCompact supersedes earlier TopicCompact ops for the same blocks.
    // We count topics that were folded into S4 by tracking TopicCompact labels
    // whose source blocks are now covered by a FullCompact.
    let mut s4CoveredTopicLabels: HashSet<String> = HashSet::new();

    // S3: active topic labels (not superseded by S4).
    let mut s3TopicLabels: Vec<String> = Vec::new();

    // First pass: identify S4 coverage.
    let mut fullCompactCount = 0usize;
    for op in &filteredOps {
        if let CompactionOp::FullCompact { sourceIds, .. } = op {
            fullCompactCount += 1;
            for id in sourceIds {
                s4Blocks.insert(id.as_str());
            }
        }
    }
    // Prior briefings = total S4 ops minus the current active one.
    let s4PriorBriefings = fullCompactCount.saturating_sub(1);

    // Second pass: classify S3 and S2, count S4 topics.
    for op in &filteredOps {
        match op {
            CompactionOp::TopicCompact { sourceBlockIds, topicLabel, .. } => {
                let coveredByS4 = sourceBlockIds.iter()
                    .all(|id| s4Blocks.contains(id.as_str()));
                if coveredByS4 {
                    s4CoveredTopicLabels.insert(topicLabel.clone());
                } else {
                    s3TopicLabels.push(topicLabel.clone());
                    for id in sourceBlockIds {
                        s3Blocks.insert(id.as_str());
                    }
                }
            }
            CompactionOp::BlockCompact { blockId, .. } => {
                // Only count if not already covered by S3 or S4.
                if !s4Blocks.contains(blockId.as_str())
                    && !s3Blocks.contains(blockId.as_str())
                {
                    s2Blocks.insert(blockId.as_str());
                }
            }
            _ => {}
        }
    }

    let s4TopicsMerged = s4CoveredTopicLabels.len();

    // Count turns (blocks) per layer.
    let s4TurnCount = allBlockIds.iter()
        .filter(|bid| s4Blocks.contains(**bid))
        .count();
    let s3TurnCount = allBlockIds.iter()
        .filter(|bid| s3Blocks.contains(**bid))
        .count();
    let s2TurnCount = allBlockIds.iter()
        .filter(|bid| s2Blocks.contains(**bid))
        .count();
    let rawTurnCount = allBlockIds.iter()
        .filter(|bid| {
            !s4Blocks.contains(**bid)
                && !s3Blocks.contains(**bid)
                && !s2Blocks.contains(**bid)
        })
        .count();

    // Estimate tokens per layer from the transformed stream.
    let mut s4Tokens = 0usize;
    let mut s3Tokens = 0usize;
    let mut s2Tokens = 0usize;
    let mut rawTokens = 0usize;

    for tt in &transformed {
        let chars = match tt {
            TransformedTurn::Original(turn) => turn.content.len(),
            TransformedTurn::Replaced { newContent, .. } => newContent.len(),
            TransformedTurn::Summary { content, .. } => content.len(),
        };
        let tokens = chars / 4;

        match tt {
            TransformedTurn::Summary { kind, .. } => match kind {
                SummaryKind::Full => s4Tokens += tokens,
                SummaryKind::Topic => s3Tokens += tokens,
                SummaryKind::Block => s2Tokens += tokens,
            },
            TransformedTurn::Original(turn) | TransformedTurn::Replaced { turn, .. } => {
                let bid = turn.blockId.as_str();
                if s4Blocks.contains(bid) {
                    s4Tokens += tokens;
                } else if s3Blocks.contains(bid) {
                    s3Tokens += tokens;
                } else if s2Blocks.contains(bid) {
                    s2Tokens += tokens;
                } else {
                    rawTokens += tokens;
                }
            }
        }
    }

    // Build layers — only present if that stage has produced output.
    let s4 = if fullCompactCount > 0 {
        Some(S4Layer {
            topicsMerged: s4TopicsMerged,
            priorBriefings: s4PriorBriefings,
            turnsCovered: s4TurnCount,
            estimatedTokens: s4Tokens,
        })
    } else {
        None
    };

    let s3 = if !s3TopicLabels.is_empty() {
        Some(S3Layer {
            topicLabels: s3TopicLabels,
            turnsCondensed: s3TurnCount,
            estimatedTokens: s3Tokens,
        })
    } else {
        None
    };

    let s2 = if !s2Blocks.is_empty() {
        Some(S2Layer {
            turnsCondensed: s2TurnCount,
            estimatedTokens: s2Tokens,
        })
    } else {
        None
    };

    let raw = RawLayer {
        turns: rawTurnCount,
        estimatedTokens: rawTokens,
    };

    // Use sum of layer estimates as the total — more accurate than
    // history char count because it reflects the actual compacted content.
    let layerTotal = s4.as_ref().map(|l| l.estimatedTokens).unwrap_or(0)
        + s3.as_ref().map(|l| l.estimatedTokens).unwrap_or(0)
        + s2.as_ref().map(|l| l.estimatedTokens).unwrap_or(0)
        + raw.estimatedTokens;

    ContextState {
        estimatedTokens: layerTotal,
        reportedTokens: input.reportedTokens,
        contextWindow: input.contextWindow,
        s4,
        s3,
        s2,
        raw,
    }
}

/// Format a token count with K/M suffixes.
pub fn formatTokenCount(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 10_000 {
        format!("{:.0}k", count as f64 / 1_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Wrap an S2 block summary in `<compressed_content>` XML.
///
/// The `<referenced_turns>` section tells the model how to retrieve the
/// original uncompressed content via `historyFetch`.
fn formatBlockSummary(content: &str, blockId: &str) -> String {
    format!(
        "<compressed_content>\n\
         <agent_work>{content}</agent_work>\n\
         <referenced_turns>Use historyFetch(\"{blockId}\") to retrieve the \
         original uncompressed content from this exchange.</referenced_turns>\n\
         </compressed_content>"
    )
}

/// Wrap an S3 topic summary in `<compressed_content>` XML.
///
/// Lists all source block IDs so the model can retrieve original
/// exchanges via `historyFetch`.
fn formatTopicSummary(content: &str, sourceBlockIds: &[String]) -> String {
    let refs = sourceBlockIds
        .iter()
        .map(|id| format!("historyFetch(\"{id}\")"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "<compressed_content>\n\
         <topic_summary>{content}</topic_summary>\n\
         <referenced_turns>Use {refs} to retrieve \
         original exchanges from this topic.</referenced_turns>\n\
         </compressed_content>"
    )
}

/// Wrap an S4 handoff briefing in `<session_briefing>` XML.
fn formatFullBriefing(content: &str, sourceBlockIds: &[String]) -> String {
    let first = sourceBlockIds.first().map(|s| s.as_str()).unwrap_or("b_00000000");
    let last = sourceBlockIds.last().map(|s| s.as_str()).unwrap_or(first);
    format!(
        "<session_briefing>\n\
         {content}\n\
         <referenced_turns>Use historyFetch or historySearch to retrieve \
         original exchanges from blocks {first} through {last}.</referenced_turns>\n\
         </session_briefing>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::CompactionLog;
    use crate::transcript::Transcript;

    // -----------------------------------------------------------------------
    // Harness
    // -----------------------------------------------------------------------

    /// Ephemeral transcript that cleans up on drop.
    struct TestSession {
        transcript: Transcript,
        headTurnId: Option<String>,
    }

    impl TestSession {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CTR: AtomicU64 = AtomicU64::new(0);
            let n = CTR.fetch_add(1, Ordering::Relaxed);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            let id = format!("test_{ts:x}_{n:04x}");
            Self {
                transcript: Transcript::create(&id).unwrap(),
                headTurnId: None,
            }
        }

        fn compactionLog(&self) -> CompactionLog {
            CompactionLog::open(self.transcript.sessionDir()).unwrap()
        }

        /// Record a user message, chaining to the current head.
        /// Mirrors session.rs:658 — `recordUser(msg, headTurnId)`.
        fn user(&mut self, content: &str) {
            let id = self.transcript
                .recordUser(content, self.headTurnId.as_deref(), None)
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a user message with image attachments.
        fn userWithImages(&mut self, content: &str, attachments: Vec<crate::transcript::TurnAttachment>) {
            let id = self.transcript
                .recordUser(content, self.headTurnId.as_deref(), Some(attachments))
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record assistant text (and optional reasoning).
        /// Mirrors session.rs:1182-1187 — content recorded before tool calls.
        fn assistant(&mut self, content: &str, reasoning: Option<&str>) {
            let id = self.transcript
                .recordAssistant(content, reasoning, None)
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a tool call.
        /// Mirrors session.rs:752-764.
        fn toolCall(&mut self, callId: &str, name: &str, args: serde_json::Value) {
            let id = self.transcript
                .recordToolCall(callId, name, &args)
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a tool result.
        /// Mirrors session.rs:2084-2093 (pushToolResult).
        fn toolResult(&mut self, callId: &str, content: &str) {
            let id = self.transcript
                .recordToolResult(callId, content, None)
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Reconstruct messages from the current head.
        fn reconstruct(&self, promptThinking: bool) -> Vec<Message> {
            let head = self.headTurnId.as_ref().expect("no turns recorded");
            let log = self.compactionLog();
            reconstruct(&self.transcript, &log, head, promptThinking).unwrap()
        }
    }

    impl Drop for TestSession {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(self.transcript.sessionDir());
        }
    }

    /// Print all messages for debugging (visible with `--nocapture`).
    fn dump(label: &str, msgs: &[Message]) {
        println!("{label} — {} messages:", msgs.len());
        for (i, m) in msgs.iter().enumerate() {
            match m {
                Message::System { content } => {
                    println!("  [{i}] System ({} chars)", content.len());
                }
                Message::User { content } => {
                    println!("  [{i}] User: {}", truncate(content.textContent(), 60));
                }
                Message::Assistant { content, tool_calls, reasoning } => {
                    println!(
                        "  [{i}] Assistant: content={:?} tool_calls={} reasoning={}",
                        content.as_deref().map(|c| truncate(c, 40)),
                        tool_calls.as_ref().map(|c| c.len()).unwrap_or(0),
                        reasoning.is_some(),
                    );
                }
                Message::Tool { tool_call_id, content } => {
                    println!("  [{i}] Tool({tool_call_id}): {}", truncate(content.textContent(), 50));
                }
            }
        }
    }

    fn truncate(s: &str, max: usize) -> String {
        if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max]) }
    }

    /// Assert a message is `Assistant` and return (content, toolCallCount, hasReasoning).
    fn assertAssistant(msg: &Message) -> (Option<&str>, usize, bool) {
        match msg {
            Message::Assistant { content, tool_calls, reasoning } => (
                content.as_deref(),
                tool_calls.as_ref().map(|c| c.len()).unwrap_or(0),
                reasoning.is_some(),
            ),
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    /// Assert a message is `Tool` and return (callId, content text).
    fn assertTool(msg: &Message) -> (&str, &str) {
        match msg {
            Message::Tool { tool_call_id, content } => (tool_call_id, content.textContent()),
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    /// Assert a message is `User` and return content text.
    fn assertUser(msg: &Message) -> &str {
        match msg {
            Message::User { content } => content.textContent(),
            other => panic!("expected User, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Baseline — no tools
    // -----------------------------------------------------------------------

    #[test]
    fn simple_exchange() {
        let mut s = TestSession::new();
        s.user("Hello");
        s.assistant("Hi there!", None);

        let msgs = s.reconstruct(false);
        assert_eq!(msgs.len(), 2);
        assert_eq!(assertUser(&msgs[0]), "Hello");
        let (content, calls, _) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some("Hi there!"));
        assert_eq!(calls, 0);
    }

    #[test]
    fn reasoning_preserved() {
        let mut s = TestSession::new();
        s.user("Explain this");
        s.assistant("Here's my answer.", Some("Thought carefully"));

        let msgs = s.reconstruct(false);
        let (content, _, hasReasoning) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some("Here's my answer."));
        assert!(hasReasoning);
    }

    // -----------------------------------------------------------------------
    // Tool call basics
    // -----------------------------------------------------------------------

    /// Single tool call + result. The bread-and-butter pattern.
    #[test]
    fn single_tool_call() {
        let mut s = TestSession::new();
        s.user("Read /tmp/foo.txt");
        s.toolCall("c01", "readFile", serde_json::json!({"path": "/tmp/foo.txt"}));
        s.toolResult("c01", "file contents here");

        let msgs = s.reconstruct(false);
        assert_eq!(msgs.len(), 3);
        assert_eq!(assertUser(&msgs[0]), "Read /tmp/foo.txt");
        let (content, calls, _) = assertAssistant(&msgs[1]);
        assert!(content.is_none() || content == Some(""));
        assert_eq!(calls, 1);
        let (id, body) = assertTool(&msgs[2]);
        assert_eq!(id, "c01");
        assert_eq!(body, "file contents here");
    }

    /// Multiple tool calls grouped into one assistant message.
    #[test]
    fn parallel_tool_calls() {
        let mut s = TestSession::new();
        s.user("Read two files");
        s.toolCall("c10", "readFile", serde_json::json!({"path": "/a.txt"}));
        s.toolCall("c11", "readFile", serde_json::json!({"path": "/b.txt"}));
        s.toolResult("c10", "aaa");
        s.toolResult("c11", "bbb");

        let msgs = s.reconstruct(false);
        assert_eq!(msgs.len(), 4, "User + Assistant(2 calls) + Tool + Tool");
        let (_, calls, _) = assertAssistant(&msgs[1]);
        assert_eq!(calls, 2);
        assert_eq!(assertTool(&msgs[2]).0, "c10");
        assert_eq!(assertTool(&msgs[3]).0, "c11");
    }

    /// Tool call args round-trip through JSON serialization.
    #[test]
    fn tool_call_args_preserved() {
        let mut s = TestSession::new();
        s.user("Do it");
        s.toolCall("c20", "editFile", serde_json::json!({
            "path": "/tmp/x.rs",
            "oldText": "fn foo()",
            "newText": "fn bar()",
        }));
        s.toolResult("c20", "ok");

        let msgs = s.reconstruct(false);
        match &msgs[1] {
            Message::Assistant { tool_calls: Some(calls), .. } => {
                let args: serde_json::Value =
                    serde_json::from_str(&calls[0].function.arguments).unwrap();
                assert_eq!(args["path"], "/tmp/x.rs");
                assert_eq!(args["oldText"], "fn foo()");
                assert_eq!(args["newText"], "fn bar()");
            }
            other => panic!("expected Assistant with tool_calls, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Agent loop — tool calls → result → more tool calls (no user between)
    // -----------------------------------------------------------------------

    /// The core agentic pattern: model calls a tool, sees the result,
    /// then immediately calls another tool without user intervention.
    /// The mid-loop text ("I see the bug...") must merge with the
    /// following tool call into a single Assistant message.
    #[test]
    fn tool_loop_no_user_between() {
        let mut s = TestSession::new();
        s.user("Fix the bug in main.rs");
        // Round 1: model reads the file.
        s.toolCall("c30", "readFile", serde_json::json!({"path": "main.rs"}));
        s.toolResult("c30", "fn main() { bug() }");
        // Round 2: model responds briefly, then edits.
        s.assistant("I see the bug, fixing it.", None);
        s.toolCall("c31", "editFile", serde_json::json!({"path": "main.rs", "old": "bug()", "new": "fix()"}));
        s.toolResult("c31", "ok");
        // Round 3: model responds.
        s.assistant("Fixed. The bug() call was replaced with fix().", None);

        let msgs = s.reconstruct(false);
        dump("tool_loop_no_user_between", &msgs);

        // [0] User
        // [1] Assistant(tool_calls=[c30])
        // [2] Tool(c30)
        // [3] Assistant("I see the bug...", tool_calls=[c31])  ← merged
        // [4] Tool(c31)
        // [5] Assistant("Fixed...")
        assert_eq!(msgs.len(), 6);

        let (content, calls, _) = assertAssistant(&msgs[1]);
        assert!(content.is_none() || content == Some(""));
        assert_eq!(calls, 1);

        let (content, calls, _) = assertAssistant(&msgs[3]);
        assert_eq!(content, Some("I see the bug, fixing it."));
        assert_eq!(calls, 1, "text and tool_calls should be merged");

        let (content, calls, _) = assertAssistant(&msgs[5]);
        assert_eq!(content, Some("Fixed. The bug() call was replaced with fix()."));
        assert_eq!(calls, 0);

        let toolMsgs: Vec<_> = msgs.iter().filter_map(|m| {
            if let Message::Tool { tool_call_id, .. } = m { Some(tool_call_id.as_str()) } else { None }
        }).collect();
        assert_eq!(toolMsgs, vec!["c30", "c31"]);
    }

    // -----------------------------------------------------------------------
    // Content + tool_calls merge
    // -----------------------------------------------------------------------

    /// When the model produces text content AND tool calls in one response,
    /// session.rs records them as separate transcript turns (Assistant then
    /// ToolCall). Reconstruct must merge them into a single Assistant message.
    #[test]
    fn content_plus_tool_calls_merged() {
        let mut s = TestSession::new();
        s.user("Do something");
        s.assistant("Let me check that file.", None);
        s.toolCall("c40", "readFile", serde_json::json!({"path": "/tmp/bar.txt"}));
        s.toolResult("c40", "bar contents");

        let msgs = s.reconstruct(false);
        dump("content_plus_tool_calls_merged", &msgs);

        assert_eq!(msgs.len(), 3, "User + Assistant(content+tool_calls) + Tool");

        let (content, calls, _) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some("Let me check that file."));
        assert_eq!(calls, 1);

        let (id, body) = assertTool(&msgs[2]);
        assert_eq!(id, "c40");
        assert_eq!(body, "bar contents");
    }

    /// Content + reasoning + tool calls all merge into one Assistant message.
    #[test]
    fn content_plus_reasoning_plus_tool_calls_merged() {
        let mut s = TestSession::new();
        s.user("Think about it, then act");
        s.assistant("I'll read the config.", Some("Need to check config before proceeding"));
        s.toolCall("c50", "readFile", serde_json::json!({"path": "config.toml"}));
        s.toolResult("c50", "[settings]\nfoo = true");

        let msgs = s.reconstruct(false);
        dump("content_plus_reasoning_plus_tool_calls_merged", &msgs);

        assert_eq!(msgs.len(), 3, "User + Assistant(content+reasoning+tool_calls) + Tool");

        let (content, calls, hasReasoning) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some("I'll read the config."));
        assert_eq!(calls, 1);
        assert!(hasReasoning);
    }

    // -----------------------------------------------------------------------
    // Multi-turn / cross-block
    // -----------------------------------------------------------------------

    /// Two full exchanges across separate blocks.
    #[test]
    fn multi_turn_interleaved() {
        let mut s = TestSession::new();

        // Block 1.
        s.user("What's in /tmp/x?");
        s.toolCall("c60", "readFile", serde_json::json!({"path": "/tmp/x"}));
        s.toolResult("c60", "x contents");
        s.assistant("The file contains x contents.", None);

        // Block 2.
        s.user("Now read /tmp/y");
        s.toolCall("c61", "readFile", serde_json::json!({"path": "/tmp/y"}));
        s.toolResult("c61", "y contents");

        let msgs = s.reconstruct(false);
        dump("multi_turn_interleaved", &msgs);

        // [0] User [1] Asst(calls) [2] Tool [3] Asst(text) [4] User [5] Asst(calls) [6] Tool
        assert_eq!(msgs.len(), 7);
        assert_eq!(assertUser(&msgs[0]), "What's in /tmp/x?");
        assert_eq!(assertUser(&msgs[4]), "Now read /tmp/y");
        assert_eq!(assertTool(&msgs[2]).0, "c60");
        assert_eq!(assertTool(&msgs[6]).0, "c61");
    }

    /// Three blocks, verifying chain integrity over longer conversations.
    #[test]
    fn three_block_chain() {
        let mut s = TestSession::new();

        s.user("Step 1");
        s.assistant("Done 1.", None);

        s.user("Step 2");
        s.toolCall("c70", "shell", serde_json::json!({"cmd": "echo hello"}));
        s.toolResult("c70", "hello\n");
        s.assistant("Done 2.", None);

        s.user("Step 3");
        s.assistant("Done 3.", None);

        let msgs = s.reconstruct(false);
        dump("three_block_chain", &msgs);

        let userMsgs: Vec<&str> = msgs.iter().filter_map(|m| {
            if let Message::User { content } = m { Some(content.textContent()) } else { None }
        }).collect();
        assert_eq!(userMsgs, vec!["Step 1", "Step 2", "Step 3"]);
    }

    // -----------------------------------------------------------------------
    // Prompt thinking mode (scratchpad)
    // -----------------------------------------------------------------------

    /// With promptThinking=true, reasoning is merged into content as
    /// `<scratchpad>` tags instead of the reasoning field.
    #[test]
    fn prompt_thinking_merges_reasoning() {
        let mut s = TestSession::new();
        s.user("Explain this");
        s.assistant("Here's the answer.", Some("Let me think step by step"));

        let msgs = s.reconstruct(true);

        let (content, _, hasReasoning) = assertAssistant(&msgs[1]);
        assert!(!hasReasoning, "reasoning field should be None in promptThinking mode");
        let text = content.unwrap();
        assert!(text.contains("<scratchpad>"), "content should contain scratchpad open tag");
        assert!(text.contains("Let me think step by step"), "reasoning text in scratchpad");
        assert!(text.contains("Here's the answer."), "content text after scratchpad");
    }

    /// promptThinking with reasoning-only (no content text).
    #[test]
    fn prompt_thinking_reasoning_only() {
        let mut s = TestSession::new();
        s.user("Think");
        // Empty content, reasoning only.
        s.assistant("", Some("Internal monologue"));

        let msgs = s.reconstruct(true);

        let (content, _, hasReasoning) = assertAssistant(&msgs[1]);
        assert!(!hasReasoning);
        let text = content.unwrap();
        assert!(text.contains("<scratchpad>"));
        assert!(text.contains("Internal monologue"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    /// Empty assistant content — should still produce an Assistant message
    /// since the API saw one.
    #[test]
    fn empty_assistant_content() {
        let mut s = TestSession::new();
        s.user("Do nothing");
        s.assistant("", None);

        let msgs = s.reconstruct(false);
        assert_eq!(msgs.len(), 2);
        let (content, _, _) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some(""));
    }

    /// Tool result with empty content.
    #[test]
    fn empty_tool_result() {
        let mut s = TestSession::new();
        s.user("Run it");
        s.toolCall("c80", "shell", serde_json::json!({"cmd": "true"}));
        s.toolResult("c80", "");

        let msgs = s.reconstruct(false);
        assert_eq!(msgs.len(), 3);
        let (_, body) = assertTool(&msgs[2]);
        assert_eq!(body, "");
    }

    /// Long agent loop: 4 consecutive tool calls before a text response.
    /// Verifies that the tool-call accumulator handles many sequential
    /// ToolCall turns correctly.
    #[test]
    fn long_tool_chain() {
        let mut s = TestSession::new();
        s.user("Do everything");

        // Round 1: read.
        s.toolCall("c90", "readFile", serde_json::json!({"path": "/a"}));
        s.toolResult("c90", "a");

        // Round 2: read another.
        s.toolCall("c91", "readFile", serde_json::json!({"path": "/b"}));
        s.toolResult("c91", "b");

        // Round 3: edit.
        s.toolCall("c92", "editFile", serde_json::json!({"path": "/a", "old": "a", "new": "aa"}));
        s.toolResult("c92", "ok");

        // Round 4: shell.
        s.toolCall("c93", "shell", serde_json::json!({"cmd": "cargo test"}));
        s.toolResult("c93", "test passed");

        // Final text response.
        s.assistant("All done.", None);

        let msgs = s.reconstruct(false);
        dump("long_tool_chain", &msgs);

        // Each tool call round = Assistant(calls) + Tool = 2 messages.
        // Total: User(1) + 4*(Asst+Tool) + Asst(text) = 10.
        //
        // BUT: consecutive ToolCall turns without an intervening non-ToolCall
        // turn get grouped. Let's verify what actually happens.
        let toolMsgs: Vec<&str> = msgs.iter().filter_map(|m| {
            if let Message::Tool { tool_call_id, .. } = m { Some(tool_call_id.as_str()) } else { None }
        }).collect();
        assert_eq!(toolMsgs, vec!["c90", "c91", "c92", "c93"],
            "all 4 tool results should be present");

        // Verify the final assistant text is present.
        if let Some(lastAssist) = msgs.iter().rev().find(|m| matches!(m, Message::Assistant { .. })) {
            let (content, calls, _) = assertAssistant(lastAssist);
            assert_eq!(content, Some("All done."));
            assert_eq!(calls, 0);
        } else {
            panic!("no final assistant message found");
        }
    }

    // -----------------------------------------------------------------------
    // Compaction interaction
    // -----------------------------------------------------------------------

    /// MiddleOut compaction truncates tool result content.
    #[test]
    fn middle_out_truncates_tool_result() {
        let mut s = TestSession::new();
        s.user("Read big file");
        s.toolCall("c100", "readFile", serde_json::json!({"path": "/big.txt"}));
        let bigContent = "x".repeat(10_000);
        s.toolResult("c100", &bigContent);
        s.assistant("Got it.", None);

        let afterTurn = s.headTurnId.clone().unwrap();
        let mut log = s.compactionLog();
        log.recordMiddleOut(vec!["c100".into()], &afterTurn, 200).unwrap();
        drop(log);

        let msgs = s.reconstruct(false);

        let (_, body) = assertTool(&msgs[2]);
        assert!(body.len() < bigContent.len(), "tool result should be truncated");
        assert!(body.contains("bytes truncated"), "should contain truncation marker");
    }

    /// FileDedup removes duplicate tool call + result pairs.
    #[test]
    fn file_dedup_removes_pairs() {
        let mut s = TestSession::new();
        s.user("Read it twice");
        s.toolCall("c110", "readFile", serde_json::json!({"path": "/x.txt"}));
        s.toolResult("c110", "first read");
        s.toolCall("c111", "readFile", serde_json::json!({"path": "/x.txt"}));
        s.toolResult("c111", "second read");
        s.assistant("Read it.", None);

        let afterTurn = s.headTurnId.clone().unwrap();
        let mut log = s.compactionLog();
        log.recordFileDedup(vec!["c110".into()], &afterTurn).unwrap();
        drop(log);

        let msgs = s.reconstruct(false);
        dump("file_dedup", &msgs);

        let toolIds: Vec<&str> = msgs.iter().filter_map(|m| {
            if let Message::Tool { tool_call_id, .. } = m { Some(tool_call_id.as_str()) } else { None }
        }).collect();
        assert_eq!(toolIds, vec!["c111"], "only second read should survive dedup");
    }

    /// S2 BlockCompact replaces assistant/tool turns with a summary.
    #[test]
    fn block_compact_replaces_with_summary() {
        let mut s = TestSession::new();
        s.user("First task");
        s.toolCall("c120", "shell", serde_json::json!({"cmd": "ls"}));
        s.toolResult("c120", "file1\nfile2");
        s.assistant("Listed files.", None);

        let turns = s.transcript.loadAll().unwrap();
        let blockId = turns.iter()
            .find(|t| t.content == "First task")
            .map(|t| t.blockId.clone())
            .unwrap();

        // Second block so we have a head to reconstruct from.
        s.user("Second task");
        s.assistant("Ok.", None);

        let afterTurn = s.headTurnId.clone().unwrap();
        let mut log = s.compactionLog();
        log.recordBlockCompact(&blockId, "User listed files, found file1 and file2.", vec![], &afterTurn).unwrap();
        drop(log);

        let msgs = s.reconstruct(false);
        dump("block_compact", &msgs);

        let hasCompressed = msgs.iter().any(|m| {
            if let Message::User { content } = m {
                content.textContent().contains("<compressed_content>")
            } else {
                false
            }
        });
        assert!(hasCompressed, "should contain a compressed_content summary message");

        let toolCount = msgs.iter().filter(|m| matches!(m, Message::Tool { .. })).count();
        assert_eq!(toolCount, 0, "compacted block's tool results should be gone");
    }

    // -----------------------------------------------------------------------
    // Image attachment reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn imageAttachmentsReconstructedOnRewind() {
        let mut s = TestSession::new();

        // Turn 1: user sends message with image.
        s.userWithImages("look at this screenshot", vec![
            crate::transcript::TurnAttachment {
                mimeType: "image/png".into(),
                data: "iVBORw0KGgo=".into(),
            },
        ]);
        s.assistant("I can see the screenshot.", None);
        let turn1Head = s.headTurnId.clone().unwrap();

        // Turn 2: user sends text-only follow-up.
        s.user("now fix the bug");
        s.assistant("Done.", None);

        // Rewind to turn 1 — images should be in the reconstructed history.
        let msgs = {
            let log = s.compactionLog();
            reconstruct(&s.transcript, &log, &turn1Head, false).unwrap()
        };
        dump("image_rewind", &msgs);

        assert_eq!(msgs.len(), 2, "User + Assistant");
        if let Message::User { content } = &msgs[0] {
            assert!(content.hasImages(), "rewound user message should have images");
            let uris = content.imageUris();
            assert_eq!(uris.len(), 1);
            assert!(uris[0].contains("iVBORw0KGgo="), "should contain original base64 data");
        } else {
            panic!("expected User message");
        }
    }

    #[test]
    fn imageAttachmentsReconstructedInFullChain() {
        let mut s = TestSession::new();

        s.userWithImages("check this", vec![
            crate::transcript::TurnAttachment {
                mimeType: "image/jpeg".into(),
                data: "/9j/4AAQ".into(),
            },
        ]);
        s.assistant("I see the image.", None);
        s.user("thanks");
        s.assistant("No problem.", None);

        // Full chain should preserve images on the first user message.
        let msgs = s.reconstruct(false);
        dump("image_full_chain", &msgs);

        if let Message::User { content } = &msgs[0] {
            assert!(content.hasImages(), "first user message should have images");
        } else {
            panic!("expected User message");
        }

        // Second user message should NOT have images.
        if let Message::User { content } = &msgs[2] {
            assert!(!content.hasImages(), "second user message should not have images");
        } else {
            panic!("expected User message at index 2");
        }
    }
}
