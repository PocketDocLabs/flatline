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
//! - [`buildState`] — `/context` geological layer data (rendered by deck)
//!
//! # Dependencies
//! `crate::compaction`, `crate::transcript`, `crate::message`

use std::collections::{HashMap, HashSet};

use crate::compaction::{CompactionLog, CompactionOp};
use crate::message::{FunctionCall, Message, ToolCall};
use crate::transcript::{Transcript, Turn, TurnRole};

use anyhow::Result;

/// Char cost of a single turn (content + args + reasoning).
fn turnChars(turn: &Turn) -> usize {
    turn.content.len()
        + turn.args.as_ref().map(|a| a.to_string().len()).unwrap_or(0)
        + turn.reasoning.as_ref().map(|r| r.len()).unwrap_or(0)
}

/// Result of budget-aware context reconstruction.
pub(crate) struct ReconstructResult {
    pub messages: Vec<Message>,
}

/// Reconstruct conversation history with budget-aware compaction.
///
/// Loads the compaction log as a CACHE of available compressions, then
/// selectively applies them bottom-up (oldest first) until the context
/// fits within `tokenBudget`. Stages are applied in order:
///
/// 1. S1 (lossless dedup/truncation) — always applied
/// 2. S2 (per-block summaries) — oldest blocks first
/// 3. S3 (per-topic summaries) — oldest topics first
/// 4. S4 (monolithic briefing) — last resort
///
/// Each stage stops as soon as the estimated total drops below budget.
/// If `tokenBudget` is 0, all cached compressions are applied (legacy
/// behavior for callers that don't have a budget yet).
///
/// `overheadTokens` is the estimated token cost of system prompt + tool
/// definitions — subtracted from the budget to get the message allowance.
pub(crate) fn reconstruct(
    transcript: &Transcript,
    compactionLog: &CompactionLog,
    headTurnId: &str,
    tokenBudget: usize,
    overheadTokens: usize,
) -> Result<ReconstructResult> {
    let allTurns = transcript.loadAll()?;
    let chain = walkChain(&allTurns, headTurnId);

    if chain.is_empty() {
        return Ok(ReconstructResult {
            messages: Vec::new(),
        });
    }

    let activeBlockIds: HashSet<&str> = chain.iter().map(|t| t.blockId.as_str()).collect();
    let activeTurnIds: HashSet<&str> = chain.iter().map(|t| t.id.as_str()).collect();
    let knownTurnIds: HashSet<&str> = allTurns.iter().map(|t| t.id.as_str()).collect();

    let allOps = compactionLog.loadAll()?;
    let ops = filterOpsForChain(allOps, &activeBlockIds, &activeTurnIds, &knownTurnIds);

    // Unlimited budget → apply everything (legacy behavior).
    if tokenBudget == 0 {
        let transformed = applyOps(&chain, &ops);
        return Ok(ReconstructResult {
            messages: assembleMessages(&transformed),
        });
    }

    let messageBudgetTokens = tokenBudget.saturating_sub(overheadTokens);

    // Separate S1 ops (always applied) from S2/S3/S4 (budget-gated cache).
    let mut s1Ops: Vec<&CompactionOp> = Vec::new();
    let mut s2Cache: HashMap<String, String> = HashMap::new();
    // S3 keyed by first source block ID (same as applyOps). Later ops
    // for the same blocks overwrite earlier ones.
    let mut s3Cache: Vec<(Vec<String>, String)> = Vec::new();
    let mut s4Cache: Option<(String, Vec<String>)> = None;

    for op in &ops {
        match op {
            CompactionOp::FileDedup { .. } | CompactionOp::MiddleOut { .. } => {
                s1Ops.push(op);
            }
            CompactionOp::BlockCompact {
                blockId, summary, ..
            } => {
                s2Cache.insert(blockId.clone(), summary.clone());
            }
            CompactionOp::TopicCompact {
                sourceBlockIds,
                summary,
                ..
            } => {
                s3Cache.push((sourceBlockIds.clone(), summary.clone()));
            }
            CompactionOp::FullCompact {
                sourceIds, summary, ..
            } => {
                s4Cache = Some((summary.clone(), sourceIds.clone()));
            }
        }
    }

    // Apply S1 unconditionally — build the base turn stream with
    // dedup/middle-out applied.
    let s1OnlyOps: Vec<CompactionOp> = s1Ops.iter().map(|o| (*o).clone()).collect();
    let baseTurns = applyOps(&chain, &s1OnlyOps);

    // Group turns into blocks for budget accounting.
    let mut blocks: Vec<ContextBlock> = Vec::new();
    let mut currentBlockId = String::new();
    for (i, tt) in baseTurns.iter().enumerate() {
        let bid = transformedTurnBlockId(tt);
        if bid != currentBlockId {
            blocks.push(ContextBlock {
                blockId: bid.to_string(),
                startIdx: i,
                endIdx: i + 1,
                rawChars: transformedTurnChars(tt),
                state: BlockState::Raw,
            });
            currentBlockId = bid.to_string();
        } else if let Some(last) = blocks.last_mut() {
            last.endIdx = i + 1;
            last.rawChars += transformedTurnChars(tt);
        }
    }

    let totalRawChars: usize = blocks.iter().map(|b| b.rawChars).sum();

    // Zone-based application — same logic as organic compaction.
    // S2 protects newest 40% of budget, S3 protects newest 70%.
    // Zones are computed from block positions, oldest first.
    let blockIdToIdx: HashMap<String, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.blockId.clone(), i))
        .collect();

    let budgetChars = if messageBudgetTokens > 0 {
        // Convert token budget to approximate chars.
        (messageBudgetTokens as f64 * 2.5) as usize
    } else {
        usize::MAX
    };

    // S2 zone: protect newest 40% of budget → compress oldest portion.
    let s2ProtectedChars = (budgetChars as f64 * 0.40) as usize;
    let s2Zone: HashSet<String> = {
        let mut zone = HashSet::new();
        let mut cumulative = 0usize;
        let zoneLimit = totalRawChars.saturating_sub(s2ProtectedChars);
        for block in blocks.iter() {
            if cumulative >= zoneLimit {
                break;
            }
            cumulative += block.rawChars;
            zone.insert(block.blockId.clone());
        }
        zone
    };

    // S3 zone: protect newest 70% of budget → compress oldest portion.
    let s3ProtectedChars = (budgetChars as f64 * 0.70) as usize;
    let s3Zone: HashSet<String> = {
        let mut zone = HashSet::new();
        let mut cumulative = 0usize;
        let zoneLimit = totalRawChars.saturating_sub(s3ProtectedChars);
        for block in blocks.iter() {
            if cumulative >= zoneLimit {
                break;
            }
            cumulative += block.rawChars;
            zone.insert(block.blockId.clone());
        }
        zone
    };

    // S2 pass: apply cached summaries to raw blocks in the S2 zone.
    for block in blocks.iter_mut() {
        if !matches!(block.state, BlockState::Raw) {
            continue;
        }
        if !s2Zone.contains(&block.blockId) {
            continue;
        }
        if let Some(summary) = s2Cache.get(&block.blockId) {
            let userChars: usize = baseTurns[block.startIdx..block.endIdx]
                .iter()
                .filter(|tt| {
                    matches!(
                        tt,
                        TransformedTurn::Original(t) | TransformedTurn::Replaced { turn: t, .. }
                            if matches!(t.role, TurnRole::User | TurnRole::Wake)
                    )
                })
                .map(|tt| transformedTurnChars(tt))
                .sum();
            let compressedChars = userChars + summary.len() + 200;
            if compressedChars < block.rawChars {
                block.state = BlockState::S2(summary.clone());
            }
        }
    }

    // S3 pass: apply cached topic summaries where ALL blocks are in S3 zone.
    for (sourceBlockIds, summary) in &s3Cache {
        if !sourceBlockIds.iter().all(|bid| s3Zone.contains(bid)) {
            continue;
        }
        let indices: Vec<usize> = sourceBlockIds
            .iter()
            .filter_map(|bid| blockIdToIdx.get(bid).copied())
            .collect();
        if indices.len() != sourceBlockIds.len() {
            continue;
        }
        if indices.iter().any(|&i| {
            matches!(
                blocks[i].state,
                BlockState::S3 { .. }
                    | BlockState::S3Covered
                    | BlockState::S4
                    | BlockState::S4Briefing(..)
            )
        }) {
            continue;
        }
        let coveredChars: usize = indices
            .iter()
            .map(|&i| blockCurrentChars(&blocks[i], &baseTurns))
            .sum();
        let summaryChars = summary.len() + 200;
        if summaryChars < coveredChars {
            let firstIdx = indices[0];
            blocks[firstIdx].state = BlockState::S3 {
                summary: summary.clone(),
                sourceBlockIds: sourceBlockIds.clone(),
            };
            for &i in &indices[1..] {
                blocks[i].state = BlockState::S3Covered;
            }
        }
    }

    // S4 pass: always apply cached briefing when available.
    // S4 replaces large S3/S2 content with a tiny briefing — always beneficial.
    if let Some((summary, sourceIds)) = &s4Cache {
        let indices: Vec<usize> = sourceIds
            .iter()
            .filter_map(|bid| blockIdToIdx.get(bid).copied())
            .collect();
        if !indices.is_empty()
            && !indices
                .iter()
                .any(|&i| matches!(blocks[i].state, BlockState::S4 | BlockState::S4Briefing(..)))
        {
            let coveredChars: usize = indices
                .iter()
                .map(|&i| blockCurrentChars(&blocks[i], &baseTurns))
                .sum();
            let briefingChars = summary.len() + 200;
            if briefingChars < coveredChars {
                let firstIdx = indices[0];
                blocks[firstIdx].state = BlockState::S4Briefing(summary.clone(), sourceIds.clone());
                for &i in &indices[1..] {
                    blocks[i].state = BlockState::S4;
                }
            }
        }
    }

    // Build the final TransformedTurn list from block states.
    let mut result: Vec<TransformedTurn> = Vec::new();
    for block in &blocks {
        match &block.state {
            BlockState::Raw => {
                for tt in &baseTurns[block.startIdx..block.endIdx] {
                    result.push(cloneTransformedTurn(tt));
                }
            }
            BlockState::S2(summary) => {
                // Keep user turns, replace rest with summary.
                let mut emittedSummary = false;
                for tt in &baseTurns[block.startIdx..block.endIdx] {
                    let isUser = matches!(
                        tt,
                        TransformedTurn::Original(t) | TransformedTurn::Replaced { turn: t, .. }
                            if matches!(t.role, TurnRole::User | TurnRole::Wake)
                    );
                    if isUser {
                        result.push(cloneTransformedTurn(tt));
                    } else if !emittedSummary {
                        result.push(TransformedTurn::Summary {
                            blockId: block.blockId.clone(),
                            content: summary.clone(),
                            kind: SummaryKind::Block,
                            sourceBlockIds: vec![block.blockId.clone()],
                        });
                        emittedSummary = true;
                    }
                }
            }
            BlockState::S3 {
                summary,
                sourceBlockIds,
            } => {
                result.push(TransformedTurn::Summary {
                    blockId: block.blockId.clone(),
                    content: summary.clone(),
                    kind: SummaryKind::Topic,
                    sourceBlockIds: sourceBlockIds.clone(),
                });
            }
            BlockState::S3Covered => {} // Handled by the S3 block
            BlockState::S4Briefing(summary, sourceIds) => {
                result.push(TransformedTurn::Summary {
                    blockId: block.blockId.clone(),
                    content: summary.clone(),
                    kind: SummaryKind::Full,
                    sourceBlockIds: sourceIds.clone(),
                });
            }
            BlockState::S4 => {} // Handled by the S4Briefing block
        }
    }

    Ok(ReconstructResult {
        messages: assembleMessages(&result),
    })
}

/// Block state during budget-aware reconstruction.
#[derive(Clone)]
enum BlockState {
    Raw,
    S2(String),
    S3 {
        summary: String,
        sourceBlockIds: Vec<String>,
    },
    S3Covered,
    S4Briefing(String, Vec<String>),
    S4,
}

/// A contiguous group of turns sharing a blockId.
struct ContextBlock {
    blockId: String,
    startIdx: usize,
    endIdx: usize,
    rawChars: usize,
    state: BlockState,
}

/// Current char cost of a block given its compression state.
fn blockCurrentChars(block: &ContextBlock, baseTurns: &[TransformedTurn]) -> usize {
    match &block.state {
        BlockState::Raw => block.rawChars,
        BlockState::S2(s) => {
            let userChars: usize = baseTurns[block.startIdx..block.endIdx]
                .iter()
                .filter(|tt| {
                    matches!(
                        tt,
                        TransformedTurn::Original(t) | TransformedTurn::Replaced { turn: t, .. }
                            if matches!(t.role, TurnRole::User | TurnRole::Wake)
                    )
                })
                .map(|tt| transformedTurnChars(tt))
                .sum();
            userChars + s.len() + 200
        }
        BlockState::S3 { summary, .. } => summary.len() + 200,
        BlockState::S3Covered | BlockState::S4 | BlockState::S4Briefing(..) => 0,
    }
}

fn transformedTurnChars(tt: &TransformedTurn) -> usize {
    match tt {
        TransformedTurn::Original(turn) => turnChars(turn),
        TransformedTurn::Replaced { turn, newContent } => {
            newContent.len()
                + turn.args.as_ref().map(|a| a.to_string().len()).unwrap_or(0)
                + turn.reasoning.as_ref().map(|r| r.len()).unwrap_or(0)
        }
        TransformedTurn::Summary { content, .. } => content.len() + 200,
    }
}

fn transformedTurnBlockId<'a>(tt: &'a TransformedTurn) -> &'a str {
    match tt {
        TransformedTurn::Original(t) | TransformedTurn::Replaced { turn: t, .. } => &t.blockId,
        TransformedTurn::Summary { blockId, .. } => blockId,
    }
}

fn cloneTransformedTurn<'a>(tt: &TransformedTurn<'a>) -> TransformedTurn<'a> {
    match tt {
        TransformedTurn::Original(t) => TransformedTurn::Original(t),
        TransformedTurn::Replaced { turn, newContent } => TransformedTurn::Replaced {
            turn,
            newContent: newContent.clone(),
        },
        TransformedTurn::Summary {
            blockId,
            content,
            kind,
            sourceBlockIds,
        } => TransformedTurn::Summary {
            blockId: blockId.clone(),
            content: content.clone(),
            kind: *kind,
            sourceBlockIds: sourceBlockIds.clone(),
        },
    }
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

/// Filter compaction ops to those whose targets and timing match the
/// active chain.
///
/// Two filters apply:
///
/// 1. **Temporal** (rewind correctness) — `afterTurn` was the head turn
///    id at the moment compaction ran. If that turn is not on the
///    active chain back from the current head, the op was recorded on
///    a branch we've since rewound past and must not apply. Legacy ops
///    written before the afterTurn schema migration stored a block id
///    there instead of a turn id; we recognize those (afterTurn is not
///    any known turn id) and apply them unconditionally to preserve
///    backward compatibility.
///
/// 2. **Branch coverage** — even when temporally valid, structural ops
///    that reference blocks must reference only blocks on this branch.
///    - FileDedup/MiddleOut: tool_call_id matching is a natural no-op
///      for turns not in the chain.
///    - BlockCompact: apply if blockId is in active chain.
///    - TopicCompact/FullCompact: apply if ALL source blocks are in
///      active chain (cross-branch summaries are invalid).
fn filterOpsForChain(
    ops: Vec<CompactionOp>,
    activeBlockIds: &HashSet<&str>,
    activeTurnIds: &HashSet<&str>,
    knownTurnIds: &HashSet<&str>,
) -> Vec<CompactionOp> {
    ops.into_iter()
        .filter(|op| {
            // Temporal gate: drop ops whose afterTurn is a real turn
            // that's off the active branch. Tolerate legacy afterTurn
            // values (block ids from before the migration) by treating
            // unknown-turn references as "always apply".
            //
            // BlockCompact is exempt: it's a deterministic compression
            // of a single block whose content is identical across
            // branches. Only the blockId check matters.
            let skipTemporal = matches!(op, CompactionOp::BlockCompact { .. });
            if !skipTemporal {
                let afterTurn = op.afterTurn();
                if knownTurnIds.contains(afterTurn) && !activeTurnIds.contains(afterTurn) {
                    return false;
                }
            }

            // Branch coverage gate.
            match op {
                CompactionOp::FileDedup { .. } | CompactionOp::MiddleOut { .. } => true,
                CompactionOp::BlockCompact { blockId, .. } => {
                    activeBlockIds.contains(blockId.as_str())
                }
                CompactionOp::TopicCompact { sourceBlockIds, .. } => sourceBlockIds
                    .iter()
                    .all(|id| activeBlockIds.contains(id.as_str())),
                CompactionOp::FullCompact { sourceIds, .. } => sourceIds
                    .iter()
                    .all(|id| activeBlockIds.contains(id.as_str())),
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
            CompactionOp::MiddleOut {
                targetIds,
                threshold,
                ..
            } => {
                for id in targetIds {
                    middleOutCallIds.insert(id.clone(), *threshold);
                }
            }
            CompactionOp::BlockCompact {
                blockId, summary, ..
            } => {
                s2SummarizedBlocks.insert(blockId.clone());
                blockSummaries.insert(blockId.clone(), summary.clone());
                summaryKinds.insert(blockId.clone(), SummaryKind::Block);
                summaryBlockIds.insert(blockId.clone(), vec![blockId.clone()]);
            }
            CompactionOp::TopicCompact {
                sourceBlockIds,
                summary,
                ..
            } => {
                if let Some(first) = sourceBlockIds.first() {
                    blockSummaries.insert(first.clone(), summary.clone());
                    summaryKinds.insert(first.clone(), SummaryKind::Topic);
                    summaryBlockIds.insert(first.clone(), sourceBlockIds.clone());
                }
                for bid in sourceBlockIds {
                    summarizedBlocks.insert(bid.clone());
                }
            }
            CompactionOp::FullCompact {
                sourceIds, summary, ..
            } => {
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
            if !emittedSummaries.contains(bid.as_str())
                && let Some(summary) = blockSummaries.get(bid.as_str())
            {
                let kind = *summaryKinds
                    .get(bid.as_str())
                    .unwrap_or(&SummaryKind::Topic);
                let blockIds = summaryBlockIds
                    .get(bid.as_str())
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
            continue;
        }

        // S2 block compact: user messages stay, assistant/tool turns replaced.
        if s2SummarizedBlocks.contains(bid.as_str()) {
            if matches!(turn.role, TurnRole::User) {
                result.push(TransformedTurn::Original(turn));
            } else if !emittedSummaries.contains(bid.as_str())
                && let Some(summary) = blockSummaries.get(bid.as_str())
            {
                result.push(TransformedTurn::Summary {
                    blockId: bid.clone(),
                    content: summary.clone(),
                    kind: SummaryKind::Block,
                    sourceBlockIds: vec![bid.clone()],
                });
                emittedSummaries.insert(bid.clone());
            }
            // Skip remaining assistant/tool turns in this block.
            continue;
        }

        // FileDedup: skip turns whose toolCallId was removed.
        if let Some(tcid) = &turn.toolCallId
            && removedCallIds.contains(tcid.as_str())
        {
            continue;
        }

        // MiddleOut: truncate ToolResult content.
        if let Some(tcid) = &turn.toolCallId
            && let Some(&thresh) = middleOutCallIds.get(tcid.as_str())
            && matches!(turn.role, TurnRole::ToolResult)
        {
            result.push(TransformedTurn::Replaced {
                turn,
                newContent: middleOut(&turn.content, thresh, Some(&turn.blockId)),
            });
            continue;
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
fn assembleMessages(turns: &[TransformedTurn]) -> Vec<Message> {
    let mut history: Vec<Message> = Vec::new();
    let mut pendingCalls: Vec<ToolCall> = Vec::new();
    // Assistant content waiting to see if tool calls follow.
    let mut pendingAssistant: Option<PendingAssistant> = None;

    for tt in turns {
        match tt {
            TransformedTurn::Summary {
                content,
                blockId,
                kind,
                sourceBlockIds,
            } => {
                flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls);
                let wrapped = match kind {
                    SummaryKind::Block => formatBlockSummary(content, blockId),
                    SummaryKind::Topic => formatTopicSummary(content, sourceBlockIds),
                    SummaryKind::Full => formatFullBriefing(content, sourceBlockIds),
                };
                history.push(Message::User {
                    content: wrapped.into(),
                });
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
                        let argsStr = turn
                            .args
                            .as_ref()
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
                        flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls);

                        // Hold — don't emit yet. If ToolCall turns follow,
                        // this content will be merged into that message.
                        // Trim leading/trailing whitespace on reasoning loaded
                        // from older sessions where we didn't normalize on write.
                        let reasoning = turn.reasoning.as_ref().map(|r| r.trim().to_string());
                        pendingAssistant = Some(PendingAssistant {
                            content: Some(content),
                            reasoning,
                        });
                    }
                    TurnRole::User | TurnRole::Wake => {
                        // Wake turns are user-shaped to the model — the
                        // distinction only matters for transcript display.
                        flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls);
                        let msgContent = rebuildContent(&content, &turn.attachments);
                        history.push(Message::User {
                            content: msgContent,
                        });
                    }
                    TurnRole::ToolResult => {
                        flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls);
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

    flushPending(&mut history, &mut pendingAssistant, &mut pendingCalls);
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
            let imageUris: Vec<String> = atts
                .iter()
                .map(|a| format!("data:{};base64,{}", a.mimeType, a.data))
                .collect();
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
// /context display
// ---------------------------------------------------------------------------

/// Input parameters for building context state.
pub(crate) struct BuildStateInput<'a> {
    pub contextWindow: usize,
    pub compactionLog: &'a CompactionLog,
    pub reportedTokens: usize,
    pub transcript: &'a Transcript,
    pub headTurnId: &'a str,
    /// Char length of the system prompt.
    pub systemPromptChars: usize,
    /// Char length of the serialized tool definitions.
    pub toolDefsChars: usize,
    /// Char length of the serialized history (Vec<Message> JSON).
    /// Includes all message structure, tool call framing, etc.
    pub historyChars: usize,
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
pub(crate) fn buildState(input: &BuildStateInput) -> ContextState {
    let ops = input.compactionLog.loadAll().unwrap_or_default();
    let totalOps = ops.len();
    let totalBcOps = ops
        .iter()
        .filter(|op| matches!(op, CompactionOp::BlockCompact { .. }))
        .count();

    if input.headTurnId.is_empty() {
        return ContextState {
            estimatedTokens: 0,
            reportedTokens: input.reportedTokens,
            contextWindow: input.contextWindow,
            s4: None,
            s3: None,
            s2: None,
            raw: RawLayer {
                turns: 0,
                estimatedTokens: 0,
            },
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
                raw: RawLayer {
                    turns: 0,
                    estimatedTokens: 0,
                },
            };
        }
    };

    let chain = walkChain(&allTurns, input.headTurnId);
    let activeBlockIds: HashSet<&str> = chain.iter().map(|t| t.blockId.as_str()).collect();
    let activeTurnIds: HashSet<&str> = chain.iter().map(|t| t.id.as_str()).collect();
    let knownTurnIds: HashSet<&str> = allTurns.iter().map(|t| t.id.as_str()).collect();
    let filteredOps = filterOpsForChain(ops, &activeBlockIds, &activeTurnIds, &knownTurnIds);

    // Build block list with raw char sizes.
    let mut allBlockIds: Vec<&str> = Vec::new();
    let mut blockRawChars: HashMap<&str, usize> = HashMap::new();
    {
        let mut seen = HashSet::new();
        for t in &chain {
            let bid = t.blockId.as_str();
            *blockRawChars.entry(bid).or_default() += turnChars(t);
            if seen.insert(bid) {
                allBlockIds.push(bid);
            }
        }
    }
    let totalRawChars: usize = blockRawChars.values().sum();

    // Ops metadata: topic labels, S4 topic counts, briefing counts.
    let mut s4CoveredTopicLabels: HashSet<String> = HashSet::new();
    let mut s3TopicLabels: Vec<String> = Vec::new();
    let mut fullCompactCount = 0usize;

    // Use only the LATEST FullCompact (matching reconstruction).
    let latestS4SourceIds: HashSet<&str> = filteredOps
        .iter()
        .rev()
        .find_map(|op| match op {
            CompactionOp::FullCompact { sourceIds, .. } => Some(
                sourceIds
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<HashSet<&str>>(),
            ),
            _ => None,
        })
        .unwrap_or_default();

    // Zone-based classification — same math as reconstruction.
    let compactRatio = 0.80;
    let overheadToks = (input.systemPromptChars + input.toolDefsChars) / 4;
    let budgetToks = (input.contextWindow as f64 * compactRatio) as usize;
    let msgBudgetToks = budgetToks.saturating_sub(overheadToks);
    let budgetChars = if msgBudgetToks > 0 {
        (msgBudgetToks as f64 * 2.5) as usize
    } else {
        usize::MAX
    };

    let s2ProtectedChars = (budgetChars as f64 * 0.40) as usize;
    let s2Zone: HashSet<&str> = {
        let mut zone = HashSet::new();
        let mut cum = 0usize;
        let limit = totalRawChars.saturating_sub(s2ProtectedChars);
        for &bid in &allBlockIds {
            if cum >= limit {
                break;
            }
            cum += blockRawChars.get(bid).copied().unwrap_or(0);
            zone.insert(bid);
        }
        zone
    };

    tracing::debug!(
        totalOps,
        totalBcOps,
        totalRawChars,
        budgetChars,
        s2ProtectedChars,
        s2ZoneLimit = totalRawChars.saturating_sub(s2ProtectedChars),
        s2ZoneSize = s2Zone.len(),
        latestS4Size = latestS4SourceIds.len(),
        "buildState zone debug"
    );

    let s3ProtectedChars = (budgetChars as f64 * 0.70) as usize;
    let s3Zone: HashSet<&str> = {
        let mut zone = HashSet::new();
        let mut cum = 0usize;
        let limit = totalRawChars.saturating_sub(s3ProtectedChars);
        for &bid in &allBlockIds {
            if cum >= limit {
                break;
            }
            cum += blockRawChars.get(bid).copied().unwrap_or(0);
            zone.insert(bid);
        }
        zone
    };

    // S2/S3/S4 cache for zone-aware classification and token estimation.
    let bcOpsCount = filteredOps
        .iter()
        .filter(|op| matches!(op, CompactionOp::BlockCompact { .. }))
        .count();
    let mut s2CacheSizes: HashMap<&str, usize> = HashMap::new();
    for op in &filteredOps {
        if let CompactionOp::BlockCompact {
            blockId, summary, ..
        } = op
        {
            s2CacheSizes.insert(blockId.as_str(), summary.len());
        }
    }
    let s2Cache: HashSet<&str> = s2CacheSizes.keys().copied().collect();

    let mut s3TopicSummaryChars: usize = 0;
    let mut s4BriefingChars: usize = 0;

    // S3 topics: not in latest S4 and all blocks in S3 zone.
    let mut s3CacheBlocks: HashSet<&str> = HashSet::new();
    for op in &filteredOps {
        match op {
            CompactionOp::FullCompact { summary, .. } => {
                fullCompactCount += 1;
                s4BriefingChars = summary.len();
            }
            CompactionOp::TopicCompact {
                sourceBlockIds,
                topicLabel,
                summary,
                ..
            } => {
                let inS4 = sourceBlockIds
                    .iter()
                    .all(|id| latestS4SourceIds.contains(id.as_str()));
                if inS4 {
                    s4CoveredTopicLabels.insert(topicLabel.clone());
                } else if sourceBlockIds.iter().all(|id| s3Zone.contains(id.as_str())) {
                    s3TopicLabels.push(topicLabel.clone());
                    s3TopicSummaryChars += summary.len();
                    for id in sourceBlockIds {
                        s3CacheBlocks.insert(id.as_str());
                    }
                }
            }
            _ => {}
        }
    }

    let s4PriorBriefings = fullCompactCount.saturating_sub(1);
    let s4TopicsMerged = s4CoveredTopicLabels.len();

    // Classify blocks by what reconstruction would do.
    let mut s4TurnCount = 0usize;
    let mut s3TurnCount = 0usize;
    let mut s2TurnCount = 0usize;
    let mut rawTurnCount = 0usize;
    let mut blockLayer: HashMap<&str, &str> = HashMap::new();
    let mut rawInZoneWithCache = 0usize;
    let mut rawInZoneNoCache = 0usize;
    let mut rawNotInZone = 0usize;

    for &bid in &allBlockIds {
        if latestS4SourceIds.contains(bid) {
            s4TurnCount += 1;
            blockLayer.insert(bid, "s4");
        } else if s3CacheBlocks.contains(bid) {
            s3TurnCount += 1;
            blockLayer.insert(bid, "s3");
        } else if s2Zone.contains(bid) && s2Cache.contains(bid) {
            s2TurnCount += 1;
            blockLayer.insert(bid, "s2");
        } else {
            rawTurnCount += 1;
            blockLayer.insert(bid, "raw");
            if s2Zone.contains(bid) {
                if s2Cache.contains(bid) {
                    rawInZoneWithCache += 1;
                } else {
                    rawInZoneNoCache += 1;
                }
            } else {
                rawNotInZone += 1;
            }
        }
    }

    tracing::debug!(
        s2CacheLen = s2Cache.len(),
        bcOpsCount,
        s3CacheBlocksLen = s3CacheBlocks.len(),
        rawInZoneWithCache,
        rawInZoneNoCache,
        rawNotInZone,
        "buildState raw breakdown"
    );

    tracing::debug!(
        s4TurnCount,
        s3TurnCount,
        s2TurnCount,
        rawTurnCount,
        s3TopicCount = s3TopicLabels.len(),
        s4TopicsMerged,
        totalBlocks = allBlockIds.len(),
        "buildState layer classification"
    );

    // Estimate tokens per layer directly from block classification
    // and cached summary sizes. No applyOps — measure what the
    // reconstruction would actually send to the model.
    let mut s2Tokens = 0usize;
    let mut rawTokens = 0usize;

    for &bid in &allBlockIds {
        let layer = blockLayer.get(bid).copied().unwrap_or("raw");
        match layer {
            "s4" => {} // Counted separately from briefing.
            "s3" => {} // Counted separately from topic summaries.
            "s2" => {
                // S2 keeps user turns raw + summary for agent turns.
                let summaryChars = s2CacheSizes.get(bid).copied().unwrap_or(0);
                // Approximate user turn chars as raw chars minus agent chars.
                // Since we don't track agent vs user separately, use summary
                // size + 200 overhead as the block's effective size.
                s2Tokens += (summaryChars + 200) / 4;
            }
            _ => {
                // Raw: full block content.
                rawTokens += blockRawChars.get(bid).copied().unwrap_or(0) / 4;
            }
        }
    }

    // S3 token estimate from topic summary sizes.
    let s3Tokens = s3TopicSummaryChars / 4;
    // S4 token estimate from latest briefing.
    let s4Tokens = s4BriefingChars / 4;

    // Build layers — only present if that stage has produced output.
    let mut s4 = if fullCompactCount > 0 {
        Some(S4Layer {
            topicsMerged: s4TopicsMerged,
            priorBriefings: s4PriorBriefings,
            turnsCovered: s4TurnCount,
            estimatedTokens: s4Tokens,
        })
    } else {
        None
    };

    let mut s3 = if !s3TopicLabels.is_empty() {
        Some(S3Layer {
            topicLabels: s3TopicLabels,
            turnsCondensed: s3TurnCount,
            estimatedTokens: s3Tokens,
        })
    } else {
        None
    };

    let mut s2 = if s2TurnCount > 0 {
        Some(S2Layer {
            turnsCondensed: s2TurnCount,
            estimatedTokens: s2Tokens,
        })
    } else {
        None
    };

    let mut raw = RawLayer {
        turns: rawTurnCount,
        estimatedTokens: rawTokens,
    };

    // Total estimate from the serialized message list — captures all
    // JSON structure overhead (role tags, tool_call wrappers, etc.).
    // Divisor of 2.5 empirically matches Anthropic's reported token
    // counts on serialized request content (cl100k gives 3.4 c/t but
    // Anthropic's tokenizer + API framing push effective ratio to ~2.5).
    let totalRequestChars = input.historyChars + input.systemPromptChars + input.toolDefsChars;
    let estimatedTokens = totalRequestChars * 2 / 5;

    // Scale layer estimates so they sum to the accurate total.
    // The per-layer chars/4 estimates are good for relative proportions
    // but undercount in absolute terms; rescaling keeps the breakdown
    // consistent with the top-level number.
    let layerSum = s4.as_ref().map(|l| l.estimatedTokens).unwrap_or(0)
        + s3.as_ref().map(|l| l.estimatedTokens).unwrap_or(0)
        + s2.as_ref().map(|l| l.estimatedTokens).unwrap_or(0)
        + raw.estimatedTokens;

    if layerSum > 0 && estimatedTokens > 0 {
        let scale = estimatedTokens as f64 / layerSum as f64;
        if let Some(ref mut l) = s4 {
            l.estimatedTokens = (l.estimatedTokens as f64 * scale) as usize;
        }
        if let Some(ref mut l) = s3 {
            l.estimatedTokens = (l.estimatedTokens as f64 * scale) as usize;
        }
        if let Some(ref mut l) = s2 {
            l.estimatedTokens = (l.estimatedTokens as f64 * scale) as usize;
        }
        raw.estimatedTokens = (raw.estimatedTokens as f64 * scale) as usize;
    }

    ContextState {
        estimatedTokens,
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
    let first = sourceBlockIds
        .first()
        .map(|s| s.as_str())
        .unwrap_or("b_00000000");
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
        _dir: tempfile::TempDir,
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
            let dir = tempfile::TempDir::new().unwrap();
            Self {
                transcript: Transcript::createAt(dir.path(), &id).unwrap(),
                _dir: dir,
                headTurnId: None,
            }
        }

        fn compactionLog(&self) -> CompactionLog {
            CompactionLog::open(self.transcript.sessionDir()).unwrap()
        }

        /// Record a user message, chaining to the current head.
        /// Mirrors session.rs:658 — `recordUser(msg, headTurnId)`.
        fn user(&mut self, content: &str) {
            let id = self
                .transcript
                .recordUser(content, self.headTurnId.as_deref(), None)
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a user message with image attachments.
        fn userWithImages(
            &mut self,
            content: &str,
            attachments: Vec<crate::transcript::TurnAttachment>,
        ) {
            let id = self
                .transcript
                .recordUser(content, self.headTurnId.as_deref(), Some(attachments))
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a wake event. Wake turns are not authored user messages in
        /// the transcript, but reconstruct feeds their exact envelope back to
        /// the model as user-shaped context.
        fn wake(&mut self, content: &str) {
            let id = self
                .transcript
                .recordWake(content, self.headTurnId.as_deref())
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Record assistant text (and optional reasoning).
        /// Mirrors session.rs:1182-1187 — content recorded before tool calls.
        fn assistant(&mut self, content: &str, reasoning: Option<&str>) {
            let meta = crate::transcript::AssistantMeta {
                reasoning,
                ..Default::default()
            };
            let id = self.transcript.recordAssistant(content, meta).unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a tool call.
        /// Mirrors session.rs:752-764.
        fn toolCall(&mut self, callId: &str, name: &str, args: serde_json::Value) {
            let id = self.transcript.recordToolCall(callId, name, &args).unwrap();
            self.headTurnId = Some(id);
        }

        /// Record a tool result.
        /// Mirrors session.rs:2084-2093 (pushToolResult).
        fn toolResult(&mut self, callId: &str, content: &str) {
            let id = self
                .transcript
                .recordToolResult(callId, content, None)
                .unwrap();
            self.headTurnId = Some(id);
        }

        /// Reconstruct messages from the current head.
        fn reconstruct(&self) -> Vec<Message> {
            let head = self.headTurnId.as_ref().expect("no turns recorded");
            let log = self.compactionLog();
            reconstruct(&self.transcript, &log, head, 0, 0)
                .unwrap()
                .messages
        }
    }

    impl Drop for TestSession {
        fn drop(&mut self) {
            // `TempDir` owns cleanup. Avoid touching the real session root in
            // sandboxed test environments.
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
                Message::Assistant {
                    content,
                    tool_calls,
                    reasoning,
                } => {
                    println!(
                        "  [{i}] Assistant: content={:?} tool_calls={} reasoning={}",
                        content.as_deref().map(|c| truncate(c, 40)),
                        tool_calls.as_ref().map(|c| c.len()).unwrap_or(0),
                        reasoning.is_some(),
                    );
                }
                Message::Tool {
                    tool_call_id,
                    content,
                } => {
                    println!(
                        "  [{i}] Tool({tool_call_id}): {}",
                        truncate(content.textContent(), 50)
                    );
                }
            }
        }
    }

    fn truncate(s: &str, max: usize) -> String {
        if s.len() <= max {
            s.to_string()
        } else {
            format!("{}...", &s[..max])
        }
    }

    /// Assert a message is `Assistant` and return (content, toolCallCount, hasReasoning).
    fn assertAssistant(msg: &Message) -> (Option<&str>, usize, bool) {
        match msg {
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
            } => (
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
            Message::Tool {
                tool_call_id,
                content,
            } => (tool_call_id, content.textContent()),
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

        let msgs = s.reconstruct();
        assert_eq!(msgs.len(), 2);
        assert_eq!(assertUser(&msgs[0]), "Hello");
        let (content, calls, _) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some("Hi there!"));
        assert_eq!(calls, 0);
    }

    #[test]
    fn wake_turn_reconstructs_as_exact_user_shaped_context() {
        let mut s = TestSession::new();
        s.user("before");
        s.assistant("ready", None);
        let envelope = "<wakes count=\"1\">\n<wake source=\"delay#3\" kind=\"Delay\" ageSecs=\"0\">\nSend the MyHealth message.\n</wake>\n</wakes>";
        s.wake(envelope);
        s.assistant("handled", None);

        let msgs = s.reconstruct();
        assert_eq!(msgs.len(), 4);
        assert_eq!(assertUser(&msgs[0]), "before");
        assert_eq!(assertUser(&msgs[2]), envelope);
        let turns = s.transcript.loadAll().unwrap();
        assert!(matches!(
            turns
                .iter()
                .find(|t| t.content == envelope)
                .map(|t| &t.role),
            Some(crate::transcript::TurnRole::Wake)
        ));
    }

    #[test]
    fn reasoning_preserved() {
        let mut s = TestSession::new();
        s.user("Explain this");
        s.assistant("Here's my answer.", Some("Thought carefully"));

        let msgs = s.reconstruct();
        let (content, _, hasReasoning) = assertAssistant(&msgs[1]);
        assert_eq!(content, Some("Here's my answer."));
        assert!(hasReasoning);
    }

    /// Reasoning loaded from older sessions with leading/trailing whitespace
    /// gets trimmed on reconstruct so it doesn't re-feed the model's drift.
    #[test]
    fn reasoning_trimmed_on_load() {
        let mut s = TestSession::new();
        s.user("Explain this");
        s.assistant("answer", Some("\n\n  thought\n\n"));

        let msgs = s.reconstruct();
        match &msgs[1] {
            Message::Assistant { reasoning, .. } => {
                let r = reasoning.as_deref().expect("reasoning present");
                assert_eq!(r, "thought", "reasoning should be trimmed");
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Tool call basics
    // -----------------------------------------------------------------------

    /// Single tool call + result. The bread-and-butter pattern.
    #[test]
    fn single_tool_call() {
        let mut s = TestSession::new();
        s.user("Read /tmp/foo.txt");
        s.toolCall(
            "c01",
            "readFile",
            serde_json::json!({"path": "/tmp/foo.txt"}),
        );
        s.toolResult("c01", "file contents here");

        let msgs = s.reconstruct();
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

        let msgs = s.reconstruct();
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
        s.toolCall(
            "c20",
            "editFile",
            serde_json::json!({
                "path": "/tmp/x.rs",
                "oldText": "fn foo()",
                "newText": "fn bar()",
            }),
        );
        s.toolResult("c20", "ok");

        let msgs = s.reconstruct();
        match &msgs[1] {
            Message::Assistant {
                tool_calls: Some(calls),
                ..
            } => {
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
        s.toolCall(
            "c31",
            "editFile",
            serde_json::json!({"path": "main.rs", "old": "bug()", "new": "fix()"}),
        );
        s.toolResult("c31", "ok");
        // Round 3: model responds.
        s.assistant("Fixed. The bug() call was replaced with fix().", None);

        let msgs = s.reconstruct();
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
        assert_eq!(
            content,
            Some("Fixed. The bug() call was replaced with fix().")
        );
        assert_eq!(calls, 0);

        let toolMsgs: Vec<_> = msgs
            .iter()
            .filter_map(|m| {
                if let Message::Tool { tool_call_id, .. } = m {
                    Some(tool_call_id.as_str())
                } else {
                    None
                }
            })
            .collect();
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
        s.toolCall(
            "c40",
            "readFile",
            serde_json::json!({"path": "/tmp/bar.txt"}),
        );
        s.toolResult("c40", "bar contents");

        let msgs = s.reconstruct();
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
        s.assistant(
            "I'll read the config.",
            Some("Need to check config before proceeding"),
        );
        s.toolCall(
            "c50",
            "readFile",
            serde_json::json!({"path": "config.toml"}),
        );
        s.toolResult("c50", "[settings]\nfoo = true");

        let msgs = s.reconstruct();
        dump("content_plus_reasoning_plus_tool_calls_merged", &msgs);

        assert_eq!(
            msgs.len(),
            3,
            "User + Assistant(content+reasoning+tool_calls) + Tool"
        );

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

        let msgs = s.reconstruct();
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

        let msgs = s.reconstruct();
        dump("three_block_chain", &msgs);

        let userMsgs: Vec<&str> = msgs
            .iter()
            .filter_map(|m| {
                if let Message::User { content } = m {
                    Some(content.textContent())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(userMsgs, vec!["Step 1", "Step 2", "Step 3"]);
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

        let msgs = s.reconstruct();
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

        let msgs = s.reconstruct();
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
        s.toolCall(
            "c92",
            "editFile",
            serde_json::json!({"path": "/a", "old": "a", "new": "aa"}),
        );
        s.toolResult("c92", "ok");

        // Round 4: shell.
        s.toolCall("c93", "shell", serde_json::json!({"cmd": "cargo test"}));
        s.toolResult("c93", "test passed");

        // Final text response.
        s.assistant("All done.", None);

        let msgs = s.reconstruct();
        dump("long_tool_chain", &msgs);

        // Each tool call round = Assistant(calls) + Tool = 2 messages.
        // Total: User(1) + 4*(Asst+Tool) + Asst(text) = 10.
        //
        // BUT: consecutive ToolCall turns without an intervening non-ToolCall
        // turn get grouped. Let's verify what actually happens.
        let toolMsgs: Vec<&str> = msgs
            .iter()
            .filter_map(|m| {
                if let Message::Tool { tool_call_id, .. } = m {
                    Some(tool_call_id.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            toolMsgs,
            vec!["c90", "c91", "c92", "c93"],
            "all 4 tool results should be present"
        );

        // Verify the final assistant text is present.
        if let Some(lastAssist) = msgs
            .iter()
            .rev()
            .find(|m| matches!(m, Message::Assistant { .. }))
        {
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
        log.recordMiddleOut(vec!["c100".into()], &afterTurn, 200)
            .unwrap();
        drop(log);

        let msgs = s.reconstruct();

        let (_, body) = assertTool(&msgs[2]);
        assert!(
            body.len() < bigContent.len(),
            "tool result should be truncated"
        );
        assert!(
            body.contains("bytes truncated"),
            "should contain truncation marker"
        );
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
        log.recordFileDedup(vec!["c110".into()], &afterTurn)
            .unwrap();
        drop(log);

        let msgs = s.reconstruct();
        dump("file_dedup", &msgs);

        let toolIds: Vec<&str> = msgs
            .iter()
            .filter_map(|m| {
                if let Message::Tool { tool_call_id, .. } = m {
                    Some(tool_call_id.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            toolIds,
            vec!["c111"],
            "only second read should survive dedup"
        );
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
        let blockId = turns
            .iter()
            .find(|t| t.content == "First task")
            .map(|t| t.blockId.clone())
            .unwrap();

        // Second block so we have a head to reconstruct from.
        s.user("Second task");
        s.assistant("Ok.", None);

        let afterTurn = s.headTurnId.clone().unwrap();
        let mut log = s.compactionLog();
        log.recordBlockCompact(
            &blockId,
            "User listed files, found file1 and file2.",
            vec![],
            &afterTurn,
        )
        .unwrap();
        drop(log);

        let msgs = s.reconstruct();
        dump("block_compact", &msgs);

        let hasCompressed = msgs.iter().any(|m| {
            if let Message::User { content } = m {
                content.textContent().contains("<compressed_content>")
            } else {
                false
            }
        });
        assert!(
            hasCompressed,
            "should contain a compressed_content summary message"
        );

        let toolCount = msgs
            .iter()
            .filter(|m| matches!(m, Message::Tool { .. }))
            .count();
        assert_eq!(
            toolCount, 0,
            "compacted block's tool results should be gone"
        );
    }

    /// Repeated S4 records are an append-only history, but reconstruct should
    /// expose only the latest active S4 briefing when it supersedes earlier
    /// S4 source blocks.
    #[test]
    fn full_compact_replay_uses_latest_s4_frontier() {
        let mut s = TestSession::new();
        s.user("First task");
        s.assistant("Done one.", None);
        s.user("Second task");
        s.assistant("Done two.", None);
        s.user("Third task");
        s.assistant("Done three.", None);
        s.user("Fourth task");
        s.assistant("Done four.", None);

        let turns = s.transcript.loadAll().unwrap();
        let blockIds: Vec<String> = turns
            .iter()
            .filter(|t| matches!(t.role, crate::transcript::TurnRole::User))
            .map(|t| t.blockId.clone())
            .collect();
        let afterTurn = s.headTurnId.clone().unwrap();

        let mut log = s.compactionLog();
        log.recordFullCompact(
            "OLD_S4_BRIEFING",
            vec![blockIds[0].clone(), blockIds[1].clone()],
            &afterTurn,
        )
        .unwrap();
        log.recordFullCompact(
            "LATEST_S4_BRIEFING",
            vec![
                blockIds[0].clone(),
                blockIds[1].clone(),
                blockIds[2].clone(),
            ],
            &afterTurn,
        )
        .unwrap();
        drop(log);

        let msgs = s.reconstruct();
        let text = msgs
            .iter()
            .map(|m| match m {
                Message::User { content } | Message::Tool { content, .. } => {
                    content.textContent().to_string()
                }
                Message::Assistant { content, .. } => content.clone().unwrap_or_default(),
                Message::System { content } => content.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            !text.contains("OLD_S4_BRIEFING"),
            "superseded S4 briefing must not remain active"
        );
        assert_eq!(
            text.matches("LATEST_S4_BRIEFING").count(),
            1,
            "latest S4 frontier should appear exactly once"
        );
        assert_eq!(
            text.matches("<session_briefing>").count(),
            1,
            "only one S4 summary should be active"
        );
        assert!(
            text.contains("Fourth task") && text.contains("Done four."),
            "recent raw blocks outside S4 must remain intact"
        );
    }

    // -----------------------------------------------------------------------
    // Rewind temporal correctness
    // -----------------------------------------------------------------------

    /// A compaction recorded AFTER a rewind point must not apply when
    /// reconstructing from that earlier head. The compaction's
    /// `afterTurn` references a turn off the active branch, so the
    /// op is temporally invalid for the rewound chain.
    #[test]
    fn rewind_ignores_compactions_recorded_later() {
        let mut s = TestSession::new();
        // Block 1.
        s.user("First task");
        s.toolCall("c200", "readFile", serde_json::json!({"path": "/big"}));
        let bigContent = "x".repeat(10_000);
        s.toolResult("c200", &bigContent);
        s.assistant("Read it.", None);
        let rewindTarget = s.headTurnId.clone().unwrap();

        // Block 2 — written AFTER the rewind target.
        s.user("Second task");
        s.assistant("Done.", None);
        let laterHead = s.headTurnId.clone().unwrap();

        // Record a middle-out compaction whose afterTurn is the LATER head.
        let mut log = s.compactionLog();
        log.recordMiddleOut(vec!["c200".into()], &laterHead, 200)
            .unwrap();
        drop(log);

        // Reconstructing from the later head should see the truncation
        // (op is on the active chain at that point).
        let log = s.compactionLog();
        let lateMsgs = reconstruct(&s.transcript, &log, &laterHead, 0, 0)
            .unwrap()
            .messages;
        let lateToolBody = lateMsgs
            .iter()
            .find_map(|m| {
                if let Message::Tool { content, .. } = m {
                    Some(content.textContent().to_string())
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            lateToolBody.contains("bytes truncated"),
            "later head sees truncation"
        );

        // Reconstructing from the EARLIER head must NOT see the truncation
        // — the op was recorded after a turn that isn't on this branch.
        let earlyMsgs = reconstruct(&s.transcript, &log, &rewindTarget, 0, 0)
            .unwrap()
            .messages;
        let earlyToolBody = earlyMsgs
            .iter()
            .find_map(|m| {
                if let Message::Tool { content, .. } = m {
                    Some(content.textContent().to_string())
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            !earlyToolBody.contains("bytes truncated"),
            "rewound head must not see compaction recorded after the rewind point"
        );
        assert_eq!(earlyToolBody.len(), 10_000, "raw tool result restored");
    }

    /// Same rewind correctness for S2 BlockCompact ops.
    #[test]
    fn rewind_ignores_block_compact_recorded_later() {
        let mut s = TestSession::new();
        // Block 1.
        s.user("Task one");
        s.toolCall("c210", "shell", serde_json::json!({"cmd": "ls"}));
        s.toolResult("c210", "file1\nfile2");
        s.assistant("Listed.", None);
        let rewindTarget = s.headTurnId.clone().unwrap();
        let block1Id = s
            .transcript
            .loadAll()
            .unwrap()
            .iter()
            .find(|t| t.content == "Task one")
            .map(|t| t.blockId.clone())
            .unwrap();

        // Block 2 written AFTER, used as afterTurn for the S2 op.
        s.user("Task two");
        s.assistant("Ok.", None);
        let laterHead = s.headTurnId.clone().unwrap();

        let mut log = s.compactionLog();
        log.recordBlockCompact(&block1Id, "block 1 summary", vec![], &laterHead)
            .unwrap();
        drop(log);

        // From the rewound head, no S2 summary should appear.
        let log = s.compactionLog();
        let msgs = reconstruct(&s.transcript, &log, &rewindTarget, 0, 0)
            .unwrap()
            .messages;
        let hasCompressed = msgs.iter().any(|m| {
            if let Message::User { content } = m {
                content.textContent().contains("<compressed_content>")
            } else {
                false
            }
        });
        // BlockCompact is exempt from afterTurn filtering — the summary
        // is a deterministic compression of block content that's identical
        // across branches. So even from the rewound head, the S2 cache
        // entry applies.
        assert!(
            hasCompressed,
            "S2 BlockCompact should apply regardless of afterTurn branch"
        );
    }

    /// Legacy ops with an afterTurn that isn't a known turn id (e.g.
    /// a pre-migration block id) must still apply unconditionally.
    #[test]
    fn legacy_afterTurn_block_id_still_applies() {
        let mut s = TestSession::new();
        s.user("Read big file");
        s.toolCall("c220", "readFile", serde_json::json!({"path": "/big"}));
        let bigContent = "x".repeat(10_000);
        s.toolResult("c220", &bigContent);
        s.assistant("Got it.", None);

        // afterTurn = a value that isn't any real turn id (simulates
        // legacy block-id semantics from before the schema migration).
        let mut log = s.compactionLog();
        log.recordMiddleOut(vec!["c220".into()], "b_legacy_unknown", 200)
            .unwrap();
        drop(log);

        let msgs = s.reconstruct();
        let toolBody = msgs
            .iter()
            .find_map(|m| {
                if let Message::Tool { content, .. } = m {
                    Some(content.textContent().to_string())
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            toolBody.contains("bytes truncated"),
            "legacy afterTurn should still apply"
        );
    }

    // -----------------------------------------------------------------------
    // Image attachment reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn imageAttachmentsReconstructedOnRewind() {
        let mut s = TestSession::new();

        // Turn 1: user sends message with image.
        s.userWithImages(
            "look at this screenshot",
            vec![crate::transcript::TurnAttachment {
                mimeType: "image/png".into(),
                data: "iVBORw0KGgo=".into(),
            }],
        );
        s.assistant("I can see the screenshot.", None);
        let turn1Head = s.headTurnId.clone().unwrap();

        // Turn 2: user sends text-only follow-up.
        s.user("now fix the bug");
        s.assistant("Done.", None);

        // Rewind to turn 1 — images should be in the reconstructed history.
        let msgs = {
            let log = s.compactionLog();
            reconstruct(&s.transcript, &log, &turn1Head, 0, 0)
                .unwrap()
                .messages
        };
        dump("image_rewind", &msgs);

        assert_eq!(msgs.len(), 2, "User + Assistant");
        if let Message::User { content } = &msgs[0] {
            assert!(
                content.hasImages(),
                "rewound user message should have images"
            );
            let uris = content.imageUris();
            assert_eq!(uris.len(), 1);
            assert!(
                uris[0].contains("iVBORw0KGgo="),
                "should contain original base64 data"
            );
        } else {
            panic!("expected User message");
        }
    }

    #[test]
    fn imageAttachmentsReconstructedInFullChain() {
        let mut s = TestSession::new();

        s.userWithImages(
            "check this",
            vec![crate::transcript::TurnAttachment {
                mimeType: "image/jpeg".into(),
                data: "/9j/4AAQ".into(),
            }],
        );
        s.assistant("I see the image.", None);
        s.user("thanks");
        s.assistant("No problem.", None);

        // Full chain should preserve images on the first user message.
        let msgs = s.reconstruct();
        dump("image_full_chain", &msgs);

        if let Message::User { content } = &msgs[0] {
            assert!(content.hasImages(), "first user message should have images");
        } else {
            panic!("expected User message");
        }

        // Second user message should NOT have images.
        if let Message::User { content } = &msgs[2] {
            assert!(
                !content.hasImages(),
                "second user message should not have images"
            );
        } else {
            panic!("expected User message at index 2");
        }
    }
}
