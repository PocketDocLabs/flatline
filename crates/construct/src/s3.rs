//! S3 — per-topic LLM compaction.
//!
//! Fires at 80% of compactLimit (after S2 is exhausted). Groups consecutive
//! blocks sharing a topicId, retrieves **original** content from the
//! transcript (never summarizes a summary), and compresses each topic into
//! a single summary.
//!
//! The compressed output replaces all turns in the topic's blocks — including
//! user messages (unlike S2 which keeps them).
//!
//! Uses the dispassionate-voice `TOPIC_COMPACT_SYSTEM` prompt: state facts,
//! not narrative. Center user intent and corrections. Compress around user
//! messages as anchors.
//!
//! # Public API
//! - [`run`] — execute S3 topic compaction
//! - [`S3Result`] / [`CompactedTopic`]
//!
//! # Dependencies
//! `crate::api`, `crate::compaction`, `crate::transcript`, `crate::context`,
//! `crate::topic`, `futures`

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use futures::future::join_all;

use crate::api;
use crate::compaction::{CompactionLog, CompactionOp};
use crate::context;
use crate::topic::TopicInfo;
use crate::transcript::{Transcript, Turn, TurnRole};

/// A single topic that was compacted.
pub struct CompactedTopic {
    pub topicId: String,
    pub topicLabel: String,
    /// Raw compressed text (XML wrapping happens in context.rs).
    pub summary: String,
    /// Block IDs that were in this topic.
    pub sourceBlockIds: Vec<String>,
    /// File paths read in this topic (for edit gate invalidation).
    pub invalidatedFiles: Vec<String>,
}

pub struct S3Result {
    pub didWork: bool,
    pub compacted: Vec<CompactedTopic>,
}

/// Run S3 topic-level compaction.
///
/// Works from transcript + compaction log. Groups blocks by topicId,
/// targets the oldest 30% of live context by char count, retrieves
/// original content from transcript, and compresses each eligible
/// topic in parallel.
pub async fn run(
    transcript: &Transcript,
    compactionLog: &CompactionLog,
    headTurnId: &str,
    topics: &[TopicInfo],
    client: &api::Client,
    utilityModel: &str,
    _contextWindow: usize,
    _compactRatio: f64,
) -> Result<S3Result> {
    let turns = transcript.loadAll()?;
    if turns.is_empty() || topics.is_empty() {
        return Ok(S3Result { didWork: false, compacted: Vec::new() });
    }

    let ops = compactionLog.loadAll()?;

    // Topics already compacted at S3 level.
    let alreadyCompacted: HashSet<String> = ops
        .iter()
        .filter_map(|op| match op {
            CompactionOp::TopicCompact { sourceBlockIds, .. } => {
                // Use the first block ID as the topic's identity key.
                sourceBlockIds.first().cloned()
            }
            _ => None,
        })
        .collect();

    // Blocks already S2-compacted (prerequisite for S3).
    let s2CompactedBlocks: HashSet<String> = ops
        .iter()
        .filter_map(|op| match op {
            CompactionOp::BlockCompact { blockId, .. } => Some(blockId.clone()),
            _ => None,
        })
        .collect();

    // Reconstruct live context to measure the 30% zone.
    // NOTE: promptThinking=false here — S3 only measures zone sizes,
    // reasoning is already excluded from char counts.
    let liveHistory = context::reconstruct(transcript, compactionLog, headTurnId, false)?;
    let charTarget = calculateZoneChars(&liveHistory);

    if charTarget == 0 {
        return Ok(S3Result { didWork: false, compacted: Vec::new() });
    }

    // Group transcript turns by blockId for content retrieval.
    let blockContent = groupTurnsByBlock(&turns);

    // Walk topics from oldest, accumulate chars from live context,
    // collect eligible topics within the 30% zone.
    let eligible = findEligibleTopics(
        topics,
        &liveHistory,
        &blockContent,
        &s2CompactedBlocks,
        &alreadyCompacted,
        charTarget,
    );

    if eligible.is_empty() {
        return Ok(S3Result { didWork: false, compacted: Vec::new() });
    }

    // Fire parallel compaction calls.
    let futures: Vec<_> = eligible
        .iter()
        .map(|topic| {
            compactTopic(topic, &blockContent, client, utilityModel)
        })
        .collect();

    let results = join_all(futures).await;

    let mut compacted = Vec::new();
    for (i, result) in results.into_iter().enumerate() {
        let topic = &eligible[i];
        match result {
            Ok(summary) => {
                compacted.push(CompactedTopic {
                    topicId: topic.topicId.clone(),
                    topicLabel: topic.label.clone(),
                    summary,
                    sourceBlockIds: topic.blockIds.clone(),
                    invalidatedFiles: collectReadFiles(&topic.blockIds, &blockContent),
                });
            }
            Err(e) => {
                tracing::warn!(
                    topicId = %topic.topicId,
                    label = %topic.label,
                    error = %e,
                    "S3 topic compaction failed, skipping"
                );
            }
        }
    }

    let didWork = !compacted.is_empty();
    Ok(S3Result { didWork, compacted })
}

/// An eligible topic ready for compaction.
struct EligibleTopic {
    topicId: String,
    label: String,
    blockIds: Vec<String>,
}

/// Turns grouped by blockId, preserving order.
struct BlockTurns {
    userMessage: String,
    agentTurns: Vec<TurnContent>,
}

struct TurnContent {
    role: TurnRole,
    content: String,
    toolName: Option<String>,
    toolArgs: Option<serde_json::Value>,
    reasoning: Option<String>,
}

/// Group transcript turns by blockId for content retrieval.
fn groupTurnsByBlock(turns: &[Turn]) -> HashMap<String, BlockTurns> {
    let mut map: HashMap<String, BlockTurns> = HashMap::new();
    // Preserve insertion order by using a separate vec.
    let mut order: Vec<String> = Vec::new();

    for turn in turns {
        if !map.contains_key(&turn.blockId) {
            order.push(turn.blockId.clone());
            map.insert(turn.blockId.clone(), BlockTurns {
                userMessage: String::new(),
                agentTurns: Vec::new(),
            });
        }

        let block = map.get_mut(&turn.blockId).unwrap();

        match turn.role {
            TurnRole::User => {
                block.userMessage = turn.content.clone();
            }
            TurnRole::Assistant => {
                block.agentTurns.push(TurnContent {
                    role: TurnRole::Assistant,
                    content: turn.content.clone(),
                    toolName: None,
                    toolArgs: None,
                    reasoning: turn.reasoning.clone(),
                });
            }
            TurnRole::ToolCall => {
                block.agentTurns.push(TurnContent {
                    role: TurnRole::ToolCall,
                    content: turn.content.clone(),
                    toolName: turn.tool.clone(),
                    toolArgs: turn.args.clone(),
                    reasoning: None,
                });
            }
            TurnRole::ToolResult => {
                block.agentTurns.push(TurnContent {
                    role: TurnRole::ToolResult,
                    content: turn.content.clone(),
                    toolName: None,
                    toolArgs: None,
                    reasoning: None,
                });
            }
            TurnRole::System => {}
        }
    }

    map
}

/// Calculate the S3 zone: oldest 30% of live context by char count.
/// Returns the char boundary.
fn calculateZoneChars(history: &[crate::message::Message]) -> usize {
    if history.len() <= 1 {
        return 0;
    }

    let totalChars: usize = history[1..].iter().map(|m| messageLen(m)).sum();
    totalChars * 30 / 100
}

fn messageLen(msg: &crate::message::Message) -> usize {
    match msg {
        crate::message::Message::System { content }
        | crate::message::Message::User { content } => content.len(),
        crate::message::Message::Assistant { content, tool_calls, .. } => {
            let textLen = content.as_ref().map_or(0, |c| c.len());
            let callsLen = tool_calls.as_ref().map_or(0, |calls| {
                calls.iter().map(|c| c.function.arguments.len() + c.function.name.len()).sum()
            });
            textLen + callsLen
        }
        crate::message::Message::Tool { content, .. } => content.len(),
    }
}

/// Find topics eligible for S3 compaction within the 30% zone.
///
/// A topic is eligible if:
/// - It has more than one block (single-block topics are already S2's domain)
/// - All its blocks are S2-compacted
/// - It hasn't been S3-compacted yet
/// - It starts within the 30% zone (or straddles the boundary)
fn findEligibleTopics(
    topics: &[TopicInfo],
    liveHistory: &[crate::message::Message],
    blockContent: &HashMap<String, BlockTurns>,
    s2Compacted: &HashSet<String>,
    alreadyCompacted: &HashSet<String>,
    charTarget: usize,
) -> Vec<EligibleTopic> {
    // Walk live history to map char positions to block boundaries.
    // We need to know which blocks fall in the 30% zone.
    let zoneBlockIds = blocksInZone(liveHistory, charTarget);

    let mut eligible = Vec::new();

    for topic in topics {
        // Expand topic into its block IDs.
        let blockIds = expandTopicBlocks(topic, blockContent);
        if blockIds.is_empty() {
            continue;
        }

        // Skip single-block topics — S2 already handles those.
        if blockIds.len() <= 1 {
            continue;
        }

        // Skip if already S3-compacted (idempotency check on first block).
        if alreadyCompacted.contains(&blockIds[0]) {
            continue;
        }

        // At least the first block must be in the zone
        // (straddling is OK — topic can extend past the boundary).
        if !zoneBlockIds.contains(&blockIds[0]) {
            continue;
        }

        // All blocks must be S2-compacted (S3 prerequisite).
        let allS2 = blockIds.iter().all(|bid| s2Compacted.contains(bid));
        if !allS2 {
            continue;
        }

        eligible.push(EligibleTopic {
            topicId: topic.topicId.clone(),
            label: topic.label.clone(),
            blockIds,
        });
    }

    eligible
}

/// Expand a TopicInfo into its block IDs by parsing startBlock and counting.
fn expandTopicBlocks(
    topic: &TopicInfo,
    blockContent: &HashMap<String, BlockTurns>,
) -> Vec<String> {
    // startBlock format: "b000001", blockCount tells us how many.
    let startNum = topic.startBlock
        .get(1..)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let mut ids = Vec::new();
    for i in 0..topic.blockCount as u32 {
        let bid = format!("b{:06}", startNum + i);
        // Only include blocks that actually exist in the transcript.
        if blockContent.contains_key(&bid) {
            ids.push(bid);
        }
    }
    ids
}

/// Find block IDs that fall within the 30% char zone of live history.
fn blocksInZone(
    history: &[crate::message::Message],
    charTarget: usize,
) -> HashSet<String> {
    let mut zoneBlocks = HashSet::new();
    let mut cumulative: usize = 0;

    // Walk messages from oldest (skip system at index 0).
    // Track block transitions by looking for User messages
    // (each User message starts a new block).
    let mut currentBlock = String::new();
    let mut blockNum: u32 = 0;

    for i in 1..history.len() {
        if cumulative >= charTarget {
            break;
        }

        let len = messageLen(&history[i]);

        // Detect block boundaries: User messages start new blocks.
        if matches!(history[i], crate::message::Message::User { .. }) {
            blockNum += 1;
            currentBlock = format!("b{:06}", blockNum);
        }

        if !currentBlock.is_empty() {
            zoneBlocks.insert(currentBlock.clone());
        }

        cumulative += len;
    }

    zoneBlocks
}

/// Collect file paths read in the given blocks (for edit gate invalidation).
fn collectReadFiles(
    blockIds: &[String],
    blockContent: &HashMap<String, BlockTurns>,
) -> Vec<String> {
    let mut paths = Vec::new();
    for bid in blockIds {
        if let Some(block) = blockContent.get(bid) {
            for turn in &block.agentTurns {
                if turn.toolName.as_deref() == Some("readFile") {
                    if let Some(args) = &turn.toolArgs {
                        if let Some(path) = args["path"].as_str() {
                            let norm = normalizePath(path);
                            if !paths.contains(&norm) {
                                paths.push(norm);
                            }
                        }
                    }
                }
            }
        }
    }
    paths
}

/// Build the prompt and call the utility model for one topic.
async fn compactTopic(
    topic: &EligibleTopic,
    blockContent: &HashMap<String, BlockTurns>,
    client: &api::Client,
    utilityModel: &str,
) -> Result<String> {
    let mut topicParts: Vec<String> = Vec::new();

    for bid in &topic.blockIds {
        let block = match blockContent.get(bid) {
            Some(b) => b,
            None => continue,
        };

        let mut exchangeParts = vec![format!("<exchange id=\"{bid}\">")];
        exchangeParts.push(format!("<user_turn>{}</user_turn>", block.userMessage));

        for turn in &block.agentTurns {
            match turn.role {
                TurnRole::Assistant => {
                    let body = if let Some(r) = &turn.reasoning {
                        format!("<reasoning>{r}</reasoning>\n{}", turn.content)
                    } else {
                        turn.content.clone()
                    };
                    exchangeParts.push(format!("<agent_turn>{body}</agent_turn>"));
                }
                TurnRole::ToolResult => {
                    exchangeParts.push(format!("<tool_output>{}</tool_output>", turn.content));
                }
                TurnRole::ToolCall => {
                    let name = turn.toolName.as_deref().unwrap_or("unknown");
                    let argStr = turn.toolArgs.as_ref()
                        .map(|v| {
                            let s = v.to_string();
                            if s.len() > 500 { format!("{}...", &s[..500]) } else { s }
                        })
                        .unwrap_or_else(|| "{}".to_string());
                    exchangeParts.push(format!(
                        "<agent_turn>[tool_call: {name}({argStr})]</agent_turn>"
                    ));
                }
                _ => {}
            }
        }

        exchangeParts.push("</exchange>".to_string());
        topicParts.push(exchangeParts.join("\n"));
    }

    let compactContent = topicParts.join("\n\n");
    if compactContent.trim().is_empty() {
        anyhow::bail!("no content to compact in topic {}", topic.topicId);
    }

    let userPrompt = format!(
        "<compact_this>\n{compactContent}\n</compact_this>\n\n\
         Write only the compacted output wrapped in \
         <compacted_monolithic_string> tags. No preamble."
    );

    let messages = vec![
        crate::message::Message::System {
            content: TOPIC_COMPACT_SYSTEM.to_string(),
        },
        crate::message::Message::User {
            content: userPrompt,
        },
    ];

    let response = client.complete(&messages, Some(utilityModel)).await?;
    let summary = extractCompactedString(&response);
    Ok(summary)
}

/// Extract content from `<compacted_monolithic_string>` tags, or return raw.
fn extractCompactedString(response: &str) -> String {
    if let Some(start) = response.find("<compacted_monolithic_string>") {
        if let Some(end) = response.find("</compacted_monolithic_string>") {
            let inner = &response[start + 29..end];
            return inner.trim().to_string();
        }
    }
    response.trim().to_string()
}

fn normalizePath(path: &str) -> String {
    std::path::Path::new(path)
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string())
}

const TOPIC_COMPACT_SYSTEM: &str = "\
You compress a sequence of related conversation exchanges into a single \
compacted block. The exchanges all belong to one logical thread of work. \
Your compacted output replaces the entire sequence in the conversation.\n\
\n\
Another agent will continue the work using only your compacted output \
as context for everything that happened in this topic.\n\
\n\
Voice and perspective:\n\
- Write as a dispassionate record of facts, not a narrative of work \
performed.\n\
- The agent is labor, not a protagonist. Do not narrate its \
contributions, achievements, or process. State what exists now and \
what the user decided.\n\
- BAD: \"Implemented a modular prompt system with enums and a build \
function\"\n\
- GOOD: \"prompt.rs exists with InterfaceMode enum (SharedTerminal, \
Headless, MultiAgent) and build() assembler\"\n\
- BAD: \"The project compiles clean and all 4 tests pass\"\n\
- GOOD: \"Compiles. 4 tests pass.\"\n\
- Center user intent and corrections. The user's words matter more \
than the agent's actions.\n\
- If a user message is short (under ~200 chars), quote it verbatim. \
The agent content is what gets compressed \u{2014} user messages are the \
anchors you compress around.\n\
- Only attribute decisions to the user if the user explicitly stated \
them. If the agent proposed something and the user didn't object, \
that is not the user deciding \u{2014} it's the agent's proposal standing \
uncontested. Say \"agent proposed X\" not \"user chose X\" unless the \
user's own words confirm it.\n\
- Distinguish agent proposals from agent commitments. \"Agent \
suggested using Redis\" is a proposal \u{2014} unconfirmed, may not matter. \
\"Agent said it will add rate limiting next\" is a commitment \u{2014} the \
next agent must follow through or acknowledge it wasn't done.\n\
\n\
Compression priorities (in order):\n\
1. Final state \u{2014} what exists now, stated as fact\n\
2. User intent \u{2014} what the user asked for and any corrections they gave\n\
3. Decisions \u{2014} choices made and why, especially rejected alternatives\n\
4. Errors \u{2014} problems hit and how they were resolved (avoid repeats)\n\
5. Unfinished work \u{2014} anything started but not completed\n\
\n\
Compression rules:\n\
- PRESERVE: exact file paths, function/variable names, error messages, \
configuration values, user corrections\n\
- MERGE: multiple rounds of similar work into outcome statements\n\
- OMIT: intermediate exploration that led nowhere, raw tool output, \
redundant reads of the same file, self-congratulatory conclusions\n\
- For code changes: describe what changed and why, not the full diff\n\
\n\
Structure:\n\
- Opening: one sentence \u{2014} what the user wanted and whether it's done\n\
- Body: facts following the priorities above\n\
- Closing: unresolved issues or noted next steps, if any\n\
\n\
Target length: 1-3 short paragraphs depending on topic complexity.";
