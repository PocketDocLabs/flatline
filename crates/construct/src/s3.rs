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
    /// Total USD cost of all utility model calls in this S3 pass.
    pub cost: Option<f64>,
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
        return Ok(S3Result { didWork: false, compacted: Vec::new(), cost: None });
    }

    let ops = compactionLog.loadAll()?;
    let blockContent = groupTurnsByBlock(&turns);
    let eligible = resolveEligible(&turns, &ops, headTurnId, topics);

    if eligible.is_empty() {
        tracing::debug!(
            topicCount = topics.len(),
            eligibleCount = 0,
            "S3: no eligible topics found"
        );
        return Ok(S3Result { didWork: false, compacted: Vec::new(), cost: None });
    }

    // Fire parallel compaction calls.
    let futures: Vec<_> = eligible
        .iter()
        .map(|topic| {
            compactTopic(topic, &blockContent, client, utilityModel)
        })
        .collect();

    let results: Vec<Result<(String, Option<f64>)>> = join_all(futures).await;

    let mut compacted = Vec::new();
    let mut totalCost: f64 = 0.0;
    for (i, result) in results.into_iter().enumerate() {
        let topic = &eligible[i];
        match result {
            Ok((summary, topicCost)) => {
                if let Some(c) = topicCost {
                    totalCost += c;
                }
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
    let cost = if totalCost > 0.0 { Some(totalCost) } else { None };
    Ok(S3Result { didWork, compacted, cost })
}

/// Resolve which topics are eligible for S3 compaction.
///
/// Builds the zone, filters superseded blocks, and finds topics with
/// >= 2 S2'd blocks in the zone. This is the complete eligibility
/// pipeline that `run` uses — extracted so tests can verify it without
/// needing an API client.
fn resolveEligible(
    turns: &[Turn],
    ops: &[CompactionOp],
    headTurnId: &str,
    topics: &[TopicInfo],
) -> Vec<EligibleTopic> {
    let alreadyCompacted: HashSet<String> = ops
        .iter()
        .filter_map(|op| match op {
            CompactionOp::TopicCompact { sourceBlockIds, .. } => {
                sourceBlockIds.first().cloned()
            }
            _ => None,
        })
        .collect();

    let compactedSizes = crate::compaction::compactedBlockSizes(ops);
    let topicBlockMap = buildTopicBlockMap(turns);

    let activeTurns = crate::transcript::walkBranchTurns(turns, headTurnId);
    if activeTurns.is_empty() {
        return Vec::new();
    }

    let superseded = crate::compaction::supersededBlocks(ops);
    let zone = crate::compaction::zoneBlocks(&activeTurns, &compactedSizes, &superseded, 0.30);

    findEligibleTopics(topics, &topicBlockMap, &zone, &compactedSizes, &alreadyCompacted)
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

/// Find topics eligible for S3 compaction within the 30% zone.
///
/// A topic is eligible if:
/// - It has more than one block (single-block topics are already S2's domain)
/// - All its blocks are S2-compacted
/// - It hasn't been S3-compacted yet
/// - It starts within the 30% zone (or straddles the boundary)
fn findEligibleTopics(
    topics: &[TopicInfo],
    topicBlockMap: &HashMap<String, Vec<String>>,
    zone: &HashSet<String>,
    compactedSizes: &HashMap<String, usize>,
    alreadyCompacted: &HashSet<String>,
) -> Vec<EligibleTopic> {
    let mut eligible = Vec::new();

    for topic in topics {
        let blockIds = match topicBlockMap.get(&topic.topicId) {
            Some(ids) => ids.clone(),
            None => {
                tracing::debug!(topicId = %topic.topicId, "S3 skip: not in transcript");
                continue;
            }
        };
        if blockIds.is_empty() {
            continue;
        }

        // Skip if already S3-compacted (idempotency check on first block).
        if alreadyCompacted.contains(&blockIds[0]) {
            tracing::debug!(topicId = %topic.topicId, "S3 skip: already compacted");
            continue;
        }

        // Narrow to blocks that are both in the zone and S2-compacted.
        // Partial topic compaction is OK — the rest gets picked up on future passes.
        let zonedBlocks: Vec<String> = blockIds
            .iter()
            .filter(|bid| zone.contains(*bid) && compactedSizes.contains_key(*bid))
            .cloned()
            .collect();

        if zonedBlocks.is_empty() {
            tracing::debug!(topicId = %topic.topicId, "S3 skip: no in-zone S2'd blocks");
            continue;
        }

        // Need at least 2 blocks — single blocks are S2's domain.
        if zonedBlocks.len() <= 1 {
            tracing::debug!(topicId = %topic.topicId, "S3 skip: only 1 block in zone");
            continue;
        }

        eligible.push(EligibleTopic {
            topicId: topic.topicId.clone(),
            label: topic.label.clone(),
            blockIds: zonedBlocks,
        });
    }

    eligible
}

/// Build a map of topicId → ordered distinct block IDs from transcript turns.
fn buildTopicBlockMap(turns: &[Turn]) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    let mut seen: HashMap<String, HashSet<String>> = HashMap::new();

    for turn in turns {
        if turn.topicId.is_empty() {
            continue;
        }
        let set = seen.entry(turn.topicId.clone()).or_default();
        if set.insert(turn.blockId.clone()) {
            map.entry(turn.topicId.clone())
                .or_default()
                .push(turn.blockId.clone());
        }
    }

    map
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
) -> Result<(String, Option<f64>)> {
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
                            if s.len() > 500 { format!("{}...", &s[..s.floor_char_boundary(500)]) } else { s }
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
            content: userPrompt.into(),
        },
    ];

    let (response, usage) = client.complete(&messages, Some(utilityModel)).await?;
    let cost = usage.and_then(|u| u.cost);
    let summary = extractCompactedString(&response);
    Ok((summary, cost))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn makeTurn(id: &str, blockId: &str, topicId: &str, role: TurnRole, content: &str, parentId: Option<&str>) -> Turn {
        Turn {
            id: id.to_string(),
            blockId: blockId.to_string(),
            topicId: topicId.to_string(),
            role,
            content: content.to_string(),
            ts: 0,
            parentId: parentId.map(|s| s.to_string()),
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments: None,
            cost: None,
        }
    }

    #[test]
    fn buildTopicBlockMap_groups_by_topicId() {
        let turns = vec![
            makeTurn("t1", "b_aaa", "topic-01", TurnRole::User, "a", None),
            makeTurn("t2", "b_aaa", "topic-01", TurnRole::Assistant, "b", Some("t1")),
            makeTurn("t3", "b_bbb", "topic-01", TurnRole::User, "c", Some("t2")),
            makeTurn("t4", "b_ccc", "topic-02", TurnRole::User, "d", Some("t3")),
            makeTurn("t5", "b_ddd", "topic-02", TurnRole::User, "e", Some("t4")),
        ];
        let map = buildTopicBlockMap(&turns);
        assert_eq!(map["topic-01"], vec!["b_aaa", "b_bbb"]);
        assert_eq!(map["topic-02"], vec!["b_ccc", "b_ddd"]);
    }

    #[test]
    fn zoneBlocks_returns_oldest_30_percent() {
        let turns = vec![
            makeTurn("t1", "b_aaa", "", TurnRole::User, &"x".repeat(50), None),
            makeTurn("t2", "b_aaa", "", TurnRole::Assistant, &"y".repeat(50), Some("t1")),
            makeTurn("t3", "b_bbb", "", TurnRole::User, &"x".repeat(50), Some("t2")),
            makeTurn("t4", "b_bbb", "", TurnRole::Assistant, &"y".repeat(50), Some("t3")),
            makeTurn("t5", "b_ccc", "", TurnRole::User, &"x".repeat(50), Some("t4")),
            makeTurn("t6", "b_ccc", "", TurnRole::Assistant, &"y".repeat(50), Some("t5")),
            makeTurn("t7", "b_ddd", "", TurnRole::User, &"x".repeat(50), Some("t6")),
            makeTurn("t8", "b_ddd", "", TurnRole::Assistant, &"y".repeat(50), Some("t7")),
        ];

        let compactedSizes = HashMap::new();
        let zone = crate::compaction::zoneBlocks(&turns, &compactedSizes, &HashSet::new(), 0.30);

        assert!(zone.contains("b_aaa"), "first block should be in zone");
        assert!(!zone.contains("b_ddd"), "last block should NOT be in zone");
    }

    #[test]
    fn zoneBlocks_extends_past_compacted_blocks() {
        // 4 blocks, first 2 compacted to 10 chars each, last 2 raw at 200 each.
        // Total effective = 10 + 10 + 200 + 200 = 420. 30% = 126.
        // Zone should include blocks aaa (10) + bbb (10) + ccc (200 pushes past 126).
        let turns = vec![
            makeTurn("t1", "b_aaa", "", TurnRole::User, &"x".repeat(100), None),
            makeTurn("t2", "b_aaa", "", TurnRole::Assistant, &"y".repeat(100), Some("t1")),
            makeTurn("t3", "b_bbb", "", TurnRole::User, &"x".repeat(100), Some("t2")),
            makeTurn("t4", "b_bbb", "", TurnRole::Assistant, &"y".repeat(100), Some("t3")),
            makeTurn("t5", "b_ccc", "", TurnRole::User, &"x".repeat(100), Some("t4")),
            makeTurn("t6", "b_ccc", "", TurnRole::Assistant, &"y".repeat(100), Some("t5")),
            makeTurn("t7", "b_ddd", "", TurnRole::User, &"x".repeat(100), Some("t6")),
            makeTurn("t8", "b_ddd", "", TurnRole::Assistant, &"y".repeat(100), Some("t7")),
        ];

        let mut compactedSizes = HashMap::new();
        compactedSizes.insert("b_aaa".to_string(), 10_usize);
        compactedSizes.insert("b_bbb".to_string(), 10_usize);

        let zone = crate::compaction::zoneBlocks(&turns, &compactedSizes, &HashSet::new(), 0.30);

        // Without compacted sizes: 30% of 800 = 240 → only aaa+bbb (400 raw) would fill it.
        // With compacted sizes: 30% of 420 = 126 → aaa(10)+bbb(10)+ccc(200) fills it.
        assert!(zone.contains("b_aaa"), "compacted block should be in zone");
        assert!(zone.contains("b_bbb"), "compacted block should be in zone");
        assert!(zone.contains("b_ccc"), "uncompacted block should now be reachable");
        assert!(!zone.contains("b_ddd"), "last block should not be in zone");
    }

    #[test]
    fn findEligibleTopics_finds_multi_block_s2d_topic_in_zone() {
        let turns = vec![
            makeTurn("t1", "b_aaa", "topic-01", TurnRole::User, &"x".repeat(100), None),
            makeTurn("t2", "b_aaa", "topic-01", TurnRole::Assistant, &"y".repeat(100), Some("t1")),
            makeTurn("t3", "b_bbb", "topic-01", TurnRole::User, &"x".repeat(100), Some("t2")),
            makeTurn("t4", "b_bbb", "topic-01", TurnRole::Assistant, &"y".repeat(100), Some("t3")),
            makeTurn("t5", "b_ccc", "topic-02", TurnRole::User, &"x".repeat(100), Some("t4")),
            makeTurn("t6", "b_ccc", "topic-02", TurnRole::Assistant, &"y".repeat(100), Some("t5")),
            makeTurn("t7", "b_ddd", "topic-02", TurnRole::User, &"x".repeat(100), Some("t6")),
            makeTurn("t8", "b_ddd", "topic-02", TurnRole::Assistant, &"y".repeat(100), Some("t7")),
        ];

        let topics = vec![
            TopicInfo { topicId: "topic-01".into(), label: "First".into(), startBlock: "b_aaa".into(), blockCount: 2 },
            TopicInfo { topicId: "topic-02".into(), label: "Second".into(), startBlock: "b_ccc".into(), blockCount: 2 },
        ];

        let topicBlockMap = buildTopicBlockMap(&turns);
        let mut compactedSizes = HashMap::new();
        for id in &["b_aaa", "b_bbb", "b_ccc", "b_ddd"] {
            compactedSizes.insert(id.to_string(), 50_usize);
        }
        let zone = crate::compaction::zoneBlocks(&turns, &compactedSizes, &HashSet::new(), 0.30);
        let alreadyCompacted: HashSet<String> = HashSet::new();

        let eligible = findEligibleTopics(
            &topics, &topicBlockMap, &zone, &compactedSizes, &alreadyCompacted,
        );

        assert!(!eligible.is_empty(), "should find at least one eligible topic");
        assert_eq!(eligible[0].topicId, "topic-01");
    }

    #[test]
    fn findEligibleTopics_allows_partial_topic_compaction() {
        let turns = vec![
            makeTurn("t1", "b_aaa", "topic-01", TurnRole::User, &"x".repeat(100), None),
            makeTurn("t2", "b_aaa", "topic-01", TurnRole::Assistant, &"y".repeat(100), Some("t1")),
            makeTurn("t3", "b_bbb", "topic-01", TurnRole::User, &"x".repeat(100), Some("t2")),
            makeTurn("t4", "b_bbb", "topic-01", TurnRole::Assistant, &"y".repeat(100), Some("t3")),
            makeTurn("t5", "b_ccc", "topic-01", TurnRole::User, &"x".repeat(100), Some("t4")),
            makeTurn("t6", "b_ccc", "topic-01", TurnRole::Assistant, &"y".repeat(100), Some("t5")),
            makeTurn("t7", "b_ddd", "topic-01", TurnRole::User, &"x".repeat(100), Some("t6")),
            makeTurn("t8", "b_ddd", "topic-01", TurnRole::Assistant, &"y".repeat(100), Some("t7")),
        ];

        let topics = vec![
            TopicInfo { topicId: "topic-01".into(), label: "Big Topic".into(), startBlock: "b_aaa".into(), blockCount: 4 },
        ];

        let topicBlockMap = buildTopicBlockMap(&turns);
        // Only first 2 blocks S2'd.
        let mut compactedSizes = HashMap::new();
        compactedSizes.insert("b_aaa".to_string(), 50_usize);
        compactedSizes.insert("b_bbb".to_string(), 50_usize);
        let zone = crate::compaction::zoneBlocks(&turns, &compactedSizes, &HashSet::new(), 0.30);
        let alreadyCompacted: HashSet<String> = HashSet::new();

        let eligible = findEligibleTopics(
            &topics, &topicBlockMap, &zone, &compactedSizes, &alreadyCompacted,
        );

        assert_eq!(eligible.len(), 1, "topic-01 should be eligible (partial)");
        assert_eq!(eligible[0].blockIds.len(), 2, "only the 2 in-zone S2'd blocks");
        assert_eq!(eligible[0].blockIds, vec!["b_aaa", "b_bbb"]);
    }

    /// Helper: build compaction ops for a set of S2'd blocks, an S3 topic,
    /// and an S4 briefing covering those blocks.
    fn makeOps(
        s2BlockIds: &[String],
        s3TopicLabel: &str,
        s3SourceBlockIds: &[String],
        s4SourceIds: &[String],
    ) -> Vec<CompactionOp> {
        let mut ops = Vec::new();
        let summary800 = "s".repeat(800);
        for bid in s2BlockIds {
            ops.push(CompactionOp::BlockCompact {
                stage: "s2".into(),
                blockId: bid.clone(),
                summary: summary800.clone(),
                sourceIds: vec![],
                afterTurn: "t_head".into(),
                ts: 0,
            });
        }
        if !s3SourceBlockIds.is_empty() {
            ops.push(CompactionOp::TopicCompact {
                stage: "s3".into(),
                topicLabel: s3TopicLabel.into(),
                summary: "topic summary".into(),
                sourceBlockIds: s3SourceBlockIds.to_vec(),
                afterTurn: "t_head".into(),
                ts: 0,
            });
        }
        if !s4SourceIds.is_empty() {
            ops.push(CompactionOp::FullCompact {
                stage: "s4".into(),
                summary: "full briefing".into(),
                sourceIds: s4SourceIds.to_vec(),
                afterTurn: "t_head".into(),
                ts: 0,
            });
        }
        ops
    }

    /// Reproduces the real-world S3 starvation bug: after S4 runs, its
    /// covered blocks should be excluded from the zone so S3 can reach
    /// newer S2'd blocks with eligible topics.
    ///
    /// Uses resolveEligible (the production code path) — no manual zone
    /// or superseded set construction.
    #[test]
    fn s3_starved_when_s4_blocks_inflate_zone() {
        let mut turns = Vec::new();
        let mut parentId: Option<&str> = None;
        let mut turnNum = 0;

        // 20 S4-covered blocks (topic-01).
        let mut s4BlockIds = Vec::new();
        for i in 0..20 {
            let bid = format!("b_s4_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-01", TurnRole::User, &"x".repeat(2000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-01", TurnRole::Assistant, &"y".repeat(2000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
            s4BlockIds.push(bid);
        }

        // 6 S2'd blocks: 3 in topic-02, 3 in topic-03.
        let mut t2Ids = Vec::new();
        let mut t3Ids = Vec::new();
        for i in 0..3 {
            let bid = format!("b_t2_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-02", TurnRole::User, &"x".repeat(2000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-02", TurnRole::Assistant, &"y".repeat(2000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
            t2Ids.push(bid);
        }
        for i in 0..3 {
            let bid = format!("b_t3_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-03", TurnRole::User, &"x".repeat(2000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-03", TurnRole::Assistant, &"y".repeat(2000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
            t3Ids.push(bid);
        }

        // 4 raw blocks (topic-04).
        for i in 0..4 {
            let bid = format!("b_raw_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-04", TurnRole::User, &"x".repeat(500), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-04", TurnRole::Assistant, &"y".repeat(500), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
        }

        let headTurnId = turns.last().unwrap().id.clone();
        let allBlockIds: Vec<String> = s4BlockIds.iter()
            .chain(t2Ids.iter()).chain(t3Ids.iter())
            .cloned().collect();

        let ops = makeOps(&allBlockIds, "Old Topic", &s4BlockIds, &s4BlockIds);

        let topics = vec![
            TopicInfo { topicId: "topic-01".into(), label: "Old".into(), startBlock: s4BlockIds[0].clone(), blockCount: 20 },
            TopicInfo { topicId: "topic-02".into(), label: "Middle A".into(), startBlock: t2Ids[0].clone(), blockCount: 3 },
            TopicInfo { topicId: "topic-03".into(), label: "Middle B".into(), startBlock: t3Ids[0].clone(), blockCount: 3 },
            TopicInfo { topicId: "topic-04".into(), label: "Recent".into(), startBlock: "b_raw_000".into(), blockCount: 4 },
        ];

        let eligible = resolveEligible(&turns, &ops, &headTurnId, &topics);

        assert!(
            !eligible.is_empty(),
            "topic-02 should be S3-eligible — S4 blocks must not inflate zone"
        );
    }

    /// Same bug at scale: 40 S4-covered, 20 S2'd across 2 topics, 10 raw.
    #[test]
    fn s3_starved_at_scale_with_many_s4_blocks() {
        let mut turns = Vec::new();
        let mut parentId: Option<&str> = None;
        let mut turnNum = 0;

        let mut s4Ids = Vec::new();
        for i in 0..40 {
            let bid = format!("b_s4_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-01", TurnRole::User, &"x".repeat(2000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-01", TurnRole::Assistant, &"y".repeat(2000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
            s4Ids.push(bid);
        }

        let mut t2Ids = Vec::new();
        let mut t3Ids = Vec::new();
        for i in 0..10 {
            let bid = format!("b_t2_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-02", TurnRole::User, &"x".repeat(2000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-02", TurnRole::Assistant, &"y".repeat(2000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
            t2Ids.push(bid);
        }
        for i in 0..10 {
            let bid = format!("b_t3_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-03", TurnRole::User, &"x".repeat(2000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-03", TurnRole::Assistant, &"y".repeat(2000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
            t3Ids.push(bid);
        }

        for i in 0..10 {
            let bid = format!("b_raw_{i:03}");
            let uid = format!("t{turnNum}"); turnNum += 1;
            let aid = format!("t{turnNum}"); turnNum += 1;
            turns.push(makeTurn(&uid, &bid, "topic-04", TurnRole::User, &"x".repeat(1000), parentId));
            turns.push(makeTurn(&aid, &bid, "topic-04", TurnRole::Assistant, &"y".repeat(1000), Some(&uid)));
            parentId = Some(Box::leak(aid.into_boxed_str()));
        }

        let headTurnId = turns.last().unwrap().id.clone();
        let allS2Ids: Vec<String> = s4Ids.iter()
            .chain(t2Ids.iter()).chain(t3Ids.iter())
            .cloned().collect();

        let ops = makeOps(&allS2Ids, "Old Topic", &s4Ids, &s4Ids);

        let topics = vec![
            TopicInfo { topicId: "topic-01".into(), label: "Old".into(), startBlock: s4Ids[0].clone(), blockCount: 40 },
            TopicInfo { topicId: "topic-02".into(), label: "Middle A".into(), startBlock: t2Ids[0].clone(), blockCount: 10 },
            TopicInfo { topicId: "topic-03".into(), label: "Middle B".into(), startBlock: t3Ids[0].clone(), blockCount: 10 },
            TopicInfo { topicId: "topic-04".into(), label: "Recent".into(), startBlock: "b_raw_000".into(), blockCount: 10 },
        ];

        let eligible = resolveEligible(&turns, &ops, &headTurnId, &topics);

        assert!(
            eligible.len() >= 2,
            "topic-02 and topic-03 should both be S3-eligible, got {} eligible",
            eligible.len(),
        );
    }
}
