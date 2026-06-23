#![allow(clippy::items_after_test_module)]

//! S4 — deep recompaction (single LLM call).
//!
//! Last-resort recompression of the OLDER compacted layers when S1–S3
//! are exhausted.
//!
//! Sources fed into S4:
//! - The latest active S4 briefing, if one exists. Older S4 ops are
//!   historical log entries already consumed by that frontier.
//! - S3 topic summaries not already covered by a later S4.
//!
//! Reads the post-compaction summaries from the log directly.
//!
//! # Public API
//! - [`run`] — execute S4 compaction
//! - [`S4Result`] — what was produced
//!
//! # Dependencies
//! `crate::api`, `crate::compaction`, `crate::message`, `crate::transcript`

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use crate::api;
use crate::compaction::{CompactionLog, CompactionOp};
use crate::config::ModelTier;
use crate::message::Message;
use crate::transcript::Transcript;

pub struct S4Result {
    pub didWork: bool,
    pub summary: String,
    pub sourceBlockIds: Vec<String>,
    /// USD cost of the utility model call (None if not reported).
    pub cost: Option<f64>,
}

/// Recompress the deepest eligible compacted layers into a single
/// handoff briefing while protecting the newest portion of context.
///
/// The protected band is computed from the active branch using each
/// block's effective size (compacted summary length if available,
/// otherwise the raw char count). Any candidate op whose blocks
/// intersect the protected band is skipped.
pub async fn run(
    transcript: &Transcript,
    compactionLog: &CompactionLog,
    headTurnId: &str,
    client: &api::Client,
    utilityModel: &str,
    maxInputChars: usize,
) -> Result<S4Result> {
    let ops = compactionLog.loadAll()?;

    let allTurns = transcript.loadAll()?;
    let activeTurns = crate::transcript::walkBranchTurns(&allTurns, headTurnId);
    let activeBlockIds: HashSet<&str> = activeTurns.iter().map(|t| t.blockId.as_str()).collect();
    let activeTurnIds: HashSet<&str> = activeTurns.iter().map(|t| t.id.as_str()).collect();
    let knownTurnIds: HashSet<&str> = allTurns.iter().map(|t| t.id.as_str()).collect();
    let activeOps: Vec<CompactionOp> = ops
        .into_iter()
        .filter(|op| opAppliesToActiveBranch(op, &activeBlockIds, &activeTurnIds, &knownTurnIds))
        .collect();

    let plan = buildRecompactPlan(&activeOps, &activeTurns, maxInputChars);

    if plan.sections.is_empty() {
        return Ok(S4Result {
            didWork: false,
            summary: String::new(),
            sourceBlockIds: Vec::new(),
            cost: None,
        });
    }

    let mixedContent = plan.sections.join("\n\n");
    let sourceBlockIds = plan.sourceBlockIds;
    let sectionCount = plan.sections.len();
    let sourceCount = sourceBlockIds.len();
    let inputChars = mixedContent.len();

    let userPrompt = format!(
        "<compact_this>\n\
         {mixedContent}\n\
         </compact_this>\n\n\
         Write the handoff briefing wrapped in \
         <compacted_monolithic_string> tags. No preamble."
    );

    tracing::info!(
        sectionCount,
        sourceBlocks = sourceCount,
        inputChars,
        "S4: recompressing older layers"
    );

    let messages = vec![
        Message::System {
            content: DEEP_RECOMPACT_SYSTEM.to_string(),
        },
        Message::User {
            content: userPrompt.into(),
        },
    ];

    let (response, usage) = client
        .complete(ModelTier::Utility, &messages, Some(utilityModel))
        .await
        .context("S4 utility model call failed")?;

    let cost = usage.and_then(|u| u.cost);
    let summary = extractCompactedString(&response);

    // Reduction gate: a "summary" longer than what we fed in is not
    // useful — drop it so the tracker treats S4 as exhausted instead
    // of growing context and re-firing.
    if summary.len() >= inputChars {
        tracing::warn!(
            inputChars,
            outputChars = summary.len(),
            "S4: model output did not reduce — discarding"
        );
        return Ok(S4Result {
            didWork: false,
            summary: String::new(),
            sourceBlockIds: Vec::new(),
            cost,
        });
    }

    tracing::info!(
        outputChars = summary.len(),
        cost = ?cost,
        "S4: briefing produced"
    );

    Ok(S4Result {
        didWork: true,
        summary,
        sourceBlockIds,
        cost,
    })
}

/// A prepared S4 compactor input.
struct RecompactPlan {
    sections: Vec<String>,
    sourceBlockIds: Vec<String>,
}

/// Select active S4/S3/S2 summary inputs for the next S4 pass.
///
/// There may be many historical `FullCompact` entries in the append-only
/// log, but only the newest active one is the current S4 frontier. Feeding
/// older S4 briefings alongside the frontier duplicates already-consumed
/// history and invites summary drift.
///
/// `maxInputChars` caps the total char count of accumulated sections so
/// S4 only compacts the oldest ~30% of effective context. The budget is
/// enforced after the latest S4 briefing (always included as anchor);
/// additional S3 topics and S2 orphans stop being added once the budget
/// is exceeded.
fn buildRecompactPlan(
    ops: &[CompactionOp],
    activeTurns: &[crate::transcript::Turn],
    maxInputChars: usize,
) -> RecompactPlan {
    // Group transcript turns by blockId for raw content rendering.
    let mut turnsByBlock: HashMap<String, Vec<&crate::transcript::Turn>> = HashMap::new();
    for turn in activeTurns {
        turnsByBlock
            .entry(turn.blockId.clone())
            .or_default()
            .push(turn);
    }

    // Find the latest S4 briefing (continuity anchor).
    let latestS4 = ops.iter().rev().find_map(|op| match op {
        CompactionOp::FullCompact {
            summary, sourceIds, ..
        } => Some((summary.as_str(), sourceIds.as_slice())),
        _ => None,
    });

    let s4Covered: HashSet<String> = latestS4
        .map(|(_, ids)| ids.iter().cloned().collect())
        .unwrap_or_default();

    // Collect S3 topics not already in S4, in op order (oldest first).
    struct TopicEntry<'a> {
        label: &'a str,
        summary: &'a str,
        sourceBlockIds: &'a [String],
    }
    let mut eligibleTopics: Vec<TopicEntry> = Vec::new();
    for op in ops {
        if let CompactionOp::TopicCompact {
            topicLabel,
            summary,
            sourceBlockIds,
            ..
        } = op
            && !sourceBlockIds.iter().all(|bid| s4Covered.contains(bid))
        {
            eligibleTopics.push(TopicEntry {
                label: topicLabel.as_str(),
                summary: summary.as_str(),
                sourceBlockIds: sourceBlockIds.as_slice(),
            });
        }
    }

    if eligibleTopics.is_empty() {
        return RecompactPlan {
            sections: Vec::new(),
            sourceBlockIds: Vec::new(),
        };
    }

    // Take the oldest 25% of S3 topics.
    let takeCount = (eligibleTopics.len() as f64 * 0.25).ceil() as usize;
    let batch = &eligibleTopics[..takeCount.min(eligibleTopics.len())];

    let mut sections: Vec<String> = Vec::new();
    let mut allSourceBlockIds: Vec<String> = Vec::new();
    let mut seenBlocks: HashSet<String> = HashSet::new();
    let mut accumulatedChars: usize = 0;

    // Include prior briefing as anchor.
    if let Some((summary, sourceIds)) = latestS4 {
        let section = format!("<prior_briefing>\n{summary}\n</prior_briefing>");
        accumulatedChars += section.len();
        sections.push(section);
        for bid in sourceIds {
            if seenBlocks.insert(bid.clone()) {
                allSourceBlockIds.push(bid.clone());
            }
        }
    }

    // Render each topic. Use raw transcript if it fits, fall back to S3 summary.
    for topic in batch {
        let rawSection = renderTopicRaw(topic.label, topic.sourceBlockIds, &turnsByBlock);
        // Use raw if it has real content and fits; otherwise fall back to S3 summary.
        let rawHasContent = rawSection.lines().count() > 2; // More than just open/close tags.
        let section = if rawHasContent && accumulatedChars + rawSection.len() <= maxInputChars {
            rawSection
        } else {
            // Fall back to S3 summary.
            let blockRange = formatBlockRange(topic.sourceBlockIds);
            format!(
                "<topic_summary label=\"{}\" blocks=\"{blockRange}\">\n\
                 {}\n\
                 </topic_summary>",
                topic.label, topic.summary,
            )
        };

        accumulatedChars += section.len();
        sections.push(section);
        for bid in topic.sourceBlockIds {
            if seenBlocks.insert(bid.clone()) {
                allSourceBlockIds.push(bid.clone());
            }
        }
    }

    RecompactPlan {
        sections,
        sourceBlockIds: allSourceBlockIds,
    }
}

/// Render a topic's blocks as raw transcript content for S4 input.
fn renderTopicRaw(
    label: &str,
    blockIds: &[String],
    turnsByBlock: &HashMap<String, Vec<&crate::transcript::Turn>>,
) -> String {
    use crate::transcript::TurnRole;

    let mut parts = vec![format!("<topic label=\"{label}\">")];
    for bid in blockIds {
        let turns = match turnsByBlock.get(bid) {
            Some(t) => t,
            None => continue,
        };
        parts.push(format!("<exchange id=\"{bid}\">"));
        for turn in turns {
            match turn.role {
                TurnRole::User | TurnRole::Wake => {
                    parts.push(format!("<user_turn>{}</user_turn>", turn.content));
                }
                TurnRole::Assistant => {
                    let body = if let Some(r) = &turn.reasoning {
                        format!("<reasoning>{r}</reasoning>\n{}", turn.content)
                    } else {
                        turn.content.clone()
                    };
                    parts.push(format!("<agent_turn>{body}</agent_turn>"));
                }
                TurnRole::ToolResult => {
                    parts.push(format!("<tool_output>{}</tool_output>", turn.content));
                }
                TurnRole::ToolCall => {
                    let name = turn.tool.as_deref().unwrap_or("unknown");
                    let argStr = turn
                        .args
                        .as_ref()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    parts.push(format!(
                        "<agent_turn>[tool_call: {name}({argStr})]</agent_turn>"
                    ));
                }
                TurnRole::System => {}
            }
        }
        parts.push("</exchange>".to_string());
    }
    parts.push("</topic>".to_string());
    parts.join("\n")
}

/// Does a compaction op belong to the current active branch?
///
/// This mirrors replay filtering in `context.rs`: if `afterTurn` is a
/// real turn id from another branch, the op is from a rewound-away future
/// and must not influence a new S4 run. Unknown `afterTurn` values are
/// treated as legacy block ids and allowed for backward compatibility.
fn opAppliesToActiveBranch(
    op: &CompactionOp,
    activeBlockIds: &HashSet<&str>,
    activeTurnIds: &HashSet<&str>,
    knownTurnIds: &HashSet<&str>,
) -> bool {
    let afterTurn = op.afterTurn();
    if knownTurnIds.contains(afterTurn) && !activeTurnIds.contains(afterTurn) {
        return false;
    }

    match op {
        CompactionOp::FileDedup { .. } | CompactionOp::MiddleOut { .. } => true,
        CompactionOp::BlockCompact { blockId, .. } => activeBlockIds.contains(blockId.as_str()),
        CompactionOp::TopicCompact { sourceBlockIds, .. } => sourceBlockIds
            .iter()
            .all(|id| activeBlockIds.contains(id.as_str())),
        CompactionOp::FullCompact { sourceIds, .. } => sourceIds
            .iter()
            .all(|id| activeBlockIds.contains(id.as_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recompact_plan_uses_only_latest_s4_frontier() {
        let ops = vec![
            CompactionOp::TopicCompact {
                stage: "s3".into(),
                topicLabel: "First Topic".into(),
                summary: "topic one summary".into(),
                sourceBlockIds: vec!["b_a".into(), "b_b".into()],
                afterTurn: "t2".into(),
                ts: 0,
            },
            CompactionOp::FullCompact {
                stage: "s4".into(),
                summary: "OLD_S4".into(),
                sourceIds: vec!["b_a".into(), "b_b".into()],
                afterTurn: "t3".into(),
                ts: 0,
            },
            CompactionOp::TopicCompact {
                stage: "s3".into(),
                topicLabel: "Second Topic".into(),
                summary: "topic two summary".into(),
                sourceBlockIds: vec!["b_c".into()],
                afterTurn: "t4".into(),
                ts: 0,
            },
            CompactionOp::FullCompact {
                stage: "s4".into(),
                summary: "LATEST_S4".into(),
                sourceIds: vec!["b_a".into(), "b_b".into(), "b_c".into()],
                afterTurn: "t5".into(),
                ts: 0,
            },
            CompactionOp::TopicCompact {
                stage: "s3".into(),
                topicLabel: "Fresh Topic".into(),
                summary: "fresh topic summary".into(),
                sourceBlockIds: vec!["b_d".into(), "b_e".into()],
                afterTurn: "t6".into(),
                ts: 0,
            },
            CompactionOp::BlockCompact {
                stage: "s2".into(),
                blockId: "b_f".into(),
                summary: "orphan block summary".into(),
                sourceIds: vec![],
                afterTurn: "t7".into(),
                ts: 0,
            },
        ];

        let plan = buildRecompactPlan(&ops, &[], usize::MAX);
        let input = plan.sections.join("\n\n");

        assert!(
            !input.contains("OLD_S4"),
            "earlier S4 briefings are historical once a later S4 consumed them"
        );
        assert_eq!(
            input.matches("LATEST_S4").count(),
            1,
            "the active S4 frontier should be fed exactly once"
        );
        // Fresh Topic is the only S3 topic not covered by S4. With 1
        // eligible topic, 25% rounds up to 1 → it should be included.
        assert!(
            input.contains("fresh topic summary"),
            "uncovered S3 topics should be included"
        );
        // S2 orphans are no longer included — S4 only merges S3 topics.
        assert!(
            !input.contains("orphan block summary"),
            "S4 no longer consumes standalone S2 blocks"
        );
        assert!(
            !input.contains("topic one summary") && !input.contains("topic two summary"),
            "summaries already covered by the latest S4 must not be fed again"
        );
        // Source blocks: S4 frontier (a,b,c) + Fresh Topic (d,e).
        assert_eq!(
            plan.sourceBlockIds,
            vec![
                "b_a".to_string(),
                "b_b".to_string(),
                "b_c".to_string(),
                "b_d".to_string(),
                "b_e".to_string(),
            ]
        );
    }
}

/// Extract content from `<compacted_monolithic_string>` tags.
fn extractCompactedString(response: &str) -> String {
    if let Some(start) = response.find("<compacted_monolithic_string>") {
        let after = &response[start + "<compacted_monolithic_string>".len()..];
        if let Some(end) = after.find("</compacted_monolithic_string>") {
            return after[..end].trim().to_string();
        }
    }
    // Fallback: use the entire response.
    response.trim().to_string()
}

/// Format block IDs into a readable range string.
fn formatBlockRange(blockIds: &[String]) -> String {
    if blockIds.is_empty() {
        return String::new();
    }
    if blockIds.len() == 1 {
        return blockIds[0].clone();
    }
    format!("{}\u{2013}{}", blockIds[0], blockIds[blockIds.len() - 1])
}

const DEEP_RECOMPACT_SYSTEM: &str = "\
You compress an entire conversation into a handoff briefing. The next \
agent taking over this session will have ONLY your output as context \
(plus tools to search the full transcript if needed).\n\
\n\
The conversation may contain a mix of raw exchanges and previously \
compressed sections. Treat both as equally authoritative.\n\
\n\
Voice and perspective:\n\
- Dispassionate record of facts. The agent is labor, not a \
protagonist.\n\
- State what exists, what the user said, what was decided. Do not \
narrate the agent's process or achievements.\n\
- Quote short user messages verbatim. User words are the primary \
source of truth.\n\
- Only attribute decisions to the user if the user explicitly stated \
them. Agent proposals that went uncontested are proposals, not user \
decisions.\n\
- Distinguish agent proposals from agent commitments. A commitment \
is something the next agent must follow through on.\n\
\n\
Structure (use these exact headers):\n\
\n\
CURRENT TASK\n\
What is being worked on right now. If interrupted mid-task, describe \
exactly where it left off \u{2014} file being edited, test being run, error \
being debugged. The next agent must resume without asking the user to \
repeat themselves. This is the most critical section.\n\
\n\
USER INTENT\n\
Every distinct request the user made, in chronological order. Include \
corrections, preferences, and redirections. Quote verbatim where \
short. The user should never have to re-state something.\n\
\n\
ACCOMPLISHED\n\
What exists now as a result of this session. File paths, function \
names, configuration values. State as facts, not achievements.\n\
\n\
ERRORS AND FIXES\n\
Every error encountered and how it was resolved. Include actual error \
messages. The next agent must not repeat solved mistakes.\n\
\n\
DECISIONS\n\
Choices made when alternatives existed, with reasoning if discussed. \
Mark whether the user confirmed or the agent proposed uncontested. \
The next agent should not revisit settled decisions without the user \
asking.\n\
\n\
ENVIRONMENT\n\
Project structure, dependencies, tooling, or context discovered \
during the session. Only include what is not obvious from reading \
project files.\n\
\n\
UNRESOLVED\n\
Anything unfinished, blocked, or noted as future work. Distinguish \
\"user asked but not started\" from \"started but hit a blocker\" from \
\"agent committed but didn't deliver.\"\n\
\n\
Rules:\n\
- Be specific. File paths, function names, error messages, exact \
values. Vague summaries are useless.\n\
- Preserve all user messages in spirit if not verbatim.\n\
- Do not invent next steps the user didn't ask for.\n\
- Do not editorialize about code quality or suggest improvements \
unless the user explicitly requested them.\n\
- Your output MUST be wrapped in <compacted_monolithic_string> tags.";
