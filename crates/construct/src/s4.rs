#![allow(clippy::items_after_test_module)]

//! S4 — deep recompaction (single LLM call).
//!
//! Last-resort recompression of the OLDER compacted layers when S1–S3
//! are exhausted. Despite the historical "full compaction" name, S4
//! does NOT touch the most recent context: a protected band of the
//! newest [`DEFAULT_PROTECTED_RATIO`] characters of effective content
//! is held verbatim, so the model always has a high-fidelity working
//! set even after repeated S4 firings.
//!
//! Sources fed into S4:
//! - The latest active S4 briefing, if one exists. Older S4 ops are
//!   historical log entries already consumed by that frontier.
//! - S3 topic summaries whose sourceBlocks are entirely outside the
//!   protected band and not already covered by a later S4.
//! - Orphan S2 block summaries whose blocks were never lifted into an
//!   S3 or S4 op AND fall outside the protected band — these are
//!   typically single-block topics or fragments S3 wouldn't touch.
//!
//! Reads the post-compaction summaries from the log directly; the
//! transcript is consulted only to compute the protected band.
//!
//! # Public API
//! - [`run`] — execute S4 compaction
//! - [`S4Result`] — what was produced
//! - [`DEFAULT_PROTECTED_RATIO`] — fraction of effective context held verbatim
//!
//! # Dependencies
//! `crate::api`, `crate::compaction`, `crate::message`, `crate::transcript`

use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::api;
use crate::compaction::{CompactionLog, CompactionOp};
use crate::message::Message;
use crate::transcript::Transcript;

/// Fraction of effective context (newest end) that S4 must NEVER
/// touch. Holds the model's working set in full fidelity even after
/// repeated S4 firings.
pub const DEFAULT_PROTECTED_RATIO: f64 = 0.30;

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

    let protectedBand = computeProtectedBand(&activeTurns, &activeOps, DEFAULT_PROTECTED_RATIO);

    let plan = buildRecompactPlan(&activeOps, &protectedBand);

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
        protectedBlocks = protectedBand.len(),
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
        .complete(&messages, Some(utilityModel))
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
fn buildRecompactPlan(ops: &[CompactionOp], protectedBand: &HashSet<String>) -> RecompactPlan {
    let latestS4 = ops.iter().rev().find_map(|op| match op {
        CompactionOp::FullCompact {
            summary, sourceIds, ..
        } => {
            if sourceIds.iter().any(|bid| protectedBand.contains(bid)) {
                None
            } else {
                Some((summary.as_str(), sourceIds.as_slice()))
            }
        }
        _ => None,
    });

    // Blocks absorbed by the active S4 frontier. Earlier S4 entries and
    // the S2/S3 summaries that fed them are dead inputs.
    let s4Covered: HashSet<String> = latestS4
        .map(|(_, sourceIds)| sourceIds.iter().cloned().collect())
        .unwrap_or_default();

    // Blocks already absorbed by S3.
    let s3Covered: HashSet<String> = ops
        .iter()
        .flat_map(|op| match op {
            CompactionOp::TopicCompact { sourceBlockIds, .. } => sourceBlockIds.clone(),
            _ => Vec::new(),
        })
        .collect();

    let mut sections: Vec<String> = Vec::new();
    let mut allSourceBlockIds: Vec<String> = Vec::new();
    let mut seenBlocks: HashSet<String> = HashSet::new();

    if let Some((summary, sourceIds)) = latestS4 {
        sections.push(format!(
            "<prior_briefing>\n\
             {summary}\n\
             </prior_briefing>"
        ));
        for bid in sourceIds {
            if seenBlocks.insert(bid.clone()) {
                allSourceBlockIds.push(bid.clone());
            }
        }
    }

    for op in ops {
        match op {
            CompactionOp::TopicCompact {
                topicLabel,
                summary,
                sourceBlockIds,
                ..
            } => {
                if sourceBlockIds.iter().any(|bid| protectedBand.contains(bid)) {
                    continue;
                }
                if sourceBlockIds.iter().all(|bid| s4Covered.contains(bid)) {
                    // Already folded into a prior S4 briefing; that
                    // briefing carries the content forward.
                    continue;
                }
                let blockRange = formatBlockRange(sourceBlockIds);
                sections.push(format!(
                    "<topic_summary label=\"{topicLabel}\" blocks=\"{blockRange}\">\n\
                     {summary}\n\
                     </topic_summary>"
                ));
                for bid in sourceBlockIds {
                    if seenBlocks.insert(bid.clone()) {
                        allSourceBlockIds.push(bid.clone());
                    }
                }
            }
            CompactionOp::BlockCompact {
                blockId, summary, ..
            } => {
                if protectedBand.contains(blockId) {
                    continue;
                }
                if s3Covered.contains(blockId) || s4Covered.contains(blockId) {
                    // Already absorbed by S3 or S4 — including it
                    // again would duplicate content.
                    continue;
                }
                sections.push(format!(
                    "<block_summary block=\"{blockId}\">\n\
                     {summary}\n\
                     </block_summary>"
                ));
                if seenBlocks.insert(blockId.clone()) {
                    allSourceBlockIds.push(blockId.clone());
                }
            }
            CompactionOp::FullCompact { .. } => {
                // Only the latest active S4 frontier is carried above.
                // Earlier S4 entries are historical and already consumed
                // by that frontier.
            }
            _ => {}
        }
    }

    RecompactPlan {
        sections,
        sourceBlockIds: allSourceBlockIds,
    }
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

/// Compute the set of block IDs in the protected recent band.
///
/// Walks the active branch from newest to oldest, accumulating each
/// block's *effective* size (its S2 summary length when one exists,
/// otherwise the raw char count). The newest blocks summing to
/// `protectRatio` of total effective chars form the protected band.
/// S2 blocks that have been superseded by S3 or S4 are excluded — they
/// no longer contribute to live context.
fn computeProtectedBand(
    activeTurns: &[crate::transcript::Turn],
    ops: &[CompactionOp],
    protectRatio: f64,
) -> HashSet<String> {
    let compactedSizes = crate::compaction::compactedBlockSizes(ops);
    let superseded = crate::compaction::supersededBlocks(ops);

    let mut blocks: Vec<(String, usize)> = Vec::new();
    let mut currentBlockId = String::new();
    let mut currentRawSize: usize = 0;
    for turn in activeTurns {
        if turn.blockId != currentBlockId {
            if !currentBlockId.is_empty() && !superseded.contains(&currentBlockId) {
                let effective = compactedSizes
                    .get(&currentBlockId)
                    .copied()
                    .unwrap_or(currentRawSize);
                blocks.push((currentBlockId.clone(), effective));
            }
            currentBlockId = turn.blockId.clone();
            currentRawSize = 0;
        }
        currentRawSize += turn.content.len();
    }
    if !currentBlockId.is_empty() && !superseded.contains(&currentBlockId) {
        let effective = compactedSizes
            .get(&currentBlockId)
            .copied()
            .unwrap_or(currentRawSize);
        blocks.push((currentBlockId, effective));
    }

    let total: usize = blocks.iter().map(|(_, s)| s).sum();
    if total == 0 {
        return HashSet::new();
    }
    let target = (total as f64 * protectRatio) as usize;

    let mut band = HashSet::new();
    let mut cumulative: usize = 0;
    for (blockId, size) in blocks.iter().rev() {
        if cumulative >= target {
            break;
        }
        band.insert(blockId.clone());
        cumulative += size;
    }
    band
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{Turn, TurnRole, TurnStatus};

    fn makeTurn(
        id: &str,
        blockId: &str,
        role: TurnRole,
        content: &str,
        parentId: Option<&str>,
    ) -> Turn {
        Turn {
            id: id.to_string(),
            blockId: blockId.to_string(),
            topicId: String::new(),
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
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
        }
    }

    /// The newest 30% of effective chars must end up in the protected
    /// band. Blocks already absorbed by S3/S4 (`superseded`) are
    /// invisible to the calculation — they aren't in live context.
    #[test]
    fn protected_band_holds_newest_thirty_percent() {
        let turns = vec![
            makeTurn("t1", "b_aaa", TurnRole::User, &"x".repeat(100), None),
            makeTurn("t2", "b_bbb", TurnRole::User, &"x".repeat(100), Some("t1")),
            makeTurn("t3", "b_ccc", TurnRole::User, &"x".repeat(100), Some("t2")),
            makeTurn("t4", "b_ddd", TurnRole::User, &"x".repeat(100), Some("t3")),
            makeTurn("t5", "b_eee", TurnRole::User, &"x".repeat(100), Some("t4")),
            makeTurn("t6", "b_fff", TurnRole::User, &"x".repeat(100), Some("t5")),
            makeTurn("t7", "b_ggg", TurnRole::User, &"x".repeat(100), Some("t6")),
            makeTurn("t8", "b_hhh", TurnRole::User, &"x".repeat(100), Some("t7")),
            makeTurn("t9", "b_iii", TurnRole::User, &"x".repeat(100), Some("t8")),
            makeTurn("t10", "b_jjj", TurnRole::User, &"x".repeat(100), Some("t9")),
        ];
        // 10 blocks × 100 chars = 1000 effective. 30% = 300, so the
        // newest 3 blocks form the band.
        let band = computeProtectedBand(&turns, &[], 0.30);
        assert!(band.contains("b_jjj"), "newest block must be protected");
        assert!(band.contains("b_iii"));
        assert!(band.contains("b_hhh"));
        assert!(
            !band.contains("b_ggg"),
            "older blocks must NOT be protected"
        );
        assert!(!band.contains("b_aaa"));
    }

    /// Blocks superseded by an S3 op are removed from live context, so
    /// they shouldn't consume protected-band budget. The band should
    /// expand backward to cover more of the still-live tail.
    #[test]
    fn protected_band_skips_superseded_blocks() {
        let turns = vec![
            makeTurn("t1", "b_old1", TurnRole::User, &"x".repeat(100), None),
            makeTurn("t2", "b_old2", TurnRole::User, &"x".repeat(100), Some("t1")),
            makeTurn("t3", "b_old3", TurnRole::User, &"x".repeat(100), Some("t2")),
            makeTurn("t4", "b_new1", TurnRole::User, &"x".repeat(100), Some("t3")),
            makeTurn("t5", "b_new2", TurnRole::User, &"x".repeat(100), Some("t4")),
        ];
        // Old three blocks are merged into one topic summary.
        let ops = vec![CompactionOp::TopicCompact {
            stage: "s3".into(),
            topicLabel: "Merged".into(),
            summary: "x".repeat(30),
            sourceBlockIds: vec!["b_old1".into(), "b_old2".into(), "b_old3".into()],
            afterTurn: "t5".into(),
            ts: 0,
        }];
        // Live: just b_new1 (100) + b_new2 (100) = 200 effective.
        // 30% = 60 → newest block (b_new2) only.
        let band = computeProtectedBand(&turns, &ops, 0.30);
        assert!(band.contains("b_new2"));
        assert!(!band.contains("b_old1"), "superseded blocks not in band");
        assert!(!band.contains("b_old2"));
        assert!(!band.contains("b_old3"));
    }

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

        let plan = buildRecompactPlan(&ops, &HashSet::new());
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
        assert!(input.contains("fresh topic summary"));
        assert!(input.contains("orphan block summary"));
        assert!(
            !input.contains("topic one summary") && !input.contains("topic two summary"),
            "summaries already covered by the latest S4 must not be fed again"
        );
        assert_eq!(
            plan.sourceBlockIds,
            vec![
                "b_a".to_string(),
                "b_b".to_string(),
                "b_c".to_string(),
                "b_d".to_string(),
                "b_e".to_string(),
                "b_f".to_string(),
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
