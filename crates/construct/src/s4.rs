//! S4 — full conversation compaction (single LLM call).
//!
//! Fires at 90% of compactLimit when S3 is exhausted. Merges all
//! existing S3 topic summaries and any prior S4 briefings into a
//! single structured handoff briefing.
//!
//! Unlike S2/S3, S4 reads **post-compaction** content (the summaries
//! themselves), not originals from the transcript. It's the one layer
//! allowed to summarize summaries.
//!
//! # Public API
//! - [`run`] — execute S4 compaction
//! - [`S4Result`] — what was produced
//!
//! # Dependencies
//! `crate::api`, `crate::compaction`, `crate::message`

use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::api;
use crate::compaction::{CompactionLog, CompactionOp};
use crate::message::Message;

pub struct S4Result {
    pub didWork: bool,
    pub summary: String,
    pub sourceBlockIds: Vec<String>,
    /// USD cost of the utility model call (None if not reported).
    pub cost: Option<f64>,
}

/// Merge S3 topic summaries and existing S4 briefings into a single
/// handoff briefing.
///
/// Reads summaries from the compaction log (not the transcript).
/// Produces a single structured briefing via one utility model call.
pub async fn run(
    compactionLog: &CompactionLog,
    client: &api::Client,
    utilityModel: &str,
) -> Result<S4Result> {
    let ops = compactionLog.loadAll()?;

    // Collect S3 topic summaries and existing S4 briefings.
    let mut sections: Vec<String> = Vec::new();
    let mut allSourceBlockIds: Vec<String> = Vec::new();
    let mut seenBlocks: HashSet<String> = HashSet::new();

    for op in &ops {
        match op {
            CompactionOp::TopicCompact {
                topicLabel,
                summary,
                sourceBlockIds,
                ..
            } => {
                // Skip if this topic's blocks are already covered by a later S4.
                if sourceBlockIds.iter().any(|bid| seenBlocks.contains(bid)) {
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
            CompactionOp::FullCompact {
                summary,
                sourceIds,
                ..
            } => {
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
            _ => {}
        }
    }

    if sections.is_empty() {
        return Ok(S4Result {
            didWork: false,
            summary: String::new(),
            sourceBlockIds: Vec::new(),
            cost: None,
        });
    }

    let mixedContent = sections.join("\n\n");
    let userPrompt = format!(
        "<compact_this>\n\
         {mixedContent}\n\
         </compact_this>\n\n\
         Write the handoff briefing wrapped in \
         <compacted_monolithic_string> tags. No preamble."
    );

    tracing::info!(
        sectionCount = sections.len(),
        sourceBlocks = allSourceBlockIds.len(),
        inputChars = mixedContent.len(),
        "S4: compacting topic summaries into briefing"
    );

    let messages = vec![
        Message::System {
            content: FULL_COMPACT_SYSTEM.to_string(),
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

    tracing::info!(
        outputChars = summary.len(),
        cost = ?cost,
        "S4: briefing produced"
    );

    Ok(S4Result {
        didWork: true,
        summary,
        sourceBlockIds: allSourceBlockIds,
        cost,
    })
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

const FULL_COMPACT_SYSTEM: &str = "\
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
