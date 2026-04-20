//! S2 — per-block LLM compaction.
//!
//! Fires at 80% of compactLimit (after S1 is exhausted). For each exchange
//! block in the oldest 60% of context that hasn't been compacted yet:
//!
//! 1. Load the **original** block content from the transcript
//! 2. Build an XML-structured prompt with preceding/following user messages
//! 3. Send to the utility model for compression (not summarization)
//! 4. Record the result for the compaction log
//!
//! User messages stay verbatim — only assistant/tool content is compressed.
//! Calls are parallelizable (each block is independent).
//!
//! The compressed output is displayed as a User message wrapped in
//! `<compressed_content>` tags, with `<referenced_turns>` pointing the
//! model at `historyFetch` for the original content.
//!
//! # Public API
//! - [`run`] — execute S2 block compaction
//! - [`S2Result`] / [`CompactedBlock`]
//!
//! # Dependencies
//! `crate::api`, `crate::compaction`, `crate::transcript`, `futures`

use anyhow::Result;
use futures::future::join_all;

use crate::api;
use crate::compaction::CompactionLog;
use crate::transcript::{Transcript, Turn, TurnRole};

/// A single block that was compacted.
pub struct CompactedBlock {
    pub blockId: String,
    /// Raw compressed text (no XML wrapper — that's added during assembly).
    pub summary: String,
    /// Turn IDs from the transcript that were in this block.
    pub sourceIds: Vec<String>,
    /// File paths read in this block (for edit gate invalidation).
    pub invalidatedFiles: Vec<String>,
}

pub struct S2Result {
    pub didWork: bool,
    pub compacted: Vec<CompactedBlock>,
    /// Total USD cost of all utility model calls in this S2 pass.
    pub cost: Option<f64>,
}

/// Run S2 block-level compaction.
///
/// Works entirely from transcript + compaction log — does not touch the
/// live history. The caller records results in the compaction log and
/// reconstructs the live history afterward.
pub async fn run(
    transcript: &Transcript,
    compactionLog: &CompactionLog,
    headTurnId: &str,
    client: &api::Client,
    utilityModel: &str,
    _contextWindow: usize,
    _compactRatio: f64,
) -> Result<S2Result> {
    let allTurns = transcript.loadAll()?;
    if allTurns.is_empty() {
        return Ok(S2Result { didWork: false, compacted: Vec::new(), cost: None });
    }

    // Walk the active branch only — dead branches from rewinds
    // must not consume the char budget.
    let turns = crate::transcript::walkBranchTurns(&allTurns, headTurnId);
    if turns.is_empty() {
        return Ok(S2Result { didWork: false, compacted: Vec::new(), cost: None });
    }

    // Build compacted sizes map and zone from shared infrastructure.
    let ops = compactionLog.loadAll()?;
    let compactedSizes = crate::compaction::compactedBlockSizes(&ops);
    let superseded = crate::compaction::supersededBlocks(&ops);
    let zone = crate::compaction::zoneBlocks(&turns, &compactedSizes, &superseded, 0.60);

    // Group turns by blockId, preserving order.
    let blocks = groupByBlock(&turns);
    if blocks.is_empty() {
        return Ok(S2Result { didWork: false, compacted: Vec::new(), cost: None });
    }

    // Filter to eligible blocks: in zone, not already compacted, has agent content.
    let mut eligible: Vec<usize> = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        if !zone.contains(&block.blockId) {
            continue;
        }
        if compactedSizes.contains_key(&block.blockId) {
            continue;
        }
        if block.agentTurns.is_empty() {
            continue;
        }
        eligible.push(i);
    }

    if eligible.is_empty() {
        return Ok(S2Result { didWork: false, compacted: Vec::new(), cost: None });
    }

    // Build and fire parallel compaction calls.
    let futures: Vec<_> = eligible
        .iter()
        .map(|&i| {
            let block = &blocks[i];
            let followupUser = blocks.get(i + 1).map(|b| b.userMessage.as_str());
            compactBlock(block, followupUser, client, utilityModel)
        })
        .collect();

    let results = join_all(futures).await;

    let mut compacted = Vec::new();
    let mut totalCost: f64 = 0.0;
    for (resultIdx, result) in results.into_iter().enumerate() {
        let blockIdx = eligible[resultIdx];
        let block = &blocks[blockIdx];

        match result {
            Ok((summary, blockCost)) => {
                if let Some(c) = blockCost {
                    totalCost += c;
                }
                compacted.push(CompactedBlock {
                    blockId: block.blockId.clone(),
                    summary,
                    sourceIds: block.turnIds(),
                    invalidatedFiles: block.readFilePaths(),
                });
            }
            Err(e) => {
                tracing::warn!(
                    blockId = %block.blockId,
                    error = %e,
                    "S2 block compaction failed, skipping"
                );
            }
        }
    }

    let didWork = !compacted.is_empty();
    let cost = if totalCost > 0.0 { Some(totalCost) } else { None };
    Ok(S2Result { didWork, compacted, cost })
}

/// A block of turns grouped by blockId.
struct Block {
    blockId: String,
    userMessage: String,
    /// Assistant and tool result turns (the content being compressed).
    agentTurns: Vec<AgentTurn>,
    /// All turn IDs in this block (user + agent + tool).
    allTurnIds: Vec<String>,
    charCount: usize,
}

struct AgentTurn {
    role: TurnRole,
    content: String,
    /// For ToolCall turns: the tool name.
    toolName: Option<String>,
    /// For ToolCall turns: the tool arguments as JSON.
    toolArgs: Option<serde_json::Value>,
    /// Assistant reasoning/thinking from this turn.
    reasoning: Option<String>,
}

impl Block {
    fn turnIds(&self) -> Vec<String> {
        self.allTurnIds.clone()
    }

    /// Extract file paths from readFile tool calls in this block.
    fn readFilePaths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        for t in &self.agentTurns {
            if t.toolName.as_deref() == Some("readFile") {
                if let Some(args) = &t.toolArgs {
                    if let Some(path) = args["path"].as_str() {
                        let norm = normalizePath(path);
                        if !paths.contains(&norm) {
                            paths.push(norm);
                        }
                    }
                }
            }
        }
        paths
    }
}

/// Group transcript turns into blocks by blockId.
fn groupByBlock(turns: &[Turn]) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut currentBlockId = String::new();

    for turn in turns {
        // Start a new block when blockId changes.
        if turn.blockId != currentBlockId {
            currentBlockId = turn.blockId.clone();
            blocks.push(Block {
                blockId: turn.blockId.clone(),
                userMessage: String::new(),
                agentTurns: Vec::new(),
                allTurnIds: Vec::new(),
                charCount: 0,
            });
        }

        let block = blocks.last_mut().unwrap();
        block.allTurnIds.push(turn.id.clone());
        block.charCount += turn.content.len();

        match turn.role {
            TurnRole::User => {
                block.userMessage = turn.content.clone();
            }
            TurnRole::System => {
                // System turns don't contribute to agent content.
            }
            TurnRole::Assistant => {
                block.agentTurns.push(AgentTurn {
                    role: TurnRole::Assistant,
                    content: turn.content.clone(),
                    toolName: None,
                    toolArgs: None,
                    reasoning: turn.reasoning.clone(),
                });
            }
            TurnRole::ToolCall => {
                block.agentTurns.push(AgentTurn {
                    role: TurnRole::ToolCall,
                    content: turn.content.clone(),
                    toolName: turn.tool.clone(),
                    toolArgs: turn.args.clone(),
                    reasoning: None,
                });
            }
            TurnRole::ToolResult => {
                block.agentTurns.push(AgentTurn {
                    role: TurnRole::ToolResult,
                    content: turn.content.clone(),
                    toolName: None,
                    toolArgs: None,
                    reasoning: None,
                });
            }
        }
    }

    blocks
}

/// Build the user prompt and call the utility model for one block.
async fn compactBlock(
    block: &Block,
    followupUser: Option<&str>,
    client: &api::Client,
    utilityModel: &str,
) -> Result<(String, Option<f64>)> {
    // Build the <compact_this> content from agent/tool turns.
    let mut compactParts: Vec<String> = Vec::new();
    for turn in &block.agentTurns {
        match turn.role {
            TurnRole::Assistant => {
                let body = if let Some(r) = &turn.reasoning {
                    format!("<reasoning>{r}</reasoning>\n{}", turn.content)
                } else {
                    turn.content.clone()
                };
                compactParts.push(format!("<agent_turn>{body}</agent_turn>"));
            }
            TurnRole::ToolResult => {
                compactParts.push(format!("<tool_output>{}</tool_output>", turn.content));
            }
            TurnRole::ToolCall => {
                // ToolCall turns have the tool name + args — format as agent action.
                let name = turn.toolName.as_deref().unwrap_or("unknown");
                let argStr = turn.toolArgs.as_ref()
                    .map(|v| {
                        let s = v.to_string();
                        if s.len() > 500 { format!("{}...", &s[..s.floor_char_boundary(500)]) } else { s }
                    })
                    .unwrap_or_else(|| "{}".to_string());
                compactParts.push(format!(
                    "<agent_turn>[tool_call: {name}({argStr})]</agent_turn>"
                ));
            }
            _ => {}
        }
    }
    let compactContent = compactParts.join("\n");

    if compactContent.trim().is_empty() {
        anyhow::bail!("no agent content to compact in block {}", block.blockId);
    }

    // Build the full user prompt.
    let mut parts = vec![
        format!(
            "<preceding_user_turn>{}</preceding_user_turn>\n",
            block.userMessage
        ),
        "<compact_this>".to_string(),
        compactContent,
        "</compact_this>\n".to_string(),
    ];
    if let Some(followup) = followupUser {
        parts.push(format!(
            "<followup_user_turn>{followup}</followup_user_turn>"
        ));
    }
    let userPrompt = parts.join("\n");

    let messages = vec![
        crate::message::Message::System {
            content: BLOCK_COMPACT_SYSTEM.to_string(),
        },
        crate::message::Message::User {
            content: userPrompt.into(),
        },
    ];

    let (response, usage) = client.complete(&messages, Some(utilityModel)).await?;

    // Extract from <compacted_monolithic_string> tags if present.
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

const BLOCK_COMPACT_SYSTEM: &str = "\
You compress agent work blocks. This is compression, not summarization \
\u{2014} preserve ALL information but strip structural padding.\n\
\n\
You will receive a block wrapped in <compact_this> containing \
<agent_turn> and <tool_output> tags. Compress ONLY this content. The \
<preceding_user_turn> and <followup_user_turn> are context \u{2014} do not \
restate or reference them in your output.\n\
\n\
Compression technique:\n\
- Strip: hedging, filler (\"Let me check...\", \"I'll now...\", \
\"Based on this...\"), pleasantries, narrative transitions\n\
- Keep verbatim: file paths, function/variable names, error messages, \
config values, command lines, type signatures \u{2014} anything a grep or \
search would match on\n\
- Keep: every action taken, every outcome, every decision and its \
reason\n\
- Style: concise past tense. Short clear sentences, not fragments. \
Readable but tight \u{2014} no filler, no fluff, still flows.\n\
\n\
Your output MUST be wrapped in <compacted_monolithic_string> tags. \
Your output MUST be shorter than the original <compact_this> content. \
If you cannot make it shorter, you are not compressing hard enough.\n\
\n\
<example>\n\
<preceding_user_turn>the auth endpoint lets expired tokens through, \
can you fix that?</preceding_user_turn>\n\
\n\
<compact_this>\n\
<agent_turn>Let me check the auth module. I'll read the file to \
understand the current implementation.</agent_turn>\n\
<tool_output>[Read src/auth.rs] \u{2014} file contents showing validateToken() \
on line 47 with `if token.expiry < now` comparison</tool_output>\n\
<agent_turn>After reading auth.rs, I can see that the token validation \
function validateToken() on line 47 is using a less-than comparison for \
the expiry check, which means tokens are accepted right up until they \
expire but not after. Actually, the logic is inverted \u{2014} it should reject \
when expiry is less than now. I'll fix this by changing the comparison \
operator. I've updated the file and the tests are now passing.\
</agent_turn>\n\
</compact_this>\n\
\n\
<followup_user_turn>nice, works now</followup_user_turn>\n\
\n\
Output:\n\
<compacted_monolithic_string>Read src/auth.rs. Found validateToken() at \
line 47 using `if token.expiry < now` \u{2014} logic was inverted, accepting \
expired tokens. Changed comparison operator. Tests pass.\
</compacted_monolithic_string>\n\
</example>\n\
\n\
<example>\n\
<preceding_user_turn>how is the config system set up?\
</preceding_user_turn>\n\
\n\
<compact_this>\n\
<agent_turn>I'll look at the config module to understand the setup.\
</agent_turn>\n\
<tool_output>[Glob **/*.toml] \u{2014} found config/default.toml, \
config/schema.toml, pyproject.toml</tool_output>\n\
<agent_turn>Let me read the main config files to understand the \
structure.</agent_turn>\n\
<tool_output>[Read config/default.toml] \u{2014} 45 lines, sections: \
[server] host/port/workers, [database] url/pool_size/timeout, \
[logging] level/format/file</tool_output>\n\
<tool_output>[Read config/schema.toml] \u{2014} validation schema defining \
types and constraints for each config key, uses custom \
SchemaValidator from src/config/validate.rs</tool_output>\n\
<agent_turn>The config system has three layers. The default.toml \
provides baseline values for server, database, and logging settings. \
The schema.toml defines validation rules that are enforced by \
SchemaValidator in src/config/validate.rs. At runtime, environment \
variables override file values using a CONFIG_ prefix pattern. The \
whole thing is loaded in src/config/mod.rs via the loadConfig() \
function which merges all three sources.</agent_turn>\n\
</compact_this>\n\
\n\
<followup_user_turn>ok can we add a new config section for rate \
limiting?</followup_user_turn>\n\
\n\
Output:\n\
<compacted_monolithic_string>Found config files: config/default.toml, \
config/schema.toml. Config has three layers: default.toml provides \
baseline values for [server], [database], [logging] sections. \
schema.toml defines validation via SchemaValidator in \
src/config/validate.rs. Env vars override with CONFIG_ prefix. All \
merged in src/config/mod.rs via loadConfig().\
</compacted_monolithic_string>\n\
</example>\n\
\n\
<example>\n\
<preceding_user_turn>run the test suite</preceding_user_turn>\n\
\n\
<compact_this>\n\
<agent_turn>I'll run the tests now.</agent_turn>\n\
<tool_output>[Bash: cargo test] \u{2014} 14 passed, 2 failed. Failures:\n\
test_parse_nested: assertion failed at tests/parser.rs:89, expected \
Node::List but got Node::Scalar\n\
test_roundtrip_unicode: panicked at src/serialize.rs:203, \
called unwrap() on Err(Utf8Error)</tool_output>\n\
<agent_turn>Two tests are failing. Let me look at the first one \u{2014} \
test_parse_nested seems like a logic issue in the parser. I'll check \
the parser code.</agent_turn>\n\
<tool_output>[Read src/parser.rs lines 80-120] \u{2014} parseValue() function \
dispatches on first token, bracket case falls through to scalar \
handler</tool_output>\n\
<agent_turn>Found it \u{2014} parseValue() in src/parser.rs has a match \
statement where the `[` case falls through to the scalar handler \
instead of calling parseList(). I'll fix that and also look at the \
unicode issue.</agent_turn>\n\
<tool_output>[Edit src/parser.rs line 94] \u{2014} added `Token::LBracket \
=> self.parseList()` arm</tool_output>\n\
<tool_output>[Read src/serialize.rs lines 195-210] \u{2014} writeValue() \
calls str::from_utf8().unwrap() on raw bytes</tool_output>\n\
<agent_turn>The unicode test failure is in writeValue() at \
src/serialize.rs:203 \u{2014} it unwraps a from_utf8 call on potentially \
invalid bytes. I'll change it to use from_utf8_lossy instead.\
</agent_turn>\n\
<tool_output>[Edit src/serialize.rs line 203] \u{2014} changed unwrap() to \
from_utf8_lossy()</tool_output>\n\
<tool_output>[Bash: cargo test] \u{2014} 16 passed, 0 failed</tool_output>\n\
<agent_turn>All 16 tests pass now. Both issues are fixed.</agent_turn>\n\
</compact_this>\n\
\n\
<followup_user_turn>great</followup_user_turn>\n\
\n\
Output:\n\
<compacted_monolithic_string>Ran cargo test: 14 passed, 2 failed. \
test_parse_nested failed at tests/parser.rs:89 \u{2014} parseValue() in \
src/parser.rs had `[` case falling through to scalar handler instead \
of calling parseList(). Added `Token::LBracket => self.parseList()` \
arm at line 94. test_roundtrip_unicode panicked at \
src/serialize.rs:203 \u{2014} writeValue() called \
str::from_utf8().unwrap() on raw bytes. Changed to \
from_utf8_lossy(). Reran tests: 16 passed, 0 failed.\
</compacted_monolithic_string>\n\
</example>\n\
\n\
<example>\n\
<preceding_user_turn>add a --verbose flag to the CLI\
</preceding_user_turn>\n\
\n\
<compact_this>\n\
<agent_turn>I'll add a verbose flag to the CLI. Let me first check \
how the CLI is currently structured.</agent_turn>\n\
<tool_output>[Read src/cli.rs] \u{2014} uses clap derive macros, main \
struct is CliArgs with subcommands Run, Build, Clean</tool_output>\n\
<agent_turn>The CLI uses clap derive in src/cli.rs. I'll add a \
global --verbose flag to the CliArgs struct and wire it through to \
the logging setup.</agent_turn>\n\
<tool_output>[Edit src/cli.rs] \u{2014} added `#[arg(long, short, global = \
true)] verbose: bool` to CliArgs</tool_output>\n\
<tool_output>[Edit src/main.rs] \u{2014} passed args.verbose to \
initLogging()</tool_output>\n\
<tool_output>[Edit src/logging.rs] \u{2014} initLogging() now accepts \
verbose: bool, sets LevelFilter::Debug when true</tool_output>\n\
</compact_this>\n\
\n\
<followup_user_turn>actually make it -v not --verbose\
</followup_user_turn>\n\
\n\
Output:\n\
<compacted_monolithic_string>Read src/cli.rs \u{2014} uses clap derive \
with CliArgs struct and Run/Build/Clean subcommands. Added \
`#[arg(long, short, global = true)] verbose: bool` to CliArgs in \
src/cli.rs. Wired args.verbose through src/main.rs to \
initLogging() in src/logging.rs, which sets LevelFilter::Debug when \
true. Note: user wants -v shorthand, not --verbose.\
</compacted_monolithic_string>\n\
</example>";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::TurnStatus;
    use std::collections::HashMap;

    fn makeTurn(id: &str, blockId: &str, role: TurnRole, content: &str, parentId: Option<&str>) -> Turn {
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

    #[test]
    fn groupByBlock_groups_turns_by_blockId() {
        let turns = vec![
            makeTurn("t1", "b_aaa", TurnRole::User, "hello", None),
            makeTurn("t2", "b_aaa", TurnRole::Assistant, "hi there", Some("t1")),
            makeTurn("t3", "b_bbb", TurnRole::User, "next", Some("t2")),
            makeTurn("t4", "b_bbb", TurnRole::Assistant, "sure", Some("t3")),
        ];
        let blocks = groupByBlock(&turns);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].blockId, "b_aaa");
        assert_eq!(blocks[1].blockId, "b_bbb");
        assert_eq!(blocks[0].allTurnIds.len(), 2);
        assert_eq!(blocks[1].allTurnIds.len(), 2);
    }

    #[test]
    fn eligibility_skips_compacted_blocks_at_summary_size() {
        // 3 blocks: first two already compacted, third not.
        // Raw charCount of compacted blocks is large (1000 each).
        // Summary size is small (50 each). charTarget = 500.
        //
        // With raw sizes: cumChars hits 2000 before block 3 → no eligible.
        // With summary sizes: cumChars hits 100 before block 3 → block 3 eligible.
        let turns = vec![
            makeTurn("t1", "b_aaa", TurnRole::User, &"x".repeat(500), None),
            makeTurn("t2", "b_aaa", TurnRole::Assistant, &"y".repeat(500), Some("t1")),
            makeTurn("t3", "b_bbb", TurnRole::User, &"x".repeat(500), Some("t2")),
            makeTurn("t4", "b_bbb", TurnRole::Assistant, &"y".repeat(500), Some("t3")),
            makeTurn("t5", "b_ccc", TurnRole::User, "question", Some("t4")),
            makeTurn("t6", "b_ccc", TurnRole::Assistant, "answer with enough content", Some("t5")),
        ];

        let blocks = groupByBlock(&turns);
        assert_eq!(blocks.len(), 3);

        // Simulate compaction log: blocks aaa and bbb are compacted with short summaries.
        let mut alreadyCompacted: HashMap<String, usize> = HashMap::new();
        alreadyCompacted.insert("b_aaa".into(), 50);
        alreadyCompacted.insert("b_bbb".into(), 50);

        // charTarget set to cover all 3 blocks at summary size but not at raw size.
        let charTarget = 500;

        let mut cumChars: usize = 0;
        let mut eligible: Vec<usize> = Vec::new();

        for (i, block) in blocks.iter().enumerate() {
            if cumChars >= charTarget {
                break;
            }
            if let Some(summaryLen) = alreadyCompacted.get(&block.blockId) {
                cumChars += summaryLen;
                continue;
            }
            cumChars += block.charCount;
            if block.agentTurns.is_empty() {
                continue;
            }
            eligible.push(i);
        }

        assert!(
            !eligible.is_empty(),
            "block b_ccc should be eligible — compacted blocks counted at summary size"
        );
        assert_eq!(eligible[0], 2, "block index 2 (b_ccc) should be eligible");
    }

    #[test]
    fn eligibility_uses_only_active_branch() {
        // Simulate rewind: turns t1→t2→t3 on original branch, rewind to t2,
        // then t4→t5 on new branch. Active branch = t1→t2→t4→t5.
        // t3 (dead branch) should NOT be included.
        let allTurns = vec![
            makeTurn("t1", "b_aaa", TurnRole::User, "first", None),
            makeTurn("t2", "b_aaa", TurnRole::Assistant, "reply", Some("t1")),
            makeTurn("t3", "b_dead", TurnRole::User, "dead branch", Some("t2")),
            makeTurn("t4", "b_bbb", TurnRole::User, "after rewind", Some("t2")),
            makeTurn("t5", "b_bbb", TurnRole::Assistant, "new reply", Some("t4")),
        ];

        let activeTurns = crate::transcript::walkBranchTurns(&allTurns, "t5");
        assert_eq!(activeTurns.len(), 4); // t1, t2, t4, t5
        let blockIds: Vec<&str> = activeTurns.iter().map(|t| t.blockId.as_str()).collect();
        assert!(!blockIds.contains(&"b_dead"), "dead branch block should not be on active branch");

        let blocks = groupByBlock(&activeTurns);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].blockId, "b_aaa");
        assert_eq!(blocks[1].blockId, "b_bbb");
    }
}
