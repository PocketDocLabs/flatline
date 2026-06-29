//! Full-text search over the transcript using tantivy.
//!
//! Builds a throwaway in-RAM index from a slice of turns, then runs a
//! tokenized BM25 query. This replaces the old substring-match approach
//! so multi-keyword queries like "copy seeds petri_rl_envs" find results
//! even when those words don't appear as one contiguous string.

use crate::transcript::{Turn, TurnRole};

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{STORED, Schema, TEXT, Value};
use tantivy::{Index, IndexWriter, TantivyDocument, doc};

/// A single search hit with metadata for formatting.
#[allow(dead_code)]
pub struct Hit {
    pub turnId: String,
    pub blockId: String,
    pub role: String,
    pub snippet: String,
    pub score: f32,
    /// Index into the original turns slice, for post-filtering.
    pub turnIndex: usize,
}

/// Build an in-RAM tantivy index from the given turns, run `query` against
/// it, and return up to `limit` hits ranked by BM25 score.
pub fn search(turns: &[Turn], query: &str, limit: usize) -> Vec<Hit> {
    let mut builder = Schema::builder();
    let fTurnId = builder.add_text_field("turnId", STORED);
    let fBlockId = builder.add_text_field("blockId", STORED);
    let fRole = builder.add_text_field("role", STORED);
    let fContent = builder.add_text_field("content", TEXT | STORED);
    let fArgs = builder.add_text_field("args", TEXT);
    let fTool = builder.add_text_field("tool", TEXT | STORED);
    // Stored-only field for the original turn index.
    let fIdx = builder.add_u64_field(
        "idx",
        tantivy::schema::NumericOptions::default().set_stored(),
    );
    let schema = builder.build();

    let index = Index::create_in_ram(schema);

    let mut writer: IndexWriter = match index.writer(15_000_000) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("tantivy writer creation failed: {e}");
            return Vec::new();
        }
    };

    for (i, turn) in turns.iter().enumerate() {
        // Skip historySearch/historyFetch tool results and calls to avoid
        // echo-chamber pollution from previous search results.
        if isHistoryTool(turn) {
            continue;
        }

        if turn.content.is_empty() && turn.args.is_none() {
            continue;
        }

        let argsText = turn.args.as_ref().map(flattenArgs).unwrap_or_default();

        let toolText = turn.tool.as_deref().unwrap_or("");
        let roleLabel = roleStr(&turn.role);

        let _ = writer.add_document(doc!(
            fTurnId => turn.id.as_str(),
            fBlockId => turn.blockId.as_str(),
            fRole => roleLabel,
            fContent => turn.content.as_str(),
            fArgs => argsText.as_str(),
            fTool => toolText,
            fIdx => i as u64,
        ));
    }

    if writer.commit().is_err() {
        return Vec::new();
    }

    let reader: tantivy::IndexReader = match index.reader() {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let searcher = reader.searcher();

    // Search content and args fields by default.
    let parser = QueryParser::for_index(&index, vec![fContent, fArgs, fTool]);

    let parsed = match parser.parse_query(query) {
        Ok(q) => q,
        Err(e) => {
            tracing::debug!("tantivy query parse error: {e}");
            return Vec::new();
        }
    };

    let topDocs: Vec<(f32, tantivy::DocAddress)> =
        match searcher.search(&parsed, &TopDocs::with_limit(limit)) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("tantivy search error: {e}");
                return Vec::new();
            }
        };

    let mut hits = Vec::with_capacity(topDocs.len());
    for (score, addr) in topDocs {
        let retrieved: TantivyDocument = match searcher.doc(addr) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let turnId = fieldStr(&retrieved, fTurnId);
        let blockId = fieldStr(&retrieved, fBlockId);
        let role = fieldStr(&retrieved, fRole);
        let content = fieldStr(&retrieved, fContent);
        let turnIndex = retrieved
            .get_first(fIdx)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let snippet = makeSnippet(&content, 160);

        hits.push(Hit {
            turnId,
            blockId,
            role,
            snippet,
            score,
            turnIndex,
        });
    }

    hits
}

fn isHistoryTool(turn: &Turn) -> bool {
    match turn.tool.as_deref() {
        Some("historySearch" | "historyFetch") => true,
        _ => {
            // Also skip tool_result turns whose parent was a history tool.
            // These have toolCallId set but no tool field — we detect them by
            // checking if the content looks like our own formatted output.
            // Simpler: just check the role + content prefix patterns.
            if matches!(turn.role, TurnRole::ToolResult) {
                let c = &turn.content;
                c.starts_with("Found ") && c.contains("matches for \"")
                    || c.starts_with("No matches found for \"")
                    || c.starts_with("## Block b_")
                    || c.starts_with("No block found with ID")
            } else {
                false
            }
        }
    }
}

/// Flatten a JSON args value into a searchable string.
/// Extracts string values from the object so file paths, commands, etc.
/// become individual tokens.
fn flattenArgs(args: &serde_json::Value) -> String {
    let mut buf = String::new();
    collectStrings(args, &mut buf);
    buf
}

fn collectStrings(val: &serde_json::Value, buf: &mut String) {
    match val {
        serde_json::Value::String(s) => {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(s);
        }
        serde_json::Value::Object(map) => {
            for v in map.values() {
                collectStrings(v, buf);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                collectStrings(v, buf);
            }
        }
        _ => {}
    }
}

fn roleStr(role: &TurnRole) -> &'static str {
    match role {
        TurnRole::User => "user",
        TurnRole::Assistant => "assistant",
        TurnRole::ToolCall => "tool_call",
        TurnRole::ToolResult => "tool_result",
        TurnRole::System => "system",
        TurnRole::Wake => "wake",
    }
}

fn fieldStr(doc: &TantivyDocument, field: tantivy::schema::Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn makeSnippet(text: &str, maxChars: usize) -> String {
    let flat: String = text
        .chars()
        .take(maxChars)
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeTurn(id: &str, blockId: &str, role: TurnRole, content: &str) -> Turn {
        Turn {
            id: id.into(),
            blockId: blockId.into(),
            topicId: String::new(),
            role,
            content: content.into(),
            ts: 0,
            parentId: None,
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments: None,
            toolMeta: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: crate::transcript::TurnStatus::Completed,
        }
    }

    #[test]
    fn multiWordQueryFindsDisjointMatches() {
        let turns = vec![
            makeTurn(
                "t1",
                "b1",
                TurnRole::User,
                "can we copy the seeds into a new dir",
            ),
            makeTurn(
                "t2",
                "b1",
                TurnRole::Assistant,
                "Copied 173 seeds from inspect_petri into petri_rl_envs/seeds/",
            ),
            makeTurn(
                "t3",
                "b2",
                TurnRole::User,
                "something completely unrelated about weather",
            ),
        ];
        let hits = search(&turns, "copy seeds petri_rl_envs", 10);
        assert!(!hits.is_empty(), "multi-word query should find results");
        // Both t1 and t2 should match — t2 more strongly since it has more terms.
        let ids: Vec<&str> = hits.iter().map(|h| h.turnId.as_str()).collect();
        assert!(
            ids.contains(&"t2"),
            "should find the turn with all three terms"
        );
        assert!(
            ids.contains(&"t1"),
            "should find the turn with partial term overlap"
        );
        assert!(!ids.contains(&"t3"), "unrelated turn should not match");
    }

    #[test]
    fn historySearchResultsAreExcluded() {
        let turns = vec![
            makeTurn(
                "t1",
                "b1",
                TurnRole::User,
                "tell me about practice_problems",
            ),
            {
                let mut t = makeTurn("t2", "b1", TurnRole::ToolCall, "");
                t.tool = Some("historySearch".into());
                t.args = Some(serde_json::json!({"query": "practice_problems"}));
                t
            },
            makeTurn(
                "t3",
                "b1",
                TurnRole::ToolResult,
                "Found 55 matches for \"practice_problems\":\n- **b_foo** ...",
            ),
        ];
        let hits = search(&turns, "practice_problems", 10);
        let ids: Vec<&str> = hits.iter().map(|h| h.turnId.as_str()).collect();
        assert!(ids.contains(&"t1"), "real user turn should match");
        assert!(
            !ids.contains(&"t2"),
            "historySearch tool_call should be excluded"
        );
        assert!(
            !ids.contains(&"t3"),
            "historySearch tool_result should be excluded"
        );
    }

    #[test]
    fn argsAreSearchable() {
        let turns = vec![{
            let mut t = makeTurn("t1", "b1", TurnRole::ToolCall, "");
            t.tool = Some("readFile".into());
            t.args = Some(serde_json::json!({"path": "/src/auth/handler.rs"}));
            t
        }];
        let hits = search(&turns, "auth handler", 10);
        assert!(!hits.is_empty(), "should find turn by searching args");
    }

    #[test]
    fn emptyQueryReturnsNothing() {
        let turns = vec![makeTurn("t1", "b1", TurnRole::User, "hello world")];
        let hits = search(&turns, "", 10);
        assert!(hits.is_empty());
    }
}
