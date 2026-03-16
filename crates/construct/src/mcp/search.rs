#![allow(non_snake_case)]

//! MCP Tool Search meta-tool — context budgeting for large MCP tool sets.
//!
//! When the total token cost of MCP tool definitions exceeds 10% of the
//! context window, all individual tool defs are replaced by a single
//! `mcpToolSearch` meta-tool. The LLM uses this to discover relevant
//! tools before calling them.
//!
//! This mirrors Claude Code's approach: 72K+ tokens of tool defs → ~200 tokens.
//!
//! # Public API
//! - [`searchToolDef`] — returns the meta-tool definition
//! - [`executeSearch`] — runs a search query against the registry
//!
//! # Dependencies
//! Registry, ToolDef

use crate::message::{FunctionDef, ToolDef};

use super::registry::ToolRegistry;

/// Returns the tool definition for `mcpToolSearch`.
///
/// This is a lightweight meta-tool (~200 tokens) that replaces all
/// individual MCP tool definitions when context budgeting is active.
pub fn searchToolDef() -> ToolDef {
    ToolDef {
        defType: "function".into(),
        function: FunctionDef {
            name: "mcpToolSearch".into(),
            description: "Search available MCP tools by name or description. \
                Use this to discover which MCP tools are available before calling them. \
                Returns matching tool names, descriptions, and parameter summaries."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to match against tool names and descriptions"
                    },
                    "server": {
                        "type": "string",
                        "description": "Optional: filter results to a specific MCP server name"
                    }
                },
                "required": ["query"]
            }),
        },
    }
}

/// Execute a search against the registry and format results for the LLM.
pub fn executeSearch(
    registry: &ToolRegistry,
    argsJson: &str,
    _serverFilterOverride: Option<&str>,
) -> String {
    // Parse the arguments.
    let args: serde_json::Value = match serde_json::from_str(argsJson) {
        Ok(v) => v,
        Err(e) => return format!("Failed to parse search arguments: {e}"),
    };

    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let serverFilter = args
        .get("server")
        .and_then(|v| v.as_str());

    let results = registry.search(query, serverFilter);

    if results.is_empty() {
        return format!(
            "No MCP tools found matching \"{query}\". \
             Try a broader search or list all tools with an empty query."
        );
    }

    // Format results compactly for the LLM.
    let mut output = format!("Found {} MCP tools matching \"{query}\":\n\n", results.len());

    for r in results.iter().take(20) {
        output.push_str(&format!("▸ {} (server: {})\n", r.qualifiedName, r.serverName));
        if !r.description.is_empty() {
            output.push_str(&format!("  {}\n", r.description));
        }
        if !r.paramSummary.is_empty() {
            output.push_str(&format!("  params: {}\n", r.paramSummary));
        }
        output.push('\n');
    }

    if results.len() > 20 {
        output.push_str(&format!(
            "... and {} more. Narrow your search for better results.\n",
            results.len() - 20
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Tool;

    fn makeTool(name: &str, desc: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".into(), "object".into());
        schema.insert(
            "properties".into(),
            serde_json::json!({
                "query": { "type": "string" }
            }),
        );
        schema.insert("required".into(), serde_json::json!(["query"]));
        Tool::new(name.to_string(), desc.to_string(), schema)
    }

    #[test]
    fn searchToolDefFormat() {
        let def = searchToolDef();
        assert_eq!(def.function.name, "mcpToolSearch");
        assert!(def.function.description.contains("Search"));
    }

    #[test]
    fn executeSearchFindsTools() {
        let mut reg = ToolRegistry::new();
        reg.registerServer("github", vec![
            makeTool("search_repos", "Search GitHub repositories"),
            makeTool("create_issue", "Create a new issue"),
        ]);

        let result = executeSearch(&reg, r#"{"query": "search"}"#, None);
        assert!(result.contains("search_repos"));
        assert!(result.contains("github"));
    }

    #[test]
    fn executeSearchNoResults() {
        let reg = ToolRegistry::new();
        let result = executeSearch(&reg, r#"{"query": "nonexistent"}"#, None);
        assert!(result.contains("No MCP tools found"));
    }

    #[test]
    fn executeSearchBadJson() {
        let reg = ToolRegistry::new();
        let result = executeSearch(&reg, "not json", None);
        assert!(result.contains("Failed to parse"));
    }
}
