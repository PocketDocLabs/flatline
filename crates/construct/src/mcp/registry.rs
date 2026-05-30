#![allow(non_snake_case)]

//! MCP tool registry — qualified names, routing table, ToolDef conversion.
//!
//! Maps `mcp__{server}__{tool}` qualified names to their server origin
//! and provides conversion to the `ToolDef` format expected by the LLM API.
//!
//! # Public API
//! - [`ToolRegistry`] — the central tool registry
//! - [`RegistryEntry`] — a single registered tool
//!
//! # Dependencies
//! `rmcp` (Tool type), `serde_json`

use std::collections::HashMap;

use rmcp::model::Tool;

use crate::message::{FunctionDef, ToolDef};

use super::output::estimateTokens;
use super::schema::{qualifyName, sanitizeJsonSchema};

/// A single registered MCP tool with routing info.
pub struct RegistryEntry {
    pub serverName: String,
    pub originalName: String,
    pub qualifiedName: String,
    pub description: String,
    pub schema: serde_json::Value,
}

/// Central registry for all MCP tools across all connected servers.
///
/// Handles name qualification, schema sanitization, collision detection,
/// and conversion to the `ToolDef` format sent to the LLM.
pub struct ToolRegistry {
    /// qualifiedName → entry
    tools: HashMap<String, RegistryEntry>,
    /// serverName → [qualifiedNames]
    serverIndex: HashMap<String, Vec<String>>,
    /// Precomputed tool defs for the LLM.
    cachedDefs: Vec<ToolDef>,
    /// Estimated total token cost of all cached defs.
    totalDefTokens: usize,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            serverIndex: HashMap::new(),
            cachedDefs: Vec::new(),
            totalDefTokens: 0,
        }
    }

    /// Register all tools from a server.
    ///
    /// Qualifies names, sanitizes schemas, detects collisions.
    /// Colliding tool names are skipped with a warning.
    pub fn registerServer(&mut self, serverName: &str, tools: Vec<Tool>) {
        let mut qualifiedNames = Vec::with_capacity(tools.len());

        for tool in tools {
            let originalName = tool.name.to_string();
            let qualifiedName = qualifyName(serverName, &originalName);

            if self.tools.contains_key(&qualifiedName) {
                tracing::warn!(
                    server = %serverName,
                    tool = %originalName,
                    qualified = %qualifiedName,
                    "tool name collision — skipping"
                );
                continue;
            }

            let description = tool.description.as_deref().unwrap_or("").to_string();

            // Convert input_schema to a Value and sanitize it.
            let mut schema = serde_json::to_value(&*tool.input_schema).unwrap_or_default();
            sanitizeJsonSchema(&mut schema);

            let entry = RegistryEntry {
                serverName: serverName.to_string(),
                originalName,
                qualifiedName: qualifiedName.clone(),
                description,
                schema,
            };

            self.tools.insert(qualifiedName.clone(), entry);
            qualifiedNames.push(qualifiedName);
        }

        if !qualifiedNames.is_empty() {
            self.serverIndex
                .insert(serverName.to_string(), qualifiedNames);
        }

        self.rebuildCache();
    }

    /// Unregister all tools for a server.
    pub fn unregisterServer(&mut self, serverName: &str) {
        if let Some(names) = self.serverIndex.remove(serverName) {
            for name in &names {
                self.tools.remove(name);
            }
        }
        self.rebuildCache();
    }

    /// Look up a tool by qualified name.
    pub fn resolve(&self, qualifiedName: &str) -> Option<&RegistryEntry> {
        self.tools.get(qualifiedName)
    }

    /// Get tool defs for the LLM.
    ///
    /// If total token cost exceeds `contextBudget * 0.10`, returns only the
    /// search meta-tool (see `search.rs`). Otherwise returns all defs.
    pub fn toolDefs(
        &self,
        contextBudget: usize,
        includePermissionEscalation: bool,
    ) -> Vec<ToolDef> {
        let mut defs = self.cachedDefs.clone();
        if includePermissionEscalation {
            crate::tool::addPermissionEscalationFieldsToDefs(&mut defs);
        }

        let threshold = contextBudget / 10;
        let totalDefTokens = estimateToolDefTokens(&defs);
        if totalDefTokens > threshold && !defs.is_empty() {
            // Context too expensive — return just the search meta-tool.
            let mut searchDefs = vec![super::search::searchToolDef()];
            if includePermissionEscalation {
                crate::tool::addPermissionEscalationFieldsToDefs(&mut searchDefs);
            }
            return searchDefs;
        }
        defs
    }

    /// Whether the search meta-tool is active (i.e. defs are deferred).
    pub fn isSearchMode(&self, contextBudget: usize, includePermissionEscalation: bool) -> bool {
        let threshold = contextBudget / 10;
        let mut defs = self.cachedDefs.clone();
        if includePermissionEscalation {
            crate::tool::addPermissionEscalationFieldsToDefs(&mut defs);
        }
        estimateToolDefTokens(&defs) > threshold && !defs.is_empty()
    }

    /// Total estimated tokens for all tool defs.
    pub fn totalTokens(&self) -> usize {
        self.totalDefTokens
    }

    /// Number of registered tools.
    pub fn toolCount(&self) -> usize {
        self.tools.len()
    }

    /// Search tools by query string. Returns matching entries with relevance score.
    pub fn search(&self, query: &str, serverFilter: Option<&str>) -> Vec<SearchResult> {
        let queryLower = query.to_lowercase();
        let queryTerms: Vec<&str> = queryLower.split_whitespace().collect();

        if queryTerms.is_empty() {
            // No query — return all tools.
            return self
                .tools
                .values()
                .filter(|e| serverFilter.is_none_or(|s| e.serverName == s))
                .map(|e| SearchResult {
                    qualifiedName: e.qualifiedName.clone(),
                    serverName: e.serverName.clone(),
                    originalName: e.originalName.clone(),
                    description: e.description.clone(),
                    paramSummary: summarizeParams(&e.schema),
                    score: 0,
                })
                .collect();
        }

        let mut results: Vec<SearchResult> = self
            .tools
            .values()
            .filter(|e| serverFilter.is_none_or(|s| e.serverName == s))
            .filter_map(|e| {
                let score = scoreMatch(&queryTerms, e);
                if score > 0 {
                    Some(SearchResult {
                        qualifiedName: e.qualifiedName.clone(),
                        serverName: e.serverName.clone(),
                        originalName: e.originalName.clone(),
                        description: e.description.clone(),
                        paramSummary: summarizeParams(&e.schema),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.cmp(&a.score));
        results
    }

    /// Rebuild cached ToolDefs and token count from current registry.
    ///
    /// Sorted by qualified tool name so the serialized tools array is
    /// byte-identical across processes. HashMap iteration order is
    /// non-deterministic in Rust's default hasher — without the sort, two
    /// flatline instances with the same MCP servers would send differently
    /// ordered tool arrays and neither could read the other's prompt cache.
    fn rebuildCache(&mut self) {
        self.cachedDefs = self
            .tools
            .values()
            .map(|entry| ToolDef {
                defType: "function".into(),
                function: FunctionDef {
                    name: entry.qualifiedName.clone(),
                    description: entry.description.clone(),
                    parameters: entry.schema.clone(),
                },
            })
            .collect();
        self.cachedDefs
            .sort_by(|a, b| a.function.name.cmp(&b.function.name));

        // Estimate token cost of all defs.
        self.totalDefTokens = estimateToolDefTokens(&self.cachedDefs);
    }
}

fn estimateToolDefTokens(defs: &[ToolDef]) -> usize {
    let jsonStr = serde_json::to_string(defs).unwrap_or_default();
    estimateTokens(&jsonStr)
}

/// A search result with relevance score.
pub struct SearchResult {
    pub qualifiedName: String,
    pub serverName: String,
    pub originalName: String,
    pub description: String,
    pub paramSummary: String,
    pub score: usize,
}

/// Score how well a tool matches a set of query terms.
///
/// Points: name exact match = 100, name contains = 10,
/// description contains = 5, server name contains = 3.
fn scoreMatch(queryTerms: &[&str], entry: &RegistryEntry) -> usize {
    let nameLower = entry.originalName.to_lowercase();
    let descLower = entry.description.to_lowercase();
    let serverLower = entry.serverName.to_lowercase();
    let qualifiedLower = entry.qualifiedName.to_lowercase();

    let mut score = 0usize;

    for &term in queryTerms {
        // Exact name match.
        if nameLower == term || qualifiedLower == term {
            score += 100;
        } else if nameLower.contains(term) || qualifiedLower.contains(term) {
            score += 10;
        }

        if descLower.contains(term) {
            score += 5;
        }

        if serverLower.contains(term) {
            score += 3;
        }
    }

    score
}

/// Summarize a JSON schema's parameters into a compact string.
fn summarizeParams(schema: &serde_json::Value) -> String {
    let props = match schema.get("properties") {
        Some(serde_json::Value::Object(m)) => m,
        _ => return String::new(),
    };

    let required: std::collections::HashSet<&str> = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let params: Vec<String> = props
        .iter()
        .map(|(name, def)| {
            let typeName = def.get("type").and_then(|v| v.as_str()).unwrap_or("any");
            let marker = if required.contains(name.as_str()) {
                ""
            } else {
                "?"
            };
            format!("{name}{marker}: {typeName}")
        })
        .collect();

    params.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Tool;

    fn makeTool(name: &str, desc: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".into(), "object".into());
        schema.insert("properties".into(), serde_json::json!({}));
        Tool::new(name.to_string(), desc.to_string(), schema)
    }

    #[test]
    fn registerAndResolve() {
        let mut reg = ToolRegistry::new();
        reg.registerServer("github", vec![makeTool("search", "Search repos")]);

        assert_eq!(reg.toolCount(), 1);
        let entry = reg.resolve("mcp__github__search").unwrap();
        assert_eq!(entry.serverName, "github");
        assert_eq!(entry.originalName, "search");
    }

    #[test]
    fn unregisterCleansUp() {
        let mut reg = ToolRegistry::new();
        reg.registerServer("github", vec![makeTool("search", "Search repos")]);
        reg.unregisterServer("github");

        assert_eq!(reg.toolCount(), 0);
        assert!(reg.resolve("mcp__github__search").is_none());
    }

    #[test]
    fn collisionSkipsSecond() {
        let mut reg = ToolRegistry::new();
        reg.registerServer(
            "github",
            vec![makeTool("search", "First"), makeTool("search", "Second")],
        );
        // Both qualify to the same name — second should be skipped.
        assert_eq!(reg.toolCount(), 1);
    }

    #[test]
    fn searchMatchesName() {
        let mut reg = ToolRegistry::new();
        reg.registerServer(
            "github",
            vec![
                makeTool("search_repos", "Search GitHub repos"),
                makeTool("create_issue", "Create a new issue"),
            ],
        );

        let results = reg.search("search", None);
        assert!(!results.is_empty());
        assert_eq!(results[0].originalName, "search_repos");
    }

    #[test]
    fn searchFiltersByServer() {
        let mut reg = ToolRegistry::new();
        reg.registerServer("github", vec![makeTool("search", "GitHub search")]);
        reg.registerServer("jira", vec![makeTool("search", "Jira search")]);

        let results = reg.search("search", Some("github"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].serverName, "github");
    }

    #[test]
    fn toolDefsSearchMode() {
        let mut reg = ToolRegistry::new();
        // Register enough tools to exceed 10% of a small context window.
        let tools: Vec<Tool> = (0..50)
            .map(|i| {
                makeTool(
                    &format!("tool_{i}"),
                    &format!("This is tool number {i} with a reasonably long description to inflate token count"),
                )
            })
            .collect();
        reg.registerServer("big", tools);

        // Small context budget should trigger search mode.
        let defs = reg.toolDefs(1000, false);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "mcpToolSearch");
    }

    #[test]
    fn toolDefsNormalMode() {
        let mut reg = ToolRegistry::new();
        reg.registerServer("small", vec![makeTool("one", "Tool one")]);

        // Large context budget should return all defs.
        let defs = reg.toolDefs(1_000_000, false);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "mcp__small__one");
        assert!(
            defs[0].function.parameters["properties"]
                .get("raiseToUser")
                .is_none()
        );
    }

    #[test]
    fn toolDefsCanIncludePermissionEscalationFields() {
        let mut reg = ToolRegistry::new();
        reg.registerServer("small", vec![makeTool("one", "Tool one")]);

        let defs = reg.toolDefs(1_000_000, true);
        assert_eq!(defs.len(), 1);
        assert!(
            defs[0].function.parameters["properties"]
                .get("raiseToUser")
                .is_some()
        );
    }

    #[test]
    fn toolDefsSortedByName() {
        // Register tools across two servers in mixed insertion order. Output
        // must be sorted by qualified name so the serialized tools array is
        // byte-stable across processes (otherwise HashMap iteration would
        // scramble it differently every run).
        let mut reg = ToolRegistry::new();
        reg.registerServer(
            "zeta",
            vec![makeTool("charlie", "Tool C"), makeTool("alpha", "Tool A")],
        );
        reg.registerServer("alpha", vec![makeTool("bravo", "Tool B")]);

        let defs = reg.toolDefs(1_000_000, false);
        let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
        let mut expected = names.clone();
        expected.sort();
        assert_eq!(names, expected, "tool defs not sorted by qualified name");

        // Run the conversion again; order must be identical.
        let defs2 = reg.toolDefs(1_000_000, false);
        let names2: Vec<&str> = defs2.iter().map(|d| d.function.name.as_str()).collect();
        assert_eq!(names, names2, "tool def order differs between calls");
    }

    #[test]
    fn paramSummaryFormat() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "limit": { "type": "number" }
            },
            "required": ["query"]
        });
        let summary = summarizeParams(&schema);
        assert!(summary.contains("query: string"));
        assert!(summary.contains("limit?: number"));
    }
}
