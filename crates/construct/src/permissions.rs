//! Permission system for tool execution.
//!
//! Every tool invocation must pass through a permission check before execution.
//! Permissions can be configured per-tool with allow/deny patterns, or globally.
//!
//! Default: deny everything, require explicit approval for each action.
//!
//! When a tool call is denied and no approval channel is available,
//! the session exits the current turn rather than hanging.
//!
//! # Public API
//! - [`Permissions`] — permission configuration
//! - [`PermitMode`] — what to do when a permit is needed
//! - [`Verdict`] — result of a permission check
//! - [`PermitResponse`] — supervisor response to a permission prompt
//! - [`suggestPatterns`] — generate scoped "always allow" patterns
//!
//! # Dependencies
//! `serde`

use crate::tool::ToolAction;
use serde::{Deserialize, Serialize};

/// Response from the supervisor (TUI or parent agent) to a permission prompt.
#[derive(Debug, Clone)]
pub enum PermitResponse {
    /// Allow this one invocation.
    Allow,
    /// Allow and persist an allow rule for the given pattern.
    AlwaysAllow { pattern: String },
    /// Deny this invocation.
    Deny,
    /// Deny and persist a deny rule for the given pattern.
    AlwaysDeny { pattern: String },
}

/// Generate suggested "always allow" patterns for a tool action.
///
/// Returns a list from most specific to broadest scope. The TUI presents
/// these as selectable options when the user presses 'a' (always allow).
pub fn suggestPatterns(action: &ToolAction) -> Vec<String> {
    match action {
        // File tools: parent dir → grandparent → project root.
        ToolAction::ReadFile { path, .. }
        | ToolAction::WriteFile { path, .. }
        | ToolAction::EditFile { path, .. }
        | ToolAction::MultiEdit { path, .. }
        | ToolAction::DeleteFile { path, .. }
        | ToolAction::MakeDirs { path }
        | ToolAction::FileOutline { path }
        | ToolAction::ViewSymbol { file: path, .. }
        | ToolAction::RelatedFiles { path } => pathPatterns(path),

        ToolAction::CopyFile { dest, .. }
        | ToolAction::MoveFile { dest, .. } => pathPatterns(dest),

        ToolAction::Glob { path, .. } => {
            if let Some(p) = path {
                pathPatterns(p)
            } else {
                vec!["./".into()]
            }
        }
        ToolAction::Grep { path, .. } => {
            if let Some(p) = path {
                pathPatterns(p)
            } else {
                vec!["./".into()]
            }
        }
        ToolAction::ListDir { path, .. } => pathPatterns(path),

        // Shell: progressively shorter token prefixes.
        ToolAction::Shell { command, .. } => shellPatterns(command),

        // Web: subdomain → domain → any.
        ToolAction::WebFetch { url, .. } | ToolAction::WebSimilar { url, .. } => urlPatterns(url),
        ToolAction::WebSearch { query, .. } => vec![query.clone()],

        // MCP: specific tool → server wildcard.
        ToolAction::Mcp { qualifiedName, .. } => {
            let mut patterns = vec![qualifiedName.clone()];
            // Extract server prefix for wildcard.
            if let Some(pos) = qualifiedName.rfind("__") {
                let serverPrefix = &qualifiedName[..pos + 2];
                patterns.push(format!("{serverPrefix}*"));
            }
            patterns
        }

        ToolAction::Task { .. } => vec!["task".into()],

        _ => Vec::new(),
    }
}

/// Get the display impact tier for any tool action.
///
/// For shell: uses the model-provided impact classification.
/// For file writes/edits/MCP: hardcoded to MinorMod.
/// For read-only tools (shouldn't reach a prompt): Read.
pub fn toolImpact(action: &ToolAction) -> crate::tool::ShellImpact {
    use crate::tool::ShellImpact;
    match action {
        ToolAction::Shell { impact, .. } => impact.clone(),
        ToolAction::WriteFile { .. }
        | ToolAction::EditFile { .. }
        | ToolAction::MultiEdit { .. }
        | ToolAction::CopyFile { .. }
        | ToolAction::MoveFile { .. }
        | ToolAction::MakeDirs { .. } => ShellImpact::MinorMod,
        ToolAction::DeleteFile { .. } => ShellImpact::Delete,
        ToolAction::Mcp { .. } => ShellImpact::MinorMod,
        ToolAction::Task { .. } => ShellImpact::MinorMod,
        _ => ShellImpact::Read,
    }
}

/// Get the shell explanation from a tool action (only present for shell commands).
pub fn toolExplanation(action: &ToolAction) -> Option<&str> {
    match action {
        ToolAction::Shell { explanation, .. } if !explanation.is_empty() => Some(explanation),
        _ => None,
    }
}

/// Generate directory-based patterns from a file path.
fn pathPatterns(path: &str) -> Vec<String> {
    let mut patterns = Vec::new();
    let normalized = path.replace('\\', "/");

    // Parent directory.
    if let Some(pos) = normalized.rfind('/') {
        let parent = &normalized[..=pos];
        patterns.push(parent.to_string());

        // Grandparent.
        if pos > 0 {
            if let Some(gpos) = normalized[..pos].rfind('/') {
                patterns.push(normalized[..=gpos].to_string());
            }
        }
    }

    // Project root.
    patterns.push("./".into());
    patterns.dedup();
    patterns
}

/// Generate shell command patterns for "always allow/deny" rules.
///
/// Parses the command to extract the binary and optional subcommand,
/// ignoring flags, quoted strings, pipes, and shell metacharacters.
/// Patterns use trailing `*` for prefix matching.
fn shellPatterns(command: &str) -> Vec<String> {
    let mut patterns = Vec::new();

    // Take only the first command (before pipes, &&, ||, ;).
    let firstCmd = command
        .split(&['|', ';'][..])
        .next()
        .unwrap_or(command)
        .split("&&")
        .next()
        .unwrap_or(command)
        .split("||")
        .next()
        .unwrap_or(command)
        .trim();

    // Tokenize, skipping env assignments (FOO=bar).
    let tokens: Vec<&str> = firstCmd
        .split_whitespace()
        .filter(|t| !(t.contains('=') && !t.starts_with('-')))
        .collect();

    if tokens.is_empty() {
        return patterns;
    }

    let binary = tokens[0];

    // A subcommand is an alphabetic word (not a flag, path, quoted string, or redirect).
    let subcommand = tokens.get(1).and_then(|t| {
        let first = t.chars().next()?;
        if first.is_alphabetic()
            && t.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            Some(*t)
        } else {
            None
        }
    });

    // Most specific: binary + subcommand.
    if let Some(sub) = subcommand {
        patterns.push(format!("{binary} {sub}*"));
    }
    // Broadest: just the binary.
    patterns.push(format!("{binary}*"));

    patterns.dedup();
    patterns
}

/// Generate domain-based patterns from a URL.
fn urlPatterns(url: &str) -> Vec<String> {
    let mut patterns = Vec::new();

    // Extract host from URL.
    let host = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .and_then(|rest| rest.split('/').next())
        .unwrap_or(url);

    patterns.push(host.to_string());

    // Domain without subdomain (if has subdomain).
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() > 2 {
        let domain = parts[parts.len() - 2..].join(".");
        patterns.push(format!("*.{domain}"));
    }

    patterns.push("*".into());
    patterns.dedup();
    patterns
}

/// What to do when a tool call isn't pre-approved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermitMode {
    /// Ask the supervisor (TUI user, parent agent, etc.) and wait for response.
    Ask,
    /// Immediately deny and continue the turn with a denial message.
    Deny,
    /// Immediately deny and abort the entire turn.
    Abort,
}

/// Result of a permission check.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Pre-approved by the permission config.
    Allow,
    /// Pre-denied by the permission config.
    Deny,
    /// Not covered by any rule — needs runtime approval per `PermitMode`.
    NeedsApproval,
}

/// Permission rule matching a tool action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Tool name to match ("shell", "readFile", "writeFile", or "*" for all).
    pub tool: String,
    /// Optional pattern to match against the action's key argument.
    /// For shell: matches against the command string.
    /// For readFile/writeFile: matches against the path.
    /// If None, matches all invocations of the tool.
    pub pattern: Option<String>,
    /// Whether this rule allows or denies.
    pub allow: bool,
}

/// Where a permission set was loaded from.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PermissionsSource {
    /// Built-in preset (allowReadOnly, askForEverything, etc.).
    #[default]
    BuiltIn,
    /// User config (`~/.config/flatline/config.toml`).
    User,
    /// Project config (`.flatline/config.toml`).
    Project,
    /// Local config (`.flatline/config.local.toml`).
    Local,
}

/// Permission configuration for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permissions {
    /// What to do when no rule matches and approval is needed.
    pub defaultMode: PermitMode,
    /// Ordered list of rules. First match wins.
    pub rules: Vec<Rule>,
    /// Which config layer these permissions came from (runtime-only, not serialized).
    #[serde(skip)]
    pub source: PermissionsSource,
}

impl Default for Permissions {
    /// Default: deny everything, ask the supervisor.
    fn default() -> Self {
        Self {
            defaultMode: PermitMode::Ask,
            rules: Vec::new(),
            source: PermissionsSource::BuiltIn,
        }
    }
}

impl Permissions {
    /// Create permissions that deny everything and ask for each action.
    pub fn askForEverything() -> Self {
        Self::default()
    }

    /// Create permissions that deny everything and abort on any tool call.
    pub fn denyAll() -> Self {
        Self {
            defaultMode: PermitMode::Abort,
            rules: Vec::new(),
            source: PermissionsSource::BuiltIn,
        }
    }

    /// Create permissions that auto-approve everything.
    pub fn allowAll() -> Self {
        Self {
            defaultMode: PermitMode::Deny,
            rules: vec![Rule { tool: "*".into(), pattern: None, allow: true }],
            source: PermissionsSource::BuiltIn,
        }
    }

    /// Create permissions that auto-approve read-only tools.
    pub fn allowReadOnly() -> Self {
        let readOnlyTools = [
            "readFile", "glob", "grep", "listDir", "structSearch", "diff",
            "fuzzyFind", "fileOutline", "viewSymbol", "relatedFiles",
            "shellHistory", "readOutput", "searchOutput", "readTerminal",
            "webSearch", "webFetch", "webSimilar", "diagnostics",
        ];
        Self {
            defaultMode: PermitMode::Ask,
            rules: readOnlyTools
                .iter()
                .map(|&tool| Rule { tool: tool.into(), pattern: None, allow: true })
                .collect(),
            source: PermissionsSource::BuiltIn,
        }
    }

    /// Add a rule.
    pub fn addRule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Check whether an action is permitted.
    pub fn check(&self, action: &ToolAction) -> Verdict {
        let (toolName, keyArg) = actionKey(action);

        for rule in &self.rules {
            if !matchesTool(&rule.tool, toolName) {
                continue;
            }
            if let Some(pattern) = &rule.pattern {
                // Trailing * = prefix match. Otherwise substring match.
                let matches = if let Some(prefix) = pattern.strip_suffix('*') {
                    keyArg.starts_with(prefix)
                } else {
                    keyArg.contains(pattern.as_str())
                };
                if !matches {
                    continue;
                }
            }
            return if rule.allow {
                Verdict::Allow
            } else {
                Verdict::Deny
            };
        }

        Verdict::NeedsApproval
    }
}

/// Extract the tool name and key argument from an action.
pub fn actionKey(action: &ToolAction) -> (&str, &str) {
    match action {
        ToolAction::Shell { command, .. } => ("shell", command),
        ToolAction::ReadFile { path, .. } => ("readFile", path),
        ToolAction::WriteFile { path, .. } => ("writeFile", path),
        ToolAction::EditFile { path, .. } => ("editFile", path),
        ToolAction::MultiEdit { path, .. } => ("multiEdit", path),
        ToolAction::CopyFile { dest, .. } => ("copyFile", dest),
        ToolAction::MoveFile { dest, .. } => ("moveFile", dest),
        ToolAction::DeleteFile { path, .. } => ("deleteFile", path),
        ToolAction::MakeDirs { path } => ("makeDirs", path),
        ToolAction::ShellHistory => ("shellHistory", ""),
        ToolAction::ReadOutput { .. } => ("readOutput", ""),
        ToolAction::SearchOutput { pattern, .. } => ("searchOutput", pattern),
        ToolAction::ReadTerminal { .. } => ("readTerminal", ""),
        ToolAction::Glob { pattern, .. } => ("glob", pattern),
        ToolAction::Grep { pattern, .. } => ("grep", pattern),
        ToolAction::ListDir { path, .. } => ("listDir", path),
        ToolAction::StructSearch { pattern, .. } => ("structSearch", pattern),
        ToolAction::Diff { path, pathA, .. } => {
            ("diff", path.as_deref().or(pathA.as_deref()).unwrap_or(""))
        }
        ToolAction::FuzzyFind { query, .. } => ("fuzzyFind", query),
        ToolAction::FileOutline { path } => ("fileOutline", path),
        ToolAction::ViewSymbol { file, .. } => ("viewSymbol", file),
        ToolAction::RelatedFiles { path } => ("relatedFiles", path),
        ToolAction::WebSearch { query, .. } => ("webSearch", query),
        ToolAction::WebFetch { url, .. } => ("webFetch", url),
        ToolAction::WebSimilar { url, .. } => ("webSimilar", url),
        ToolAction::HistoryFetch { blockId } => ("historyFetch", blockId),
        ToolAction::HistorySearch { query, .. } => ("historySearch", query),
        ToolAction::Task { prompt, .. } => ("task", prompt),
        ToolAction::Diagnostics { path, .. } => ("diagnostics", path),
        ToolAction::Mcp { qualifiedName, args } => (qualifiedName, args),
        ToolAction::Unknown { name, args } => (name, args),
    }
}

/// Check if a rule's tool field matches a tool name.
///
/// Supports exact match, global wildcard `"*"`, and prefix wildcards
/// like `"mcp__github__*"` to match all tools from a server.
fn matchesTool(rulePattern: &str, toolName: &str) -> bool {
    if rulePattern == "*" {
        return true;
    }
    if let Some(prefix) = rulePattern.strip_suffix('*') {
        return toolName.starts_with(prefix);
    }
    rulePattern == toolName
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matchesToolExact() {
        assert!(matchesTool("shell", "shell"));
        assert!(!matchesTool("shell", "readFile"));
    }

    #[test]
    fn matchesToolGlobalWildcard() {
        assert!(matchesTool("*", "shell"));
        assert!(matchesTool("*", "mcp__github__search"));
    }

    #[test]
    fn matchesToolPrefixWildcard() {
        assert!(matchesTool("mcp__github__*", "mcp__github__search_repos"));
        assert!(matchesTool("mcp__github__*", "mcp__github__create_issue"));
        assert!(!matchesTool("mcp__github__*", "mcp__jira__search"));
        assert!(!matchesTool("mcp__github__*", "shell"));
    }

    #[test]
    fn mcpToolPermissionCheck() {
        let mut perms = Permissions::default();
        perms.addRule(Rule {
            tool: "mcp__github__*".into(),
            pattern: None,
            allow: true,
        });

        let action = ToolAction::Mcp {
            qualifiedName: "mcp__github__search_repos".into(),
            args: "{}".into(),
        };
        assert_eq!(perms.check(&action), Verdict::Allow);

        let action2 = ToolAction::Mcp {
            qualifiedName: "mcp__jira__search".into(),
            args: "{}".into(),
        };
        assert_eq!(perms.check(&action2), Verdict::NeedsApproval);
    }

    #[test]
    fn mcpToolExactRule() {
        let mut perms = Permissions::default();
        perms.addRule(Rule {
            tool: "mcp__github__search_repos".into(),
            pattern: None,
            allow: true,
        });

        let action = ToolAction::Mcp {
            qualifiedName: "mcp__github__search_repos".into(),
            args: "{}".into(),
        };
        assert_eq!(perms.check(&action), Verdict::Allow);

        let action2 = ToolAction::Mcp {
            qualifiedName: "mcp__github__create_issue".into(),
            args: "{}".into(),
        };
        assert_eq!(perms.check(&action2), Verdict::NeedsApproval);
    }

    #[test]
    fn firstMatchWins() {
        let mut perms = Permissions::default();
        // Deny all MCP tools, then allow github specifically.
        // Since first match wins, the deny should take precedence.
        perms.addRule(Rule {
            tool: "mcp__*".into(),
            pattern: None,
            allow: false,
        });
        perms.addRule(Rule {
            tool: "mcp__github__*".into(),
            pattern: None,
            allow: true,
        });

        let action = ToolAction::Mcp {
            qualifiedName: "mcp__github__search_repos".into(),
            args: "{}".into(),
        };
        assert_eq!(perms.check(&action), Verdict::Deny);
    }
}
