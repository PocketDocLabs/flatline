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

/// Read-only tools. These read state without side effects and are
/// unconditionally auto-approved by [`Permissions::check`] after the
/// rules loop. [`Permissions::allowReadOnly`] also emits explicit rules
/// for documentation/serialization, but the approval no longer depends
/// on those rules being present.
const READ_ONLY_TOOLS: &[&str] = &[
    "readFile",
    "glob",
    "grep",
    "listDir",
    "structSearch",
    "diff",
    "fuzzyFind",
    "fileOutline",
    "viewSymbol",
    "relatedFiles",
    "shellHistory",
    "readOutput",
    "searchOutput",
    "readTerminal",
    "terminalList",
    "terminalRunList",
    "terminalRunStop",
    "jobOutput",
    "waitForSubagent",
    "jobList",
    "monitorList",
    "webSearch",
    "webFetch",
    "webSimilar",
    "historySearch",
    "historyFetch",
    "diagnostics",
    "cronList",
];

/// Response from the supervisor (TUI or parent agent) to a permission prompt.
#[derive(Debug, Clone)]
pub enum PermitResponse {
    /// Allow this one invocation.
    Allow,
    /// Allow and persist an allow rule for the given pattern.
    AlwaysAllow { pattern: String },
    /// Deny this invocation.
    Deny,
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

        ToolAction::CopyFile { dest, .. } | ToolAction::MoveFile { dest, .. } => pathPatterns(dest),

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

        // Shell: progressively shorter token prefixes. Same pattern shape
        // for foreground and background — `runInBackground` only changes
        // where the result goes, not what's executed.
        ToolAction::Shell { command, .. } => shellPatterns(command),

        // Read-only / management tools — exact tool name only. No
        // user-typed pattern can broaden these incorrectly.
        ToolAction::JobList => vec!["jobList".into()],
        ToolAction::WaitForSubagent { .. } => vec!["waitForSubagent".into()],
        ToolAction::JobOutput { .. } => vec!["jobOutput".into()],
        ToolAction::JobStop { .. } => vec!["jobStop".into()],
        ToolAction::TerminalRunList => vec!["terminalRunList".into()],
        ToolAction::TerminalRunStop { .. } => vec!["terminalRunStop".into()],
        ToolAction::MonitorList => vec!["monitorList".into()],
        ToolAction::MonitorStop { .. } => vec!["monitorStop".into()],

        // Monitors attach to an existing terminal stream; they do not
        // execute commands, so scope by tool name only.
        ToolAction::Monitor { .. } => vec!["monitor".into()],

        // Wake registry tools. Each creation tool's keyArg is the most
        // identifying scope: cron spec, delay duration, watched path.
        ToolAction::ScheduleWakeup { .. } => vec!["scheduleWakeup".into()],
        ToolAction::CronCreate { spec, .. } => {
            vec![spec.clone(), "cronCreate".into()]
        }
        ToolAction::CronList => vec!["cronList".into()],
        ToolAction::CronDelete { .. } => vec!["cronDelete".into()],
        ToolAction::FileWatch { path, .. } => {
            let mut patterns = pathPatterns(path);
            patterns.push("fileWatch".into());
            patterns
        }

        // Terminal mutators. Suggest the specific terminal name first
        // (so a click-through is the narrower rule) then the tool-wide
        // fallback. Without these, the permit prompt would have no
        // suggestions, hide the custom field by default, and trap the
        // user in a state where Shift+A says "type one in the custom
        // field" but no keystrokes are accepted.
        ToolAction::TerminalSpawn { name } => {
            let mut patterns = Vec::new();
            if let Some(n) = name.as_deref().filter(|s| !s.is_empty()) {
                patterns.push(n.to_string());
            }
            patterns.push("terminalSpawn".into());
            patterns
        }
        ToolAction::TerminalSwitch { name } => {
            let mut patterns = Vec::new();
            if !name.is_empty() {
                patterns.push(name.clone());
            }
            patterns.push("terminalSwitch".into());
            patterns
        }
        ToolAction::TerminalKill { name } => {
            let mut patterns = Vec::new();
            if !name.is_empty() {
                patterns.push(name.clone());
            }
            patterns.push("terminalKill".into());
            patterns
        }

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
        // Wake registry tools schedule autonomous future LLM calls —
        // conservatively MinorMod so the confirm UI surfaces them
        // above read-tier (file reads/grep/etc.).
        ToolAction::ScheduleWakeup { .. }
        | ToolAction::CronCreate { .. }
        | ToolAction::CronDelete { .. }
        | ToolAction::FileWatch { .. } => ShellImpact::MinorMod,
        // cronList is read-only; let it fall through to the bottom.
        // Monitors subscribe to terminal output and schedule future wake
        // turns — still a session-visible behavior change.
        ToolAction::Monitor { .. } => ShellImpact::MinorMod,
        ToolAction::MonitorStop { .. } => ShellImpact::MinorMod,
        ToolAction::WriteFile { .. }
        | ToolAction::EditFile { .. }
        | ToolAction::MultiEdit { .. }
        | ToolAction::CopyFile { .. }
        | ToolAction::MoveFile { .. }
        | ToolAction::MakeDirs { .. } => ShellImpact::MinorMod,
        ToolAction::DeleteFile { .. } => ShellImpact::Delete,
        ToolAction::Mcp { .. } => ShellImpact::MinorMod,
        ToolAction::Task { .. } => ShellImpact::MinorMod,
        // jobStop kills a process group — surface as MinorMod so the
        // confirm UI doesn't downgrade the prompt.
        ToolAction::JobStop { .. } => ShellImpact::MinorMod,
        // terminalRunStop interrupts the visible terminal owning the run.
        ToolAction::TerminalRunStop { .. } => ShellImpact::MinorMod,
        // terminalSpawn allocates a real PTY — session-visible state
        // change beyond what Read covers. MinorMod tier.
        ToolAction::TerminalSpawn { .. } => ShellImpact::MinorMod,
        // terminalSwitch changes the agent's default routing for
        // subsequent shell calls — also session-visible state. MinorMod.
        ToolAction::TerminalSwitch { .. } => ShellImpact::MinorMod,
        // terminalKill destroys a visible PTY and any process tree
        // running in it — closer to a delete than an edit.
        ToolAction::TerminalKill { .. } => ShellImpact::Delete,
        _ => ShellImpact::Read,
    }
}

/// Get the user-facing explanation from a tool action. Present for any
/// tool whose schema requires an `explanation` field — currently just
/// `shell` (foreground and background share the same shape).
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
        if pos > 0
            && let Some(gpos) = normalized[..pos].rfind('/')
        {
            patterns.push(normalized[..=gpos].to_string());
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
        .filter(|t| !t.contains('=') || t.starts_with('-'))
        .collect();

    if tokens.is_empty() {
        return patterns;
    }

    let binary = tokens[0];

    // A subcommand is an alphabetic word (not a flag, path, quoted string, or redirect).
    let subcommand = tokens.get(1).and_then(|t| {
        let first = t.chars().next()?;
        if first.is_alphabetic()
            && t.chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
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
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PermitMode {
    /// Ask the supervisor (TUI user, parent agent, etc.) and wait for response.
    Ask,
    /// Ask an automatic reviewer first; escalate to the supervisor only when
    /// the reviewer explicitly grants a raise-to-user retry.
    Auto,
}

impl<'de> Deserialize<'de> for PermitMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "ask" => Ok(PermitMode::Ask),
            "auto" => Ok(PermitMode::Auto),
            _ => Ok(PermitMode::Ask),
        }
    }
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

    /// Create permissions that auto-approve everything.
    pub fn allowAll() -> Self {
        Self {
            defaultMode: PermitMode::Auto,
            rules: vec![Rule {
                tool: "*".into(),
                pattern: None,
                allow: true,
            }],
            source: PermissionsSource::BuiltIn,
        }
    }

    /// Create permissions that auto-approve read-only tools and prompt
    /// for anything else.
    pub fn allowReadOnly() -> Self {
        Self {
            defaultMode: PermitMode::Ask,
            rules: READ_ONLY_TOOLS
                .iter()
                .map(|&tool| Rule {
                    tool: tool.into(),
                    pattern: None,
                    allow: true,
                })
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

        // Read-only tools are unconditionally allowed. They read state
        // without side effects, so there is no security reason to block
        // or prompt for them — regardless of permit mode or config.
        if isReadOnlyTool(action) {
            return Verdict::Allow;
        }

        Verdict::NeedsApproval
    }
}

/// Read-only tools have no side effects — never prompt or deny them,
/// regardless of configured permit mode or rules (explicit deny rules
/// still take precedence, caught by the rules loop in `check`).
fn isReadOnlyTool(action: &ToolAction) -> bool {
    let (toolName, _) = actionKey(action);
    READ_ONLY_TOOLS.contains(&toolName)
        || matches!(
            action,
            ToolAction::Shell {
                impact: crate::tool::ShellImpact::Read,
                ..
            }
        )
}

/// Decide how a user-supplied pattern from the permit prompt should be
/// stored on a persisted [`Rule`] for the given action. Most tools want
/// the pattern to remain a substring constraint, but tools whose
/// [`actionKey`] returns an empty keyArg (e.g. `jobList`, `jobOutput`,
/// `jobStop`, `terminalList`) need the pattern stripped to `None` —
/// otherwise the substring matcher in [`Permissions::check`] would never
/// fire (`"".contains("jobStop")` is false).
///
/// The "exact tool match" normalization only fires when the pattern
/// equals the tool name AND the tool exposes no keyArg, so a user typing
/// `"shell"` as a custom pattern for the `shell` tool is preserved as a
/// real substring rule rather than silently widened to allow-all.
pub fn normalizeRulePattern(action: &ToolAction, pattern: &str) -> Option<String> {
    let (toolName, keyArg) = actionKey(action);
    if keyArg.is_empty() && pattern == toolName {
        None
    } else {
        Some(pattern.to_string())
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
        ToolAction::ShellHistory { .. } => ("shellHistory", ""),
        ToolAction::ReadOutput { .. } => ("readOutput", ""),
        ToolAction::SearchOutput { pattern, .. } => ("searchOutput", pattern),
        ToolAction::ReadTerminal { .. } => ("readTerminal", ""),
        ToolAction::TerminalSpawn { name } => ("terminalSpawn", name.as_deref().unwrap_or("")),
        ToolAction::TerminalSwitch { name } => ("terminalSwitch", name),
        ToolAction::TerminalKill { name } => ("terminalKill", name),
        ToolAction::TerminalList => ("terminalList", ""),
        ToolAction::TerminalRunList => ("terminalRunList", ""),
        ToolAction::TerminalRunStop { .. } => ("terminalRunStop", ""),
        ToolAction::JobOutput { .. } => ("jobOutput", ""),
        ToolAction::JobStop { .. } => ("jobStop", ""),
        ToolAction::Monitor { .. } => ("monitor", ""),
        ToolAction::MonitorStop { .. } => ("monitorStop", ""),
        ToolAction::MonitorList => ("monitorList", ""),
        ToolAction::JobList => ("jobList", ""),
        ToolAction::WaitForSubagent { .. } => ("waitForSubagent", ""),
        ToolAction::ScheduleWakeup { .. } => ("scheduleWakeup", ""),
        ToolAction::CronCreate { spec, .. } => ("cronCreate", spec),
        ToolAction::CronList => ("cronList", ""),
        ToolAction::CronDelete { .. } => ("cronDelete", ""),
        ToolAction::FileWatch { path, .. } => ("fileWatch", path),
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
        ToolAction::Mcp {
            qualifiedName,
            args,
        } => (qualifiedName, args),
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

    #[test]
    fn readOnlyShellAutoApprovedUnderReadOnly() {
        use crate::tool::ShellImpact;
        let perms = Permissions::allowReadOnly();
        let action = ToolAction::Shell {
            command: "cat foo.txt".into(),
            explanation: "read a file".into(),
            impact: ShellImpact::Read,
            timeout: None,
            terminal: None,
            runInBackground: false,
        };
        assert_eq!(perms.check(&action), Verdict::Allow);
    }

    #[test]
    fn mutatingShellStillNeedsApprovalUnderReadOnly() {
        use crate::tool::ShellImpact;
        let perms = Permissions::allowReadOnly();
        let action = ToolAction::Shell {
            command: "cargo fmt".into(),
            explanation: "format the project".into(),
            impact: ShellImpact::MinorMod,
            timeout: None,
            terminal: None,
            runInBackground: false,
        };
        assert_eq!(perms.check(&action), Verdict::NeedsApproval);
    }

    #[test]
    fn readOnlyShellAutoApprovalSurvivesPersistedPresetRules() {
        use crate::tool::ShellImpact;
        let persisted = Permissions {
            defaultMode: PermitMode::Ask,
            rules: Permissions::allowReadOnly().rules,
            source: PermissionsSource::Project,
        };
        let action = ToolAction::Shell {
            command: "git status --short".into(),
            explanation: "inspect working tree status".into(),
            impact: ShellImpact::Read,
            timeout: None,
            terminal: None,
            runInBackground: false,
        };
        assert_eq!(persisted.check(&action), Verdict::Allow);
    }

    #[test]
    fn explicitShellDenyBeatsReadOnlyShellAutoApproval() {
        use crate::tool::ShellImpact;
        let mut perms = Permissions::allowReadOnly();
        perms.rules.insert(
            0,
            Rule {
                tool: "shell".into(),
                pattern: None,
                allow: false,
            },
        );
        let action = ToolAction::Shell {
            command: "git status --short".into(),
            explanation: "inspect working tree status".into(),
            impact: ShellImpact::Read,
            timeout: None,
            terminal: None,
            runInBackground: false,
        };
        assert_eq!(perms.check(&action), Verdict::Deny);
    }

    #[test]
    fn terminalListAutoApprovedUnderReadOnly() {
        // Inventory is harmless — terminalList should be auto-approved.
        let perms = Permissions::allowReadOnly();
        assert_eq!(perms.check(&ToolAction::TerminalList), Verdict::Allow,);
    }

    #[test]
    fn readOnlyToolsAutoApproveWithEmptyRulesAndAskMode() {
        // Regression: bare [permissions] with defaultMode="ask" and no
        // rules must not force read-only tools to prompt the user.
        let perms = Permissions::askForEverything();
        for toolName in ["readFile", "grep", "glob", "listDir"] {
            let action = match toolName {
                "readFile" => ToolAction::ReadFile {
                    path: "foo.txt".into(),
                    offset: None,
                    limit: None,
                    anchor: None,
                },
                "grep" => ToolAction::Grep {
                    pattern: "x".into(),
                    path: None,
                    include: None,
                    fileType: None,
                    outputMode: "files".into(),
                    caseSensitive: None,
                    contextLines: None,
                    multiline: false,
                },
                "glob" => ToolAction::Glob {
                    pattern: "*.rs".into(),
                    path: None,
                    metadata: false,
                },
                "listDir" => ToolAction::ListDir {
                    path: ".".into(),
                    depth: 1,
                    offset: 0,
                    limit: 50,
                    metadata: false,
                },
                _ => unreachable!(),
            };
            assert_eq!(
                perms.check(&action),
                Verdict::Allow,
                "read-only tool {toolName} must auto-approve even with ask mode and no rules"
            );
        }
    }

    #[test]
    fn normalizeStripsPatternForEmptyKeyArgTools() {
        // jobStop has empty keyArg — a Some(pattern) rule would never
        // fire, so the helper must strip the pattern.
        let action = ToolAction::JobStop { jobId: 1 };
        assert_eq!(normalizeRulePattern(&action, "jobStop"), None);
    }

    #[test]
    fn normalizePreservesPatternForShellEvenWhenItEqualsToolName() {
        use crate::tool::ShellImpact;
        // The danger case: a user types "shell" as a custom pattern for
        // the shell tool. That's a legitimate substring constraint —
        // never widen it to allow-all.
        let action = ToolAction::Shell {
            command: "shell --help".into(),
            explanation: "test".into(),
            impact: ShellImpact::Read,
            timeout: None,
            terminal: None,
            runInBackground: false,
        };
        assert_eq!(normalizeRulePattern(&action, "shell"), Some("shell".into()),);

        // A normal substring also stays as-is.
        assert_eq!(
            normalizeRulePattern(&action, "git status"),
            Some("git status".into()),
        );
    }

    #[test]
    fn normalizePreservesPatternForBackgroundShellEvenWhenItEqualsToolName() {
        use crate::tool::ShellImpact;
        // Background shell calls share the keyArg shape with foreground
        // shell — any user-typed pattern is a real substring constraint.
        let action = ToolAction::Shell {
            command: "shell".into(),
            explanation: "test".into(),
            impact: ShellImpact::Read,
            timeout: None,
            terminal: None,
            runInBackground: true,
        };
        assert_eq!(normalizeRulePattern(&action, "shell"), Some("shell".into()),);
    }

    #[test]
    fn terminalMutatorsHaveSuggestedPatterns() {
        // Without these suggestions the permit prompt would hang the
        // user in a dead state where Shift+A points at the custom field
        // but the keystroke pipeline doesn't accept input. Each mutator
        // gets both the specific-name pattern and a tool-wide fallback.
        let kill = suggestPatterns(&ToolAction::TerminalKill {
            name: "build".into(),
        });
        assert_eq!(kill, vec!["build", "terminalKill"]);

        let switch = suggestPatterns(&ToolAction::TerminalSwitch {
            name: "logs".into(),
        });
        assert_eq!(switch, vec!["logs", "terminalSwitch"]);

        let spawnNamed = suggestPatterns(&ToolAction::TerminalSpawn {
            name: Some("worker".into()),
        });
        assert_eq!(spawnNamed, vec!["worker", "terminalSpawn"]);

        // Spawn with no name has no specific suggestion — just tool-wide.
        let spawnAuto = suggestPatterns(&ToolAction::TerminalSpawn { name: None });
        assert_eq!(spawnAuto, vec!["terminalSpawn"]);
    }

    #[test]
    fn terminalMutatorsImpactAboveRead() {
        use crate::tool::ShellImpact;
        // Read-tier in the confirm UI implies "harmless." None of these
        // are harmless: spawn/switch mutate routing state, kill destroys
        // a PTY and any work in it.
        assert!(matches!(
            toolImpact(&ToolAction::TerminalSpawn { name: None }),
            ShellImpact::MinorMod,
        ));
        assert!(matches!(
            toolImpact(&ToolAction::TerminalSwitch {
                name: "main".into()
            }),
            ShellImpact::MinorMod,
        ));
        assert!(matches!(
            toolImpact(&ToolAction::TerminalKill {
                name: "build".into()
            }),
            ShellImpact::Delete,
        ));
        // Inventory is genuinely read-only.
        assert!(matches!(
            toolImpact(&ToolAction::TerminalList),
            ShellImpact::Read,
        ));
    }

    #[test]
    fn backgroundShellExplanationForwarded() {
        use crate::tool::ShellImpact;
        // The agent supplies an `explanation` on every shell call,
        // foreground or background — the approval UI must surface it
        // identically in both cases.
        let action = ToolAction::Shell {
            command: "cargo build --release".into(),
            explanation: "build the release binary".into(),
            impact: ShellImpact::MinorMod,
            timeout: None,
            terminal: None,
            runInBackground: true,
        };
        assert_eq!(toolExplanation(&action), Some("build the release binary"),);
        // Empty explanation still reports None so the UI doesn't render
        // an empty section.
        let empty = ToolAction::Shell {
            command: "ls".into(),
            explanation: String::new(),
            impact: ShellImpact::Read,
            timeout: None,
            terminal: None,
            runInBackground: true,
        };
        assert_eq!(toolExplanation(&empty), None);
    }

    #[test]
    fn toolWideRuleMatchesEmptyKeyArgTools() {
        // `jobStop` / `jobOutput` / `jobList` have an empty keyArg, so
        // a rule with `pattern: Some("jobStop")` would never fire
        // (`"".contains("jobStop")` is false). Tool-wide rules
        // (pattern: None) must work for these.
        let mut perms = Permissions::askForEverything();
        perms.addRule(Rule {
            tool: "jobStop".into(),
            pattern: None,
            allow: true,
        });
        assert_eq!(
            perms.check(&ToolAction::JobStop { jobId: 1 }),
            Verdict::Allow,
        );

        // Sanity: a pattern: Some("jobStop") rule must NOT match the
        // same action — that's the bug we normalized at the construction
        // site. This test guarantees future refactors don't regress the
        // matcher into pretending the substring match works.
        let mut bad = Permissions::askForEverything();
        bad.addRule(Rule {
            tool: "jobStop".into(),
            pattern: Some("jobStop".into()),
            allow: true,
        });
        assert_eq!(
            bad.check(&ToolAction::JobStop { jobId: 1 }),
            Verdict::NeedsApproval,
        );
    }
}
