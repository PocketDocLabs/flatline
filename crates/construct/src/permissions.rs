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
//!
//! # Dependencies
//! None.

use crate::tool::ToolAction;

/// What to do when a tool call isn't pre-approved.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone)]
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

/// Permission configuration for a session.
#[derive(Debug, Clone)]
pub struct Permissions {
    /// What to do when no rule matches and approval is needed.
    pub defaultMode: PermitMode,
    /// Ordered list of rules. First match wins.
    pub rules: Vec<Rule>,
}

impl Default for Permissions {
    /// Default: deny everything, ask the supervisor.
    fn default() -> Self {
        Self {
            defaultMode: PermitMode::Ask,
            rules: Vec::new(),
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
        }
    }

    /// Create permissions that auto-approve read-only tools.
    pub fn allowReadOnly() -> Self {
        Self {
            defaultMode: PermitMode::Ask,
            rules: vec![
                Rule {
                    tool: "readFile".into(),
                    pattern: None,
                    allow: true,
                },
            ],
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
                if !keyArg.contains(pattern.as_str()) {
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
fn actionKey(action: &ToolAction) -> (&str, &str) {
    match action {
        ToolAction::Shell { command } => ("shell", command),
        ToolAction::ReadFile { path, .. } => ("readFile", path),
        ToolAction::WriteFile { path, .. } => ("writeFile", path),
        ToolAction::EditFile { path, .. } => ("editFile", path),
        ToolAction::ShellHistory => ("shellHistory", ""),
        ToolAction::ReadOutput { .. } => ("readOutput", ""),
        ToolAction::SearchOutput { pattern, .. } => ("searchOutput", pattern),
        ToolAction::ReadTerminal { .. } => ("readTerminal", ""),
        ToolAction::Unknown { name, args } => (name, args),
    }
}

/// Check if a rule's tool field matches a tool name.
fn matchesTool(rulePattern: &str, toolName: &str) -> bool {
    rulePattern == "*" || rulePattern == toolName
}
