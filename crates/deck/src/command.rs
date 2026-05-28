#![allow(non_snake_case)]

//! Slash commands — TUI-local commands intercepted before reaching construct.
//!
//! Parses `/command args` input, dispatches to handler functions, and
//! returns output for the agent panel to render. Also provides a command
//! registry for tab completion.
//!
//! Commands that need construct state (session, transcript, compaction) return
//! an `Action` variant. The caller (app.rs) is responsible for dispatching
//! these to construct.
//!
//! # Public API
//! - [`tryHandle`] — parse raw input, returning output if it was a command
//! - [`completions`] — return commands matching a prefix
//! - [`CommandOutput`] — where/how to display the result
//! - [`CommandDef`] — command metadata for completion
//! - [`COMMANDS`] — static registry of all commands
//!
//! # Dependencies
//! None (deck-internal only)
//!
//! # Usage
//! ```ignore
//! match command::tryHandle("/context") {
//!     Some(CommandOutput::Inline(text)) => render(text),
//!     Some(CommandOutput::Action(action)) => dispatchToConstruct(action),
//!     None => sendToConstruct(input),
//! }
//! ```

/// Command metadata for completion and help.
pub struct CommandDef {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
}

/// All registered commands.
pub const COMMANDS: &[CommandDef] = &[
    CommandDef {
        name: "help",
        aliases: &["h", "?"],
        description: "Show help",
    },
    CommandDef {
        name: "context",
        aliases: &["ctx"],
        description: "Show context usage and compaction state",
    },
    CommandDef {
        name: "undo",
        aliases: &[],
        description: "Restore project to before last tool execution",
    },
    CommandDef {
        name: "rewind",
        aliases: &[],
        description: "Rewind conversation; pass a turnId to skip the picker",
    },
    CommandDef {
        name: "resume",
        aliases: &[],
        description: "List or resume a previous session",
    },
    CommandDef {
        name: "clear",
        aliases: &["cls", "new"],
        description: "Clear display and start a fresh session",
    },
    CommandDef {
        name: "forks",
        aliases: &[],
        description: "List saved forks or switch to one",
    },
    CommandDef {
        name: "mcp",
        aliases: &[],
        description: "Show MCP server status and tool counts",
    },
    CommandDef {
        name: "lsp",
        aliases: &[],
        description: "Show LSP server status and install hints",
    },
    CommandDef {
        name: "permissions",
        aliases: &["perms"],
        description: "View and manage permission rules",
    },
    CommandDef {
        name: "model",
        aliases: &["models"],
        description: "View and switch model profiles",
    },
    CommandDef {
        name: "cost",
        aliases: &[],
        description: "Show session and rolling cost breakdown",
    },
    CommandDef {
        name: "tasks",
        aliases: &["jobs"],
        description: "Show background jobs, monitors, and wake schedules",
    },
    CommandDef {
        name: "logs",
        aliases: &["log"],
        description: "Show developer log history",
    },
    CommandDef {
        name: "layout",
        aliases: &[],
        description: "Open layout controls",
    },
];

/// Return commands whose name or aliases start with the given prefix.
pub fn completions(prefix: &str) -> Vec<&'static CommandDef> {
    if prefix.is_empty() {
        return COMMANDS.iter().collect();
    }
    COMMANDS
        .iter()
        .filter(|cmd| {
            cmd.name.starts_with(prefix) || cmd.aliases.iter().any(|a| a.starts_with(prefix))
        })
        .collect()
}

/// Actions that require construct state to execute.
#[derive(Debug)]
pub enum CommandAction {
    /// Show context usage stats from the compaction tracker.
    ShowContext,
    /// Restore project to before the last file-modifying tool.
    Undo,
    /// Rewind conversation. Empty `target` opens the picker; a non-empty
    /// `target` is dispatched directly (used for `/rewind <turnId>`).
    Rewind { target: String },
    /// List saved forks or switch to one by ID.
    Forks { forkId: Option<String> },
    /// List available sessions or resume a specific one.
    Resume { sessionId: Option<String> },
    /// Clear display and start a fresh session.
    Clear,
    /// Show MCP server status.
    Mcp,
    /// Show LSP server status.
    Lsp,
    /// Show permissions panel.
    Permissions,
    /// Show model profile panel.
    Model,
    /// Show cost breakdown.
    ShowCost,
    /// Open the background jobs / monitors / schedules panel.
    Tasks,
    /// Open the developer log history panel.
    Logs,
    /// Open the same layout controls as Ctrl+O.
    ShowLayout,
}

/// How command output should be rendered.
pub enum CommandOutput {
    /// Display inline in the conversation panel as a system-style entry.
    Inline(String),
    /// Requires construct state — caller must dispatch.
    Action(CommandAction),
}

/// Try to handle input as a slash command.
///
/// Returns `None` if the input is not a slash command (should be sent
/// to construct). Returns `Some(output)` for any `/`-prefixed input.
pub fn tryHandle(input: &str) -> Option<CommandOutput> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let mut parts = trimmed[1..].splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim();

    // Match against registry names and aliases.
    let matched = COMMANDS
        .iter()
        .find(|cmd| cmd.name == name || cmd.aliases.contains(&name));

    let result = match matched {
        Some(cmd) => dispatch(cmd.name, args),
        None => CommandOutput::Inline(format!(
            "Unknown command: /{name}. Type /help for available commands."
        )),
    };

    Some(result)
}

/// Dispatch to the handler for a canonical command name.
fn dispatch(name: &str, args: &str) -> CommandOutput {
    match name {
        "help" => executeHelp(args),
        "context" => CommandOutput::Action(CommandAction::ShowContext),
        "undo" => CommandOutput::Action(CommandAction::Undo),
        "rewind" => CommandOutput::Action(CommandAction::Rewind {
            target: args.to_string(),
        }),
        "resume" => {
            let sessionId = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            CommandOutput::Action(CommandAction::Resume { sessionId })
        }
        "forks" => {
            let forkId = if args.is_empty() {
                None
            } else {
                Some(args.to_string())
            };
            CommandOutput::Action(CommandAction::Forks { forkId })
        }
        "clear" => CommandOutput::Action(CommandAction::Clear),
        "mcp" => CommandOutput::Action(CommandAction::Mcp),
        "lsp" => CommandOutput::Action(CommandAction::Lsp),
        "permissions" => CommandOutput::Action(CommandAction::Permissions),
        "model" => CommandOutput::Action(CommandAction::Model),
        "cost" => CommandOutput::Action(CommandAction::ShowCost),
        "tasks" => CommandOutput::Action(CommandAction::Tasks),
        "logs" => CommandOutput::Action(CommandAction::Logs),
        "layout" => CommandOutput::Action(CommandAction::ShowLayout),
        _ => CommandOutput::Inline(format!("/{name} is not yet implemented.")),
    }
}

fn executeHelp(args: &str) -> CommandOutput {
    if args.is_empty() {
        let mut text = String::from("**Commands**\n");
        for cmd in COMMANDS {
            text.push_str(&format!("`/{}` \u{2014} {}\n", cmd.name, cmd.description));
        }
        CommandOutput::Inline(text)
    } else {
        CommandOutput::Inline(format!(
            "No help topic: \"{args}\". Type /help for available commands."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasksCommandAndJobsAliasReturnSameAction() {
        // Slice 6a: /tasks (canonical) and /jobs (alias) both open the
        // jobs panel via CommandAction::Tasks. Guards against the alias
        // silently breaking if either string is edited.
        for input in ["/tasks", "/jobs"] {
            match tryHandle(input) {
                Some(CommandOutput::Action(CommandAction::Tasks)) => {}
                other => panic!(
                    "expected CommandAction::Tasks for {input}, got {:?}",
                    match other {
                        Some(CommandOutput::Action(a)) => format!("Action({a:?})"),
                        Some(CommandOutput::Inline(s)) => format!("Inline({s})"),
                        None => "None".into(),
                    },
                ),
            }
        }
    }

    #[test]
    fn layoutCommandReturnsShowLayoutAction() {
        match tryHandle("/layout") {
            Some(CommandOutput::Action(CommandAction::ShowLayout)) => {}
            other => panic!(
                "expected CommandAction::ShowLayout, got {:?}",
                match other {
                    Some(CommandOutput::Action(a)) => format!("Action({a:?})"),
                    Some(CommandOutput::Inline(s)) => format!("Inline({s})"),
                    None => "None".into(),
                },
            ),
        }
    }

    #[test]
    fn logsCommandAndAliasReturnLogsAction() {
        for input in ["/logs", "/log"] {
            match tryHandle(input) {
                Some(CommandOutput::Action(CommandAction::Logs)) => {}
                other => panic!(
                    "expected CommandAction::Logs for {input}, got {:?}",
                    match other {
                        Some(CommandOutput::Action(a)) => format!("Action({a:?})"),
                        Some(CommandOutput::Inline(s)) => format!("Inline({s})"),
                        None => "None".into(),
                    },
                ),
            }
        }
    }

    #[test]
    fn modelCommandAndAliasReturnModelAction() {
        for input in ["/model", "/models"] {
            match tryHandle(input) {
                Some(CommandOutput::Action(CommandAction::Model)) => {}
                other => panic!(
                    "expected CommandAction::Model for {input}, got {:?}",
                    match other {
                        Some(CommandOutput::Action(a)) => format!("Action({a:?})"),
                        Some(CommandOutput::Inline(s)) => format!("Inline({s})"),
                        None => "None".into(),
                    },
                ),
            }
        }
    }

    #[test]
    fn tasksAppearsInCompletions() {
        let matches = completions("ta");
        assert!(
            matches.iter().any(|c| c.name == "tasks"),
            "completions for `ta` should include /tasks",
        );
        let matches = completions("jo");
        assert!(
            matches.iter().any(|c| c.name == "tasks"),
            "completions for `jo` should match the /jobs alias and surface /tasks",
        );
    }
}
