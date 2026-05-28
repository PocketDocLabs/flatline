#![allow(non_snake_case)]

//! MCP prompt management — list, get, slash command generation.
//!
//! MCP servers can expose reusable prompt templates that become
//! `/mcp__server__prompt` slash commands in the TUI.
//!
//! # Public API
//! - [`PromptManager`] — tracks prompts across all connected servers
//! - [`PromptCommand`] — a prompt exposed as a slash command
//!
//! # Dependencies
//! None.

use std::collections::HashMap;

/// Information about a single MCP prompt.
#[derive(Debug, Clone)]
pub struct PromptInfo {
    pub serverName: String,
    pub name: String,
    pub qualifiedName: String,
    pub description: String,
    pub args: Vec<PromptArgInfo>,
}

/// A prompt argument definition.
#[derive(Debug, Clone)]
pub struct PromptArgInfo {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// An MCP prompt exposed as a slash command.
#[derive(Debug, Clone)]
pub struct PromptCommand {
    /// Slash command name: `/mcp__server__prompt`.
    pub slashName: String,
    pub description: String,
    pub args: Vec<PromptArgInfo>,
}

/// Central tracker for MCP prompts across all connected servers.
pub struct PromptManager {
    /// serverName → list of prompts.
    prompts: HashMap<String, Vec<PromptInfo>>,
}

impl Default for PromptManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptManager {
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
        }
    }

    /// Register prompts discovered from a server.
    pub fn registerServer(&mut self, serverName: &str, prompts: Vec<PromptInfo>) {
        tracing::debug!(
            server = %serverName,
            count = prompts.len(),
            "registered MCP prompts"
        );
        self.prompts.insert(serverName.to_string(), prompts);
    }

    /// Remove all prompts for a server.
    pub fn unregisterServer(&mut self, serverName: &str) {
        self.prompts.remove(serverName);
    }

    /// Generate slash commands from all registered prompts.
    pub fn slashCommands(&self) -> Vec<PromptCommand> {
        self.prompts
            .values()
            .flat_map(|prompts| {
                prompts.iter().map(|p| PromptCommand {
                    slashName: format!("/{}", p.qualifiedName),
                    description: p.description.clone(),
                    args: p.args.clone(),
                })
            })
            .collect()
    }

    /// Look up a prompt by qualified name.
    pub fn resolve(&self, qualifiedName: &str) -> Option<&PromptInfo> {
        self.prompts
            .values()
            .flat_map(|v| v.iter())
            .find(|p| p.qualifiedName == qualifiedName)
    }

    /// Total number of prompts.
    pub fn totalCount(&self) -> usize {
        self.prompts.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makePrompt(server: &str, name: &str) -> PromptInfo {
        PromptInfo {
            serverName: server.into(),
            name: name.into(),
            qualifiedName: format!("mcp__{server}__{name}"),
            description: format!("{name} prompt"),
            args: vec![PromptArgInfo {
                name: "input".into(),
                description: "The input text".into(),
                required: true,
            }],
        }
    }

    #[test]
    fn registerAndResolve() {
        let mut mgr = PromptManager::new();
        mgr.registerServer("github", vec![makePrompt("github", "review")]);

        assert_eq!(mgr.totalCount(), 1);
        let prompt = mgr.resolve("mcp__github__review").unwrap();
        assert_eq!(prompt.name, "review");
    }

    #[test]
    fn slashCommandsGenerated() {
        let mut mgr = PromptManager::new();
        mgr.registerServer("github", vec![makePrompt("github", "review")]);
        mgr.registerServer("jira", vec![makePrompt("jira", "create_ticket")]);

        let commands = mgr.slashCommands();
        assert_eq!(commands.len(), 2);

        let names: Vec<&str> = commands.iter().map(|c| c.slashName.as_str()).collect();
        assert!(names.contains(&"/mcp__github__review"));
        assert!(names.contains(&"/mcp__jira__create_ticket"));
    }

    #[test]
    fn unregisterCleansUp() {
        let mut mgr = PromptManager::new();
        mgr.registerServer("github", vec![makePrompt("github", "review")]);
        mgr.unregisterServer("github");
        assert_eq!(mgr.totalCount(), 0);
    }
}
