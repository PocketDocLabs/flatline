use std::collections::HashMap;

use tokio::sync::mpsc;

use super::Session;
use crate::message::Message;
use crate::{lsp, mcp, prompt};

impl Session {
    /// Initialize MCP server connections.
    ///
    /// Starts all configured servers in parallel and merges their tool
    /// definitions into the session's tool list. Failures are logged
    /// but not fatal — the session continues without the failed servers.
    ///
    /// Args:
    ///     servers: Server name → config map from the config file.
    pub async fn initMcp(&mut self, servers: HashMap<String, mcp::config::ServerConfig>) {
        if servers.is_empty() {
            return;
        }

        let (elicitationTx, _elicitationRx) = mpsc::channel(8);
        let mut mgr = mcp::McpManager::new(elicitationTx);
        self.mcpConfigs = servers.clone();
        let statuses = mgr.startAll(servers).await;

        for status in &statuses {
            let stateStr = format!("{:?}", status.state);
            tracing::info!(
                server = %status.name,
                state = %stateStr,
                "MCP server status"
            );
        }

        let contextBudget = self.config.heavy.contextWindow;
        let mcpDefs = mgr.toolDefs(contextBudget).await;
        if !mcpDefs.is_empty() {
            self.tools.extend(mcpDefs);
            let mcpToolCount = mgr.toolCount().await;
            tracing::info!(
                totalTools = self.tools.len(),
                mcpTools = mcpToolCount,
                "merged MCP tools"
            );
        }

        let searchMode = mgr.isSearchMode(contextBudget).await;
        let serverInfos: Vec<prompt::McpServerInfo> = statuses
            .iter()
            .map(|s| prompt::McpServerInfo {
                name: s.name.clone(),
                toolCount: 0,
                status: format!("{:?}", s.state),
            })
            .collect();

        let mcpPrompt = prompt::mcpSection(&serverInfos, searchMode);
        if !mcpPrompt.is_empty()
            && let Some(Message::System { content }) = self.history.first_mut()
        {
            content.push_str("\n\n");
            content.push_str(&mcpPrompt);
        }

        self.mcpManager = Some(mgr);
    }

    /// Replace the current permission set.
    pub fn setPermissions(&mut self, permissions: crate::permissions::Permissions) {
        self.permissions = permissions;
    }

    /// Get permissions data for the /permissions panel.
    pub fn permissionsStatusData(
        &self,
    ) -> (
        crate::permissions::PermitMode,
        Vec<crate::permissions::Rule>,
        crate::permissions::PermissionsSource,
        String,
    ) {
        let configPath = self
            .config
            .projectRoot
            .as_ref()
            .map(|r| r.join(".flatline/config.toml").display().to_string())
            .unwrap_or_else(|| "~/.config/flatline/config.toml".into());
        (
            self.permissions.defaultMode.clone(),
            self.permissions.rules.clone(),
            self.permissions.source,
            configPath,
        )
    }

    /// Gather structured MCP status data for the TUI panel.
    pub async fn mcpStatusData(
        &self,
    ) -> (
        Vec<(String, String, usize, Vec<(String, String)>, String)>,
        usize,
        bool,
        String,
    ) {
        let configPath = ".mcp.json".to_string();

        let mgr = match &self.mcpManager {
            Some(m) => m,
            None => return (Vec::new(), 0, false, configPath),
        };

        let statuses = mgr.serverStatuses();
        let totalTools = mgr.toolCount().await;
        let searchMode = mgr.isSearchMode(self.config.heavy.contextWindow).await;

        let registry = mgr.registry().read().await;

        let servers = statuses
            .iter()
            .map(|s| {
                let stateStr = format!("{:?}", s.state);

                let tools: Vec<(String, String)> = registry
                    .search("", Some(&s.name))
                    .iter()
                    .map(|r| (r.qualifiedName.clone(), r.description.clone()))
                    .collect();

                let toolCount = tools.len();

                let transport = self
                    .mcpConfigs
                    .get(&s.name)
                    .map(|cfg| {
                        if let Some(ref cmd) = cfg.command {
                            let args = if cfg.args.is_empty() {
                                String::new()
                            } else {
                                format!(" {}", cfg.args.join(" "))
                            };
                            format!("stdio: {cmd}{args}")
                        } else if let Some(ref url) = cfg.url {
                            format!("http: {url}")
                        } else {
                            "unknown".into()
                        }
                    })
                    .unwrap_or_else(|| "unknown".into());

                (s.name.clone(), stateStr, toolCount, tools, transport)
            })
            .collect();

        (servers, totalTools, searchMode, configPath)
    }

    /// Gracefully shut down all MCP server connections.
    pub async fn shutdownMcp(&mut self) {
        if let Some(ref mut mgr) = self.mcpManager {
            mgr.shutdown().await;
        }
    }

    /// Gracefully shut down all LSP server connections.
    pub async fn shutdownLsp(&mut self) {
        self.lspManager.shutdown().await;
    }

    /// Get LSP server status data for the /lsp panel.
    pub fn lspStatusData(&self) -> Vec<lsp::FullServerStatus> {
        self.lspManager.allServerStatuses()
    }
}
