use std::collections::HashMap;

use tokio::sync::mpsc;

use super::request::buildRiders;
use super::{Session, toolDefsForPermitMode};
use crate::{api, compaction_trigger, lsp, mcp, prompt, web};

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

        self.mcpManager = Some(mgr);
        self.refreshToolDefs().await;
        self.refreshSystemPrompt().await;
    }

    /// Replace the current permission set.
    pub async fn setPermissions(&mut self, permissions: crate::permissions::Permissions) {
        self.permissions = permissions;
        self.refreshToolDefs().await;
        self.refreshSystemPrompt().await;
    }

    /// Change only the default permission mode, preserving rules/source.
    pub async fn setPermitMode(&mut self, mode: crate::permissions::PermitMode) {
        self.permissions.defaultMode = mode;
        self.refreshToolDefs().await;
        self.refreshSystemPrompt().await;
    }

    /// Apply a freshly loaded config to this live session.
    ///
    /// This hot-swaps model clients, prompt/reasoning behavior, context
    /// thresholds, riders, and MCP tool presentation without clearing the
    /// transcript or replacing terminals/background planes. In-flight turns
    /// continue with the config they started with; the next turn sees this.
    pub async fn applyConfig(&mut self, config: &crate::config::Config) -> anyhow::Result<()> {
        let client = api::Client::new(config)?;
        let lastTokens = self.compactionTracker.lastTokens();
        let mut tracker =
            compaction_trigger::Tracker::new(config.heavy.contextWindow, config.compactRatio);
        tracker.updateTokens(lastTokens);

        self.client = client;
        self.config = config.clone();
        self.reasoning =
            self.config
                .heavy
                .reasoning
                .as_ref()
                .map(|r| crate::message::ReasoningConfig {
                    effort: r.effort.clone(),
                    summary: r.summary.clone(),
                });
        self.compactionTracker = tracker;
        self.exaClient = web::ExaClient::new(&self.config.web.searchKey);
        self.riders = buildRiders(&self.config);
        self.refreshToolDefs().await;
        self.refreshSystemPrompt().await;

        tracing::info!(
            model = %self.config.heavy.model,
            provider = %self.config.heavy.provider,
            contextWindow = self.config.heavy.contextWindow,
            "live model config applied"
        );
        Ok(())
    }

    pub(super) async fn refreshToolDefs(&mut self) {
        let includePermissionEscalation = matches!(
            self.permissions.defaultMode,
            crate::permissions::PermitMode::Auto
        );
        let mut defs = toolDefsForPermitMode(&self.permissions.defaultMode);
        if let Some(mgr) = &self.mcpManager {
            let mcpDefs = mgr
                .toolDefs(self.config.heavy.contextWindow, includePermissionEscalation)
                .await;
            if !mcpDefs.is_empty() {
                defs.extend(mcpDefs);
                let mcpToolCount = mgr.toolCount().await;
                tracing::info!(
                    totalTools = defs.len(),
                    mcpTools = mcpToolCount,
                    "merged MCP tools"
                );
            }
        }
        self.tools = defs;
    }

    pub(super) async fn refreshSystemPrompt(&mut self) {
        let contextOptions = crate::config::resolveContextOptions(&self.config.modules);
        let mut systemPrompt = prompt::build(
            self.interface,
            &self.domains,
            self.config.heavy.promptThinking,
            &contextOptions,
        );
        let mcpPrompt = self.mcpPromptSection().await;
        if !mcpPrompt.is_empty() {
            systemPrompt.push_str("\n\n");
            systemPrompt.push_str(&mcpPrompt);
        }

        self.systemPrompt = systemPrompt;
    }

    async fn mcpPromptSection(&self) -> String {
        let Some(mgr) = &self.mcpManager else {
            return String::new();
        };

        let searchMode = mgr
            .isSearchMode(
                self.config.heavy.contextWindow,
                matches!(
                    self.permissions.defaultMode,
                    crate::permissions::PermitMode::Auto
                ),
            )
            .await;
        let statuses = mgr.serverStatuses();
        let registry = mgr.registry().read().await;
        let serverInfos: Vec<prompt::McpServerInfo> = statuses
            .iter()
            .map(|s| {
                let toolCount = registry.search("", Some(&s.name)).len();
                prompt::McpServerInfo {
                    name: s.name.clone(),
                    toolCount,
                    status: format!("{:?}", s.state),
                }
            })
            .collect();

        prompt::mcpSection(&serverInfos, searchMode)
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
        let searchMode = mgr
            .isSearchMode(
                self.config.heavy.contextWindow,
                matches!(
                    self.permissions.defaultMode,
                    crate::permissions::PermitMode::Auto
                ),
            )
            .await;

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
