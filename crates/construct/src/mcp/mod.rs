#![allow(non_snake_case)]

//! MCP (Model Context Protocol) subsystem.
//!
//! Provides full MCP client support: stdio + HTTP transports, tool discovery
//! and execution, resources, prompts, notifications, sampling, elicitation,
//! context budgeting (tool search), and serve mode.
//!
//! # Module Structure
//! - [`config`] — server configuration and TOML parsing
//! - [`schema`] — JSON schema sanitization and name qualification
//! - [`output`] — output limits and token estimation
//! - [`connection`] — per-server connection lifecycle
//! - [`handler`] — ClientHandler impl for rmcp
//! - [`registry`] — tool registry and routing table
//! - [`search`] — tool search meta-tool (context budgeting)
//!
//! # Public API
//! - [`McpManager`] — central coordinator for all MCP servers
//!
//! # Dependencies
//! `rmcp`, `sha1_smol`, `tokio-util`

pub mod config;
pub mod connection;
pub mod handler;
pub mod output;
pub mod prompt;
pub mod registry;
pub mod resource;
pub mod schema;
pub mod search;
pub mod serve;

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Notify, RwLock, mpsc};
use tokio::task::JoinSet;

use crate::message::ToolDef;

use connection::{ConnectionState, ServerConnection};
use handler::{ElicitationRequest, FlatlineHandler};
use registry::ToolRegistry;

/// Status event emitted during server startup.
pub struct ServerStatus {
    pub name: String,
    pub state: ConnectionState,
}

/// Central coordinator for all MCP server connections.
///
/// Owns connections, the tool registry, and notification tasks.
/// Provides the unified API consumed by `Session`.
pub struct McpManager {
    connections: HashMap<String, ServerConnection>,
    registry: Arc<RwLock<ToolRegistry>>,
    elicitationTx: mpsc::Sender<ElicitationRequest>,
    toolsChanged: Arc<Notify>,
    resourcesChanged: Arc<Notify>,
    promptsChanged: Arc<Notify>,
}

impl McpManager {
    /// Create a new McpManager.
    ///
    /// Does NOT start connections — call `startAll()` separately.
    pub fn new(elicitationTx: mpsc::Sender<ElicitationRequest>) -> Self {
        Self {
            connections: HashMap::new(),
            registry: Arc::new(RwLock::new(ToolRegistry::new())),
            elicitationTx,
            toolsChanged: Arc::new(Notify::new()),
            resourcesChanged: Arc::new(Notify::new()),
            promptsChanged: Arc::new(Notify::new()),
        }
    }

    /// Start all enabled servers in parallel.
    ///
    /// Returns a status for each server (connected or failed).
    /// Failures are logged but not fatal — the session continues.
    pub async fn startAll(
        &mut self,
        servers: HashMap<String, config::ServerConfig>,
    ) -> Vec<ServerStatus> {
        let enabledServers: Vec<(String, config::ServerConfig)> = servers
            .into_iter()
            .filter(|(name, cfg)| {
                if !cfg.enabled {
                    tracing::debug!(server = %name, "MCP server disabled, skipping");
                    false
                } else {
                    true
                }
            })
            .collect();

        if enabledServers.is_empty() {
            return Vec::new();
        }

        // Start all servers in parallel.
        let mut joinSet = JoinSet::new();

        for (name, serverConfig) in enabledServers {
            let elicitationTx = self.elicitationTx.clone();
            let toolsChanged = self.toolsChanged.clone();
            let resourcesChanged = self.resourcesChanged.clone();
            let promptsChanged = self.promptsChanged.clone();

            joinSet.spawn(async move {
                let handler = FlatlineHandler {
                    elicitationTx,
                    toolsChanged,
                    resourcesChanged,
                    promptsChanged,
                    serverName: name.clone(),
                };

                let mut conn = ServerConnection::new(name.clone(), serverConfig);
                let result = conn.connect(handler).await;
                (name, conn, result)
            });
        }

        let mut statuses = Vec::new();

        while let Some(result) = joinSet.join_next().await {
            match result {
                Ok((name, conn, _connectResult)) => {
                    let state = conn.state().clone();
                    statuses.push(ServerStatus {
                        name: name.clone(),
                        state: state.clone(),
                    });

                    // If connected, list tools and register them.
                    if matches!(state, ConnectionState::Connected) {
                        match conn.listTools().await {
                            Ok(tools) => {
                                let count = tools.len();
                                self.registry.write().await.registerServer(&name, tools);
                                tracing::info!(
                                    server = %name,
                                    tools = count,
                                    "registered MCP tools"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    server = %name,
                                    error = %e,
                                    "failed to list tools"
                                );
                            }
                        }
                    }

                    self.connections.insert(name, conn);
                }
                Err(e) => {
                    tracing::error!("MCP server startup task panicked: {e}");
                }
            }
        }

        statuses
    }

    /// Get tool definitions for the LLM, applying context budgeting.
    pub async fn toolDefs(
        &self,
        contextBudget: usize,
        includePermissionEscalation: bool,
    ) -> Vec<ToolDef> {
        self.registry
            .read()
            .await
            .toolDefs(contextBudget, includePermissionEscalation)
    }

    /// Whether tool search mode is active.
    pub async fn isSearchMode(
        &self,
        contextBudget: usize,
        includePermissionEscalation: bool,
    ) -> bool {
        self.registry
            .read()
            .await
            .isSearchMode(contextBudget, includePermissionEscalation)
    }

    /// Route a tool call to the correct server connection.
    ///
    /// Resolves the qualified name to server + original tool name,
    /// calls the tool, and applies output limits.
    pub async fn routeToolCall(&self, qualifiedName: &str, argsJson: &str) -> String {
        // Resolve from registry.
        let (serverName, originalName) = {
            let reg = self.registry.read().await;
            match reg.resolve(qualifiedName) {
                Some(entry) => (entry.serverName.clone(), entry.originalName.clone()),
                None => {
                    return format!(
                        "Unknown MCP tool \"{qualifiedName}\". \
                         Use mcpToolSearch to discover available tools."
                    );
                }
            }
        };

        // Find the connection.
        let conn = match self.connections.get(&serverName) {
            Some(c) => c,
            None => {
                return format!("MCP server \"{serverName}\" is not connected.");
            }
        };

        let maxOutput = conn.maxOutputTokens();

        // Parse args.
        let args: Option<serde_json::Map<String, serde_json::Value>> = if argsJson.is_empty()
            || argsJson == "{}"
        {
            None
        } else {
            match serde_json::from_str(argsJson) {
                Ok(serde_json::Value::Object(mut m)) => {
                    crate::tool::stripPermissionEscalationObject(&mut m);
                    if m.is_empty() { None } else { Some(m) }
                }
                Ok(_) => {
                    return format!("MCP tool arguments must be a JSON object, got: {argsJson}");
                }
                Err(e) => {
                    return format!("Failed to parse MCP tool arguments: {e}");
                }
            }
        };

        // Call the tool.
        match conn.callTool(&originalName, args).await {
            Ok(result) => {
                // Convert CallToolResult to a string.
                let text = formatCallToolResult(&result);
                output::limitOutput(&text, maxOutput)
            }
            Err(e) => e,
        }
    }

    /// Execute the mcpToolSearch meta-tool.
    pub async fn executeSearch(&self, argsJson: &str) -> String {
        let reg = self.registry.read().await;
        search::executeSearch(&reg, argsJson, None)
    }

    /// Check if a tool name is an MCP tool.
    pub fn isMcpTool(name: &str) -> bool {
        schema::isMcpTool(name)
    }

    /// Get current server statuses.
    pub fn serverStatuses(&self) -> Vec<ServerStatus> {
        self.connections
            .values()
            .map(|c| ServerStatus {
                name: c.name.clone(),
                state: c.state().clone(),
            })
            .collect()
    }

    /// Total number of registered MCP tools.
    pub async fn toolCount(&self) -> usize {
        self.registry.read().await.toolCount()
    }

    /// Get a read lock on the registry (for search operations).
    pub fn registry(&self) -> &Arc<RwLock<ToolRegistry>> {
        &self.registry
    }

    /// Get the tools-changed notifier (for notification-driven re-listing).
    pub fn toolsChangedNotify(&self) -> Arc<Notify> {
        self.toolsChanged.clone()
    }

    /// Gracefully shut down all MCP server connections.
    pub async fn shutdown(&mut self) {
        tracing::info!("shutting down {} MCP servers", self.connections.len());

        let mut joinSet = JoinSet::new();

        // Take ownership of all connections.
        let connections: Vec<(String, ServerConnection)> = self.connections.drain().collect();

        for (name, mut conn) in connections {
            joinSet.spawn(async move {
                conn.disconnect().await;
                tracing::debug!(server = %name, "MCP server shut down");
            });
        }

        // Wait for all shutdowns with a global timeout.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while joinSet.join_next().await.is_some() {}
        })
        .await;

        // Clear registry.
        self.registry.write().await.unregisterServer("*");
    }
}

/// Convert a CallToolResult to a string representation.
fn formatCallToolResult(result: &rmcp::model::CallToolResult) -> String {
    use rmcp::model::{RawContent, ResourceContents};

    let mut parts = Vec::new();

    for content in &result.content {
        match &content.raw {
            RawContent::Text(t) => {
                parts.push(t.text.to_string());
            }
            RawContent::Image(img) => {
                parts.push(format!("[Image: {}]", img.mime_type));
            }
            RawContent::Audio(audio) => {
                parts.push(format!("[Audio: {}]", audio.mime_type));
            }
            RawContent::Resource(res) => {
                let uri = match &res.resource {
                    ResourceContents::TextResourceContents { uri, .. } => uri,
                    ResourceContents::BlobResourceContents { uri, .. } => uri,
                };
                parts.push(format!("[Resource: {uri}]"));
            }
            RawContent::ResourceLink(link) => {
                parts.push(format!("[ResourceLink: {}]", link.uri));
            }
        }
    }

    if parts.is_empty() {
        "(empty result)".into()
    } else {
        parts.join("\n")
    }
}
