#![allow(non_snake_case)]

//! Per-server MCP connection lifecycle.
//!
//! Manages spawning/connecting to a single MCP server, tool filtering,
//! and graceful shutdown with process group cleanup.
//!
//! # Public API
//! - [`ServerConnection`] — owns a connection to one MCP server
//! - [`ConnectionState`] — current state of the connection
//!
//! # Dependencies
//! `rmcp`, `tokio`

use std::collections::HashSet;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, GetPromptRequestParams, GetPromptResult, Prompt,
    ReadResourceRequestParams, ReadResourceResult, Resource, Tool,
};
use rmcp::service::{DynService, RoleClient, RunningService};
use tokio::process::Command;

use super::config::{ServerConfig, TransportType};
use super::handler::FlatlineHandler;

/// Current state of a server connection.
#[derive(Debug, Clone)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
    ShuttingDown,
}

impl std::fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "disconnected"),
            Self::Connecting => write!(f, "connecting"),
            Self::Connected => write!(f, "connected"),
            Self::Failed(e) => write!(f, "failed: {e}"),
            Self::ShuttingDown => write!(f, "shutting down"),
        }
    }
}

/// Tool allowlist/blocklist filter.
struct ToolFilter {
    enabled: Option<HashSet<String>>,
    disabled: HashSet<String>,
}

impl ToolFilter {
    fn new(config: &ServerConfig) -> Self {
        Self {
            enabled: config
                .enabledTools
                .as_ref()
                .map(|v| v.iter().cloned().collect()),
            disabled: config
                .disabledTools
                .as_ref()
                .map(|v| v.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }

    fn allows(&self, toolName: &str) -> bool {
        if let Some(ref enabled) = self.enabled
            && !enabled.contains(toolName)
        {
            return false;
        }
        !self.disabled.contains(toolName)
    }
}

/// A live connection to a single MCP server.
pub struct ServerConnection {
    pub name: String,
    config: ServerConfig,
    service: Option<RunningService<RoleClient, Box<dyn DynService<RoleClient>>>>,
    state: ConnectionState,
    toolFilter: ToolFilter,
}

impl ServerConnection {
    pub fn new(name: String, config: ServerConfig) -> Self {
        let toolFilter = ToolFilter::new(&config);
        Self {
            name,
            config,
            service: None,
            state: ConnectionState::Disconnected,
            toolFilter,
        }
    }

    /// Current connection state.
    pub fn state(&self) -> &ConnectionState {
        &self.state
    }

    /// Connect to the server using the configured transport.
    pub async fn connect(&mut self, handler: FlatlineHandler) -> Result<(), String> {
        self.state = ConnectionState::Connecting;

        let timeout = Duration::from_secs(self.config.startupTimeout);
        let transport = self.config.transport();

        let result = match transport {
            TransportType::Stdio {
                command,
                args,
                cwd,
                env,
            } => {
                self.connectStdio(handler, &command, &args, cwd.as_deref(), &env, timeout)
                    .await
            }
            TransportType::Http { url, auth, headers } => {
                self.connectHttp(handler, &url, auth.as_deref(), &headers, timeout)
                    .await
            }
            TransportType::Invalid => {
                Err("server config must specify either 'command' (stdio) or 'url' (HTTP)".into())
            }
        };

        match result {
            Ok(()) => {
                self.state = ConnectionState::Connected;
                tracing::info!(server = %self.name, "MCP server connected");
                Ok(())
            }
            Err(e) => {
                let msg = formatStartupError(&self.name, &e);
                self.state = ConnectionState::Failed(msg.clone());
                tracing::warn!(server = %self.name, error = %msg, "MCP server failed to connect");
                Err(msg)
            }
        }
    }

    async fn connectStdio(
        &mut self,
        handler: FlatlineHandler,
        command: &str,
        args: &[String],
        cwd: Option<&str>,
        env: &std::collections::HashMap<String, String>,
        timeout: Duration,
    ) -> Result<(), String> {
        let mut cmd = Command::new(command);
        cmd.args(args);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        // Create a new process group so we can kill the whole tree.
        #[cfg(unix)]
        cmd.process_group(0);
        // Use the builder so we can override stderr to piped (the default
        // is inherit, which dumps MCP server banners into the TUI).
        let (transport, stderrHandle) = rmcp::transport::TokioChildProcess::builder(cmd)
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn \"{command}\": {e}"))?;

        // Drain stderr to tracing in the background.
        if let Some(stderr) = stderrHandle {
            let serverName = self.name.clone();
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(server = %serverName, "[stderr] {line}");
                }
            });
        }

        let service = tokio::time::timeout(timeout, handler.into_dyn().serve(transport))
            .await
            .map_err(|_| format!("startup timed out after {timeout:?}"))?
            .map_err(|e| format!("initialization failed: {e}"))?;

        self.service = Some(service);
        Ok(())
    }

    async fn connectHttp(
        &mut self,
        handler: FlatlineHandler,
        url: &str,
        auth: Option<&str>,
        _headers: &std::collections::HashMap<String, String>,
        timeout: Duration,
    ) -> Result<(), String> {
        let mut config =
            rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(
                url,
            );

        if let Some(authVal) = auth {
            config.auth_header = Some(authVal.to_string());
        }

        // NOTE: Use from_config() so rmcp constructs its own reqwest::Client
        // internally — avoids version mismatch with our workspace reqwest.
        let transport =
            rmcp::transport::streamable_http_client::StreamableHttpClientTransport::from_config(
                config,
            );

        let service = tokio::time::timeout(timeout, handler.into_dyn().serve(transport))
            .await
            .map_err(|_| format!("startup timed out after {timeout:?}"))?
            .map_err(|e| format!("initialization failed: {e}"))?;

        self.service = Some(service);
        Ok(())
    }

    /// List tools from the server, applying the tool filter.
    pub async fn listTools(&self) -> Result<Vec<Tool>, String> {
        let service = self.requireConnected()?;
        let tools = service
            .list_all_tools()
            .await
            .map_err(|e| format!("list_tools failed: {e}"))?;

        Ok(tools
            .into_iter()
            .filter(|t| self.toolFilter.allows(&t.name))
            .collect())
    }

    /// Call a tool on the server.
    pub async fn callTool(
        &self,
        toolName: &str,
        args: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, String> {
        let service = self.requireConnected()?;

        if !self.toolFilter.allows(toolName) {
            return Err(format!(
                "tool \"{toolName}\" is disabled for server \"{}\"",
                self.name
            ));
        }

        let timeout = Duration::from_secs(self.config.toolTimeout);
        let mut params = CallToolRequestParams::new(toolName.to_string());
        if let Some(a) = args {
            params = params.with_arguments(a);
        }

        let result = tokio::time::timeout(timeout, service.call_tool(params))
            .await
            .map_err(|_| {
                format!(
                    "MCP tool \"{toolName}\" timed out after {}s",
                    self.config.toolTimeout
                )
            })?
            .map_err(|e| format!("tool call failed: {e}"))?;

        Ok(result)
    }

    /// List resources from the server.
    pub async fn listResources(&self) -> Result<Vec<Resource>, String> {
        let service = self.requireConnected()?;
        service
            .list_all_resources()
            .await
            .map_err(|e| format!("list_resources failed: {e}"))
    }

    /// Read a resource from the server.
    pub async fn readResource(&self, uri: &str) -> Result<ReadResourceResult, String> {
        let service = self.requireConnected()?;
        service
            .read_resource(ReadResourceRequestParams::new(uri))
            .await
            .map_err(|e| format!("read_resource failed: {e}"))
    }

    /// List prompts from the server.
    pub async fn listPrompts(&self) -> Result<Vec<Prompt>, String> {
        let service = self.requireConnected()?;
        service
            .list_all_prompts()
            .await
            .map_err(|e| format!("list_prompts failed: {e}"))
    }

    /// Get a prompt from the server.
    pub async fn getPrompt(
        &self,
        name: &str,
        args: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<GetPromptResult, String> {
        let service = self.requireConnected()?;
        let mut params = GetPromptRequestParams::new(name);
        if let Some(a) = args {
            params = params.with_arguments(a);
        }
        service
            .get_prompt(params)
            .await
            .map_err(|e| format!("get_prompt failed: {e}"))
    }

    /// Gracefully disconnect from the server.
    pub async fn disconnect(&mut self) {
        self.state = ConnectionState::ShuttingDown;
        if let Some(mut service) = self.service.take() {
            let name = self.name.clone();
            match service.close_with_timeout(Duration::from_secs(5)).await {
                Ok(_) => {
                    tracing::debug!(server = %name, "MCP server disconnected gracefully");
                }
                Err(e) => {
                    tracing::warn!(server = %name, error = %e, "MCP server disconnect error");
                }
            }
        }
        self.state = ConnectionState::Disconnected;
    }

    /// Check if connected. Returns error string if not.
    fn requireConnected(
        &self,
    ) -> Result<&RunningService<RoleClient, Box<dyn DynService<RoleClient>>>, String> {
        self.service.as_ref().ok_or_else(|| {
            format!(
                "MCP server \"{}\" is not connected (state: {})",
                self.name, self.state
            )
        })
    }

    /// Per-server output token limit.
    pub fn maxOutputTokens(&self) -> usize {
        self.config.maxOutputTokens
    }
}

/// Format a startup error with context-aware suggestions.
fn formatStartupError(serverName: &str, error: &str) -> String {
    let lowerErr = error.to_lowercase();

    if lowerErr.contains("not found") || lowerErr.contains("no such file") {
        return format!(
            "Command not found for MCP server \"{serverName}\". \
             Check that the command is installed and in PATH. Error: {error}"
        );
    }

    if lowerErr.contains("auth") || lowerErr.contains("unauthorized") || lowerErr.contains("401") {
        return format!(
            "Authentication failed for MCP server \"{serverName}\". \
             Check your API key or auth configuration. Error: {error}"
        );
    }

    if lowerErr.contains("timed out") {
        return format!(
            "MCP server \"{serverName}\" did not respond within the startup timeout. \
             Try increasing startup_timeout in config. Error: {error}"
        );
    }

    format!("MCP server \"{serverName}\" failed: {error}")
}
