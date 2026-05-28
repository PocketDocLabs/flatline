#![allow(non_snake_case)]

//! MCP client handler — routes server requests back through Flatline.
//!
//! Implements rmcp's `ClientHandler` trait to handle server-initiated requests
//! (sampling, elicitation) and notifications (list_changed, progress, logging).
//!
//! # Public API
//! - [`FlatlineHandler`] — the handler struct
//! - [`ElicitationRequest`] / [`ElicitationResponse`] — elicitation types
//!
//! # Dependencies
//! `rmcp`, `tokio`

use std::sync::Arc;

use rmcp::ClientHandler;
use rmcp::ErrorData as McpError;
use rmcp::model::{
    ClientCapabilities, ClientInfo, CreateElicitationRequestParams, CreateElicitationResult,
    CreateMessageRequestMethod, CreateMessageRequestParams, CreateMessageResult, ElicitationAction,
    Implementation, ListRootsResult, LoggingMessageNotificationParam, ProgressNotificationParam,
    ProtocolVersion, Root,
};
use rmcp::service::{NotificationContext, RequestContext, RoleClient};
use tokio::sync::{Notify, mpsc, oneshot};

/// Elicitation request forwarded to the supervisor (TUI or headless controller).
pub struct ElicitationRequest {
    pub serverName: String,
    pub message: String,
    pub schema: serde_json::Value,
    pub responseTx: oneshot::Sender<ElicitationResponse>,
}

/// Supervisor's response to an elicitation request.
pub enum ElicitationResponse {
    Accept(serde_json::Map<String, serde_json::Value>),
    Decline,
}

/// ClientHandler implementation that routes MCP server requests through Flatline.
///
/// - Sampling (`create_message`): routes to our LLM API client.
/// - Elicitation: forwards to the supervisor via channel.
/// - Notifications: signals Notify handles for list_changed events.
pub struct FlatlineHandler {
    /// Channel for elicitation requests to the supervisor.
    pub elicitationTx: mpsc::Sender<ElicitationRequest>,

    /// Signaled when a server's tool list changes.
    pub toolsChanged: Arc<Notify>,

    /// Signaled when a server's resource list changes.
    pub resourcesChanged: Arc<Notify>,

    /// Signaled when a server's prompt list changes.
    pub promptsChanged: Arc<Notify>,

    /// Server name (for logging and elicitation context).
    pub serverName: String,
}

impl ClientHandler for FlatlineHandler {
    fn get_info(&self) -> ClientInfo {
        ClientInfo::new(
            ClientCapabilities::builder()
                .enable_elicitation()
                .enable_roots()
                .build(),
            Implementation::new("flatline", env!("CARGO_PKG_VERSION")),
        )
        .with_protocol_version(ProtocolVersion::V_2025_03_26)
    }

    fn create_message(
        &self,
        _params: CreateMessageRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> impl std::future::Future<Output = Result<CreateMessageResult, McpError>> + Send + '_ {
        async move {
            // TODO(pocketdoc, 2026-03-11): Route to api::Client for real sampling.
            // For now, decline — most servers don't use sampling.
            Err(McpError::method_not_found::<CreateMessageRequestMethod>())
        }
    }

    fn create_elicitation(
        &self,
        request: CreateElicitationRequestParams,
        _context: RequestContext<RoleClient>,
    ) -> impl std::future::Future<Output = Result<CreateElicitationResult, McpError>> + Send + '_
    {
        async move {
            match request {
                CreateElicitationRequestParams::FormElicitationParams {
                    message,
                    requested_schema,
                    ..
                } => {
                    let schema = serde_json::to_value(&requested_schema).unwrap_or_default();
                    let (responseTx, responseRx) = oneshot::channel();
                    let elicitation = ElicitationRequest {
                        serverName: self.serverName.clone(),
                        message,
                        schema,
                        responseTx,
                    };

                    // Try to send to supervisor. If the channel is closed, decline.
                    if self.elicitationTx.send(elicitation).await.is_err() {
                        return Ok(CreateElicitationResult::new(ElicitationAction::Decline));
                    }

                    // Wait for supervisor response with a timeout.
                    match tokio::time::timeout(std::time::Duration::from_secs(120), responseRx)
                        .await
                    {
                        Ok(Ok(ElicitationResponse::Accept(content))) => {
                            Ok(CreateElicitationResult::new(ElicitationAction::Accept)
                                .with_content(serde_json::Value::Object(content)))
                        }
                        _ => Ok(CreateElicitationResult::new(ElicitationAction::Decline)),
                    }
                }
                // URL-based elicitation — not supported yet, decline.
                CreateElicitationRequestParams::UrlElicitationParams { .. } => {
                    Ok(CreateElicitationResult::new(ElicitationAction::Decline))
                }
            }
        }
    }

    fn list_roots(
        &self,
        _context: RequestContext<RoleClient>,
    ) -> impl std::future::Future<Output = Result<ListRootsResult, McpError>> + Send + '_ {
        async move {
            let cwd = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".into());
            Ok(ListRootsResult::new(vec![
                Root::new(format!("file://{cwd}")).with_name("project"),
            ]))
        }
    }

    fn on_tool_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {
            tracing::info!(server = %self.serverName, "tool list changed");
            self.toolsChanged.notify_waiters();
        }
    }

    fn on_resource_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {
            tracing::info!(server = %self.serverName, "resource list changed");
            self.resourcesChanged.notify_waiters();
        }
    }

    fn on_prompt_list_changed(
        &self,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {
            tracing::info!(server = %self.serverName, "prompt list changed");
            self.promptsChanged.notify_waiters();
        }
    }

    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            tracing::debug!(
                server = %self.serverName,
                progress = %params.progress,
                total = ?params.total,
                message = ?params.message,
                "MCP progress"
            );
        }
    }

    fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async move {
            tracing::debug!(
                server = %self.serverName,
                level = ?params.level,
                logger = ?params.logger,
                "MCP log: {:?}",
                params.data,
            );
        }
    }
}
