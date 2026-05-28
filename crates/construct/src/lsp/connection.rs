//! Single language server connection lifecycle.
//!
//! Manages a child process running a language server, communicates via
//! JSON-RPC over stdio using async-lsp, and collects diagnostics from
//! `publishDiagnostics` notifications.
//!
//! # Public API
//! - [`LspConnection`] — owns one server process + client handle
//! - [`ConnectionState`] — lifecycle state
//!
//! # Dependencies
//! `async-lsp`, `dashmap`, `tokio`

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_lsp::lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
    PublishDiagnostics,
};
use async_lsp::lsp_types::request::Initialize;
use async_lsp::lsp_types::{
    ClientCapabilities, Diagnostic, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DidSaveTextDocumentParams, GeneralClientCapabilities,
    InitializeParams, InitializedParams, PublishDiagnosticsClientCapabilities,
    TextDocumentClientCapabilities, TextDocumentContentChangeEvent, TextDocumentIdentifier,
    TextDocumentItem, TextDocumentSyncClientCapabilities, Url, VersionedTextDocumentIdentifier,
};
use async_lsp::router::Router;
use async_lsp::{MainLoop, ServerSocket};
use dashmap::DashMap;

use super::config::ResolvedServer;

/// Connection state for a single language server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Initializing,
    Ready,
    Failed(String),
    ShuttingDown,
}

/// Client-side state for the async-lsp router.
struct ClientState {
    diagnostics: Arc<DashMap<Url, Vec<Diagnostic>>>,
    diagnosticsNotify: Arc<tokio::sync::Notify>,
}

/// A connection to a single language server process.
pub struct LspConnection {
    config: ResolvedServer,
    server: Option<ServerSocket>,
    state: ConnectionState,
    diagnostics: Arc<DashMap<Url, Vec<Diagnostic>>>,
    diagnosticsNotify: Arc<tokio::sync::Notify>,
    versions: HashMap<String, i32>,
    loopHandle: Option<tokio::task::JoinHandle<()>>,
    childHandle: Option<tokio::task::JoinHandle<()>>,
}

impl LspConnection {
    /// Create a new connection (does not start the server).
    pub fn new(config: ResolvedServer) -> Self {
        Self {
            config,
            server: None,
            state: ConnectionState::Disconnected,
            diagnostics: Arc::new(DashMap::new()),
            diagnosticsNotify: Arc::new(tokio::sync::Notify::new()),
            versions: HashMap::new(),
            loopHandle: None,
            childHandle: None,
        }
    }

    /// Current connection state.
    pub fn state(&self) -> &ConnectionState {
        &self.state
    }

    /// Server configuration.
    pub fn config(&self) -> &ResolvedServer {
        &self.config
    }

    /// Reference to the shared diagnostics store.
    pub fn diagnosticsStore(&self) -> &Arc<DashMap<Url, Vec<Diagnostic>>> {
        &self.diagnostics
    }

    /// Spawn the language server and run the initialize handshake.
    #[allow(deprecated)]
    pub async fn start(&mut self, projectRoot: &Path) -> anyhow::Result<()> {
        self.state = ConnectionState::Initializing;

        let diagnostics = self.diagnostics.clone();
        let diagnosticsNotify = self.diagnosticsNotify.clone();

        // Build the async-lsp router to handle server notifications.
        let (mainloop, serverSocket) = MainLoop::new_client(|_serverSocket| {
            let mut router = Router::new(ClientState {
                diagnostics: diagnostics.clone(),
                diagnosticsNotify: diagnosticsNotify.clone(),
            });

            router.notification::<PublishDiagnostics>(|state, params| {
                tracing::debug!(
                    uri = %params.uri,
                    count = params.diagnostics.len(),
                    "publishDiagnostics received"
                );
                state
                    .diagnostics
                    .insert(params.uri.clone(), params.diagnostics);
                state.diagnosticsNotify.notify_waiters();
                ControlFlow::Continue(())
            });

            // Ignore optional notifications we don't care about.
            router.unhandled_notification(|_, _| ControlFlow::Continue(()));

            router
        });

        // Spawn the server child process.
        let mut cmd = tokio::process::Command::new(&self.config.command);
        cmd.args(&self.config.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        for (k, v) in &self.config.env {
            cmd.env(k, v);
        }

        // Process group isolation for clean shutdown.
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd.spawn().map_err(|e| {
            self.state = ConnectionState::Failed(format!("Failed to spawn: {e}"));
            anyhow::anyhow!(
                "Failed to start {} ({}): {e}",
                self.config.id,
                self.config.command,
            )
        })?;

        let childStdin = child.stdin.take().expect("stdin piped");
        let childStdout = child.stdout.take().expect("stdout piped");

        // Drain stderr to tracing.
        let stderrHandle = if let Some(stderr) = child.stderr.take() {
            let serverId = self.config.id.clone();
            Some(tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(server = %serverId, stderr = %line);
                }
            }))
        } else {
            None
        };

        // Adapt tokio IO to futures IO for async-lsp.
        let stdout = tokio_util::compat::TokioAsyncReadCompatExt::compat(childStdout);
        let stdin = tokio_util::compat::TokioAsyncWriteCompatExt::compat_write(childStdin);

        // Spawn the main loop.
        // WARNING: async-lsp panics with "Sender is alive" on channel close
        // during shutdown. This is an upstream bug. We catch it via catch_unwind.
        let loopHandle = tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(mainloop.run_buffered(stdout, stdin));
            match futures::FutureExt::catch_unwind(result).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::debug!("LSP main loop exited: {e}"),
                Err(_) => tracing::debug!("LSP main loop exited during shutdown"),
            }
        });

        // Spawn child process waiter.
        let childJoin = tokio::spawn(async move {
            let _ = child.wait().await;
            if let Some(h) = stderrHandle {
                let _ = h.await;
            }
        });

        self.loopHandle = Some(loopHandle);
        self.childHandle = Some(childJoin);
        self.server = Some(serverSocket.clone());

        // Send initialize request.
        let rootUri = Url::from_file_path(projectRoot).ok();
        let initParams = InitializeParams {
            root_uri: rootUri,
            capabilities: clientCapabilities(),
            ..Default::default()
        };

        let timeout = Duration::from_secs(self.config.startupTimeout);
        let initResult =
            tokio::time::timeout(timeout, serverSocket.request::<Initialize>(initParams)).await;

        match initResult {
            Ok(Ok(_caps)) => {
                if let Err(e) = serverSocket
                    .notify::<async_lsp::lsp_types::notification::Initialized>(InitializedParams {})
                {
                    tracing::warn!(server = %self.config.id, "failed to send initialized: {e}");
                }
                self.state = ConnectionState::Ready;
                tracing::info!(server = %self.config.id, "LSP server ready");
                Ok(())
            }
            Ok(Err(e)) => {
                self.state = ConnectionState::Failed(format!("Initialize error: {e}"));
                Err(anyhow::anyhow!(
                    "LSP {} initialize failed: {e}",
                    self.config.id,
                ))
            }
            Err(_) => {
                self.state = ConnectionState::Failed("Initialize timed out".into());
                Err(anyhow::anyhow!(
                    "LSP {} initialize timed out after {timeout:?}",
                    self.config.id,
                ))
            }
        }
    }

    /// Notify the server that a file was opened.
    pub fn didOpen(&mut self, path: &str, content: &str, languageId: &str) {
        let Some(ref server) = self.server else {
            tracing::debug!(path = %path, "didOpen: no server socket");
            return;
        };
        let Some(uri) = Url::from_file_path(path).ok() else {
            tracing::debug!(path = %path, "didOpen: invalid path for URI");
            return;
        };
        let version = self.versions.entry(path.to_string()).or_insert(0);
        *version += 1;

        tracing::debug!(uri = %uri, version = *version, contentLen = content.len(), "didOpen sent");
        let _ = server.notify::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri,
                language_id: languageId.to_string(),
                version: *version,
                text: content.to_string(),
            },
        });
    }

    /// Notify the server that a file's content changed (full sync).
    pub fn didChange(&mut self, path: &str, content: &str) {
        let Some(ref server) = self.server else {
            tracing::debug!(path = %path, "didChange: no server socket");
            return;
        };
        let Some(uri) = Url::from_file_path(path).ok() else {
            tracing::debug!(path = %path, "didChange: invalid path for URI");
            return;
        };
        let version = self.versions.entry(path.to_string()).or_insert(0);
        *version += 1;

        tracing::debug!(uri = %uri, version = *version, contentLen = content.len(), "didChange sent");

        // NOTE: We do NOT clear the DashMap here. RA's initial workspace scan
        // diagnostics are still valid and useful. Clearing them causes
        // awaitDiagnostics to wait for fresh results that RA may take 30+
        // seconds to produce on large workspaces. The DashMap updates
        // naturally when RA eventually re-publishes.

        let _ = server.notify::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri,
                version: *version,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: content.to_string(),
            }],
        });
    }

    /// Notify the server that a file was saved to disk.
    ///
    /// Triggers cargo check / flycheck in rust-analyzer, which produces
    /// the full rustc diagnostic suite (borrow checker, lifetimes, etc.).
    /// Native RA diagnostics come from didChange; cargo diagnostics need didSave.
    pub fn didSave(&self, path: &str) {
        let Some(ref server) = self.server else {
            tracing::debug!(path = %path, "didSave: no server socket");
            return;
        };
        let Some(uri) = Url::from_file_path(path).ok() else {
            tracing::debug!(path = %path, "didSave: invalid path for URI");
            return;
        };

        tracing::debug!(uri = %uri, "didSave sent");
        let _ = server.notify::<DidSaveTextDocument>(DidSaveTextDocumentParams {
            text_document: TextDocumentIdentifier { uri },
            text: None,
        });
    }

    /// Notify the server that a file was closed.
    pub fn didClose(&mut self, path: &str) {
        let Some(ref server) = self.server else {
            return;
        };
        let Some(uri) = Url::from_file_path(path).ok() else {
            return;
        };
        self.versions.remove(path);

        let _ = server.notify::<DidCloseTextDocument>(DidCloseTextDocumentParams {
            text_document: TextDocumentIdentifier { uri },
        });
    }

    /// Wait for diagnostics to arrive for a specific file URI.
    ///
    /// Uses a 150ms debounce — waits until the server stops publishing
    /// updates before returning. This lets RA finish its full analysis
    /// pass (syntax → types → borrow checker) instead of returning
    /// partial results on the first notification.
    ///
    /// Returns diagnostics for the URI, or empty vec on timeout.
    pub async fn awaitDiagnostics(&self, path: &str, timeout: Duration) -> Vec<Diagnostic> {
        const DEBOUNCE: Duration = Duration::from_millis(150);

        let Some(uri) = Url::from_file_path(path).ok() else {
            tracing::debug!(path = %path, "awaitDiagnostics: invalid file path for URI");
            return Vec::new();
        };

        tracing::debug!(uri = %uri, timeout = ?timeout, "awaitDiagnostics: waiting");

        let notify = self.diagnosticsNotify.clone();
        let diagnostics = self.diagnostics.clone();

        let result = tokio::time::timeout(timeout, async {
            // Wait for first non-empty diagnostics for this URI.
            // Register the waiter BEFORE checking the condition to avoid
            // a race where notify_waiters fires between check and await.
            loop {
                let waiter = notify.notified();
                if let Some(diags) = diagnostics.get(&uri)
                    && !diags.is_empty()
                {
                    break;
                }
                waiter.await;
            }

            // Debounce: keep waiting while the server is still sending updates.
            loop {
                let waiter = notify.notified();
                match tokio::time::timeout(DEBOUNCE, waiter).await {
                    Ok(()) => {
                        // Got another notification — server is still active, reset debounce.
                        continue;
                    }
                    Err(_) => {
                        // 150ms of silence — server is done. Collect final results.
                        break;
                    }
                }
            }

            diagnostics.get(&uri).map(|d| d.clone()).unwrap_or_default()
        })
        .await;

        result.unwrap_or_default()
    }

    /// Shutdown the language server gracefully.
    pub async fn shutdown(&mut self) {
        self.state = ConnectionState::ShuttingDown;

        // Close all tracked files before shutting down.
        let openPaths: Vec<String> = self.versions.keys().cloned().collect();
        for path in &openPaths {
            self.didClose(path);
        }

        if let Some(server) = self.server.take() {
            // Send shutdown request, then exit notification.
            let _ = tokio::time::timeout(
                Duration::from_secs(5),
                server.request::<async_lsp::lsp_types::request::Shutdown>(()),
            )
            .await;

            let _ = server.notify::<async_lsp::lsp_types::notification::Exit>(());

            // Drop the socket so the MainLoop sees channel close and exits.
            drop(server);
        }

        // Suppress the panic hook temporarily — async-lsp panics on channel close.
        let prevHook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        if let Some(handle) = self.loopHandle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(3), handle).await;
        }

        std::panic::set_hook(prevHook);

        if let Some(handle) = self.childHandle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(3), handle).await;
        }

        self.state = ConnectionState::Disconnected;
        tracing::debug!(server = %self.config.id, "LSP server shut down");
    }
}

/// Build the client capabilities we advertise to servers.
fn clientCapabilities() -> ClientCapabilities {
    ClientCapabilities {
        text_document: Some(TextDocumentClientCapabilities {
            synchronization: Some(TextDocumentSyncClientCapabilities {
                dynamic_registration: Some(false),
                will_save: Some(false),
                will_save_wait_until: Some(false),
                did_save: Some(true),
            }),
            publish_diagnostics: Some(PublishDiagnosticsClientCapabilities {
                related_information: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        }),
        general: Some(GeneralClientCapabilities {
            ..Default::default()
        }),
        ..Default::default()
    }
}
