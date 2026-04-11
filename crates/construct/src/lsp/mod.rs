//! LSP integration — language server lifecycle and diagnostics.
//!
//! Manages language server processes, routes file events, and collects
//! diagnostics to surface in tool output after edits.
//!
//! # Public API
//! - [`LspManager`] — owns connections, routes file events, collects diagnostics
//! - [`LspConfig`] — per-server configuration from config.toml
//! - [`ConnectionState`] — server connection state
//!
//! # Dependencies
//! `async-lsp`, `dashmap`, `tokio`

pub mod config;
mod connection;
pub mod diagnostics;
pub mod servers;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_lsp::lsp_types::DiagnosticSeverity;

pub use config::{LspConfig, LspServerConfig};
pub use connection::ConnectionState;

use config::{ResolvedServer, resolveServers};
use connection::LspConnection;
/// Manages all active LSP connections and routes file events.
pub struct LspManager {
    /// Active connections keyed by "serverId:projectRoot".
    connections: HashMap<String, LspConnection>,
    /// Resolved server configurations.
    serverDefs: Vec<ResolvedServer>,
    /// Servers whose binary was not found — don't retry.
    unavailable: HashSet<String>,
    /// Files already opened with a server (keyed by "serverId:path").
    openedFiles: HashSet<String>,
    /// Servers we've already hinted about (one hint per server per session).
    hintedServers: HashSet<String>,
}

/// Hint about a missing server that could enhance the experience.
#[derive(Debug, Clone)]
pub struct LspHint {
    pub serverId: String,
    pub installHint: String,
}

impl LspManager {
    /// Create a new manager with resolved server definitions.
    pub fn new(
        userConfig: &LspConfig,
        projectConfig: &LspConfig,
    ) -> Self {
        let serverDefs = resolveServers(userConfig, projectConfig);
        tracing::debug!(
            serverCount = serverDefs.len(),
            "LSP manager initialized"
        );
        Self {
            connections: HashMap::new(),
            serverDefs,
            unavailable: HashSet::new(),
            openedFiles: HashSet::new(),
            hintedServers: HashSet::new(),
        }
    }

    /// Pre-start servers relevant to the current project so indexing begins immediately.
    ///
    /// Scans the project directory for file extensions and project markers,
    /// then starts matching servers in the background. Called at session init.
    pub async fn warmUp(&mut self, projectDir: &Path) {
        let knownExtensions: HashSet<String> = self
            .serverDefs
            .iter()
            .flat_map(|d| d.extensions.iter().cloned())
            .collect();

        // Scan project for relevant extensions.
        let projectExts = scanExtensions(projectDir);
        let relevantExts: Vec<&String> = projectExts
            .iter()
            .filter(|e| knownExtensions.contains(e.as_str()))
            .collect();

        if relevantExts.is_empty() {
            return;
        }

        // Deduplicate servers — each server only needs to start once.
        let mut startedServers = HashSet::new();

        for ext in relevantExts {
            let matching = self.findServersForExtension(ext);
            for (serverId, _) in matching {
                if startedServers.contains(&serverId) {
                    continue;
                }

                let Some(def) = self.serverDefs.iter().find(|d| d.id == serverId).cloned() else {
                    continue;
                };

                // Check if binary exists before trying to spawn.
                if !which(&def.command) {
                    self.unavailable.insert(serverId.clone());
                    tracing::debug!(server = %serverId, "not installed, skipping warm-up");
                    continue;
                }

                let projectRoot = self.findProjectRoot(
                    &projectDir.to_string_lossy(),
                    &serverId,
                );
                let connKey = format!("{serverId}:{}", projectRoot.display());

                if self.connections.contains_key(&connKey) {
                    startedServers.insert(serverId);
                    continue;
                }

                let mut conn = LspConnection::new(def);
                match conn.start(&projectRoot).await {
                    Ok(()) => {
                        tracing::info!(server = %serverId, "warm-up started");
                        self.connections.insert(connKey, conn);
                        startedServers.insert(serverId);
                    }
                    Err(e) => {
                        tracing::warn!(server = %serverId, "warm-up failed: {e}");
                        self.unavailable.insert(serverId);
                    }
                }
            }
        }
    }

    /// Touch a file — ensure the appropriate server is running and aware of the content.
    ///
    /// If the server is not yet started, it will be spawned lazily.
    /// Returns a hint if a matching server is not installed.
    pub async fn touchFile(&mut self, path: &str, content: &str) -> Option<LspHint> {
        let path = &absolutePath(path);
        let ext = fileExtension(path)?;
        let matchingDefs = self.findServersForExtension(&ext);

        if matchingDefs.is_empty() {
            return None;
        }

        let mut hint = None;

        for (serverId, languageId) in matchingDefs {
            // Skip servers we know are unavailable.
            if self.unavailable.contains(&serverId) {
                // Generate a one-time hint.
                if !self.hintedServers.contains(&serverId) {
                    if let Some(def) = self.serverDefs.iter().find(|d| d.id == serverId) {
                        if !def.installHint.is_empty() {
                            self.hintedServers.insert(serverId.clone());
                            hint = Some(LspHint {
                                serverId: serverId.clone(),
                                installHint: def.installHint.clone(),
                            });
                        }
                    }
                }
                continue;
            }

            // Find the project root.
            let projectRoot = self.findProjectRoot(path, &serverId);
            let connKey = format!("{serverId}:{}", projectRoot.display());

            // Ensure connection exists and is started.
            if !self.connections.contains_key(&connKey) {
                let Some(def) = self.serverDefs.iter().find(|d| d.id == serverId).cloned() else {
                    continue;
                };

                let mut conn = LspConnection::new(def);
                match conn.start(&projectRoot).await {
                    Ok(()) => {
                        self.connections.insert(connKey.clone(), conn);
                    }
                    Err(e) => {
                        let errMsg = e.to_string();
                        if errMsg.contains("Failed to start")
                            || errMsg.contains("No such file")
                            || errMsg.contains("not found")
                        {
                            tracing::info!(server = %serverId, "marking unavailable: {e}");
                            self.unavailable.insert(serverId.clone());
                            if !self.hintedServers.contains(&serverId) {
                                if let Some(def) =
                                    self.serverDefs.iter().find(|d| d.id == serverId)
                                {
                                    if !def.installHint.is_empty() {
                                        self.hintedServers.insert(serverId.clone());
                                        hint = Some(LspHint {
                                            serverId: serverId.clone(),
                                            installHint: def.installHint.clone(),
                                        });
                                    }
                                }
                            }
                        } else {
                            tracing::warn!(server = %serverId, "start failed: {e}");
                        }
                        continue;
                    }
                }
            }

            // Send didOpen or didChange.
            let fileKey = format!("{serverId}:{path}");
            if let Some(conn) = self.connections.get_mut(&connKey) {
                if *conn.state() != ConnectionState::Ready {
                    continue;
                }
                if self.openedFiles.contains(&fileKey) {
                    conn.didChange(path, content);
                } else {
                    conn.didOpen(path, content, &languageId);
                    self.openedFiles.insert(fileKey);
                }
            }
        }

        hint
    }

    /// Get diagnostics for a file, returning (formatted_output, optional_hint).
    ///
    /// Sends didOpen/didChange + didSave to the server and waits for diagnostics.
    /// Used after edits where the content has changed on disk.
    /// The didSave triggers cargo check / flycheck for full rustc diagnostics.
    ///
    /// Clears cached diagnostics first so we only return results from this
    /// analysis pass (not stale data from a prior file state).
    pub async fn getDiagnostics(
        &mut self,
        path: &str,
        content: &str,
        timeout: Duration,
    ) -> (String, Option<LspHint>) {
        let path = &absolutePath(path);
        let hint = self.touchFile(path, content).await;
        // Clear stale diagnostics so awaitDiagnostics waits for fresh results.
        self.clearDiagnostics(path);
        // File is already on disk — notify servers so cargo check runs.
        self.notifySave(path);
        let diags = self.collectDiagnostics(path, timeout).await;
        if diags.is_empty() {
            return (String::new(), hint);
        }
        let formatted = diagnostics::formatDiagnostics(path, &diags, DiagnosticSeverity::ERROR);
        (formatted, hint)
    }

    /// Get raw diagnostics for a file after a mutation (clears cache, waits for fresh results).
    ///
    /// Returns the raw Diagnostic objects for caller-side filtering (e.g. baseline diff).
    pub async fn getRawDiagnostics(
        &mut self,
        path: &str,
        content: &str,
        timeout: Duration,
    ) -> (Vec<async_lsp::lsp_types::Diagnostic>, Option<LspHint>) {
        let path = &absolutePath(path);
        let hint = self.touchFile(path, content).await;
        self.clearDiagnostics(path);
        self.notifySave(path);
        let diags = self.collectDiagnostics(path, timeout).await;
        (diags, hint)
    }

    /// Get raw cached diagnostics without re-analysis.
    ///
    /// Returns whatever diagnostics are already in the store for this file.
    /// Used to snapshot the baseline before an edit.
    pub fn getRawCachedDiagnostics(&self, path: &str) -> Vec<async_lsp::lsp_types::Diagnostic> {
        let path = &absolutePath(path);
        self.collectExistingDiagnostics(path)
    }

    /// Clear cached diagnostics for a file across all matching connections.
    fn clearDiagnostics(&self, path: &str) {
        let ext = match fileExtension(path) {
            Some(e) => e,
            None => return,
        };
        let uri = match async_lsp::lsp_types::Url::from_file_path(path).ok() {
            Some(u) => u,
            None => return,
        };
        let matchingDefs = self.findServersForExtension(&ext);
        for (serverId, _) in matchingDefs {
            if self.unavailable.contains(&serverId) {
                continue;
            }
            let projectRoot = self.findProjectRoot(path, &serverId);
            let connKey = format!("{serverId}:{}", projectRoot.display());
            if let Some(conn) = self.connections.get(&connKey) {
                conn.diagnosticsStore().remove(&uri);
            }
        }
    }

    /// Check if any active (Ready) server covers this file's extension.
    fn hasActiveServer(&self, path: &str) -> bool {
        let ext = match fileExtension(path) {
            Some(e) => e,
            None => return false,
        };
        let defs = self.findServersForExtension(&ext);
        defs.iter().any(|(id, _)| {
            !self.unavailable.contains(id)
                && self.connections.values().any(|c| {
                    c.config().id == *id && *c.state() == ConnectionState::Ready
                })
        })
    }

    /// Send didSave to all matching connections for a file.
    ///
    /// This triggers flycheck (cargo check) in rust-analyzer, which produces
    /// the full rustc diagnostic suite. Only useful after the file is on disk.
    fn notifySave(&self, path: &str) {
        let ext = match fileExtension(path) {
            Some(e) => e,
            None => return,
        };
        let matchingDefs = self.findServersForExtension(&ext);
        for (serverId, _) in matchingDefs {
            if self.unavailable.contains(&serverId) {
                continue;
            }
            let projectRoot = self.findProjectRoot(path, &serverId);
            let connKey = format!("{serverId}:{}", projectRoot.display());
            if let Some(conn) = self.connections.get(&connKey) {
                if *conn.state() == ConnectionState::Ready {
                    conn.didSave(path);
                }
            }
        }
    }

    /// Get existing diagnostics without re-touching the file.
    ///
    /// Returns cached diagnostics from RA's workspace scan. Only sends
    /// didOpen if the file hasn't been opened yet (no DashMap clear).
    /// Used by the diagnostics tool for proactive checks.
    pub async fn getCachedDiagnostics(
        &mut self,
        path: &str,
        content: &str,
        timeout: Duration,
    ) -> (String, Option<LspHint>) {
        let path = &absolutePath(path);

        // Check if any connection already has diagnostics for this file.
        let existing = self.collectExistingDiagnostics(path);
        if !existing.is_empty() {
            let formatted = diagnostics::formatDiagnostics(
                path, &existing, DiagnosticSeverity::ERROR,
            );
            return (formatted, None);
        }

        // No cached diagnostics — touch the file and wait.
        let hint = self.touchFile(path, content).await;
        let diags = self.collectDiagnostics(path, timeout).await;
        if diags.is_empty() {
            return (String::new(), hint);
        }
        let formatted = diagnostics::formatDiagnostics(path, &diags, DiagnosticSeverity::ERROR);
        (formatted, hint)
    }

    /// Collect diagnostics from all matching connections, waiting for results.
    async fn collectDiagnostics(&self, path: &str, timeout: Duration) -> Vec<async_lsp::lsp_types::Diagnostic> {
        let ext = match fileExtension(path) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let matchingDefs = self.findServersForExtension(&ext);
        let mut allDiagnostics = Vec::new();

        for (serverId, _) in matchingDefs {
            if self.unavailable.contains(&serverId) {
                continue;
            }
            let projectRoot = self.findProjectRoot(path, &serverId);
            let connKey = format!("{serverId}:{}", projectRoot.display());

            if let Some(conn) = self.connections.get(&connKey) {
                if *conn.state() != ConnectionState::Ready {
                    continue;
                }
                let diags = conn.awaitDiagnostics(path, timeout).await;
                if !diags.is_empty() {
                    allDiagnostics.extend(diags);
                }
            }
        }

        allDiagnostics
    }

    /// Collect diagnostics already in the DashMap without waiting.
    fn collectExistingDiagnostics(&self, path: &str) -> Vec<async_lsp::lsp_types::Diagnostic> {
        let ext = match fileExtension(path) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let matchingDefs = self.findServersForExtension(&ext);
        let mut allDiagnostics = Vec::new();

        for (serverId, _) in matchingDefs {
            if self.unavailable.contains(&serverId) {
                continue;
            }
            let projectRoot = self.findProjectRoot(path, &serverId);
            let connKey = format!("{serverId}:{}", projectRoot.display());

            if let Some(conn) = self.connections.get(&connKey) {
                let uri = async_lsp::lsp_types::Url::from_file_path(path).ok();
                if let Some(uri) = uri {
                    if let Some(diags) = conn.diagnosticsStore().get(&uri) {
                        if !diags.is_empty() {
                            allDiagnostics.extend(diags.clone());
                        }
                    }
                }
            }
        }

        allDiagnostics
    }

    /// Get full status for ALL known servers (for /lsp panel).
    ///
    /// Probes the system to check which binaries are on PATH.
    /// Returns every server def with its current status.
    pub fn allServerStatuses(&self) -> Vec<FullServerStatus> {
        self.serverDefs
            .iter()
            .map(|def| {
                // Check if actively connected.
                let activeState = self.connections.values().find_map(|conn| {
                    if conn.config().id == def.id {
                        Some(conn.state().clone())
                    } else {
                        None
                    }
                });

                let status = if let Some(state) = activeState {
                    match state {
                        ConnectionState::Ready => ServerAvailability::Active,
                        ConnectionState::Initializing => ServerAvailability::Starting,
                        ConnectionState::Failed(e) => ServerAvailability::Failed(e),
                        _ => ServerAvailability::Installed,
                    }
                } else if self.unavailable.contains(&def.id) {
                    ServerAvailability::NotInstalled
                } else {
                    // Probe: check if binary exists on PATH.
                    match which(&def.command) {
                        true => ServerAvailability::Installed,
                        false => ServerAvailability::NotInstalled,
                    }
                };

                FullServerStatus {
                    id: def.id.clone(),
                    extensions: def.extensions.clone(),
                    installHint: def.installHint.clone(),
                    runtime: def.runtime.clone(),
                    status,
                }
            })
            .collect()
    }

    /// Shutdown all active connections.
    pub async fn shutdown(&mut self) {
        let keys: Vec<String> = self.connections.keys().cloned().collect();
        let mut handles = tokio::task::JoinSet::new();

        // NOTE: Can't move connections into JoinSet directly due to borrow rules.
        // Drain and shutdown sequentially with a global timeout.
        let mut connections: Vec<LspConnection> = keys
            .iter()
            .filter_map(|k| self.connections.remove(k))
            .collect();

        for mut conn in connections.drain(..) {
            handles.spawn(async move {
                conn.shutdown().await;
            });
        }

        // Global timeout for all shutdowns.
        let _ = tokio::time::timeout(Duration::from_secs(10), async {
            while handles.join_next().await.is_some() {}
        })
        .await;

        self.connections.clear();
        tracing::debug!("all LSP servers shut down");
    }

    /// Get formatted diagnostics for the diagnostics tool.
    ///
    /// Touches the file, waits for diagnostics, formats output. Handles
    /// both file and directory paths.
    pub async fn getDiagnosticsForTool(
        &mut self,
        path: &str,
        minSeverity: async_lsp::lsp_types::DiagnosticSeverity,
        timeout: Duration,
    ) -> String {
        let path = &absolutePath(path);
        let filePath = std::path::Path::new(path);

        if filePath.is_dir() {
            return self.getDiagnosticsForDirectory(path, minSeverity, timeout).await;
        }

        if !self.hasActiveServer(path) {
            return format!("No LSP server active for {path}.");
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return format!("Failed to read {path}: {e}"),
        };

        let (result, _hint) = self.getCachedDiagnostics(path, &content, timeout).await;
        if result.is_empty() {
            return format!("Clean \u{2014} no errors in {path}.");
        }
        result
    }

    /// Collect diagnostics across all files in a directory matching known extensions.
    async fn getDiagnosticsForDirectory(
        &mut self,
        dirPath: &str,
        minSeverity: async_lsp::lsp_types::DiagnosticSeverity,
        timeout: Duration,
    ) -> String {
        const MAX_DIR_FILES: usize = 50;

        let knownExtensions: Vec<String> = self
            .serverDefs
            .iter()
            .flat_map(|d| d.extensions.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dirPath) {
            for entry in entries.flatten() {
                if files.len() >= MAX_DIR_FILES {
                    break;
                }
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension() {
                    let dotExt = format!(".{}", ext.to_string_lossy());
                    if knownExtensions.contains(&dotExt) {
                        files.push(path);
                    }
                }
            }
        }

        if files.is_empty() {
            return format!("No files with known LSP extensions in {dirPath}.");
        }

        let mut fileDiagnostics = Vec::new();
        for file in &files {
            let pathStr = file.to_string_lossy();
            let content = match std::fs::read_to_string(file) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let _ = self.touchFile(&pathStr, &content).await;
        }

        // Wait once, then collect all diagnostics.
        tokio::time::sleep(timeout).await;

        for file in &files {
            let pathStr = file.to_string_lossy();
            let ext = match fileExtension(&pathStr) {
                Some(e) => e,
                None => continue,
            };
            let matching = self.findServersForExtension(&ext);
            for (serverId, _) in matching {
                if self.unavailable.contains(&serverId) {
                    continue;
                }
                let root = self.findProjectRoot(&pathStr, &serverId);
                let key = format!("{serverId}:{}", root.display());
                if let Some(conn) = self.connections.get(&key) {
                    let diags = conn
                        .awaitDiagnostics(&pathStr, Duration::from_millis(100))
                        .await;
                    if !diags.is_empty() {
                        fileDiagnostics.push((pathStr.to_string(), diags));
                    }
                }
            }
        }

        if fileDiagnostics.is_empty() {
            return format!("Clean \u{2014} no errors in {dirPath}.");
        }

        let pairs: Vec<(&str, &[async_lsp::lsp_types::Diagnostic])> = fileDiagnostics
            .iter()
            .map(|(p, d)| (p.as_str(), d.as_slice()))
            .collect();

        diagnostics::formatMultiFileDiagnostics(&pairs, minSeverity)
    }

    /// Find server defs matching a file extension, returning (serverId, languageId) pairs.
    fn findServersForExtension(&self, ext: &str) -> Vec<(String, String)> {
        let mut results = Vec::new();
        for def in &self.serverDefs {
            for (i, defExt) in def.extensions.iter().enumerate() {
                if defExt == ext {
                    let langId = def
                        .languageIds
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| ext.trim_start_matches('.').to_string());
                    results.push((def.id.clone(), langId));
                    break;
                }
            }
        }

        // Resolve conflicts: biome and typescript-language-server both handle JS/TS.
        // Keep only one per conflict group.
        resolveConflicts(&mut results);

        results
    }

    /// Walk up from file path to find a project root using root markers.
    ///
    /// Returns the outermost matching directory so that workspace roots
    /// (e.g. a Cargo workspace) are preferred over subcrate roots. This
    /// prevents spawning redundant server instances for workspace members.
    fn findProjectRoot(&self, filePath: &str, serverId: &str) -> PathBuf {
        let Some(def) = self.serverDefs.iter().find(|d| d.id == serverId) else {
            return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        };

        if def.rootMarkers.is_empty() {
            return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        }

        let path = Path::new(filePath);
        let mut dir = if path.is_file() {
            path.parent().map(|p| p.to_path_buf())
        } else {
            Some(path.to_path_buf())
        };

        // Walk all the way up, keeping the outermost match.
        let mut bestMatch: Option<PathBuf> = None;
        while let Some(ref d) = dir {
            for marker in &def.rootMarkers {
                if d.join(marker).exists() {
                    bestMatch = Some(d.clone());
                    break;
                }
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }

        bestMatch.unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        })
    }
}

/// Availability status of a server.
#[derive(Debug, Clone)]
pub enum ServerAvailability {
    /// Server is running and connected.
    Active,
    /// Server is currently starting up.
    Starting,
    /// Binary found on PATH but not currently running.
    Installed,
    /// Server failed to start.
    Failed(String),
    /// Binary not found on PATH.
    NotInstalled,
}

/// Full status for a server definition (for the /lsp panel).
#[derive(Debug, Clone)]
pub struct FullServerStatus {
    pub id: String,
    pub extensions: Vec<String>,
    pub installHint: String,
    pub runtime: Option<String>,
    pub status: ServerAvailability,
}

/// Scan a project directory for file extensions (top-level, src/, and project markers).
fn scanExtensions(projectDir: &Path) -> HashSet<String> {
    let mut exts = HashSet::new();
    let dirs = [projectDir.to_path_buf(), projectDir.join("src")];

    for dir in &dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension() {
                    exts.insert(format!(".{}", ext.to_string_lossy()));
                }
            }
        }
    }

    let markerToExt: &[(&str, &str)] = &[
        ("Cargo.toml", ".rs"),
        ("pyproject.toml", ".py"),
        ("package.json", ".ts"),
        ("go.mod", ".go"),
        ("CMakeLists.txt", ".c"),
        ("compile_commands.json", ".c"),
    ];
    for (marker, ext) in markerToExt {
        if projectDir.join(marker).exists() {
            exts.insert(ext.to_string());
        }
    }

    exts
}

/// Resolve a path to absolute, using cwd if relative.
fn absolutePath(path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path).to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string())
    }
}

/// Check if a binary exists on PATH.
fn which(command: &str) -> bool {
    // Handle commands with arguments (e.g. "ty server" — check "ty").
    let binary = command.split_whitespace().next().unwrap_or(command);
    std::process::Command::new("which")
        .arg(binary)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Extract the file extension (including the dot) from a path.
fn fileExtension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
}

/// Conflict groups — servers that handle overlapping extensions.
/// Within each group, only the first match (by priority order) is kept.
const CONFLICT_GROUPS: &[&[&str]] = &[
    // Biome wins over typescript-language-server for JS/TS.
    &["biome", "typescript-language-server"],
];

/// Remove lower-priority servers when multiple servers from the same
/// conflict group match.
fn resolveConflicts(results: &mut Vec<(String, String)>) {
    for group in CONFLICT_GROUPS {
        let matched: Vec<&str> = group
            .iter()
            .filter(|id| results.iter().any(|(rid, _)| rid == **id))
            .copied()
            .collect();

        // If more than one server from this group matched, keep only the highest priority.
        if matched.len() > 1 {
            let keep = matched[0];
            results.retain(|(id, _)| !group.contains(&id.as_str()) || id == keep);
        }
    }
}
