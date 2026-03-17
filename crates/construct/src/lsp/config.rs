//! LSP configuration loading and merging.
//!
//! Servers have built-in definitions in [`servers.rs`]. User config in
//! `~/.config/flatline/config.toml` under `[lsp."server-id"]` tables can
//! override built-in settings or add custom servers. Project config in
//! `.flatline/lsp.toml` merges on top.
//!
//! # Public API
//! - [`LspServerConfig`] — per-server configuration overrides
//! - [`LspConfig`] — resolved config for all servers
//! - [`loadProjectLsp`] — load project-scoped LSP config
//! - [`resolveServers`] — merge built-ins with user/project overrides
//!
//! # Dependencies
//! `serde`, `toml`

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::servers::{ServerDef, BUILTIN_SERVERS};

/// Per-server configuration override from user/project config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerConfig {
    /// Override the command to run.
    #[serde(default)]
    pub command: Option<String>,

    /// Override the arguments.
    #[serde(default)]
    pub args: Option<Vec<String>>,

    /// Environment variables for the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Whether this server is enabled. Default: true.
    #[serde(default = "defaults::enabled")]
    pub enabled: bool,

    /// File extensions this server handles (required for custom servers).
    #[serde(default)]
    pub extensions: Option<Vec<String>>,

    /// Project root marker files (optional for custom servers).
    #[serde(default)]
    pub rootMarkers: Option<Vec<String>>,

    /// LSP languageId values (parallel to extensions, for custom servers).
    #[serde(default)]
    pub languageIds: Option<Vec<String>>,

    /// Startup/initialize timeout in seconds. Default: 30.
    #[serde(default = "defaults::startupTimeout", rename = "startup_timeout")]
    pub startupTimeout: u64,

    /// Diagnostics wait timeout in seconds. Default: 3.
    #[serde(default = "defaults::diagnosticsTimeout", rename = "diagnostics_timeout")]
    pub diagnosticsTimeout: u64,
}

mod defaults {
    pub fn enabled() -> bool {
        true
    }
    pub fn startupTimeout() -> u64 {
        30
    }
    pub fn diagnosticsTimeout() -> u64 {
        3
    }
}

/// Fully resolved server configuration ready for connection.
#[derive(Debug, Clone)]
pub struct ResolvedServer {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub extensions: Vec<String>,
    pub rootMarkers: Vec<String>,
    pub languageIds: Vec<String>,
    pub installHint: String,
    pub runtime: Option<String>,
    pub startupTimeout: u64,
    pub diagnosticsTimeout: u64,
}

/// Top-level LSP config wrapper for TOML deserialization.
pub type LspConfig = HashMap<String, LspServerConfig>;

/// Resolve built-in server definitions with user/project overrides.
///
/// Built-in servers can be overridden by matching id. Custom servers
/// (ids not in BUILTIN_SERVERS) are added if they have a command and
/// extensions defined.
///
/// Args:
///     userConfig: User-level overrides from config.toml.
///     projectConfig: Project-level overrides from .flatline/lsp.toml.
///
/// Returns:
///     Vec of fully resolved server configs.
pub fn resolveServers(
    userConfig: &LspConfig,
    projectConfig: &LspConfig,
) -> Vec<ResolvedServer> {
    // Project overrides user.
    let mut merged: HashMap<String, &LspServerConfig> = HashMap::new();
    for (id, cfg) in userConfig {
        merged.insert(id.clone(), cfg);
    }
    for (id, cfg) in projectConfig {
        merged.insert(id.clone(), cfg);
    }

    let mut servers = Vec::new();

    // Resolve built-ins, applying overrides.
    for builtin in BUILTIN_SERVERS {
        let resolved = if let Some(override_) = merged.get(builtin.id) {
            if !override_.enabled {
                continue;
            }
            resolveBuiltinWithOverride(builtin, override_)
        } else {
            resolveBuiltinDefault(builtin)
        };
        servers.push(resolved);
    }

    // Add custom servers (ids not matching any built-in).
    let builtinIds: Vec<&str> = BUILTIN_SERVERS.iter().map(|s| s.id).collect();
    for (id, cfg) in &merged {
        if builtinIds.contains(&id.as_str()) {
            continue;
        }
        if !cfg.enabled {
            continue;
        }
        if let Some(resolved) = resolveCustom(id, cfg) {
            servers.push(resolved);
        } else {
            tracing::warn!(
                id = %id,
                "custom LSP server missing required fields (command, extensions)"
            );
        }
    }

    servers
}

fn resolveBuiltinDefault(def: &ServerDef) -> ResolvedServer {
    ResolvedServer {
        id: def.id.to_string(),
        command: def.command.to_string(),
        args: def.args.iter().map(|s| s.to_string()).collect(),
        env: HashMap::new(),
        extensions: def.extensions.iter().map(|s| s.to_string()).collect(),
        rootMarkers: def.rootMarkers.iter().map(|s| s.to_string()).collect(),
        languageIds: def.languageIds.iter().map(|s| s.to_string()).collect(),
        installHint: def.installHint.to_string(),
        runtime: def.runtime.map(|s| s.to_string()),
        startupTimeout: defaults::startupTimeout(),
        diagnosticsTimeout: defaults::diagnosticsTimeout(),
    }
}

fn resolveBuiltinWithOverride(def: &ServerDef, cfg: &LspServerConfig) -> ResolvedServer {
    ResolvedServer {
        id: def.id.to_string(),
        command: cfg
            .command
            .clone()
            .unwrap_or_else(|| def.command.to_string()),
        args: cfg.args.clone().unwrap_or_else(|| {
            def.args.iter().map(|s| s.to_string()).collect()
        }),
        env: cfg.env.clone(),
        extensions: cfg.extensions.clone().unwrap_or_else(|| {
            def.extensions.iter().map(|s| s.to_string()).collect()
        }),
        rootMarkers: cfg.rootMarkers.clone().unwrap_or_else(|| {
            def.rootMarkers.iter().map(|s| s.to_string()).collect()
        }),
        languageIds: cfg.languageIds.clone().unwrap_or_else(|| {
            def.languageIds.iter().map(|s| s.to_string()).collect()
        }),
        installHint: def.installHint.to_string(),
        runtime: def.runtime.map(|s| s.to_string()),
        startupTimeout: cfg.startupTimeout,
        diagnosticsTimeout: cfg.diagnosticsTimeout,
    }
}

fn resolveCustom(id: &str, cfg: &LspServerConfig) -> Option<ResolvedServer> {
    let command = cfg.command.as_ref()?;
    let extensions = cfg.extensions.as_ref()?;
    if extensions.is_empty() {
        return None;
    }

    let languageIds = cfg.languageIds.clone().unwrap_or_else(|| {
        // Default languageId: strip the dot from extension.
        extensions
            .iter()
            .map(|e| e.trim_start_matches('.').to_string())
            .collect()
    });

    Some(ResolvedServer {
        id: id.to_string(),
        command: command.clone(),
        args: cfg.args.clone().unwrap_or_default(),
        env: cfg.env.clone(),
        extensions: extensions.clone(),
        rootMarkers: cfg.rootMarkers.clone().unwrap_or_default(),
        languageIds,
        installHint: String::new(),
        runtime: None,
        startupTimeout: cfg.startupTimeout,
        diagnosticsTimeout: cfg.diagnosticsTimeout,
    })
}

/// Load project-scoped LSP config from `.flatline/lsp.toml`.
///
/// Returns an empty map if the file doesn't exist.
pub fn loadProjectLsp(projectDir: &Path) -> Result<LspConfig, anyhow::Error> {
    let path = projectDir.join(".flatline").join("lsp.toml");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let contents = std::fs::read_to_string(&path)?;

    #[derive(Deserialize)]
    struct ProjectLsp {
        #[serde(default)]
        lsp: LspConfig,
    }

    let parsed: ProjectLsp = toml::from_str(&contents)?;
    Ok(parsed.lsp)
}
