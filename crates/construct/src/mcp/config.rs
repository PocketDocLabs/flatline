//! MCP server configuration — JSON parsing, env interpolation, multi-source merge.
//!
//! Servers are configured in `.mcp.json` files using the cross-tool standard format.
//! Two discovery locations (lowest → highest priority):
//! 1. `~/.config/flatline/mcp.json` — user-level servers
//! 2. `.mcp.json` in CWD — project-level servers
//!
//! Flatline-specific fields (`enabled`, `enabledTools`, `disabledTools`, timeouts)
//! are JSON extensions — other tools ignore them.
//!
//! # Public API
//! - [`ServerConfig`] — configuration for a single MCP server
//! - [`TransportType`] — stdio or HTTP transport settings
//! - [`interpolateEnv`] — resolve `{env:VAR_NAME}` patterns
//! - [`loadMcpJson`] — load servers from a `.mcp.json` file
//! - [`loadMcpServers`] — full discovery: user + project merge
//!
//! # Dependencies
//! `serde`, `serde_json`

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::schema::validateServerName;

/// Configuration for a single MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Command to run (stdio transport). Mutually exclusive with `url`.
    #[serde(default)]
    pub command: Option<String>,

    /// Arguments for the command (stdio transport).
    #[serde(default)]
    pub args: Vec<String>,

    /// URL for HTTP transport. Mutually exclusive with `command`.
    #[serde(default)]
    pub url: Option<String>,

    /// Auth header value for HTTP transport (e.g. "Bearer {env:TOKEN}").
    #[serde(default)]
    pub auth: Option<String>,

    /// Additional HTTP headers.
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Environment variables to pass to the server process (stdio only).
    /// Values support `{env:VAR_NAME}` interpolation.
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Working directory for the server process (stdio only).
    #[serde(default)]
    pub cwd: Option<String>,

    /// Whether this server is enabled. Default: true.
    #[serde(default = "defaults::enabled")]
    pub enabled: bool,

    /// Allowlist of tool names. If set, only these tools are exposed.
    #[serde(default, rename = "enabled_tools")]
    pub enabledTools: Option<Vec<String>>,

    /// Blocklist of tool names. These tools are hidden even if in the allowlist.
    #[serde(default, rename = "disabled_tools")]
    pub disabledTools: Option<Vec<String>>,

    /// Startup timeout in seconds. Default: 10.
    #[serde(default = "defaults::startupTimeout", rename = "startup_timeout")]
    pub startupTimeout: u64,

    /// Tool call timeout in seconds. Default: 120.
    #[serde(default = "defaults::toolTimeout", rename = "tool_timeout")]
    pub toolTimeout: u64,

    /// Maximum output tokens per tool call. Default: 25000.
    #[serde(default = "defaults::maxOutputTokens", rename = "max_output_tokens")]
    pub maxOutputTokens: usize,
}

impl ServerConfig {
    /// Determine the transport type from config fields.
    pub fn transport(&self) -> TransportType {
        if let Some(ref url) = self.url {
            TransportType::Http {
                url: interpolateEnv(url),
                auth: self.auth.as_ref().map(|a| interpolateEnv(a)),
                headers: self
                    .headers
                    .iter()
                    .map(|(k, v)| (k.clone(), interpolateEnv(v)))
                    .collect(),
            }
        } else if let Some(ref cmd) = self.command {
            TransportType::Stdio {
                command: interpolateEnv(cmd),
                args: self.args.iter().map(|a| interpolateEnv(a)).collect(),
                cwd: self.cwd.as_ref().map(|c| interpolateEnv(c)),
                env: self
                    .env
                    .iter()
                    .map(|(k, v)| (k.clone(), interpolateEnv(v)))
                    .collect(),
            }
        } else {
            TransportType::Invalid
        }
    }
}

/// Resolved transport configuration.
#[derive(Debug)]
pub enum TransportType {
    Stdio {
        command: String,
        args: Vec<String>,
        cwd: Option<String>,
        env: HashMap<String, String>,
    },
    Http {
        url: String,
        auth: Option<String>,
        headers: HashMap<String, String>,
    },
    Invalid,
}

mod defaults {
    pub fn enabled() -> bool {
        true
    }
    pub fn startupTimeout() -> u64 {
        10
    }
    pub fn toolTimeout() -> u64 {
        120
    }
    pub fn maxOutputTokens() -> usize {
        25_000
    }
}

/// Resolve `{env:VAR_NAME}` patterns in a string from environment variables.
///
/// Unset variables resolve to empty string (with a debug log).
pub fn interpolateEnv(input: &str) -> String {
    let mut result = input.to_string();
    // Simple scan for {env:...} patterns.
    while let Some(start) = result.find("{env:") {
        let afterPrefix = start + 5; // len of "{env:"
        if let Some(end) = result[afterPrefix..].find('}') {
            let varName = &result[afterPrefix..afterPrefix + end];
            let value = std::env::var(varName).unwrap_or_else(|_| {
                tracing::debug!(var = %varName, "env var not set, using empty string");
                String::new()
            });
            result = format!("{}{value}{}", &result[..start], &result[afterPrefix + end + 1..]);
        } else {
            // Malformed pattern — stop processing.
            break;
        }
    }
    result
}

/// Validate all server names in a config map.
pub fn validateServerNames(
    servers: &HashMap<String, ServerConfig>,
) -> Result<(), String> {
    for name in servers.keys() {
        validateServerName(name)?;
    }
    Ok(())
}

/// Load MCP servers from a `.mcp.json` file at the given path.
///
/// Returns an empty map if the file doesn't exist. The standard format is:
/// ```json
/// { "mcpServers": { "name": { "command": "...", "args": [...] } } }
/// ```
///
/// Flatline extensions (`enabled`, `enabledTools`, etc.) are supported inline
/// and ignored by other tools.
pub fn loadMcpJson(
    path: &Path,
) -> Result<HashMap<String, ServerConfig>, anyhow::Error> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let contents = std::fs::read_to_string(path)?;
    parseMcpJson(&contents)
}

/// Parse a `.mcp.json` string into server configs.
fn parseMcpJson(contents: &str) -> Result<HashMap<String, ServerConfig>, anyhow::Error> {
    let file: McpJsonFile = serde_json::from_str(contents)?;
    let mut servers = HashMap::new();
    for (name, entry) in file.mcpServers {
        servers.insert(name, entry.intoServerConfig());
    }
    Ok(servers)
}

/// JSON deserialization for the standard `.mcp.json` format.
#[derive(Deserialize)]
struct McpJsonFile {
    #[serde(default, rename = "mcpServers")]
    mcpServers: HashMap<String, McpJsonServer>,
}

/// A single server entry in `.mcp.json`.
#[derive(Deserialize)]
struct McpJsonServer {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    cwd: Option<String>,

    /// Standard field — inverted to our `enabled`.
    #[serde(default)]
    disabled: bool,

    // Flatline extensions (ignored by other tools).
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    auth: Option<String>,
    #[serde(default, rename = "enabledTools")]
    enabledTools: Option<Vec<String>>,
    #[serde(default, rename = "disabledTools")]
    disabledTools: Option<Vec<String>>,
    #[serde(default, rename = "startupTimeout")]
    startupTimeout: Option<u64>,
    #[serde(default, rename = "toolTimeout")]
    toolTimeout: Option<u64>,
    #[serde(default, rename = "maxOutputTokens")]
    maxOutputTokens: Option<usize>,
}

impl McpJsonServer {
    fn intoServerConfig(self) -> ServerConfig {
        ServerConfig {
            command: self.command.map(|s| convertEnvSyntax(&s)),
            args: self.args.iter().map(|s| convertEnvSyntax(s)).collect(),
            url: self.url.map(|s| convertEnvSyntax(&s)),
            auth: self.auth.map(|s| convertEnvSyntax(&s)),
            headers: self
                .headers
                .iter()
                .map(|(k, v)| (k.clone(), convertEnvSyntax(v)))
                .collect(),
            env: self
                .env
                .iter()
                .map(|(k, v)| (k.clone(), convertEnvSyntax(v)))
                .collect(),
            cwd: self.cwd.map(|s| convertEnvSyntax(&s)),
            // Explicit `enabled` takes priority over `disabled`.
            enabled: self.enabled.unwrap_or(!self.disabled),
            enabledTools: self.enabledTools,
            disabledTools: self.disabledTools,
            startupTimeout: self.startupTimeout.unwrap_or(10),
            toolTimeout: self.toolTimeout.unwrap_or(120),
            maxOutputTokens: self.maxOutputTokens.unwrap_or(25_000),
        }
    }
}

/// Convert `.mcp.json` env var syntax (`${VAR}`) to flatline syntax (`{env:VAR}`).
///
/// Strips `:-default` fallback suffixes (not supported).
fn convertEnvSyntax(input: &str) -> String {
    let mut result = input.to_string();
    while let Some(start) = result.find("${") {
        let afterPrefix = start + 2;
        if let Some(end) = result[afterPrefix..].find('}') {
            let varExpr = &result[afterPrefix..afterPrefix + end];
            // Strip :-default suffix if present.
            let varName = varExpr.split(":-").next().unwrap_or(varExpr);
            result = format!(
                "{}{{env:{varName}}}{}",
                &result[..start],
                &result[afterPrefix + end + 1..]
            );
        } else {
            break;
        }
    }
    result
}

/// Load all MCP servers from user-level and project-level `.mcp.json` files.
///
/// Discovery order (lowest → highest priority):
/// 1. `~/.config/flatline/mcp.json`
/// 2. `.mcp.json` at project root (or CWD if no project root given)
pub fn loadMcpServers(
    projectRoot: Option<&std::path::Path>,
) -> Result<HashMap<String, ServerConfig>, anyhow::Error> {
    // User-level.
    let userDir = crate::config::configDir();
    let userServers = loadMcpJson(&userDir.join("mcp.json"))?;

    // Project-level.
    let projectDir = projectRoot
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let projectServers = loadMcpJson(&projectDir.join(".mcp.json"))?;

    let mut merged = mergeConfigs(userServers, projectServers);

    if !merged.is_empty() {
        tracing::debug!(
            count = merged.len(),
            "loaded MCP servers from .mcp.json"
        );
    }

    // Validate and remove invalid names.
    if let Err(e) = validateServerNames(&merged) {
        tracing::error!("invalid MCP server name: {e}");
        merged.retain(|name, _| {
            crate::mcp::schema::validateServerName(name).is_ok()
        });
    }

    Ok(merged)
}

/// Merge project MCP servers on top of user MCP servers.
///
/// Project servers with the same name override user servers entirely.
/// Project-only servers are added.
pub fn mergeConfigs(
    user: HashMap<String, ServerConfig>,
    project: HashMap<String, ServerConfig>,
) -> HashMap<String, ServerConfig> {
    let mut merged = user;
    for (name, config) in project {
        merged.insert(name, config);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolateEnvBasic() {
        // NOTE: set_var is unsafe in Rust 2024 edition. These tests use
        // unique var names to avoid races with other tests.
        unsafe { std::env::set_var("FLATLINE_TEST_KEY_A1", "hello") };
        let result = interpolateEnv("token={env:FLATLINE_TEST_KEY_A1}");
        assert_eq!(result, "token=hello");
        unsafe { std::env::remove_var("FLATLINE_TEST_KEY_A1") };
    }

    #[test]
    fn interpolateEnvMissing() {
        let result = interpolateEnv("{env:FLATLINE_NONEXISTENT_VAR_12345}");
        assert_eq!(result, "");
    }

    #[test]
    fn interpolateEnvMultiple() {
        unsafe { std::env::set_var("FLATLINE_TEST_A2", "one") };
        unsafe { std::env::set_var("FLATLINE_TEST_B2", "two") };
        let result = interpolateEnv("{env:FLATLINE_TEST_A2}-{env:FLATLINE_TEST_B2}");
        assert_eq!(result, "one-two");
        unsafe { std::env::remove_var("FLATLINE_TEST_A2") };
        unsafe { std::env::remove_var("FLATLINE_TEST_B2") };
    }

    #[test]
    fn interpolateEnvNoPattern() {
        assert_eq!(interpolateEnv("plain string"), "plain string");
    }

    #[test]
    fn parseMcpJsonStdio() {
        let json = r#"{
            "mcpServers": {
                "test": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-everything"]
                }
            }
        }"#;
        let servers = parseMcpJson(json).unwrap();
        let config = &servers["test"];
        assert_eq!(config.command.as_deref(), Some("npx"));
        assert_eq!(config.args, vec!["-y", "@modelcontextprotocol/server-everything"]);
        assert!(config.url.is_none());
        assert!(matches!(config.transport(), TransportType::Stdio { .. }));
    }

    #[test]
    fn parseMcpJsonHttp() {
        let json = r#"{
            "mcpServers": {
                "remote": {
                    "url": "https://api.example.com/mcp",
                    "headers": { "Authorization": "Bearer test-token" }
                }
            }
        }"#;
        let servers = parseMcpJson(json).unwrap();
        let config = &servers["remote"];
        assert!(config.url.is_some());
        assert!(config.command.is_none());
        assert!(matches!(config.transport(), TransportType::Http { .. }));
    }

    #[test]
    fn parseMcpJsonDefaults() {
        let json = r#"{ "mcpServers": { "s": { "command": "test" } } }"#;
        let servers = parseMcpJson(json).unwrap();
        let config = &servers["s"];
        assert!(config.enabled);
        assert_eq!(config.startupTimeout, 10);
        assert_eq!(config.toolTimeout, 120);
        assert_eq!(config.maxOutputTokens, 25_000);
    }

    #[test]
    fn parseMcpJsonDisabled() {
        let json = r#"{ "mcpServers": { "s": { "command": "test", "disabled": true } } }"#;
        let servers = parseMcpJson(json).unwrap();
        assert!(!servers["s"].enabled);
    }

    #[test]
    fn parseMcpJsonEnabledOverridesDisabled() {
        // Explicit `enabled` takes priority over `disabled`.
        let json = r#"{ "mcpServers": { "s": { "command": "test", "disabled": true, "enabled": true } } }"#;
        let servers = parseMcpJson(json).unwrap();
        assert!(servers["s"].enabled);
    }

    #[test]
    fn parseMcpJsonFlatlineExtensions() {
        let json = r#"{
            "mcpServers": {
                "s": {
                    "command": "test",
                    "startupTimeout": 30,
                    "toolTimeout": 60,
                    "maxOutputTokens": 50000,
                    "enabledTools": ["tool1", "tool2"],
                    "disabledTools": ["tool3"]
                }
            }
        }"#;
        let servers = parseMcpJson(json).unwrap();
        let config = &servers["s"];
        assert_eq!(config.startupTimeout, 30);
        assert_eq!(config.toolTimeout, 60);
        assert_eq!(config.maxOutputTokens, 50_000);
        assert_eq!(config.enabledTools.as_deref(), Some(&["tool1".to_string(), "tool2".to_string()][..]));
        assert_eq!(config.disabledTools.as_deref(), Some(&["tool3".to_string()][..]));
    }

    #[test]
    fn parseMcpJsonEnvVarConversion() {
        let json = r#"{
            "mcpServers": {
                "s": {
                    "command": "${MY_CMD}",
                    "args": ["--key", "${API_KEY}"],
                    "env": { "TOKEN": "${SECRET}" }
                }
            }
        }"#;
        let servers = parseMcpJson(json).unwrap();
        let config = &servers["s"];
        assert_eq!(config.command.as_deref(), Some("{env:MY_CMD}"));
        assert_eq!(config.args, vec!["--key", "{env:API_KEY}"]);
        assert_eq!(config.env.get("TOKEN").map(|s| s.as_str()), Some("{env:SECRET}"));
    }

    #[test]
    fn convertEnvSyntaxBasic() {
        assert_eq!(convertEnvSyntax("${VAR}"), "{env:VAR}");
    }

    #[test]
    fn convertEnvSyntaxMultiple() {
        assert_eq!(
            convertEnvSyntax("${A}-${B}"),
            "{env:A}-{env:B}"
        );
    }

    #[test]
    fn convertEnvSyntaxPassthrough() {
        assert_eq!(convertEnvSyntax("plain string"), "plain string");
    }

    #[test]
    fn convertEnvSyntaxWithDefault() {
        // :-default suffix is stripped.
        assert_eq!(convertEnvSyntax("${VAR:-fallback}"), "{env:VAR}");
    }

    #[test]
    fn convertEnvSyntaxPreservesExisting() {
        // Already in flatline syntax — untouched.
        assert_eq!(convertEnvSyntax("{env:VAR}"), "{env:VAR}");
    }

    #[test]
    fn mergeConfigsOverrides() {
        let user = parseMcpJson(
            r#"{ "mcpServers": { "s": { "command": "old" } } }"#,
        ).unwrap();
        let project = parseMcpJson(
            r#"{ "mcpServers": { "s": { "command": "new" } } }"#,
        ).unwrap();

        let merged = mergeConfigs(user, project);
        assert_eq!(merged["s"].command.as_deref(), Some("new"));
    }

    #[test]
    fn mergeConfigsAddsNew() {
        let user = parseMcpJson(
            r#"{ "mcpServers": { "a": { "command": "cmd-a" } } }"#,
        ).unwrap();
        let project = parseMcpJson(
            r#"{ "mcpServers": { "b": { "command": "cmd-b" } } }"#,
        ).unwrap();

        let merged = mergeConfigs(user, project);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged["a"].command.as_deref(), Some("cmd-a"));
        assert_eq!(merged["b"].command.as_deref(), Some("cmd-b"));
    }

    #[test]
    fn parseMcpJsonEmpty() {
        let json = r#"{ "mcpServers": {} }"#;
        let servers = parseMcpJson(json).unwrap();
        assert!(servers.is_empty());
    }

    #[test]
    fn parseMcpJsonMissingKey() {
        // No mcpServers key — treated as empty.
        let json = r#"{ }"#;
        let servers = parseMcpJson(json).unwrap();
        assert!(servers.is_empty());
    }
}
