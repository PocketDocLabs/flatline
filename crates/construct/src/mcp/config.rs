//! MCP server configuration — TOML parsing, env interpolation, project/user merge.
//!
//! Servers are configured in `~/.config/flatline/config.toml` under `[mcp."name"]`
//! tables, or in a project-scoped `.flatline/mcp.toml`. Project config merges
//! on top of user config (project servers override same-name user servers).
//!
//! # Public API
//! - [`ServerConfig`] — configuration for a single MCP server
//! - [`TransportConfig`] — stdio or HTTP transport settings
//! - [`interpolateEnv`] — resolve `{env:VAR_NAME}` patterns
//! - [`loadProjectMcp`] — load project-scoped MCP config
//!
//! # Dependencies
//! `serde`, `toml`, `regex`

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

/// Load project-scoped MCP config from `.flatline/mcp.toml` in the given directory.
///
/// Returns an empty map if the file doesn't exist.
pub fn loadProjectMcp(
    projectDir: &Path,
) -> Result<HashMap<String, ServerConfig>, anyhow::Error> {
    let path = projectDir.join(".flatline").join("mcp.toml");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let contents = std::fs::read_to_string(&path)?;

    // The project file uses the same [mcp."name"] table structure.
    #[derive(Deserialize)]
    struct ProjectMcp {
        #[serde(default)]
        mcp: HashMap<String, ServerConfig>,
    }

    let parsed: ProjectMcp = toml::from_str(&contents)?;
    Ok(parsed.mcp)
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
    fn serverConfigStdio() {
        let toml = r#"
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-everything"]
        "#;
        let config: ServerConfig = toml::from_str(toml).unwrap();
        assert!(config.command.is_some());
        assert!(config.url.is_none());
        assert!(matches!(config.transport(), TransportType::Stdio { .. }));
    }

    #[test]
    fn serverConfigHttp() {
        let toml = r#"
            url = "https://api.example.com/mcp"
            auth = "Bearer test-token"
        "#;
        let config: ServerConfig = toml::from_str(toml).unwrap();
        assert!(config.url.is_some());
        assert!(config.command.is_none());
        assert!(matches!(config.transport(), TransportType::Http { .. }));
    }

    #[test]
    fn serverConfigDefaults() {
        let config: ServerConfig = toml::from_str("command = \"test\"").unwrap();
        assert!(config.enabled);
        assert_eq!(config.startupTimeout, 10);
        assert_eq!(config.toolTimeout, 120);
        assert_eq!(config.maxOutputTokens, 25_000);
    }

    #[test]
    fn mergeConfigsOverrides() {
        let mut user = HashMap::new();
        user.insert(
            "server1".into(),
            toml::from_str::<ServerConfig>("command = \"old\"").unwrap(),
        );

        let mut project = HashMap::new();
        project.insert(
            "server1".into(),
            toml::from_str::<ServerConfig>("command = \"new\"").unwrap(),
        );

        let merged = mergeConfigs(user, project);
        assert_eq!(
            merged["server1"].command.as_deref(),
            Some("new")
        );
    }
}
