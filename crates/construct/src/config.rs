//! User configuration — loaded from ~/.config/flatline/config.toml.
//!
//! # Public API
//! - [`Config`] — the full configuration struct
//! - [`load`] — load config from disk, creating defaults if missing
//!
//! # Dependencies
//! `serde`, `toml`, `dirs`

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const CONFIG_DIR: &str = "flatline";
const CONFIG_FILE: &str = "config.toml";

/// Full application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "defaults::api")]
    pub api: ApiConfig,
}

/// API provider settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API provider. Currently only "openrouter".
    #[serde(default = "defaults::provider")]
    pub provider: String,

    /// API key.
    #[serde(default)]
    pub key: String,

    /// Default model identifier.
    #[serde(default = "defaults::model")]
    pub model: String,

    /// Base URL override.
    #[serde(default = "defaults::baseUrl")]
    pub baseUrl: String,

    /// Reasoning config for thinking models.
    #[serde(default)]
    pub reasoning: Option<ReasoningSettings>,

    /// Preferred OpenRouter providers in priority order (e.g. ["Fireworks", "Moonshot"]).
    /// When set, disables fallbacks automatically.
    #[serde(default)]
    pub providerOrder: Vec<String>,
}

/// Reasoning/thinking settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSettings {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

mod defaults {
    use super::ApiConfig;

    pub fn api() -> ApiConfig {
        ApiConfig {
            provider: provider(),
            key: String::new(),
            model: model(),
            baseUrl: baseUrl(),
            reasoning: None,
            providerOrder: Vec::new(),
        }
    }

    pub fn provider() -> String {
        "openrouter".into()
    }

    pub fn model() -> String {
        "moonshotai/kimi-k2.5".into()
    }

    pub fn baseUrl() -> String {
        "https://openrouter.ai/api/v1".into()
    }
}

/// Get the config directory path.
pub fn configDir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR)
}

/// Load config from disk. Creates a default config file if none exists.
///
/// The API key is resolved in priority order:
/// 1. `OPENROUTER_API_KEY` env var
/// 2. `key` field in config.toml
pub fn load() -> Result<Config> {
    let dir = configDir();
    let path = dir.join(CONFIG_FILE);

    let mut config = if path.exists() {
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        toml::from_str(&contents).context("Failed to parse config.toml")?
    } else {
        let config = Config {
            api: defaults::api(),
        };

        // Write default config so the user can edit it.
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create config dir: {}", dir.display()))?;
        let contents = toml::to_string_pretty(&config).context("Failed to serialize config")?;
        fs::write(&path, &contents)
            .with_context(|| format!("Failed to write config: {}", path.display()))?;

        config
    };

    // Env var overrides config file.
    if let Ok(envKey) = std::env::var("OPENROUTER_API_KEY") {
        if !envKey.is_empty() {
            config.api.key = envKey;
        }
    }

    Ok(config)
}
