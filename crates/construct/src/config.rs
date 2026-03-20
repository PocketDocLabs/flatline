//! User configuration — loaded from ~/Library/Application Support/flatline/config.toml.
//!
//! # Public API
//! - [`Config`] — the full configuration struct
//! - [`ModelConfig`] — per-model API settings (main and utility)
//! - [`load`] — load config from disk, creating defaults if missing
//!
//! # Dependencies
//! `serde`, `toml`, `dirs`

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const CONFIG_DIR: &str = "flatline";
const CONFIG_FILE: &str = "config.toml";

/// Full application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Main conversation model settings.
    #[serde(default = "defaults::mainModel")]
    pub main: ModelConfig,

    /// Utility model settings (topic tracking, compaction, web summaries).
    #[serde(default = "defaults::utilityModel")]
    pub utility: ModelConfig,

    /// Context usage ratio (0.0–1.0) at which to trigger compaction.
    #[serde(default = "defaults::compactRatio")]
    pub compactRatio: f64,

    /// Web tool settings (Exa API).
    #[serde(default)]
    pub web: WebConfig,

    /// LSP server configuration overrides. Keys are server IDs.
    #[serde(default)]
    pub lsp: crate::lsp::LspConfig,
}

/// Web tool settings (Exa API).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebConfig {
    /// Exa API key for web search/fetch/similar.
    #[serde(default)]
    pub searchKey: String,
}

/// Per-model API settings — used for both main and utility models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// API provider. Currently only "openrouter".
    #[serde(default = "defaults::provider")]
    pub provider: String,

    /// API key.
    #[serde(default)]
    pub key: String,

    /// Model identifier.
    #[serde(default = "defaults::model")]
    pub model: String,

    /// Base URL override.
    #[serde(default = "defaults::baseUrl")]
    pub baseUrl: String,

    /// Reasoning config for thinking models.
    #[serde(default)]
    pub reasoning: Option<ReasoningSettings>,

    /// Prompt-injected thinking — tells the model to reason in <thinking> blocks
    /// instead of using the official thinking API. Avoids reasoning summarization.
    #[serde(default = "defaults::promptThinkingDefault")]
    pub promptThinking: bool,

    /// Preferred OpenRouter providers in priority order (e.g. ["Moonshot", "Fireworks"]).
    /// When set, disables fallbacks automatically.
    #[serde(default = "defaults::providerOrder")]
    pub providerOrder: Vec<String>,

    /// Maximum completion tokens per response.
    #[serde(default)]
    pub maxTokens: Option<usize>,

    /// Model context window size in tokens.
    #[serde(default = "defaults::contextWindow")]
    pub contextWindow: usize,
}

/// Reasoning/thinking settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSettings {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

mod defaults {
    use super::ModelConfig;

    pub fn mainModel() -> ModelConfig {
        ModelConfig {
            provider: provider(),
            key: String::new(),
            model: model(),
            baseUrl: baseUrl(),
            reasoning: None,
            promptThinking: promptThinkingDefault(),
            providerOrder: providerOrder(),
            maxTokens: maxTokens(),
            contextWindow: contextWindow(),
        }
    }

    pub fn utilityModel() -> ModelConfig {
        ModelConfig {
            provider: provider(),
            key: String::new(),
            model: model(),
            baseUrl: baseUrl(),
            reasoning: None,
            promptThinking: promptThinkingDefault(),
            providerOrder: providerOrder(),
            maxTokens: maxTokens(),
            contextWindow: contextWindow(),
        }
    }

    pub fn provider() -> String {
        "openrouter".into()
    }

    pub fn model() -> String {
        "anthropic/claude-sonnet-4-6".into()
    }

    pub fn baseUrl() -> String {
        "https://openrouter.ai/api/v1".into()
    }

    pub fn contextWindow() -> usize {
        1_000_000
    }

    pub fn maxTokens() -> Option<usize> {
        Some(100_000)
    }

    pub fn compactRatio() -> f64 {
        0.8
    }

    pub fn promptThinkingDefault() -> bool {
        true
    }

    pub fn providerOrder() -> Vec<String> {
        vec!["Anthropic".into()]
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
/// API keys are resolved in priority order:
/// 1. `OPENROUTER_API_KEY` env var (applied to both main and utility)
/// 2. `key` field in each model's config section
pub fn load() -> Result<Config> {
    let dir = configDir();
    let path = dir.join(CONFIG_FILE);

    let mut config = if path.exists() {
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        toml::from_str(&contents).context("Failed to parse config.toml")?
    } else {
        let config = Config {
            main: defaults::mainModel(),
            utility: defaults::utilityModel(),
            compactRatio: defaults::compactRatio(),
            web: WebConfig::default(),
            lsp: HashMap::new(),
        };

        // Write default config so the user can edit it.
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create config dir: {}", dir.display()))?;
        let contents = toml::to_string_pretty(&config).context("Failed to serialize config")?;
        fs::write(&path, &contents)
            .with_context(|| format!("Failed to write config: {}", path.display()))?;

        config
    };

    // Env var overrides config file for both models.
    if let Ok(envKey) = std::env::var("OPENROUTER_API_KEY") {
        if !envKey.is_empty() {
            if config.main.key.is_empty() {
                config.main.key = envKey.clone();
            }
            if config.utility.key.is_empty() {
                config.utility.key = envKey;
            }
        }
    }

    if let Ok(exaKey) = std::env::var("EXA_API_KEY") {
        if !exaKey.is_empty() {
            config.web.searchKey = exaKey;
        }
    }

    Ok(config)
}
