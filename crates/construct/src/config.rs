//! Layered configuration — defaults < user < project < local.
//!
//! User config lives at `~/.config/flatline/config.toml`. Project config
//! lives at `<project-root>/.flatline/config.toml`, with an optional
//! gitignored `config.local.toml` for personal overrides.
//!
//! # Public API
//! - [`Config`] — the resolved configuration struct
//! - [`ModelConfig`] — per-model API settings (main and utility)
//! - [`load`] — load and merge config from all layers
//! - [`discoverProjectRoot`] — find the project root by walking up to `.git`
//! - [`persistPermissionRule`] — write an "always allow" rule to project config
//!
//! # Dependencies
//! `serde`, `toml`, `dirs`

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::permissions::{PermissionsSource, PermitMode, Permissions, Rule};

const CONFIG_DIR: &str = "flatline";
const CONFIG_FILE: &str = "config.toml";
const PROJECT_DIR: &str = ".flatline";
const PROJECT_CONFIG: &str = "config.toml";
const LOCAL_CONFIG: &str = "config.local.toml";

/// Resolved application configuration (all layers merged, no Option fields).
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

    /// Permission rules. None means use built-in defaults (allowReadOnly).
    #[serde(default, skip_serializing)]
    pub permissions: Option<Permissions>,

    /// Budget and cost warning settings.
    #[serde(default)]
    pub budget: BudgetConfig,

    /// Discovered project root (not serialized — derived at load time).
    #[serde(skip)]
    pub projectRoot: Option<PathBuf>,
}

/// Budget and cost warning settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BudgetConfig {
    /// Session cost warning threshold (USD). Emits a warning when exceeded.
    #[serde(default)]
    pub sessionLimit: Option<f64>,
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
    /// API provider: "openrouter" or "fireworks".
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

/// Get the user config directory path.
pub fn configDir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR)
}

/// Load config from all layers, merge, and resolve.
///
/// Layer order (lowest → highest priority):
/// 1. Defaults (hardcoded)
/// 2. User (`~/.config/flatline/config.toml`)
/// 3. Project (`.flatline/config.toml`)
/// 4. Local (`.flatline/config.local.toml`, gitignored)
/// 5. Env vars (`OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`, `EXA_API_KEY`)
pub fn load() -> Result<Config> {
    let userDir = configDir();
    let userPath = userDir.join(CONFIG_FILE);

    // Layer 1+2: defaults merged with user config.
    let userLayer = loadPartial(&userPath)?;

    // If no user config exists, create a default one.
    if !userPath.exists() {
        let defaultConfig = Config {
            main: defaults::mainModel(),
            utility: defaults::utilityModel(),
            compactRatio: defaults::compactRatio(),
            web: WebConfig::default(),
            lsp: HashMap::new(),
            permissions: None,
            budget: BudgetConfig::default(),
            projectRoot: None,
        };
        fs::create_dir_all(&userDir)
            .with_context(|| format!("Failed to create config dir: {}", userDir.display()))?;
        let contents =
            toml::to_string_pretty(&defaultConfig).context("Failed to serialize config")?;
        fs::write(&userPath, &contents)
            .with_context(|| format!("Failed to write config: {}", userPath.display()))?;
    }

    // Discover project root and load project/local layers.
    let projectRoot = discoverProjectRoot();
    let mut merged = userLayer;

    // Track which layer last set [permissions] for source tagging.
    let mut permsSource = if merged.permissions.is_some() {
        PermissionsSource::User
    } else {
        PermissionsSource::BuiltIn
    };

    if let Some(ref root) = projectRoot {
        let projectDir = root.join(PROJECT_DIR);

        // Layer 3: project config.
        let projectPath = projectDir.join(PROJECT_CONFIG);
        let projectLayer = loadPartial(&projectPath)?;
        if projectLayer.permissions.is_some() {
            permsSource = PermissionsSource::Project;
        }
        merged = merged.merge(projectLayer);

        // Layer 4: local config (personal, gitignored).
        let localPath = projectDir.join(LOCAL_CONFIG);
        let localLayer = loadPartial(&localPath)?;
        if localLayer.permissions.is_some() {
            permsSource = PermissionsSource::Local;
        }
        merged = merged.merge(localLayer);

        // Ensure config.local.toml is gitignored.
        if projectDir.exists() {
            ensureLocalGitignored(&projectDir);
        }
    }

    // Resolve partial into full config.
    let mut config = merged.resolve();
    config.projectRoot = projectRoot;

    // Tag permissions with their source layer.
    if let Some(ref mut perms) = config.permissions {
        perms.source = permsSource;
    }

    // Layer 5: env vars always win.
    // Apply provider-specific API key env vars.
    applyEnvKey(&mut config.main);
    applyEnvKey(&mut config.utility);

    if let Ok(exaKey) = std::env::var("EXA_API_KEY") {
        if !exaKey.is_empty() {
            config.web.searchKey = exaKey;
        }
    }

    Ok(config)
}

/// Apply provider-specific env var to a model config if its key is empty.
fn applyEnvKey(config: &mut ModelConfig) {
    if !config.key.is_empty() {
        return;
    }

    let envVar = match config.provider.as_str() {
        "fireworks" => "FIREWORKS_API_KEY",
        // Default to OpenRouter for backward compatibility.
        _ => "OPENROUTER_API_KEY",
    };

    if let Ok(key) = std::env::var(envVar) {
        if !key.is_empty() {
            config.key = key;
        }
    }
}

// ── Project root discovery ──────────────────────────────────────────

/// Walk from CWD upward to find the project root (first directory containing `.git`).
/// Falls back to CWD if no `.git` is found.
pub fn discoverProjectRoot() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut dir = cwd.as_path();

    loop {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return Some(cwd),
        }
    }
}

/// Ensure `.flatline/.gitignore` contains `config.local.toml`.
fn ensureLocalGitignored(projectDir: &Path) {
    let gitignorePath = projectDir.join(".gitignore");

    let contents = fs::read_to_string(&gitignorePath).unwrap_or_default();
    if contents.lines().any(|line| line.trim() == LOCAL_CONFIG) {
        return;
    }

    let entry = if contents.is_empty() || contents.ends_with('\n') {
        format!("{LOCAL_CONFIG}\n")
    } else {
        format!("\n{LOCAL_CONFIG}\n")
    };

    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&gitignorePath)
        .and_then(|mut f| std::io::Write::write_all(&mut f, entry.as_bytes()))
    {
        tracing::warn!("failed to update .flatline/.gitignore: {e}");
    }
}

// ── Partial config types (for layered merging) ──────────────────────

/// Partial config — every field optional for layer merging.
#[derive(Debug, Clone, Default, Deserialize)]
struct PartialConfig {
    main: Option<PartialModelConfig>,
    utility: Option<PartialModelConfig>,
    compactRatio: Option<f64>,
    web: Option<PartialWebConfig>,
    lsp: Option<crate::lsp::LspConfig>,
    permissions: Option<Permissions>,
    budget: Option<BudgetConfig>,
}

/// Partial model config — mirrors ModelConfig with all fields optional.
#[derive(Debug, Clone, Default, Deserialize)]
struct PartialModelConfig {
    provider: Option<String>,
    key: Option<String>,
    model: Option<String>,
    baseUrl: Option<String>,
    reasoning: Option<ReasoningSettings>,
    promptThinking: Option<bool>,
    providerOrder: Option<Vec<String>>,
    maxTokens: Option<usize>,
    contextWindow: Option<usize>,
}

/// Partial web config.
#[derive(Debug, Clone, Default, Deserialize)]
struct PartialWebConfig {
    searchKey: Option<String>,
}

/// Load a config file as a PartialConfig. Returns default (all None) if file doesn't exist.
fn loadPartial(path: &Path) -> Result<PartialConfig> {
    if !path.exists() {
        return Ok(PartialConfig::default());
    }
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config: {}", path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))
}

impl PartialConfig {
    /// Merge self (base) with overlay (higher priority). Overlay wins for set fields.
    fn merge(self, overlay: Self) -> Self {
        Self {
            main: mergePartialModel(self.main, overlay.main),
            utility: mergePartialModel(self.utility, overlay.utility),
            compactRatio: overlay.compactRatio.or(self.compactRatio),
            web: mergePartialWeb(self.web, overlay.web),
            lsp: mergeLsp(self.lsp, overlay.lsp),
            // Permissions use replace semantics — overlay wins entirely.
            permissions: overlay.permissions.or(self.permissions),
            budget: overlay.budget.or(self.budget),
        }
    }

    /// Resolve partial config into a full Config by filling in defaults.
    fn resolve(self) -> Config {
        Config {
            main: resolveModel(self.main, defaults::mainModel()),
            utility: resolveModel(self.utility, defaults::utilityModel()),
            compactRatio: self.compactRatio.unwrap_or_else(defaults::compactRatio),
            web: resolveWeb(self.web),
            lsp: self.lsp.unwrap_or_default(),
            permissions: self.permissions,
            budget: self.budget.unwrap_or_default(),
            projectRoot: None,
        }
    }
}

fn mergePartialModel(
    base: Option<PartialModelConfig>,
    overlay: Option<PartialModelConfig>,
) -> Option<PartialModelConfig> {
    match (base, overlay) {
        (None, None) => None,
        (Some(b), None) => Some(b),
        (None, Some(o)) => Some(o),
        (Some(b), Some(o)) => Some(PartialModelConfig {
            provider: o.provider.or(b.provider),
            key: o.key.or(b.key),
            model: o.model.or(b.model),
            baseUrl: o.baseUrl.or(b.baseUrl),
            reasoning: o.reasoning.or(b.reasoning),
            promptThinking: o.promptThinking.or(b.promptThinking),
            providerOrder: o.providerOrder.or(b.providerOrder),
            maxTokens: o.maxTokens.or(b.maxTokens),
            contextWindow: o.contextWindow.or(b.contextWindow),
        }),
    }
}

fn mergePartialWeb(
    base: Option<PartialWebConfig>,
    overlay: Option<PartialWebConfig>,
) -> Option<PartialWebConfig> {
    match (base, overlay) {
        (None, None) => None,
        (Some(b), None) => Some(b),
        (None, Some(o)) => Some(o),
        (Some(b), Some(o)) => Some(PartialWebConfig {
            searchKey: o.searchKey.or(b.searchKey),
        }),
    }
}

fn mergeLsp(
    base: Option<crate::lsp::LspConfig>,
    overlay: Option<crate::lsp::LspConfig>,
) -> Option<crate::lsp::LspConfig> {
    match (base, overlay) {
        (None, None) => None,
        (Some(b), None) => Some(b),
        (None, Some(o)) => Some(o),
        (Some(mut b), Some(o)) => {
            // Deep merge: overlay entries override base entries by key.
            for (k, v) in o {
                b.insert(k, v);
            }
            Some(b)
        }
    }
}

fn resolveModel(partial: Option<PartialModelConfig>, fallback: ModelConfig) -> ModelConfig {
    match partial {
        None => fallback,
        Some(p) => ModelConfig {
            provider: p.provider.unwrap_or(fallback.provider),
            key: p.key.unwrap_or(fallback.key),
            model: p.model.unwrap_or(fallback.model),
            baseUrl: p.baseUrl.unwrap_or(fallback.baseUrl),
            reasoning: p.reasoning.or(fallback.reasoning),
            promptThinking: p.promptThinking.unwrap_or(fallback.promptThinking),
            providerOrder: p.providerOrder.unwrap_or(fallback.providerOrder),
            maxTokens: p.maxTokens.or(fallback.maxTokens),
            contextWindow: p.contextWindow.unwrap_or(fallback.contextWindow),
        },
    }
}

fn resolveWeb(partial: Option<PartialWebConfig>) -> WebConfig {
    match partial {
        None => WebConfig::default(),
        Some(p) => WebConfig {
            searchKey: p.searchKey.unwrap_or_default(),
        },
    }
}

// ── Permission persistence ──────────────────────────────────────────

/// Persist a new permission rule to `.flatline/config.toml`.
///
/// If no `[permissions]` section exists yet, snapshots the current effective
/// rules as the starting set before appending the new rule. Creates the
/// `.flatline/` directory and config file if needed.
pub fn persistPermissionRule(
    projectRoot: &Path,
    currentPermissions: &Permissions,
    toolName: &str,
    pattern: &str,
    allow: bool,
) -> Result<()> {
    let projectDir = projectRoot.join(PROJECT_DIR);
    let configPath = projectDir.join(PROJECT_CONFIG);

    fs::create_dir_all(&projectDir)
        .with_context(|| format!("failed to create {}", projectDir.display()))?;

    // Load existing project config (or empty).
    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table =
        toml::from_str(&existing).unwrap_or_default();

    let newRule = Rule {
        tool: toolName.to_string(),
        pattern: if pattern.is_empty() { None } else { Some(pattern.to_string()) },
        allow,
    };

    if doc.contains_key("permissions") {
        // Append to existing permissions section.
        if let Some(toml::Value::Table(permTable)) = doc.get_mut("permissions") {
            let rules = permTable
                .entry("rules")
                .or_insert_with(|| toml::Value::Array(Vec::new()));
            if let toml::Value::Array(arr) = rules {
                arr.push(ruleToToml(&newRule));
            }
        }
    } else {
        // Snapshot current effective rules + new rule.
        let mut rules: Vec<toml::Value> = currentPermissions
            .rules
            .iter()
            .map(ruleToToml)
            .collect();
        rules.push(ruleToToml(&newRule));

        let mut permTable = toml::Table::new();
        permTable.insert(
            "defaultMode".to_string(),
            toml::Value::String(permitModeToStr(&currentPermissions.defaultMode).to_string()),
        );
        permTable.insert("rules".to_string(), toml::Value::Array(rules));
        doc.insert("permissions".to_string(), toml::Value::Table(permTable));
    }

    let output = toml::to_string_pretty(&doc).context("failed to serialize config")?;
    fs::write(&configPath, &output)
        .with_context(|| format!("failed to write {}", configPath.display()))?;

    // Ensure local config is gitignored while we're here.
    ensureLocalGitignored(&projectDir);

    Ok(())
}

fn ruleToToml(rule: &Rule) -> toml::Value {
    let mut table = toml::Table::new();
    table.insert("tool".to_string(), toml::Value::String(rule.tool.clone()));
    if let Some(ref pattern) = rule.pattern {
        table.insert("pattern".to_string(), toml::Value::String(pattern.clone()));
    }
    table.insert("allow".to_string(), toml::Value::Boolean(rule.allow));
    toml::Value::Table(table)
}

/// Save a full permissions set to `.flatline/config.toml`, replacing the existing section.
pub fn savePermissions(
    projectRoot: &Path,
    defaultMode: &PermitMode,
    rules: &[Rule],
) -> Result<()> {
    let projectDir = projectRoot.join(PROJECT_DIR);
    let configPath = projectDir.join(PROJECT_CONFIG);

    fs::create_dir_all(&projectDir)
        .with_context(|| format!("failed to create {}", projectDir.display()))?;

    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&existing).unwrap_or_default();

    // Build the full permissions section.
    let ruleValues: Vec<toml::Value> = rules.iter().map(ruleToToml).collect();
    let mut permTable = toml::Table::new();
    permTable.insert(
        "defaultMode".to_string(),
        toml::Value::String(permitModeToStr(defaultMode).to_string()),
    );
    permTable.insert("rules".to_string(), toml::Value::Array(ruleValues));
    doc.insert("permissions".to_string(), toml::Value::Table(permTable));

    let output = toml::to_string_pretty(&doc).context("failed to serialize config")?;
    fs::write(&configPath, &output)
        .with_context(|| format!("failed to write {}", configPath.display()))?;

    ensureLocalGitignored(&projectDir);
    Ok(())
}

fn permitModeToStr(mode: &PermitMode) -> &'static str {
    match mode {
        PermitMode::Ask => "ask",
        PermitMode::Deny => "deny",
        PermitMode::Abort => "abort",
    }
}
