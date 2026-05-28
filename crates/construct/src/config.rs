//! Layered configuration — defaults < user < project < local.
//!
//! User config lives at `~/.config/flatline/config.toml`. Project config
//! lives at `<project-root>/.flatline/config.toml`, with an optional
//! gitignored `config.local.toml` for personal overrides.
//!
//! Model settings live in **named profiles**. Each profile is one flat
//! `ModelConfig` — a single model identity. Users select which profile
//! plays each tier via top-level `heavyProfile` / `lightProfile` /
//! `utilityProfile` keys (or `FLATLINE_HEAVY_PROFILE` /
//! `FLATLINE_LIGHT_PROFILE` / `FLATLINE_UTILITY_PROFILE` env vars).
//! Fallback chain: light → heavy; utility → light → heavy.
//!
//! Profiles are **atomic** across layers — if two layers both define
//! `[profile.foo]`, the higher-priority layer fully wins; fields do not
//! composite. This prevents provider-specific fields (e.g. OpenRouter
//! `providerOrder`) from leaking into other-provider configs.
//!
//! Non-profile sections (permissions, lsp, web, budget) continue to
//! composite field-by-field across layers.
//!
//! # Public API
//! - [`Config`] — the resolved configuration struct
//! - [`ModelConfig`] — a model identity (profile contents)
//! - [`load`] — load and merge config from all layers
//! - [`discoverProjectRoot`] — find the project root by walking up to `.git`
//! - [`persistPermissionRule`] — write an "always allow" rule to project config
//!
//! # Env vars
//! - `FLATLINE_CONFIG` — explicit config file path (bypasses layer discovery)
//! - `FLATLINE_HEAVY_PROFILE` — override `heavyProfile` selection
//! - `FLATLINE_LIGHT_PROFILE` — override `lightProfile` selection
//! - `FLATLINE_UTILITY_PROFILE` — override `utilityProfile` selection
//! - `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `EXA_API_KEY` — API keys
//!
//! # Dependencies
//! `serde`, `toml`, `dirs`

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::model_catalog::ModelCatalogEntry;
use crate::permissions::{Permissions, PermissionsSource, PermitMode, Rule};

const CONFIG_DIR: &str = "flatline";
const CONFIG_FILE: &str = "config.toml";
const PROJECT_DIR: &str = ".flatline";
const PROJECT_CONFIG: &str = "config.toml";
const LOCAL_CONFIG: &str = "config.local.toml";
const DEFAULT_PROFILE: &str = "default";

/// Resolved application configuration (profiles selected and resolved).
#[derive(Debug, Clone)]
pub struct Config {
    /// Name of the profile chosen for the heavy tier (display / debugging).
    pub heavyProfile: String,

    /// Name of the profile chosen for the light tier (display / debugging).
    /// Falls back to `heavyProfile` when unset.
    pub lightProfile: String,

    /// Name of the profile chosen for the utility tier (display / debugging).
    /// Falls back to `lightProfile`, then `heavyProfile` when unset.
    pub utilityProfile: String,

    /// Heavy tier — primary pair-programmer session. Resolved from `heavyProfile`.
    pub heavy: ModelConfig,

    /// Light tier — mid-weight subagents. Resolved from `lightProfile`,
    /// falling back to `heavyProfile` when unset.
    pub light: ModelConfig,

    /// Utility tier — topics, compaction, web summaries. Resolved from
    /// `utilityProfile`, falling back to `lightProfile` / `heavyProfile`.
    pub utility: ModelConfig,

    /// Resolved named model profiles available to the in-app model UI.
    pub profiles: BTreeMap<String, ModelConfig>,

    /// Context usage ratio (0.0–1.0) at which to trigger compaction.
    pub compactRatio: f64,

    /// Web tool settings (Exa API).
    pub web: WebConfig,

    /// LSP server configuration overrides. Keys are server IDs.
    pub lsp: crate::lsp::LspConfig,

    /// Permission rules. None means use built-in defaults (allowReadOnly).
    pub permissions: Option<Permissions>,

    /// Budget and cost warning settings.
    pub budget: BudgetConfig,

    /// Discovered project root (not serialized — derived at load time).
    pub projectRoot: Option<PathBuf>,

    /// Directory Flatline was launched from. Used for launch-scoped config
    /// files when the agent starts below a larger project root.
    pub launchDir: PathBuf,
}

/// Which model tier is being edited by the in-app model profile UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Heavy,
    Light,
    Utility,
}

/// Where an in-app config edit should be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigScope {
    User,
    Project,
    ProjectLocal,
    Launch,
    LaunchLocal,
}

impl ConfigScope {
    pub fn label(self) -> &'static str {
        match self {
            ConfigScope::User => "User",
            ConfigScope::Project => "Project",
            ConfigScope::ProjectLocal => "Project local",
            ConfigScope::Launch => "Launch dir",
            ConfigScope::LaunchLocal => "Launch local",
        }
    }

    pub fn shortLabel(self) -> &'static str {
        match self {
            ConfigScope::User => "user",
            ConfigScope::Project => "project",
            ConfigScope::ProjectLocal => "project local",
            ConfigScope::Launch => "launch",
            ConfigScope::LaunchLocal => "launch local",
        }
    }

    pub fn isLocal(self) -> bool {
        matches!(self, ConfigScope::ProjectLocal | ConfigScope::LaunchLocal)
    }
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
    /// API provider: "openrouter", "fireworks", "deepseek", "openai", or
    /// "openai-codex".
    pub provider: String,

    /// API key.
    #[serde(default)]
    pub key: String,

    /// Model identifier.
    pub model: String,

    /// Base URL override.
    pub baseUrl: String,

    /// Reasoning config for thinking models.
    #[serde(default)]
    pub reasoning: Option<ReasoningSettings>,

    /// Prompt-injected thinking — tells the model to reason in <thinking>
    /// blocks instead of using the official thinking API. Avoids reasoning
    /// summarization.
    pub promptThinking: bool,

    /// Preferred OpenRouter providers in priority order (e.g. ["Moonshot",
    /// "Fireworks"]). When set, disables fallbacks automatically.
    /// Only meaningful for the OpenRouter provider; defaults to empty for
    /// other providers.
    #[serde(default)]
    pub providerOrder: Vec<String>,

    /// Maximum completion tokens per response.
    #[serde(default)]
    pub maxTokens: Option<usize>,

    /// Model context window size in tokens.
    pub contextWindow: usize,

    /// Model's maximum context window in tokens. `contextWindow` may be set
    /// lower when a profile should run with a smaller working budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maxContextWindow: Option<usize>,

    /// Enable Anthropic-style prompt caching (cache_control markers on
    /// outgoing requests). None = auto-detect from model name; explicit
    /// true/false overrides. Auto-detect returns true for any model string
    /// containing "claude" or prefixed "anthropic/".
    #[serde(default)]
    pub supportsAnthropicCache: Option<bool>,
}

/// Heuristic: does this model string route to a Claude model that supports
/// Anthropic prompt caching? Matches "anthropic/…", "claude-…", or any
/// model string containing "claude" (Bedrock/Vertex variants).
pub fn isClaudeModel(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.starts_with("anthropic/") || lower.contains("claude")
}

impl ModelConfig {
    /// Resolved answer to "should we send cache_control markers?" —
    /// explicit override wins, otherwise auto-detected from model name.
    pub fn cachingActive(&self) -> bool {
        self.supportsAnthropicCache
            .unwrap_or_else(|| isClaudeModel(&self.model))
    }
}

/// Reasoning/thinking settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSettings {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

/// Which tier a profile is resolving for. Drives which set of defaults fills
/// in missing fields when the user's profile table is empty.
#[derive(Debug, Clone, Copy)]
enum Tier {
    Heavy,
    Light,
    Utility,
}

/// Provider-aware defaults for a ModelConfig.
///
/// `baseUrl` and `providerOrder` default based on the provider — this is
/// the root fix for the old leak where e.g. `providerOrder = ["Anthropic"]`
/// would flow into a Fireworks config.
///
/// `promptThinking` defaults to `false` regardless of provider or model —
/// it's an instruction-tuned technique that works with anything, so the
/// safe default is off and users flip it on per profile when they want it.
fn modelDefaults(provider: &str) -> ModelConfig {
    match provider {
        "fireworks" => ModelConfig {
            provider: "fireworks".into(),
            key: String::new(),
            model: "accounts/fireworks/models/kimi-k2p5".into(),
            baseUrl: "https://api.fireworks.ai/inference/v1".into(),
            reasoning: None,
            promptThinking: false,
            providerOrder: Vec::new(),
            maxTokens: Some(8_000),
            contextWindow: 256_000,
            maxContextWindow: Some(256_000),
            supportsAnthropicCache: None,
        },
        "deepseek" => ModelConfig {
            provider: "deepseek".into(),
            key: String::new(),
            model: "deepseek-v4-pro".into(),
            baseUrl: "https://api.deepseek.com".into(),
            reasoning: None,
            promptThinking: false,
            providerOrder: Vec::new(),
            maxTokens: Some(8_000),
            contextWindow: 128_000,
            maxContextWindow: Some(128_000),
            supportsAnthropicCache: None,
        },
        "openai" => ModelConfig {
            provider: "openai".into(),
            key: String::new(),
            model: "gpt-5.4".into(),
            baseUrl: "https://api.openai.com/v1".into(),
            reasoning: Some(ReasoningSettings {
                effort: Some("high".into()),
                summary: None,
            }),
            promptThinking: false,
            providerOrder: Vec::new(),
            maxTokens: Some(128_000),
            contextWindow: 1_050_000,
            maxContextWindow: Some(1_050_000),
            supportsAnthropicCache: Some(false),
        },
        "openai-codex" => ModelConfig {
            provider: "openai-codex".into(),
            key: String::new(),
            model: "gpt-5.3-codex".into(),
            baseUrl: "https://chatgpt.com/backend-api/codex".into(),
            reasoning: Some(ReasoningSettings {
                effort: Some("high".into()),
                summary: Some("auto".into()),
            }),
            promptThinking: false,
            providerOrder: Vec::new(),
            maxTokens: None,
            contextWindow: 272_000,
            maxContextWindow: Some(272_000),
            supportsAnthropicCache: Some(false),
        },
        // Default to OpenRouter for anything unrecognized.
        _ => ModelConfig {
            provider: "openrouter".into(),
            key: String::new(),
            model: "anthropic/claude-sonnet-4.6".into(),
            baseUrl: "https://openrouter.ai/api/v1".into(),
            reasoning: None,
            promptThinking: false,
            providerOrder: vec!["Anthropic".into()],
            maxTokens: Some(100_000),
            contextWindow: 250_000,
            maxContextWindow: Some(250_000),
            supportsAnthropicCache: None,
        },
    }
}

/// Tier-specific starter defaults, used when the profile map is empty entirely.
/// Heavy = Opus w/ prompt-thinking, Light = Sonnet w/ prompt-thinking,
/// Utility = Kimi K2.6 on OpenRouter, no prompt-thinking.
fn tierDefaults(tier: Tier) -> ModelConfig {
    match tier {
        Tier::Heavy => ModelConfig {
            provider: "openrouter".into(),
            key: String::new(),
            model: "anthropic/claude-opus-4.6".into(),
            baseUrl: "https://openrouter.ai/api/v1".into(),
            reasoning: None,
            promptThinking: true,
            providerOrder: vec!["Anthropic".into()],
            maxTokens: Some(100_000),
            contextWindow: 250_000,
            maxContextWindow: Some(250_000),
            supportsAnthropicCache: None,
        },
        Tier::Light => ModelConfig {
            provider: "openrouter".into(),
            key: String::new(),
            model: "anthropic/claude-sonnet-4.6".into(),
            baseUrl: "https://openrouter.ai/api/v1".into(),
            reasoning: None,
            promptThinking: true,
            providerOrder: vec!["Anthropic".into()],
            maxTokens: Some(100_000),
            contextWindow: 250_000,
            maxContextWindow: Some(250_000),
            supportsAnthropicCache: None,
        },
        Tier::Utility => ModelConfig {
            provider: "openrouter".into(),
            key: String::new(),
            model: "moonshotai/kimi-k2.6".into(),
            baseUrl: "https://openrouter.ai/api/v1".into(),
            reasoning: None,
            promptThinking: false,
            providerOrder: Vec::new(),
            maxTokens: Some(8_000),
            contextWindow: 256_000,
            maxContextWindow: Some(256_000),
            supportsAnthropicCache: None,
        },
    }
}

fn defaultCompactRatio() -> f64 {
    0.8
}

/// Get the user config directory path.
pub fn configDir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR)
}

/// Directory Flatline was launched from.
pub fn launchDir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn samePath(a: &Path, b: &Path) -> bool {
    a == b
}

pub fn configPathForScope(
    scope: ConfigScope,
    projectRoot: Option<&Path>,
    launchDir: &Path,
) -> Option<PathBuf> {
    match scope {
        ConfigScope::User => Some(configDir().join(CONFIG_FILE)),
        ConfigScope::Project => projectRoot.map(|root| root.join(PROJECT_DIR).join(PROJECT_CONFIG)),
        ConfigScope::ProjectLocal => {
            projectRoot.map(|root| root.join(PROJECT_DIR).join(LOCAL_CONFIG))
        }
        ConfigScope::Launch => Some(launchDir.join(PROJECT_DIR).join(PROJECT_CONFIG)),
        ConfigScope::LaunchLocal => Some(launchDir.join(PROJECT_DIR).join(LOCAL_CONFIG)),
    }
}

pub fn modelConfigScopes(config: &Config) -> Vec<ConfigScope> {
    let mut scopes = vec![ConfigScope::User];
    if config.projectRoot.is_some() {
        scopes.push(ConfigScope::Project);
        scopes.push(ConfigScope::ProjectLocal);
    }
    let launchMatchesProject = config
        .projectRoot
        .as_deref()
        .is_some_and(|root| samePath(root, &config.launchDir));
    if !launchMatchesProject {
        scopes.push(ConfigScope::Launch);
        scopes.push(ConfigScope::LaunchLocal);
    }
    scopes
}

pub fn defaultModelSaveScope(config: &Config) -> ConfigScope {
    let launchMatchesProject = config
        .projectRoot
        .as_deref()
        .is_some_and(|root| samePath(root, &config.launchDir));
    if launchMatchesProject {
        ConfigScope::ProjectLocal
    } else {
        ConfigScope::LaunchLocal
    }
}

/// Load config from all layers, merge, and resolve.
///
/// Layer order (lowest → highest priority):
/// 1. Defaults (hardcoded, provider-aware)
/// 2. User (`~/.config/flatline/config.toml`)
/// 3. Project (`.flatline/config.toml`)
/// 4. Launch directory (`<launch-dir>/.flatline/config.toml`, when distinct)
/// 5. Project local (`.flatline/config.local.toml`, gitignored)
/// 6. Launch directory local (`<launch-dir>/.flatline/config.local.toml`, when distinct)
/// 7. Env vars (`FLATLINE_HEAVY_PROFILE`, `FLATLINE_LIGHT_PROFILE`,
///    `FLATLINE_UTILITY_PROFILE`, `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`,
///    `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `EXA_API_KEY`)
pub fn load() -> Result<Config> {
    // Explicit-path override: FLATLINE_CONFIG=/path/to/config.toml bypasses
    // user/project/local discovery and loads exactly that file.
    if let Ok(explicit) = std::env::var("FLATLINE_CONFIG") {
        if !explicit.is_empty() {
            return loadExplicit(PathBuf::from(explicit));
        }
    }

    let userDir = configDir();
    let userPath = userDir.join(CONFIG_FILE);

    // If no user config exists, create a default one in new-profile shape.
    if !userPath.exists() {
        fs::create_dir_all(&userDir)
            .with_context(|| format!("Failed to create config dir: {}", userDir.display()))?;
        fs::write(&userPath, defaultConfigToml())
            .with_context(|| format!("Failed to write config: {}", userPath.display()))?;
    }

    let userLayer = loadPartial(&userPath)?;

    let launchDir = launchDir();
    let projectRoot = discoverProjectRoot();
    let mut merged = userLayer;

    let mut permsSource = if merged.permissions.is_some() {
        PermissionsSource::User
    } else {
        PermissionsSource::BuiltIn
    };

    let launchMatchesProject = projectRoot
        .as_deref()
        .is_some_and(|root| samePath(root, &launchDir));

    if let Some(ref root) = projectRoot {
        let projectDir = root.join(PROJECT_DIR);

        let projectPath = projectDir.join(PROJECT_CONFIG);
        let projectLayer = loadPartial(&projectPath)?;
        if projectLayer.permissions.is_some() {
            permsSource = PermissionsSource::Project;
        }
        merged = merged.merge(projectLayer);
    }

    if !launchMatchesProject {
        let launchProjectDir = launchDir.join(PROJECT_DIR);
        let launchPath = launchProjectDir.join(PROJECT_CONFIG);
        let launchLayer = loadPartial(&launchPath)?;
        if launchLayer.permissions.is_some() {
            permsSource = PermissionsSource::Project;
        }
        merged = merged.merge(launchLayer);
    }

    if let Some(ref root) = projectRoot {
        let projectDir = root.join(PROJECT_DIR);
        let localPath = projectDir.join(LOCAL_CONFIG);
        let localLayer = loadPartial(&localPath)?;
        if localLayer.permissions.is_some() {
            permsSource = PermissionsSource::Local;
        }
        merged = merged.merge(localLayer);

        if projectDir.exists() {
            ensureLocalGitignored(&projectDir);
        }
    }

    if !launchMatchesProject {
        let launchProjectDir = launchDir.join(PROJECT_DIR);
        let launchLocalPath = launchProjectDir.join(LOCAL_CONFIG);
        let launchLocalLayer = loadPartial(&launchLocalPath)?;
        if launchLocalLayer.permissions.is_some() {
            permsSource = PermissionsSource::Local;
        }
        merged = merged.merge(launchLocalLayer);

        if launchProjectDir.exists() {
            ensureLocalGitignored(&launchProjectDir);
        }
    }

    let heavyEnv = std::env::var("FLATLINE_HEAVY_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    let lightEnv = std::env::var("FLATLINE_LIGHT_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    let utilityEnv = std::env::var("FLATLINE_UTILITY_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    let mut config = resolveMerged(
        merged,
        ProfileOverrides {
            heavy: heavyEnv.as_deref(),
            light: lightEnv.as_deref(),
            utility: utilityEnv.as_deref(),
        },
    )?;
    config.projectRoot = projectRoot;
    config.launchDir = launchDir;

    if let Some(ref mut perms) = config.permissions {
        perms.source = permsSource;
    }

    applyEnvKey(&mut config.heavy);
    applyEnvKey(&mut config.light);
    applyEnvKey(&mut config.utility);
    applyEnvKeysToProfiles(&mut config.profiles);

    if let Ok(exaKey) = std::env::var("EXA_API_KEY") {
        if !exaKey.is_empty() {
            config.web.searchKey = exaKey;
        }
    }

    Ok(config)
}

/// Load a single config file (no user/project/local discovery).
fn loadExplicit(path: PathBuf) -> Result<Config> {
    let layer = loadPartial(&path)?;
    let heavyEnv = std::env::var("FLATLINE_HEAVY_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    let lightEnv = std::env::var("FLATLINE_LIGHT_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    let utilityEnv = std::env::var("FLATLINE_UTILITY_PROFILE")
        .ok()
        .filter(|s| !s.is_empty());
    let mut config = resolveMerged(
        layer,
        ProfileOverrides {
            heavy: heavyEnv.as_deref(),
            light: lightEnv.as_deref(),
            utility: utilityEnv.as_deref(),
        },
    )?;
    config.projectRoot = discoverProjectRoot();
    config.launchDir = launchDir();
    applyEnvKey(&mut config.heavy);
    applyEnvKey(&mut config.light);
    applyEnvKey(&mut config.utility);
    applyEnvKeysToProfiles(&mut config.profiles);
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
        "deepseek" => "DEEPSEEK_API_KEY",
        "openai" => "OPENAI_API_KEY",
        "openai-codex" => return,
        _ => "OPENROUTER_API_KEY",
    };

    if let Ok(key) = std::env::var(envVar) {
        if !key.is_empty() {
            config.key = key;
        }
    }
}

fn applyEnvKeysToProfiles(profiles: &mut BTreeMap<String, ModelConfig>) {
    for profile in profiles.values_mut() {
        applyEnvKey(profile);
    }
}

// ── Project root discovery ──────────────────────────────────────────

/// Walk from CWD upward to find the project root (first directory containing `.git`).
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
    #[serde(default)]
    heavyProfile: Option<String>,
    #[serde(default)]
    lightProfile: Option<String>,
    #[serde(default)]
    utilityProfile: Option<String>,
    /// Map of profile name → flat ModelConfig. Each profile is a single
    /// model identity; `heavyProfile` / `lightProfile` / `utilityProfile`
    /// select which one plays which tier.
    #[serde(default)]
    profile: HashMap<String, PartialModelConfig>,
    #[serde(default)]
    compactRatio: Option<f64>,
    #[serde(default)]
    web: Option<PartialWebConfig>,
    #[serde(default)]
    lsp: Option<crate::lsp::LspConfig>,
    #[serde(default)]
    permissions: Option<Permissions>,
    #[serde(default)]
    budget: Option<BudgetConfig>,
}

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
    maxContextWindow: Option<usize>,
    supportsAnthropicCache: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PartialWebConfig {
    searchKey: Option<String>,
}

/// Load a config file as a PartialConfig, detecting legacy shapes first.
fn loadPartial(path: &Path) -> Result<PartialConfig> {
    if !path.exists() {
        return Ok(PartialConfig::default());
    }
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config: {}", path.display()))?;

    // Legacy detection: top-level [main] or [utility] means pre-profile config.
    detectLegacy(&contents, path)?;

    toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

/// Emit a human-readable error and exit(2) if the config uses the old
/// top-level `[main]` / `[utility]` shape. Called from `loadPartial`.
fn detectLegacy(contents: &str, path: &Path) -> Result<()> {
    let parsed: toml::Value = match toml::from_str(contents) {
        Ok(v) => v,
        Err(_) => return Ok(()), // let the real parser emit the error below
    };
    let Some(table) = parsed.as_table() else {
        return Ok(());
    };
    if table.contains_key("main") || table.contains_key("utility") {
        eprintln!(
            "error: legacy config format detected at {}\n\n\
             Model settings must now live inside named profiles. Each\n\
             profile is one flat ModelConfig; top-level `heavyProfile` /\n\
             `lightProfile` / `utilityProfile` pick which profile plays\n\
             which tier. Example:\n\n    \
             heavyProfile   = \"default\"\n    \
             # lightProfile   = \"default\"   # defaults to heavyProfile when unset\n    \
             # utilityProfile = \"default\"   # defaults to lightProfile when unset\n\n    \
             [profile.default]\n    \
             provider = \"...\"\n    \
             model = \"...\"\n    \
             ...\n",
            path.display()
        );
        std::process::exit(2);
    }
    Ok(())
}

impl PartialConfig {
    /// Merge self (base) with overlay (higher priority).
    ///
    /// Profile map entries are **atomic**: whichever layer defines
    /// `profile.foo` last fully owns it. No field-level merge inside
    /// a profile. Non-profile fields keep their existing composite
    /// semantics.
    fn merge(mut self, overlay: Self) -> Self {
        // Atomic profile replace: overlay's profile.foo fully replaces base's.
        for (name, profile) in overlay.profile {
            self.profile.insert(name, profile);
        }

        Self {
            heavyProfile: overlay.heavyProfile.or(self.heavyProfile),
            lightProfile: overlay.lightProfile.or(self.lightProfile),
            utilityProfile: overlay.utilityProfile.or(self.utilityProfile),
            profile: self.profile,
            compactRatio: overlay.compactRatio.or(self.compactRatio),
            web: mergePartialWeb(self.web, overlay.web),
            lsp: mergeLsp(self.lsp, overlay.lsp),
            permissions: overlay.permissions.or(self.permissions),
            budget: overlay.budget.or(self.budget),
        }
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
            for (k, v) in o {
                b.insert(k, v);
            }
            Some(b)
        }
    }
}

/// Overrides passed from the CLI / env layer to influence which profiles
/// get picked without threading env-var reads into the resolver itself.
#[derive(Debug, Clone, Default)]
struct ProfileOverrides<'a> {
    heavy: Option<&'a str>,
    light: Option<&'a str>,
    utility: Option<&'a str>,
}

/// Resolve a PartialConfig into a full Config by:
/// 1. Picking `heavyProfile` (override > field > "default")
/// 2. Picking `lightProfile` (override > field > heavyProfile)
/// 3. Picking `utilityProfile` (override > field > lightProfile)
/// 4. Looking each up in the profile map and resolving with tier-aware defaults
/// 5. Filling remaining top-level fields
fn resolveMerged(partial: PartialConfig, overrides: ProfileOverrides<'_>) -> Result<Config> {
    let heavyName = overrides
        .heavy
        .map(|s| s.to_string())
        .or(partial.heavyProfile.clone())
        .unwrap_or_else(|| DEFAULT_PROFILE.to_string());
    let lightName = overrides
        .light
        .map(|s| s.to_string())
        .or(partial.lightProfile.clone())
        .unwrap_or_else(|| heavyName.clone());
    let utilityName = overrides
        .utility
        .map(|s| s.to_string())
        .or(partial.utilityProfile.clone())
        .unwrap_or_else(|| lightName.clone());

    // Always route through lookupProfile so empty configs get tier-specific
    // defaults. When the profile map has entries, two tiers pointing at the
    // same named profile naturally resolve to identical ModelConfigs.
    let heavy = lookupProfile(&partial.profile, &heavyName, "heavyProfile", Tier::Heavy)?;
    let light = lookupProfile(&partial.profile, &lightName, "lightProfile", Tier::Light)?;
    let utility = lookupProfile(
        &partial.profile,
        &utilityName,
        "utilityProfile",
        Tier::Utility,
    )?;
    let profiles = resolveProfiles(&partial.profile, &heavyName, &heavy);

    Ok(Config {
        heavyProfile: heavyName,
        lightProfile: lightName,
        utilityProfile: utilityName,
        heavy,
        light,
        utility,
        profiles,
        compactRatio: partial.compactRatio.unwrap_or_else(defaultCompactRatio),
        web: resolveWeb(partial.web),
        lsp: partial.lsp.unwrap_or_default(),
        permissions: partial.permissions,
        budget: partial.budget.unwrap_or_default(),
        projectRoot: None,
        launchDir: launchDir(),
    })
}

fn resolveProfiles(
    profiles: &HashMap<String, PartialModelConfig>,
    fallbackName: &str,
    fallbackConfig: &ModelConfig,
) -> BTreeMap<String, ModelConfig> {
    if profiles.is_empty() {
        return BTreeMap::from([(fallbackName.to_string(), fallbackConfig.clone())]);
    }

    profiles
        .iter()
        .map(|(name, profile)| (name.clone(), resolveModel(Some(profile.clone()))))
        .collect()
}

/// Look up a profile by name and resolve it, applying provider-aware defaults.
///
/// If the map is empty entirely, synthesizes tier-appropriate defaults (so
/// an empty config gets Opus/Sonnet/Kimi for heavy/light/utility). Otherwise
/// a missing-but-referenced profile is a hard error.
fn lookupProfile(
    profiles: &HashMap<String, PartialModelConfig>,
    name: &str,
    role: &str,
    tier: Tier,
) -> Result<ModelConfig> {
    if profiles.is_empty() {
        return Ok(tierDefaults(tier));
    }
    match profiles.get(name) {
        Some(p) => Ok(resolveModel(Some(p.clone()))),
        None => anyhow::bail!(
            "{role} references profile {name:?}, which is not defined. Available: {:?}",
            profiles.keys().collect::<Vec<_>>()
        ),
    }
}

/// Fill a PartialModelConfig with provider-aware defaults.
///
/// The provider field is picked first (from partial or default), then the
/// remaining fields fall back to defaults keyed off that provider.
fn resolveModel(partial: Option<PartialModelConfig>) -> ModelConfig {
    let partial = partial.unwrap_or_default();
    let provider = partial
        .provider
        .clone()
        .unwrap_or_else(|| "openrouter".to_string());
    let defaults = modelDefaults(&provider);
    let model = partial.model.unwrap_or(defaults.model);
    let maxContextWindow = partial
        .maxContextWindow
        .or_else(|| crate::model_catalog::knownModelContextWindow(&provider, &model))
        .or(defaults.maxContextWindow);
    let mut contextWindow = partial.contextWindow.unwrap_or(defaults.contextWindow);
    if let Some(max) = maxContextWindow {
        contextWindow = contextWindow.min(max);
    }
    ModelConfig {
        provider,
        key: partial.key.unwrap_or(defaults.key),
        model,
        baseUrl: partial.baseUrl.unwrap_or(defaults.baseUrl),
        reasoning: partial.reasoning.or(defaults.reasoning),
        promptThinking: partial.promptThinking.unwrap_or(defaults.promptThinking),
        providerOrder: partial.providerOrder.unwrap_or(defaults.providerOrder),
        maxTokens: partial.maxTokens.or(defaults.maxTokens),
        contextWindow,
        maxContextWindow,
        supportsAnthropicCache: partial
            .supportsAnthropicCache
            .or(defaults.supportsAnthropicCache),
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

/// TOML text written when no user config exists yet.
///
/// Three-tier starter config: Opus (heavy), Sonnet (light), Kimi K2.5 via
/// OpenRouter (utility). All three active profiles share the OpenRouter
/// provider so a user with only `OPENROUTER_API_KEY` gets a working setup
/// out of the box.
///
/// DeepSeek V4 Pro/Flash profiles are also defined but not selected by
/// default — switch `heavyProfile` / `lightProfile` / `utilityProfile` to
/// `deepseekPro` / `deepseekFlash` / `deepseekUtility` and set
/// `DEEPSEEK_API_KEY` to use the official DeepSeek API instead.
fn defaultConfigToml() -> String {
    format!(
        "heavyProfile   = \"opus\"\n\
         lightProfile   = \"sonnet\"\n\
         utilityProfile = \"kimi\"\n\
         compactRatio   = {compact}\n\n\
         [profile.opus]\n\
         provider       = \"openrouter\"\n\
         model          = \"anthropic/claude-opus-4.6\"\n\
         promptThinking = true\n\
         providerOrder  = [\"Anthropic\"]\n\
         contextWindow  = 250000\n\n\
         [profile.sonnet]\n\
         provider       = \"openrouter\"\n\
         model          = \"anthropic/claude-sonnet-4.6\"\n\
         promptThinking = true\n\
         providerOrder  = [\"Anthropic\"]\n\
         contextWindow  = 250000\n\n\
         [profile.kimi]\n\
         provider       = \"openrouter\"\n\
         model          = \"moonshotai/kimi-k2.6\"\n\
         contextWindow  = 256000\n\n\
         [profile.deepseekPro]\n\
         provider       = \"deepseek\"\n\
         model          = \"deepseek-v4-pro\"\n\
         contextWindow  = 128000\n\
         reasoning      = {{ effort = \"max\" }}\n\n\
         [profile.deepseekFlash]\n\
         provider       = \"deepseek\"\n\
         model          = \"deepseek-v4-flash\"\n\
         contextWindow  = 128000\n\
         reasoning      = {{ effort = \"high\" }}\n\n\
         [profile.deepseekUtility]\n\
         provider       = \"deepseek\"\n\
         model          = \"deepseek-v4-flash\"\n\
         contextWindow  = 128000\n\
         reasoning      = {{ effort = \"disabled\" }}\n\n\
         [profile.openaiCodex]\n\
         provider       = \"openai-codex\"\n\
         model          = \"gpt-5.3-codex\"\n\
         contextWindow  = 272000\n\
         reasoning      = {{ effort = \"high\", summary = \"auto\" }}\n\n\
         [profile.openaiGpt54]\n\
         provider       = \"openai\"\n\
         model          = \"gpt-5.4\"\n\
         contextWindow  = 1050000\n\
         reasoning      = {{ effort = \"high\" }}\n",
        compact = defaultCompactRatio(),
    )
}

// ── Model profile persistence ───────────────────────────────────────

/// Persist a model tier selection to `.flatline/config.local.toml`.
///
/// Model switching is personal/editor-local state, so the UI writes the
/// gitignored local config rather than the shared project config.
pub fn saveModelSelection(projectRoot: &Path, tier: ModelTier, profile: &str) -> Result<()> {
    let projectDir = projectRoot.join(PROJECT_DIR);
    let configPath = projectDir.join(LOCAL_CONFIG);
    saveModelSelectionToPath(&configPath, tier, profile)?;
    ensureLocalGitignored(&projectDir);
    Ok(())
}

/// Persist a model tier selection to the requested config scope.
pub fn saveModelSelectionInScope(
    config: &Config,
    scope: ConfigScope,
    tier: ModelTier,
    profile: &str,
) -> Result<PathBuf> {
    let configPath = configPathForScope(scope, config.projectRoot.as_deref(), &config.launchDir)
        .with_context(|| format!("config scope {} is not available", scope.label()))?;
    saveModelSelectionToPath(&configPath, tier, profile)?;
    if scope.isLocal() {
        if let Some(projectDir) = configPath.parent() {
            ensureLocalGitignored(projectDir);
        }
    }
    Ok(configPath)
}

/// Persist a discovered model choice into an existing profile in the requested
/// config scope.
pub fn saveDiscoveredModelInScope(
    config: &Config,
    scope: ConfigScope,
    profileName: &str,
    entry: &ModelCatalogEntry,
) -> Result<PathBuf> {
    let existing = config
        .profiles
        .get(profileName)
        .with_context(|| format!("unknown model profile: {profileName}"))?;
    let mut model = if existing.provider == entry.provider {
        existing.clone()
    } else {
        modelDefaults(&entry.provider)
    };
    model.provider = entry.provider.clone();
    model.model = entry.id.clone();
    if let Some(contextWindow) = entry.contextWindow {
        model.contextWindow = contextWindow;
        model.maxContextWindow = Some(contextWindow);
    }
    if entry.provider == "openai-codex" {
        model.maxTokens = None;
    }
    let shouldSeedReasoning = existing.provider != entry.provider
        || model
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_ref())
            .is_none();
    if shouldSeedReasoning {
        if let Some(effort) = &entry.defaultReasoningEffort {
            let summary = model.reasoning.as_ref().and_then(|r| r.summary.clone());
            model.reasoning = Some(ReasoningSettings {
                effort: Some(effort.clone()),
                summary,
            });
        }
    }
    saveModelProfileInScope(config, scope, profileName, &model)
}

pub fn saveModelProfileContextInScope(
    config: &Config,
    scope: ConfigScope,
    profileName: &str,
    contextWindow: usize,
) -> Result<PathBuf> {
    if contextWindow == 0 {
        bail!("context window must be greater than zero");
    }
    let existing = config
        .profiles
        .get(profileName)
        .with_context(|| format!("unknown model profile: {profileName}"))?;
    let maxContextWindow = existing.maxContextWindow.or_else(|| {
        crate::model_catalog::knownModelContextWindow(&existing.provider, &existing.model)
    });
    if let Some(max) = maxContextWindow {
        if contextWindow > max {
            bail!("context window {contextWindow} exceeds model max {max}");
        }
    }

    let mut model = existing.clone();
    model.contextWindow = contextWindow;
    model.maxContextWindow = maxContextWindow.or(model.maxContextWindow);
    saveModelProfileInScope(config, scope, profileName, &model)
}

pub fn saveModelProfileThinkingInScope(
    config: &Config,
    scope: ConfigScope,
    profileName: &str,
    promptThinking: bool,
    reasoningEffort: Option<String>,
    reasoningSummary: Option<String>,
) -> Result<PathBuf> {
    let existing = config
        .profiles
        .get(profileName)
        .with_context(|| format!("unknown model profile: {profileName}"))?;
    let mut model = existing.clone();
    model.promptThinking = promptThinking;
    model.reasoning = if reasoningEffort.is_some() || reasoningSummary.is_some() {
        Some(ReasoningSettings {
            effort: reasoningEffort,
            summary: reasoningSummary,
        })
    } else {
        None
    };
    saveModelProfileInScope(config, scope, profileName, &model)
}

pub fn saveModelProfileInScope(
    config: &Config,
    scope: ConfigScope,
    profileName: &str,
    model: &ModelConfig,
) -> Result<PathBuf> {
    let configPath = configPathForScope(scope, config.projectRoot.as_deref(), &config.launchDir)
        .with_context(|| format!("config scope {} is not available", scope.label()))?;
    saveModelProfileToPath(&configPath, profileName, model)?;
    if scope.isLocal() {
        if let Some(projectDir) = configPath.parent() {
            ensureLocalGitignored(projectDir);
        }
    }
    Ok(configPath)
}

pub fn createModelProfileInScope(
    config: &Config,
    scope: ConfigScope,
    profileName: &str,
    sourceProfile: &str,
) -> Result<PathBuf> {
    validateProfileName(profileName)?;
    if config.profiles.contains_key(profileName) {
        bail!("model profile already exists: {profileName}");
    }
    let source = config
        .profiles
        .get(sourceProfile)
        .with_context(|| format!("unknown source model profile: {sourceProfile}"))?;
    saveModelProfileInScope(config, scope, profileName, source)
}

pub fn renameModelProfileInScope(
    config: &Config,
    scope: ConfigScope,
    oldName: &str,
    newName: &str,
) -> Result<PathBuf> {
    validateProfileName(newName)?;
    if oldName == newName {
        bail!("profile already has that name");
    }
    if !config.profiles.contains_key(oldName) {
        bail!("unknown model profile: {oldName}");
    }
    if config.profiles.contains_key(newName) {
        bail!("model profile already exists: {newName}");
    }

    let configPath = configPathForScope(scope, config.projectRoot.as_deref(), &config.launchDir)
        .with_context(|| format!("config scope {} is not available", scope.label()))?;
    renameModelProfileInPath(&configPath, oldName, newName)?;
    if scope.isLocal() {
        if let Some(projectDir) = configPath.parent() {
            ensureLocalGitignored(projectDir);
        }
    }
    Ok(configPath)
}

pub fn deleteModelProfileInScope(
    config: &Config,
    scope: ConfigScope,
    profileName: &str,
) -> Result<PathBuf> {
    if config.heavyProfile == profileName
        || config.lightProfile == profileName
        || config.utilityProfile == profileName
    {
        bail!("profile {profileName} is assigned to a tier; switch that tier before deleting it");
    }

    let configPath = configPathForScope(scope, config.projectRoot.as_deref(), &config.launchDir)
        .with_context(|| format!("config scope {} is not available", scope.label()))?;
    deleteModelProfileFromPath(&configPath, profileName)?;
    if scope.isLocal() {
        if let Some(projectDir) = configPath.parent() {
            ensureLocalGitignored(projectDir);
        }
    }
    Ok(configPath)
}

fn saveModelSelectionToPath(configPath: &Path, tier: ModelTier, profile: &str) -> Result<()> {
    let projectDir = configPath
        .parent()
        .with_context(|| format!("config path has no parent: {}", configPath.display()))?;

    fs::create_dir_all(&projectDir)
        .with_context(|| format!("failed to create {}", projectDir.display()))?;

    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&existing).unwrap_or_default();
    let key = match tier {
        ModelTier::Heavy => "heavyProfile",
        ModelTier::Light => "lightProfile",
        ModelTier::Utility => "utilityProfile",
    };
    doc.insert(key.to_string(), toml::Value::String(profile.to_string()));

    let output = toml::to_string_pretty(&doc).context("failed to serialize config")?;
    fs::write(&configPath, output)
        .with_context(|| format!("failed to write {}", configPath.display()))?;
    Ok(())
}

fn renameModelProfileInPath(configPath: &Path, oldName: &str, newName: &str) -> Result<()> {
    let projectDir = configPath
        .parent()
        .with_context(|| format!("config path has no parent: {}", configPath.display()))?;

    fs::create_dir_all(&projectDir)
        .with_context(|| format!("failed to create {}", projectDir.display()))?;

    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&existing).unwrap_or_default();
    let profileValue = doc
        .get_mut("profile")
        .context("selected config file has no profile table")?;
    let profileTable = profileValue
        .as_table_mut()
        .context("config `profile` value must be a table")?;
    if profileTable.contains_key(newName) {
        bail!("selected config file already has profile {newName}");
    }
    let Some(modelValue) = profileTable.remove(oldName) else {
        bail!("profile {oldName} is not defined in the selected config file");
    };
    profileTable.insert(newName.to_string(), modelValue);

    for tierKey in ["heavyProfile", "lightProfile", "utilityProfile"] {
        if doc.get(tierKey).and_then(toml::Value::as_str) == Some(oldName) {
            doc.insert(
                tierKey.to_string(),
                toml::Value::String(newName.to_string()),
            );
        }
    }

    let output = toml::to_string_pretty(&doc).context("failed to serialize config")?;
    fs::write(&configPath, output)
        .with_context(|| format!("failed to write {}", configPath.display()))?;
    Ok(())
}

fn deleteModelProfileFromPath(configPath: &Path, profileName: &str) -> Result<()> {
    let projectDir = configPath
        .parent()
        .with_context(|| format!("config path has no parent: {}", configPath.display()))?;

    fs::create_dir_all(&projectDir)
        .with_context(|| format!("failed to create {}", projectDir.display()))?;

    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&existing).unwrap_or_default();
    let profileValue = doc
        .get_mut("profile")
        .context("selected config file has no profile table")?;
    let profileTable = profileValue
        .as_table_mut()
        .context("config `profile` value must be a table")?;
    if profileTable.remove(profileName).is_none() {
        bail!("profile {profileName} is not defined in the selected config file");
    }

    let output = toml::to_string_pretty(&doc).context("failed to serialize config")?;
    fs::write(&configPath, output)
        .with_context(|| format!("failed to write {}", configPath.display()))?;
    Ok(())
}

fn saveModelProfileToPath(configPath: &Path, profileName: &str, model: &ModelConfig) -> Result<()> {
    let projectDir = configPath
        .parent()
        .with_context(|| format!("config path has no parent: {}", configPath.display()))?;

    fs::create_dir_all(&projectDir)
        .with_context(|| format!("failed to create {}", projectDir.display()))?;

    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&existing).unwrap_or_default();
    let profileValue = doc
        .entry("profile".to_string())
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let profileTable = profileValue
        .as_table_mut()
        .context("config `profile` value must be a table")?;
    let modelValue = toml::Value::try_from(model).context("failed to serialize model profile")?;
    profileTable.insert(profileName.to_string(), modelValue);

    let output = toml::to_string_pretty(&doc).context("failed to serialize config")?;
    fs::write(&configPath, output)
        .with_context(|| format!("failed to write {}", configPath.display()))?;
    Ok(())
}

fn validateProfileName(name: &str) -> Result<()> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        bail!("profile name cannot be empty");
    }
    if trimmed != name {
        bail!("profile name cannot start or end with whitespace");
    }
    if !name.chars().all(validProfileNameChar) {
        bail!("profile names may contain only letters, numbers, dot, dash, and underscore");
    }
    Ok(())
}

fn validProfileNameChar(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_')
}

// ── Permission persistence ──────────────────────────────────────────

/// Persist a new permission rule to `.flatline/config.toml`.
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

    let existing = if configPath.exists() {
        fs::read_to_string(&configPath)
            .with_context(|| format!("failed to read {}", configPath.display()))?
    } else {
        String::new()
    };

    let mut doc: toml::Table = toml::from_str(&existing).unwrap_or_default();

    let newRule = Rule {
        tool: toolName.to_string(),
        pattern: if pattern.is_empty() {
            None
        } else {
            Some(pattern.to_string())
        },
        allow,
    };

    if doc.contains_key("permissions") {
        if let Some(toml::Value::Table(permTable)) = doc.get_mut("permissions") {
            let rules = permTable
                .entry("rules")
                .or_insert_with(|| toml::Value::Array(Vec::new()));
            if let toml::Value::Array(arr) = rules {
                arr.push(ruleToToml(&newRule));
            }
        }
    } else {
        // Seeding a brand-new [permissions] table — the caller already
        // applied `addRule(newRule)` to `currentPermissions` before
        // invoking us, so `currentPermissions.rules` is the complete,
        // correct set. Pushing `newRule` again here would duplicate it
        // in the first-ever persisted file. `_ = newRule` keeps the
        // contract on the function signature unchanged; the rule is
        // persisted via the rules collection below.
        let _ = &newRule;
        let rules: Vec<toml::Value> = currentPermissions.rules.iter().map(ruleToToml).collect();

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

/// Save a full permissions set to `.flatline/config.toml`.
pub fn savePermissions(projectRoot: &Path, defaultMode: &PermitMode, rules: &[Rule]) -> Result<()> {
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

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parseToml(src: &str) -> PartialConfig {
        toml::from_str(src).expect("parse test toml")
    }

    fn resolveOk(partial: PartialConfig) -> Config {
        resolveMerged(partial, ProfileOverrides::default()).expect("resolve")
    }

    fn resolveWithHeavy(partial: PartialConfig, heavy: &str) -> Config {
        resolveMerged(
            partial,
            ProfileOverrides {
                heavy: Some(heavy),
                light: None,
                utility: None,
            },
        )
        .expect("resolve")
    }

    #[test]
    fn modelScopesPreferLaunchLocalWhenLaunchDiffersFromProject() {
        let mut cfg = resolveOk(PartialConfig::default());
        let project = tempfile::tempdir().expect("project tempdir");
        let launch = tempfile::tempdir().expect("launch tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = launch.path().to_path_buf();

        let scopes = modelConfigScopes(&cfg);
        assert_eq!(defaultModelSaveScope(&cfg), ConfigScope::LaunchLocal);
        assert!(scopes.contains(&ConfigScope::User));
        assert!(scopes.contains(&ConfigScope::Project));
        assert!(scopes.contains(&ConfigScope::ProjectLocal));
        assert!(scopes.contains(&ConfigScope::Launch));
        assert!(scopes.contains(&ConfigScope::LaunchLocal));
    }

    #[test]
    fn modelScopesCollapseLaunchWhenItMatchesProject() {
        let mut cfg = resolveOk(PartialConfig::default());
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();

        let scopes = modelConfigScopes(&cfg);
        assert_eq!(defaultModelSaveScope(&cfg), ConfigScope::ProjectLocal);
        assert!(scopes.contains(&ConfigScope::ProjectLocal));
        assert!(!scopes.contains(&ConfigScope::Launch));
        assert!(!scopes.contains(&ConfigScope::LaunchLocal));
    }

    #[test]
    fn saveModelSelectionCanTargetLaunchLocal() {
        let mut cfg = resolveOk(PartialConfig::default());
        let project = tempfile::tempdir().expect("project tempdir");
        let launch = tempfile::tempdir().expect("launch tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = launch.path().to_path_buf();

        let path = saveModelSelectionInScope(
            &cfg,
            ConfigScope::LaunchLocal,
            ModelTier::Heavy,
            "openaiCodex",
        )
        .expect("save model selection");

        assert_eq!(
            path,
            launch.path().join(".flatline").join("config.local.toml")
        );
        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("heavyProfile = \"openaiCodex\""));
        let ignore = fs::read_to_string(launch.path().join(".flatline").join(".gitignore"))
            .expect("read local gitignore");
        assert!(
            ignore
                .lines()
                .any(|line| line.trim() == "config.local.toml")
        );
    }

    #[test]
    fn saveDiscoveredModelWritesProfileOverride() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "sonnet"
            [profile.sonnet]
            provider = "openrouter"
            model = "anthropic/claude-sonnet-4.6"
            contextWindow = 250000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        let launch = tempfile::tempdir().expect("launch tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = launch.path().to_path_buf();

        let entry = ModelCatalogEntry {
            id: "anthropic/claude-opus-4.6".to_string(),
            name: "Claude Opus 4.6".to_string(),
            provider: "openrouter".to_string(),
            contextWindow: Some(250000),
            promptPrice: None,
            completionPrice: None,
            reasoningEfforts: Vec::new(),
            defaultReasoningEffort: None,
            description: None,
        };
        let path = saveDiscoveredModelInScope(&cfg, ConfigScope::LaunchLocal, "sonnet", &entry)
            .expect("save discovered model");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("[profile.sonnet]"));
        assert!(contents.contains("model = \"anthropic/claude-opus-4.6\""));
        assert!(contents.contains("contextWindow = 250000"));
    }

    #[test]
    fn saveDiscoveredCodexModelOmitsUnsupportedMaxTokens() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "codex"
            [profile.codex]
            provider = "openai-codex"
            model = "gpt-5.3-codex"
            maxTokens = 128000
            contextWindow = 400000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();

        let entry = ModelCatalogEntry {
            id: "gpt-5.5".to_string(),
            name: "GPT-5.5".to_string(),
            provider: "openai-codex".to_string(),
            contextWindow: Some(272_000),
            promptPrice: None,
            completionPrice: None,
            reasoningEfforts: vec!["low".into(), "medium".into(), "high".into(), "xhigh".into()],
            defaultReasoningEffort: Some("medium".into()),
            description: None,
        };
        let path = saveDiscoveredModelInScope(&cfg, ConfigScope::ProjectLocal, "codex", &entry)
            .expect("save discovered codex model");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("model = \"gpt-5.5\""));
        assert!(contents.contains("contextWindow = 272000"));
        assert!(!contents.contains("maxTokens"));
    }

    #[test]
    fn saveModelProfileContextCanLowerBudgetBelowMax() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "codex"
            [profile.codex]
            provider = "openai-codex"
            model = "gpt-5.5"
            contextWindow = 272000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();

        let path =
            saveModelProfileContextInScope(&cfg, ConfigScope::ProjectLocal, "codex", 128_000)
                .expect("save model profile context");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("contextWindow = 128000"));
        assert!(contents.contains("maxContextWindow = 272000"));
    }

    #[test]
    fn saveModelProfileContextRejectsAboveMax() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "codex"
            [profile.codex]
            provider = "openai-codex"
            model = "gpt-5.5"
            contextWindow = 272000
            maxContextWindow = 272000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();

        let err = saveModelProfileContextInScope(&cfg, ConfigScope::ProjectLocal, "codex", 300_000)
            .expect_err("context above max should fail");
        assert!(err.to_string().contains("exceeds model max"));
    }

    #[test]
    fn saveModelProfileThinkingWritesPromptAndNativeReasoning() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "opus"
            [profile.opus]
            provider = "openrouter"
            model = "anthropic/claude-opus-4.6"
            contextWindow = 250000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();

        let path = saveModelProfileThinkingInScope(
            &cfg,
            ConfigScope::ProjectLocal,
            "opus",
            true,
            Some("high".to_string()),
            Some("auto".to_string()),
        )
        .expect("save thinking settings");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("promptThinking = true"));
        assert!(contents.contains("[profile.opus.reasoning]"));
        assert!(contents.contains("effort = \"high\""));
        assert!(contents.contains("summary = \"auto\""));
    }

    #[test]
    fn createModelProfileCopiesSourceProfile() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "sonnet"
            [profile.sonnet]
            provider = "openrouter"
            model = "anthropic/claude-sonnet-4.6"
            contextWindow = 250000
            promptThinking = true
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();

        let path =
            createModelProfileInScope(&cfg, ConfigScope::ProjectLocal, "sonnetFast", "sonnet")
                .expect("create model profile");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("[profile.sonnetFast]"));
        assert!(contents.contains("model = \"anthropic/claude-sonnet-4.6\""));
        assert!(contents.contains("promptThinking = true"));
    }

    #[test]
    fn renameModelProfileUpdatesScopedTierSelection() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "sonnet"
            [profile.sonnet]
            provider = "openrouter"
            model = "anthropic/claude-sonnet-4.6"
            contextWindow = 250000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();
        let path = saveModelProfileInScope(
            &cfg,
            ConfigScope::ProjectLocal,
            "sonnet",
            cfg.profiles.get("sonnet").expect("sonnet profile"),
        )
        .expect("seed model profile");
        fs::write(
            &path,
            r#"
            heavyProfile = "sonnet"
            [profile.sonnet]
            provider = "openrouter"
            model = "anthropic/claude-sonnet-4.6"
            contextWindow = 250000
            "#,
        )
        .expect("write scoped config");

        renameModelProfileInScope(&cfg, ConfigScope::ProjectLocal, "sonnet", "sonnetWork")
            .expect("rename model profile");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(contents.contains("heavyProfile = \"sonnetWork\""));
        assert!(contents.contains("[profile.sonnetWork]"));
        assert!(!contents.contains("[profile.sonnet]"));
    }

    #[test]
    fn deleteModelProfileRemovesScopedProfileOnlyWhenUnused() {
        let mut cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "sonnet"
            [profile.sonnet]
            provider = "openrouter"
            model = "anthropic/claude-sonnet-4.6"
            contextWindow = 250000
            [profile.scratch]
            provider = "openrouter"
            model = "moonshotai/kimi-k2.6"
            contextWindow = 256000
            "#,
        ));
        let project = tempfile::tempdir().expect("project tempdir");
        cfg.projectRoot = Some(project.path().to_path_buf());
        cfg.launchDir = project.path().to_path_buf();
        let path = saveModelProfileInScope(
            &cfg,
            ConfigScope::ProjectLocal,
            "scratch",
            cfg.profiles.get("scratch").expect("scratch profile"),
        )
        .expect("seed model profile");

        deleteModelProfileInScope(&cfg, ConfigScope::ProjectLocal, "scratch")
            .expect("delete model profile");

        let contents = fs::read_to_string(&path).expect("read saved config");
        assert!(!contents.contains("[profile.scratch]"));
    }

    #[test]
    fn atomicProfileReplace() {
        // Base defines provider=openrouter + providerOrder. Overlay redefines
        // profile.foo to fireworks/kimi with no providerOrder. Merge must NOT
        // carry over base's providerOrder — that's the leak we're killing.
        let base = parseToml(
            r#"
            heavyProfile = "foo"
            [profile.foo]
            provider = "openrouter"
            model = "anthropic/claude-opus-4.6"
            providerOrder = ["Anthropic"]
            "#,
        );
        let overlay = parseToml(
            r#"
            [profile.foo]
            provider = "fireworks"
            model = "accounts/fireworks/models/kimi-k2p5"
            "#,
        );
        let merged = base.merge(overlay);
        let cfg = resolveOk(merged);

        assert_eq!(cfg.heavy.provider, "fireworks");
        assert_eq!(cfg.heavy.providerOrder, Vec::<String>::new());
        assert_eq!(cfg.heavy.baseUrl, "https://api.fireworks.ai/inference/v1");
    }

    #[test]
    fn profileMapUnionsAcrossLayers() {
        let base = parseToml(
            r#"
            [profile.foo]
            provider = "openrouter"
            model = "a"
            "#,
        );
        let overlay = parseToml(
            r#"
            [profile.bar]
            provider = "fireworks"
            model = "b"
            "#,
        );
        let merged = base.merge(overlay);
        assert_eq!(merged.profile.len(), 2);
        assert!(merged.profile.contains_key("foo"));
        assert!(merged.profile.contains_key("bar"));
    }

    #[test]
    fn lightAndUtilityDefaultToHeavy() {
        let cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "x"
            [profile.x]
            provider = "fireworks"
            model = "my-model"
            "#,
        ));
        assert_eq!(cfg.heavy.model, "my-model");
        assert_eq!(cfg.light.model, "my-model");
        assert_eq!(cfg.utility.model, "my-model");
        assert_eq!(cfg.heavyProfile, "x");
        assert_eq!(cfg.lightProfile, "x");
        assert_eq!(cfg.utilityProfile, "x");
    }

    #[test]
    fn utilityFallsBackToLightNotHeavy() {
        let cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "big"
            lightProfile = "mid"
            [profile.big]
            provider = "fireworks"
            model = "big-model"
            [profile.mid]
            provider = "fireworks"
            model = "mid-model"
            "#,
        ));
        assert_eq!(cfg.heavy.model, "big-model");
        assert_eq!(cfg.light.model, "mid-model");
        assert_eq!(cfg.utility.model, "mid-model");
        assert_eq!(cfg.utilityProfile, "mid");
    }

    #[test]
    fn allThreeTiersIndependent() {
        let cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "big"
            lightProfile = "mid"
            utilityProfile = "small"
            [profile.big]
            provider = "fireworks"
            model = "big-model"
            [profile.mid]
            provider = "fireworks"
            model = "mid-model"
            [profile.small]
            provider = "fireworks"
            model = "small-model"
            "#,
        ));
        assert_eq!(cfg.heavy.model, "big-model");
        assert_eq!(cfg.light.model, "mid-model");
        assert_eq!(cfg.utility.model, "small-model");
        assert_eq!(cfg.heavyProfile, "big");
        assert_eq!(cfg.lightProfile, "mid");
        assert_eq!(cfg.utilityProfile, "small");
    }

    #[test]
    fn providerAwareDefaults() {
        let cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "x"
            [profile.x]
            provider = "fireworks"
            model = "m"
            "#,
        ));
        assert_eq!(cfg.heavy.baseUrl, "https://api.fireworks.ai/inference/v1");
        assert!(cfg.heavy.providerOrder.is_empty());
        assert!(!cfg.heavy.promptThinking);
    }

    #[test]
    fn openaiProviderDefaults() {
        let cfg = resolveOk(parseToml(
            r#"
            heavyProfile = "codex"
            utilityProfile = "api"
            [profile.codex]
            provider = "openai-codex"
            [profile.api]
            provider = "openai"
            "#,
        ));
        assert_eq!(cfg.heavy.baseUrl, "https://chatgpt.com/backend-api/codex");
        assert_eq!(cfg.heavy.model, "gpt-5.3-codex");
        assert_eq!(cfg.heavy.key, "");
        assert_eq!(cfg.heavy.maxTokens, None);
        assert_eq!(cfg.heavy.contextWindow, 272_000);
        assert_eq!(cfg.utility.baseUrl, "https://api.openai.com/v1");
        assert_eq!(cfg.utility.model, "gpt-5.4");
        assert!(cfg.profiles.contains_key("codex"));
        assert!(cfg.profiles.contains_key("api"));
    }

    #[test]
    fn heavyProfileOverrideWins() {
        let cfg = resolveWithHeavy(
            parseToml(
                r#"
                heavyProfile = "default"
                [profile.default]
                model = "wrong"
                [profile.picked]
                provider = "fireworks"
                model = "right"
                "#,
            ),
            "picked",
        );
        assert_eq!(cfg.heavy.model, "right");
        assert_eq!(cfg.heavyProfile, "picked");
    }

    #[test]
    fn missingNamedProfileIsError() {
        let result = resolveMerged(
            parseToml(
                r#"
                heavyProfile = "nope"
                [profile.default]
                model = "x"
                "#,
            ),
            ProfileOverrides::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn emptyConfigSynthesizesTierAppropriateDefaults() {
        // No profiles defined anywhere — each tier gets its own starter
        // (Opus for heavy, Sonnet for light, Kimi for utility) instead of
        // all three sharing the heavy default.
        let cfg = resolveOk(PartialConfig::default());
        assert_eq!(cfg.heavyProfile, "default");
        assert!(cfg.heavy.model.contains("opus"));
        assert!(cfg.heavy.promptThinking);
        assert!(cfg.light.model.contains("sonnet"));
        assert!(cfg.light.promptThinking);
        assert!(cfg.utility.model.contains("kimi"));
        assert!(!cfg.utility.promptThinking);
    }

    #[test]
    fn firstPersistDoesNotDuplicateNewRule() {
        // Setup: a project root in a tempdir, no existing config file,
        // and a Permissions instance that already has `addRule(newRule)`
        // applied — matching the in-session call order.
        use crate::permissions::{Permissions, Rule};
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        let mut perms = Permissions::askForEverything();
        let newRule = Rule {
            tool: "shell".into(),
            pattern: Some("git status".into()),
            allow: true,
        };
        perms.addRule(newRule);

        persistPermissionRule(root, &perms, "shell", "git status", true).expect("persist rule");

        let written = std::fs::read_to_string(root.join(".flatline").join("config.toml"))
            .expect("read persisted config");
        // Should appear exactly once in the file. Two copies would be the
        // duplicate-on-first-persist bug.
        let occurrences = written.matches("pattern = \"git status\"").count();
        assert_eq!(
            occurrences, 1,
            "rule duplicated on first persist:\n{written}",
        );

        // Second persist of a different rule appends, not duplicates.
        let secondRule = Rule {
            tool: "shell".into(),
            pattern: Some("ls -la".into()),
            allow: true,
        };
        perms.addRule(secondRule);
        persistPermissionRule(root, &perms, "shell", "ls -la", true).expect("persist second");
        let written2 = std::fs::read_to_string(root.join(".flatline").join("config.toml"))
            .expect("read again");
        assert_eq!(written2.matches("pattern = \"git status\"").count(), 1);
        assert_eq!(written2.matches("pattern = \"ls -la\"").count(), 1);
    }

    #[test]
    fn starterTomlParsesAndExposesProviderProfiles() {
        // Guard against escape-brace mistakes in the format! template and
        // confirm the three DeepSeek profiles land in the profile map with
        // their configured reasoning effort.
        let toml = defaultConfigToml();
        let partial: PartialConfig = ::toml::from_str(&toml).expect("parse starter toml");

        for name in [
            "opus",
            "sonnet",
            "kimi",
            "deepseekPro",
            "deepseekFlash",
            "deepseekUtility",
            "openaiCodex",
            "openaiGpt54",
        ] {
            assert!(
                partial.profile.contains_key(name),
                "starter toml missing profile {name}"
            );
        }

        let pro = partial.profile.get("deepseekPro").unwrap();
        assert_eq!(pro.provider.as_deref(), Some("deepseek"));
        assert_eq!(pro.model.as_deref(), Some("deepseek-v4-pro"));
        assert_eq!(
            pro.reasoning.as_ref().unwrap().effort.as_deref(),
            Some("max")
        );

        let flash = partial.profile.get("deepseekFlash").unwrap();
        assert_eq!(
            flash.reasoning.as_ref().unwrap().effort.as_deref(),
            Some("high")
        );

        let util = partial.profile.get("deepseekUtility").unwrap();
        assert_eq!(
            util.reasoning.as_ref().unwrap().effort.as_deref(),
            Some("disabled")
        );

        let codex = partial.profile.get("openaiCodex").unwrap();
        assert_eq!(codex.provider.as_deref(), Some("openai-codex"));
        assert_eq!(codex.model.as_deref(), Some("gpt-5.3-codex"));
    }

    /// Hit OpenRouter's public model catalog and `/endpoints` per-model
    /// API and assert every default OpenRouter `model` slug we ship —
    /// across `tierDefaults`, `modelDefaults`, and `defaultConfigToml` —
    /// (a) exists in the catalog and (b) is still served by every
    /// provider in its `providerOrder` pin. A pinned provider that has
    /// dropped the model is functionally identical to a 404 — OR will
    /// return "no endpoints found" at runtime — so the test fails it.
    ///
    /// `#[ignore]` because it requires network. Run with
    /// `cargo test --package construct defaultOpenrouterModelsAndProvidersExist -- --ignored`.
    #[test]
    #[ignore = "network: hits openrouter.ai/api/v1/models{,/endpoints}"]
    fn defaultOpenrouterModelsAndProvidersExist() {
        // (model, providerOrder) for every default OR profile/model we ship.
        let starter: PartialConfig = ::toml::from_str(&defaultConfigToml()).unwrap();
        let mut shipped: Vec<(String, Vec<String>)> = starter
            .profile
            .values()
            .filter(|p| p.provider.as_deref() == Some("openrouter"))
            .filter_map(|p| {
                p.model
                    .clone()
                    .map(|m| (m, p.providerOrder.clone().unwrap_or_default()))
            })
            .collect();
        for tier in [Tier::Heavy, Tier::Light, Tier::Utility] {
            let m = tierDefaults(tier);
            if m.provider == "openrouter" {
                shipped.push((m.model, m.providerOrder));
            }
        }
        let mFallback = modelDefaults("openrouter");
        if mFallback.provider == "openrouter" {
            shipped.push((mFallback.model, mFallback.providerOrder));
        }
        shipped.sort();
        shipped.dedup();
        assert!(!shipped.is_empty(), "no OpenRouter defaults collected");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let client = reqwest::Client::new();

        // Catalog: every model slug currently listed.
        let catalog: Vec<String> = rt.block_on(async {
            let body = client
                .get("https://openrouter.ai/api/v1/models")
                .send()
                .await
                .expect("GET /models")
                .text()
                .await
                .expect("body");
            let v: serde_json::Value = serde_json::from_str(&body).expect("json");
            v["data"]
                .as_array()
                .expect("data array")
                .iter()
                .filter_map(|m| m["id"].as_str().map(String::from))
                .collect()
        });
        assert!(
            catalog.len() > 50,
            "suspiciously small catalog: {}",
            catalog.len()
        );

        // Step 1 — every shipped model exists in the catalog.
        let missingModels: Vec<&String> = shipped
            .iter()
            .map(|(m, _)| m)
            .filter(|s| !catalog.contains(s))
            .collect();
        assert!(
            missingModels.is_empty(),
            "default OpenRouter model slugs not in catalog: {missingModels:?}",
        );

        // Step 2 — for each model with a providerOrder pin, fetch its
        // endpoints and confirm every pinned provider is still serving
        // it. OR's `/api/v1/models/{slug}/endpoints` returns provider
        // names under `data.endpoints[].provider_name`.
        let mut failures: Vec<String> = Vec::new();
        for (model, order) in &shipped {
            if order.is_empty() {
                continue;
            }
            let url = format!("https://openrouter.ai/api/v1/models/{model}/endpoints");
            let providers: Vec<String> = rt.block_on(async {
                let body = client.get(&url).send().await.unwrap().text().await.unwrap();
                let v: serde_json::Value = serde_json::from_str(&body).unwrap();
                v["data"]["endpoints"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|e| e["provider_name"].as_str().map(String::from))
                    .collect()
            });
            for pin in order {
                let needle = pin.to_ascii_lowercase();
                let found = providers
                    .iter()
                    .any(|p| p.to_ascii_lowercase().contains(&needle));
                if !found {
                    failures.push(format!(
                        "{model}: pinned provider {pin:?} not in active endpoints {providers:?}",
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
