#![allow(non_snake_case)]

//! Provider model catalog discovery for the `/model` panel.

use std::collections::BTreeMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use reqwest::header::AUTHORIZATION;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{Config, configDir};

const CACHE_FILE: &str = "model-catalog.json";
const CACHE_TTL_SECS: u64 = 6 * 60 * 60;
const CODEX_CACHE_TTL_SECS: u64 = 5 * 60;
const CODEX_MODELS_CLIENT_VERSION: &str = "0.124.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalogEntry {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub contextWindow: Option<usize>,
    pub promptPrice: Option<String>,
    pub completionPrice: Option<String>,
    #[serde(default)]
    pub reasoningEfforts: Vec<String>,
    #[serde(default)]
    pub defaultReasoningEffort: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CachedProviderCatalog {
    fetchedAt: u64,
    models: Vec<ModelCatalogEntry>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelCatalogCache {
    providers: BTreeMap<String, CachedProviderCatalog>,
}

pub async fn discoverModels(config: &Config, provider: &str) -> Result<Vec<ModelCatalogEntry>> {
    let cache = readCache();
    if let Some(cached) = cache.providers.get(provider)
        && nowSecs().saturating_sub(cached.fetchedAt) <= cacheTtlSecs(provider)
    {
        return Ok(cached.models.clone());
    }

    match fetchProviderModels(config, provider).await {
        Ok(models) => {
            writeProviderCache(provider, &models);
            Ok(models)
        }
        Err(err) => {
            if let Some(cached) = cache.providers.get(provider) {
                return Ok(cached.models.clone());
            }
            if provider == "openai-codex" {
                return Ok(codexPresets());
            }
            Err(err)
        }
    }
}

pub fn knownModelContextWindow(provider: &str, model: &str) -> Option<usize> {
    readCache()
        .providers
        .get(provider)
        .into_iter()
        .flat_map(|catalog| catalog.models.iter())
        .find(|entry| entry.id == model)
        .and_then(|entry| entry.contextWindow)
        .or_else(|| match provider {
            "openai" => officialOpenAiModelSpec(model).and_then(|spec| spec.contextWindow),
            "openai-codex" => codexPresets()
                .into_iter()
                .find(|entry| entry.id == model)
                .and_then(|entry| entry.contextWindow),
            _ => None,
        })
}

pub fn knownModelReasoningEfforts(provider: &str, model: &str) -> Vec<String> {
    if let Some(efforts) = readCache()
        .providers
        .get(provider)
        .into_iter()
        .flat_map(|catalog| catalog.models.iter())
        .find(|entry| entry.id == model)
        .map(|entry| entry.reasoningEfforts.clone())
        .filter(|efforts| !efforts.is_empty())
    {
        return efforts;
    }

    match provider {
        "openai" => officialOpenAiModelSpec(model)
            .map(|spec| spec.reasoningEfforts())
            .unwrap_or_default(),
        "openai-codex" => codexPresets()
            .into_iter()
            .find(|entry| entry.id == model)
            .map(|entry| entry.reasoningEfforts)
            .unwrap_or_default(),
        "deepseek" => vec![
            "disabled".to_string(),
            "high".to_string(),
            "max".to_string(),
        ],
        _ => Vec::new(),
    }
}

async fn fetchProviderModels(config: &Config, provider: &str) -> Result<Vec<ModelCatalogEntry>> {
    let http = reqwest::Client::new();
    match provider {
        "openrouter" => fetchOpenRouter(&http).await,
        "openai" => fetchOpenAi(config, &http).await,
        "deepseek" => fetchDeepSeek(config, &http).await,
        "openai-codex" => fetchOpenAiCodex(config, &http).await,
        other => bail!("model discovery is not implemented for provider {other:?}"),
    }
}

async fn fetchOpenRouter(http: &reqwest::Client) -> Result<Vec<ModelCatalogEntry>> {
    let body = http
        .get("https://openrouter.ai/api/v1/models")
        .send()
        .await
        .context("failed to fetch OpenRouter models")?
        .error_for_status()
        .context("OpenRouter model discovery failed")?
        .json::<Value>()
        .await
        .context("failed to parse OpenRouter model catalog")?;
    Ok(parseOpenRouterModels(&body))
}

async fn fetchOpenAi(config: &Config, http: &reqwest::Client) -> Result<Vec<ModelCatalogEntry>> {
    let key = providerKey(config, "openai")
        .context("OpenAI model discovery needs an OpenAI API profile or OPENAI_API_KEY")?;
    let base = providerBaseUrl(config, "openai", "https://api.openai.com/v1");
    let body = http
        .get(format!("{base}/models"))
        .header(AUTHORIZATION, format!("Bearer {key}"))
        .send()
        .await
        .context("failed to fetch OpenAI models")?
        .error_for_status()
        .context("OpenAI model discovery failed")?
        .json::<Value>()
        .await
        .context("failed to parse OpenAI model catalog")?;
    Ok(parseOpenAiCompatibleModels(&body, "openai"))
}

async fn fetchDeepSeek(config: &Config, http: &reqwest::Client) -> Result<Vec<ModelCatalogEntry>> {
    let key = providerKey(config, "deepseek")
        .context("DeepSeek model discovery needs a DeepSeek API profile or DEEPSEEK_API_KEY")?;
    let base = providerBaseUrl(config, "deepseek", "https://api.deepseek.com");
    let body = http
        .get(format!("{base}/models"))
        .header(AUTHORIZATION, format!("Bearer {key}"))
        .send()
        .await
        .context("failed to fetch DeepSeek models")?
        .error_for_status()
        .context("DeepSeek model discovery failed")?
        .json::<Value>()
        .await
        .context("failed to parse DeepSeek model catalog")?;
    Ok(parseOpenAiCompatibleModels(&body, "deepseek"))
}

async fn fetchOpenAiCodex(
    config: &Config,
    http: &reqwest::Client,
) -> Result<Vec<ModelCatalogEntry>> {
    let access = crate::auth::codexAccessToken()
        .await
        .context("OpenAI Codex model discovery needs `flatline auth login openai-codex`")?;
    let base = providerBaseUrl(
        config,
        "openai-codex",
        "https://chatgpt.com/backend-api/codex",
    );
    let separator = if base.contains('?') { '&' } else { '?' };
    let url = format!("{base}/models{separator}client_version={CODEX_MODELS_CLIENT_VERSION}");
    let mut req = http
        .get(url)
        .header(AUTHORIZATION, format!("Bearer {}", access.accessToken));
    if let Some(accountId) = access.accountId {
        req = req.header("ChatGPT-Account-Id", accountId);
    }
    let body = req
        .send()
        .await
        .context("failed to fetch OpenAI Codex models")?
        .error_for_status()
        .context("OpenAI Codex model discovery failed")?
        .json::<Value>()
        .await
        .context("failed to parse OpenAI Codex model catalog")?;
    let models = parseCodexModels(&body);
    if models.is_empty() {
        bail!("OpenAI Codex model discovery returned no selectable models");
    }
    Ok(models)
}

fn parseOpenRouterModels(value: &Value) -> Vec<ModelCatalogEntry> {
    value
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?.to_string();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(id.as_str())
                .to_string();
            let contextWindow = item
                .get("context_length")
                .and_then(valueAsUsize)
                .or_else(|| {
                    item.pointer("/top_provider/context_length")
                        .and_then(valueAsUsize)
                });
            let promptPrice = item
                .pointer("/pricing/prompt")
                .and_then(Value::as_str)
                .map(str::to_string);
            let completionPrice = item
                .pointer("/pricing/completion")
                .and_then(Value::as_str)
                .map(str::to_string);
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(ModelCatalogEntry {
                id,
                name,
                provider: "openrouter".to_string(),
                contextWindow,
                promptPrice,
                completionPrice,
                reasoningEfforts: Vec::new(),
                defaultReasoningEffort: None,
                description,
            })
        })
        .collect()
}

fn parseOpenAiCompatibleModels(value: &Value, provider: &str) -> Vec<ModelCatalogEntry> {
    value
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?.to_string();
            let spec = if provider == "openai" {
                officialOpenAiModelSpec(&id)
            } else {
                None
            };
            Some(ModelCatalogEntry {
                name: spec
                    .as_ref()
                    .map(|spec| spec.name.to_string())
                    .unwrap_or_else(|| id.clone()),
                id: id.clone(),
                provider: provider.to_string(),
                contextWindow: spec.as_ref().and_then(|spec| spec.contextWindow),
                promptPrice: spec
                    .as_ref()
                    .and_then(|spec| spec.promptPrice.map(str::to_string)),
                completionPrice: spec
                    .as_ref()
                    .and_then(|spec| spec.completionPrice.map(str::to_string)),
                reasoningEfforts: spec
                    .as_ref()
                    .map(OfficialOpenAiModelSpec::reasoningEfforts)
                    .unwrap_or_default(),
                defaultReasoningEffort: spec
                    .as_ref()
                    .and_then(|spec| spec.defaultReasoningEffort.map(str::to_string)),
                description: spec
                    .as_ref()
                    .map(|spec| spec.description.to_string())
                    .or_else(|| {
                        item.get("owned_by")
                            .and_then(Value::as_str)
                            .map(|owner| format!("owned by {owner}"))
                    }),
            })
        })
        .collect()
}

fn parseCodexModels(value: &Value) -> Vec<ModelCatalogEntry> {
    let mut entries: Vec<(i64, ModelCatalogEntry)> = value
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = item.get("slug")?.as_str()?.to_string();
            let visible = item
                .get("visibility")
                .and_then(Value::as_str)
                .map(|visibility| visibility == "list")
                .unwrap_or(true);
            if !visible {
                return None;
            }
            let name = item
                .get("display_name")
                .and_then(Value::as_str)
                .unwrap_or(id.as_str())
                .to_string();
            let contextWindow = item
                .get("context_window")
                .and_then(valueAsUsize)
                .or_else(|| item.get("max_context_window").and_then(valueAsUsize));
            let reasoningEfforts = item
                .get("supported_reasoning_levels")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|level| {
                    level
                        .get("effort")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect();
            let defaultReasoningEffort = item
                .get("default_reasoning_level")
                .and_then(Value::as_str)
                .map(str::to_string);
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string);
            let priority = item.get("priority").and_then(Value::as_i64).unwrap_or(100);
            Some((
                priority,
                ModelCatalogEntry {
                    id,
                    name,
                    provider: "openai-codex".to_string(),
                    contextWindow,
                    promptPrice: None,
                    completionPrice: None,
                    reasoningEfforts,
                    defaultReasoningEffort,
                    description,
                },
            ))
        })
        .collect();
    entries.sort_by(|(leftPriority, left), (rightPriority, right)| {
        leftPriority
            .cmp(rightPriority)
            .then_with(|| left.id.cmp(&right.id))
    });
    entries.into_iter().map(|(_, entry)| entry).collect()
}

fn codexPresets() -> Vec<ModelCatalogEntry> {
    [
        CodexPreset {
            id: "gpt-5.5",
            name: "GPT-5.5",
            description: "Frontier model for complex coding, research, and real-world work.",
            contextWindow: Some(272_000),
        },
        CodexPreset {
            id: "gpt-5.4",
            name: "GPT-5.4",
            description: "Strong model for everyday coding.",
            contextWindow: Some(272_000),
        },
        CodexPreset {
            id: "gpt-5.4-mini",
            name: "GPT-5.4-Mini",
            description: "Small, fast, and cost-efficient model for simpler coding tasks.",
            contextWindow: Some(272_000),
        },
        CodexPreset {
            id: "gpt-5.3-codex",
            name: "GPT-5.3-Codex",
            description: "Coding-optimized model.",
            contextWindow: Some(272_000),
        },
        CodexPreset {
            id: "gpt-5.2",
            name: "GPT-5.2",
            description: "Optimized for professional work and long-running agents.",
            contextWindow: Some(272_000),
        },
    ]
    .into_iter()
    .map(|preset| ModelCatalogEntry {
        id: preset.id.to_string(),
        name: preset.name.to_string(),
        provider: "openai-codex".to_string(),
        contextWindow: preset.contextWindow,
        promptPrice: None,
        completionPrice: None,
        reasoningEfforts: codexReasoningEfforts(),
        defaultReasoningEffort: Some("medium".to_string()),
        description: Some(format!(
            "{} Built-in fallback from Codex model catalog metadata.",
            preset.description
        )),
    })
    .collect()
}

struct CodexPreset {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    contextWindow: Option<usize>,
}

fn codexReasoningEfforts() -> Vec<String> {
    ["low", "medium", "high", "xhigh"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

#[derive(Clone)]
struct OfficialOpenAiModelSpec {
    name: &'static str,
    contextWindow: Option<usize>,
    promptPrice: Option<&'static str>,
    completionPrice: Option<&'static str>,
    reasoningEfforts: &'static [&'static str],
    defaultReasoningEffort: Option<&'static str>,
    description: &'static str,
}

impl OfficialOpenAiModelSpec {
    fn reasoningEfforts(&self) -> Vec<String> {
        self.reasoningEfforts
            .iter()
            .map(|effort| (*effort).to_string())
            .collect()
    }
}

fn officialOpenAiModelSpec(id: &str) -> Option<OfficialOpenAiModelSpec> {
    match id {
        "gpt-5.5" | "gpt-5.5-2026-04-23" => Some(OfficialOpenAiModelSpec {
            name: "GPT-5.5",
            contextWindow: Some(1_050_000),
            promptPrice: Some("$5/M"),
            completionPrice: Some("$30/M"),
            reasoningEfforts: &["none", "low", "medium", "high", "xhigh"],
            defaultReasoningEffort: Some("medium"),
            description: "Frontier model for coding and professional work.",
        }),
        "gpt-5.3-codex" => Some(OfficialOpenAiModelSpec {
            name: "GPT-5.3-Codex",
            contextWindow: Some(400_000),
            promptPrice: Some("$1.75/M"),
            completionPrice: Some("$14/M"),
            reasoningEfforts: &["low", "medium", "high", "xhigh"],
            defaultReasoningEffort: Some("high"),
            description: "Agentic coding model optimized for Codex-like environments.",
        }),
        "gpt-5.2-codex" => Some(OfficialOpenAiModelSpec {
            name: "GPT-5.2-Codex",
            contextWindow: Some(400_000),
            promptPrice: Some("$1.75/M"),
            completionPrice: Some("$14/M"),
            reasoningEfforts: &["low", "medium", "high", "xhigh"],
            defaultReasoningEffort: Some("high"),
            description: "Deprecated long-horizon agentic coding model.",
        }),
        _ => None,
    }
}

fn providerKey(config: &Config, provider: &str) -> Option<String> {
    config
        .profiles
        .values()
        .find(|profile| profile.provider == provider && !profile.key.is_empty())
        .map(|profile| profile.key.clone())
}

fn providerBaseUrl(config: &Config, provider: &str, fallback: &str) -> String {
    config
        .profiles
        .values()
        .find(|profile| profile.provider == provider && !profile.baseUrl.is_empty())
        .map(|profile| profile.baseUrl.trim_end_matches('/').to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn valueAsUsize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .or_else(|| value.as_str()?.parse::<usize>().ok())
}

fn cacheTtlSecs(provider: &str) -> u64 {
    if provider == "openai-codex" {
        CODEX_CACHE_TTL_SECS
    } else {
        CACHE_TTL_SECS
    }
}

fn cachePath() -> std::path::PathBuf {
    configDir().join(CACHE_FILE)
}

fn readCache() -> ModelCatalogCache {
    let path = cachePath();
    let Ok(text) = fs::read_to_string(path) else {
        return ModelCatalogCache::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

fn writeProviderCache(provider: &str, models: &[ModelCatalogEntry]) {
    let path = cachePath();
    let mut cache = readCache();
    cache.providers.insert(
        provider.to_string(),
        CachedProviderCatalog {
            fetchedAt: nowSecs(),
            models: models.to_vec(),
        },
    );

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(&cache) {
        let _ = fs::write(path, text);
    }
}

fn nowSecs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsesOpenRouterCatalog() {
        let value = serde_json::json!({
            "data": [{
                "id": "anthropic/claude-sonnet-4.6",
                "name": "Claude Sonnet 4.6",
                "context_length": 250000,
                "pricing": { "prompt": "0.000003", "completion": "0.000015" },
                "description": "coding model"
            }]
        });

        let entries = parseOpenRouterModels(&value);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "anthropic/claude-sonnet-4.6");
        assert_eq!(entries[0].contextWindow, Some(250000));
        assert_eq!(entries[0].promptPrice.as_deref(), Some("0.000003"));
        assert!(entries[0].reasoningEfforts.is_empty());
    }

    #[test]
    fn parsesOpenAiCompatibleCatalog() {
        let value = serde_json::json!({
            "data": [{ "id": "gpt-5.5", "owned_by": "openai" }]
        });

        let entries = parseOpenAiCompatibleModels(&value, "openai");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "gpt-5.5");
        assert_eq!(entries[0].provider, "openai");
        assert_eq!(entries[0].contextWindow, Some(1_050_000));
        assert_eq!(
            entries[0].reasoningEfforts,
            ["none", "low", "medium", "high", "xhigh"]
        );
        assert_eq!(entries[0].defaultReasoningEffort.as_deref(), Some("medium"));
    }

    #[test]
    fn parsesCodexModelCatalog() {
        let value = serde_json::json!({
            "models": [
                {
                    "slug": "hidden-model",
                    "display_name": "Hidden",
                    "visibility": "hide",
                    "supported_in_api": true,
                    "priority": 0
                },
                {
                    "slug": "gpt-5.3-codex",
                    "display_name": "gpt-5.3-codex",
                    "description": "Coding-optimized model.",
                    "context_window": 272000,
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [
                        { "effort": "low", "description": "Fast responses" },
                        { "effort": "medium", "description": "Balanced" },
                        { "effort": "high", "description": "More reasoning" },
                        { "effort": "xhigh", "description": "Extra reasoning" }
                    ],
                    "visibility": "list",
                    "supported_in_api": true,
                    "priority": 6
                },
                {
                    "slug": "gpt-5.5",
                    "display_name": "GPT-5.5",
                    "description": "Frontier model.",
                    "context_window": 272000,
                    "max_context_window": 272000,
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [
                        { "effort": "low", "description": "Fast responses" },
                        { "effort": "medium", "description": "Balanced" },
                        { "effort": "high", "description": "More reasoning" },
                        { "effort": "xhigh", "description": "Extra reasoning" }
                    ],
                    "visibility": "list",
                    "supported_in_api": true,
                    "priority": 0
                }
            ]
        });

        let entries = parseCodexModels(&value);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "gpt-5.5");
        assert_eq!(entries[0].contextWindow, Some(272_000));
        assert_eq!(entries[0].defaultReasoningEffort.as_deref(), Some("medium"));
        assert_eq!(
            entries[0].reasoningEfforts,
            ["low", "medium", "high", "xhigh"]
        );
        assert_eq!(entries[1].id, "gpt-5.3-codex");
    }

    #[test]
    fn codexPresetsIncludeCatalogFallbackModels() {
        let entries = codexPresets();
        let gpt55 = entries
            .iter()
            .find(|entry| entry.id == "gpt-5.5")
            .expect("gpt-5.5 preset");
        assert_eq!(gpt55.contextWindow, Some(272_000));
        assert_eq!(gpt55.defaultReasoningEffort.as_deref(), Some("medium"));
        assert!(
            gpt55
                .reasoningEfforts
                .iter()
                .any(|effort| effort == "xhigh")
        );

        let codex = entries
            .iter()
            .find(|entry| entry.id == "gpt-5.3-codex")
            .expect("gpt-5.3-codex preset");
        assert_eq!(codex.contextWindow, Some(272_000));
        assert!(
            codex
                .reasoningEfforts
                .iter()
                .any(|effort| effort == "xhigh")
        );
        assert!(entries.iter().any(|entry| entry.id == "gpt-5.4"));
        assert!(entries.iter().any(|entry| entry.id == "gpt-5.4-mini"));
    }
}
