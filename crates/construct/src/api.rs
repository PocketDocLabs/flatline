//! Multi-provider chat completion client (OpenRouter, Fireworks, DeepSeek).
//!
//! Speaks the OpenAI-compatible chat completions format with provider-
//! specific extensions for reasoning, caching, and routing.
//!
//! # Public API
//! - [`Client`] — HTTP client for the LLM API
//! - [`Client::stream`] — send a prompt, get streaming events
//! - [`Client::complete`] — non-streaming completion for utility calls
//!
//! # Dependencies
//! `reqwest`, `tokio`, `serde_json`, `backon`

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use backon::{ExponentialBuilder, Retryable};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use tokio::sync::mpsc;

use crate::config::{Config, ModelConfig};
use crate::message::{
    Message, ReasoningConfig, StreamChunk, StreamEvent, TokenUsage, ToolDef,
};

// Emit the "providerOrder not pinned" warning once per process. Cache hit
// rates depend on every request landing on the same back-end, so we want the
// operator to see this once at startup without spamming every turn.
static PROVIDER_PIN_WARNED: AtomicBool = AtomicBool::new(false);

/// Marker error for API failures that should not be retried (400, 401, 403, etc.).
#[derive(Debug)]
struct PermanentApiError(String);

impl std::fmt::Display for PermanentApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PermanentApiError {}

/// LLM API client. Supports OpenRouter, Fireworks, and the official DeepSeek API
/// (any OpenAI-compatible endpoint).
///
/// Holds only heavy (streaming, main session) and utility (non-streaming,
/// compaction + topic tracking). The light tier is reached by subagents
/// re-entering `Session::new` with `config.heavy` swapped to the light
/// profile (see `session::executeTask`), so no light HTTP client lives
/// on the parent.
#[derive(Clone)]
pub struct Client {
    heavyHttp: reqwest::Client,
    utilityHttp: reqwest::Client,
    heavy: ModelConfig,
    utility: ModelConfig,
}

impl Client {
    /// Create a new API client from config.
    pub fn new(config: &Config) -> Result<Self> {
        if config.heavy.key.is_empty() {
            bail!("API key not set. Set the heavy profile's key in config.toml, or OPENROUTER_API_KEY / FIREWORKS_API_KEY / DEEPSEEK_API_KEY");
        }

        let heavyHttp = buildHttpClient(&config.heavy)
            .context("Failed to build heavy HTTP client")?;
        let utilityHttp = buildHttpClient(&config.utility)
            .context("Failed to build utility HTTP client")?;

        Ok(Self {
            heavyHttp,
            utilityHttp,
            heavy: config.heavy.clone(),
            utility: config.utility.clone(),
        })
    }

    /// Send a chat completion request and stream events back.
    ///
    /// Args:
    ///     messages: Conversation history.
    ///     tools: Available tool definitions.
    ///     reasoning: Optional reasoning config for thinking models.
    pub async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        reasoning: Option<&ReasoningConfig>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let url = format!("{}/chat/completions", self.heavy.baseUrl);

        let mut adaptedMessages = adaptMessages(messages, &self.heavy.provider);

        if self.heavy.cachingActive() {
            injectCacheControl(&mut adaptedMessages);
            warnIfProviderNotPinned(&self.heavy);
        }

        let mut body = serde_json::json!({
            "model": self.heavy.model,
            "messages": adaptedMessages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });

        if let Some(max) = self.heavy.maxTokens {
            body["max_tokens"] = serde_json::json!(max);
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools)?;
            body["tool_choice"] = serde_json::json!("auto");
        }

        if let Some(r) = reasoning {
            match self.heavy.provider.as_str() {
                "fireworks" => {
                    // Fireworks uses a top-level `reasoning_effort` string.
                    if let Some(ref effort) = r.effort {
                        body["reasoning_effort"] = serde_json::json!(effort);
                    }
                }
                "deepseek" => {
                    // DeepSeek wraps reasoning in a `thinking` object with
                    // `type` ("enabled" | "disabled") and `reasoning_effort`
                    // ("high" | "max"). The effort string "disabled" / "off"
                    // turns thinking off entirely; anything else passes
                    // through as the effort level.
                    if let Some(ref effort) = r.effort {
                        body["thinking"] = deepseekThinking(effort);
                    }
                }
                _ => {
                    // OpenRouter uses a nested `reasoning` object.
                    body["reasoning"] = serde_json::to_value(r)?;
                }
            }
        }

        // OpenRouter-specific provider routing.
        if self.heavy.provider == "openrouter" && !self.heavy.providerOrder.is_empty() {
            body["provider"] = serde_json::json!({
                "order": self.heavy.providerOrder,
                "allow_fallbacks": false,
            });
        }

        tracing::debug!(
            model = %self.heavy.model,
            messageCount = messages.len(),
            toolCount = tools.len(),
            hasReasoning = reasoning.is_some(),
            "sending API request"
        );
        tracing::trace!(body = %body, "request body");

        let http = self.heavyHttp.clone();
        let response = (|| async {
            let response = http
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Failed to send API request")?;

            if response.status().is_success() {
                return Ok(response);
            }

            let status = response.status();
            let errorBody = response.text().await.unwrap_or_default();

            // Only retry on rate limits and server errors.
            if status.as_u16() == 429 || status.is_server_error() {
                tracing::warn!(%status, body = %errorBody, "retryable API error");
                bail!("API error {status}: {errorBody}");
            }

            // Client errors (400, 401, 403, etc.) are not retryable.
            tracing::error!(%status, body = %errorBody, "API error (not retryable)");
            Err(PermanentApiError(format!("API error {status}: {errorBody}")).into())
        })
        .retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_secs(1))
                .with_max_delay(Duration::from_secs(120))
                .with_max_times(8)
                .with_jitter(),
        )
        .when(|e| e.downcast_ref::<PermanentApiError>().is_none())
        .notify(|err, dur| {
            tracing::warn!(error = %err, delay = ?dur, "retrying API request");
        })
        .await?;

        tracing::debug!(status = %response.status(), "stream started");

        let (tx, rx) = mpsc::channel(256);

        // Spawn a task to read the SSE stream.
        tokio::spawn(async move {
            if let Err(e) = readStream(response, &tx).await {
                let _ = tx.send(StreamEvent::Error(e.to_string())).await;
            }
        });

        Ok(rx)
    }

    /// Non-streaming completion for utility calls (topic tracking, compaction).
    ///
    /// Args:
    ///     messages: Conversation messages (typically system + user).
    ///     model: Model override. Uses the client's configured model if None.
    pub async fn complete(
        &self,
        messages: &[Message],
        model: Option<&str>,
    ) -> Result<(String, Option<TokenUsage>)> {
        let url = format!("{}/chat/completions", self.utility.baseUrl);
        let modelId = model.unwrap_or(&self.utility.model);

        let mut body = serde_json::json!({
            "model": modelId,
            "messages": messages,
            "stream": false,
        });

        // OpenRouter-specific provider routing.
        if self.utility.provider == "openrouter" && !self.utility.providerOrder.is_empty() {
            body["provider"] = serde_json::json!({
                "order": self.utility.providerOrder,
                "allow_fallbacks": false,
            });
        }

        // DeepSeek defaults reasoning to `high` server-side, which wastes
        // tokens on mechanical utility calls (topic naming, compaction).
        // Honour the utility profile's reasoning.effort — including
        // "disabled" — so the operator can opt out.
        if self.utility.provider == "deepseek" {
            if let Some(ref settings) = self.utility.reasoning {
                if let Some(ref effort) = settings.effort {
                    body["thinking"] = deepseekThinking(effort);
                }
            }
        }

        tracing::debug!(
            model = %modelId,
            messageCount = messages.len(),
            "sending utility completion request"
        );

        let http = self.utilityHttp.clone();
        let response = (|| async {
            let response = http
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Failed to send utility request")?;

            if response.status().is_success() {
                return Ok(response);
            }

            let status = response.status();
            let errorBody = response.text().await.unwrap_or_default();

            if status.as_u16() == 429 || status.is_server_error() {
                tracing::warn!(%status, body = %errorBody, "retryable utility API error");
                bail!("API error {status}: {errorBody}");
            }

            tracing::error!(%status, body = %errorBody, "utility API error (not retryable)");
            Err(PermanentApiError(format!("API error {status}: {errorBody}")).into())
        })
        .retry(
            ExponentialBuilder::default()
                .with_min_delay(Duration::from_secs(1))
                .with_max_delay(Duration::from_secs(60))
                .with_max_times(5)
                .with_jitter(),
        )
        .when(|e| e.downcast_ref::<PermanentApiError>().is_none())
        .notify(|err, dur| {
            tracing::warn!(error = %err, delay = ?dur, "retrying utility request");
        })
        .await?;

        let responseBody: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse utility response")?;

        let content = responseBody["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = responseBody.get("usage").and_then(|u| {
            Some(TokenUsage {
                promptTokens: u.get("prompt_tokens")?.as_u64()? as usize,
                completionTokens: u.get("completion_tokens")?.as_u64()? as usize,
                totalTokens: u.get("total_tokens")?.as_u64()? as usize,
                cost: u.get("cost").and_then(|c| c.as_f64()),
                cacheReadTokens: u.get("cache_read_input_tokens")
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as usize,
                cacheCreationTokens: u.get("cache_creation_input_tokens")
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as usize,
            })
        });

        tracing::debug!(
            model = %modelId,
            responseLen = content.len(),
            cost = ?usage.as_ref().and_then(|u| u.cost),
            "utility completion received"
        );

        Ok((content, usage))
    }
}

/// Inject Anthropic `cache_control: {type: "ephemeral"}` markers onto the
/// system prompt tail and the last user message of the serialized messages
/// array.
///
/// Placement rationale:
/// - System: caches the full `tools + system` prefix in one breakpoint (the
///   Anthropic prefix hierarchy orders tools → system → messages, so a marker
///   on the system tail implicitly covers tools too).
/// - Last user message: rolling breakpoint that advances each turn, keeping
///   accumulated conversation history cacheable within the TTL window.
///
/// Uses 2 of the 4 available breakpoint slots. Remaining 2 slots are reserved
/// for tier-2/3 work (compacted-region marker, subagent fork point).
///
/// No-op if there is no system or no user message in the array.
fn injectCacheControl(messages: &mut serde_json::Value) {
    let Some(arr) = messages.as_array_mut() else {
        return;
    };

    // System marker — find the (usually first) system message. If its
    // content contains the CACHE_BOUNDARY sentinel, split into two blocks
    // with a 1-hour marker on the static prefix. Otherwise fall back to a
    // single 5-minute marker at the tail.
    if let Some(sys) = arr.iter_mut().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
    }) {
        if !splitSystemAtBoundary(sys) {
            markContentBlock(sys, Ttl::FiveMin);
        }
    }

    // Rolling marker — last user message in the array. Always 5m.
    if let Some(user) = arr.iter_mut().rev().find(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("user")
    }) {
        markContentBlock(user, Ttl::FiveMin);
    }
}

/// Cache-control TTL. Serializes to Anthropic's string form (absent field for
/// 5-minute default, explicit `"1h"` for the one-hour variant).
#[derive(Copy, Clone)]
enum Ttl {
    FiveMin,
    OneHour,
}

fn cacheControlJson(ttl: Ttl) -> serde_json::Value {
    match ttl {
        Ttl::FiveMin => serde_json::json!({ "type": "ephemeral" }),
        Ttl::OneHour => serde_json::json!({ "type": "ephemeral", "ttl": "1h" }),
    }
}

/// If the system message's content string contains the [`prompt::CACHE_BOUNDARY`]
/// sentinel, split it into two text blocks — the static prefix (marked with a
/// 1-hour cache_control) and the dynamic suffix (unmarked). Returns true if a
/// split happened, false if the sentinel was absent.
fn splitSystemAtBoundary(msg: &mut serde_json::Value) -> bool {
    let Some(content) = msg.get("content") else {
        return false;
    };
    let Some(text) = content.as_str() else {
        // Already blocks-shaped — leave it to markContentBlock.
        return false;
    };
    let Some((staticPart, dynamicPart)) =
        text.split_once(crate::prompt::CACHE_BOUNDARY)
    else {
        return false;
    };

    let blocks = serde_json::json!([
        {
            "type": "text",
            "text": staticPart.trim_end(),
            "cache_control": cacheControlJson(Ttl::OneHour),
        },
        {
            "type": "text",
            "text": dynamicPart.trim_start(),
        },
    ]);
    msg["content"] = blocks;
    true
}

/// Rewrite a message's `content` field to carry a `cache_control` marker on
/// its trailing text block.
///
/// Handles three shapes:
/// - String content → converted to a single-element blocks array with the
///   marker on that block.
/// - Blocks array with a trailing text block → marker added to the last text
///   block in place.
/// - Blocks array with a trailing non-text block (image) → marker added to
///   the last block regardless (cache_control is valid on image blocks too).
fn markContentBlock(msg: &mut serde_json::Value, ttl: Ttl) {
    let Some(content) = msg.get_mut("content") else {
        return;
    };

    match content {
        serde_json::Value::String(s) => {
            let text = std::mem::take(s);
            *content = serde_json::json!([{
                "type": "text",
                "text": text,
                "cache_control": cacheControlJson(ttl),
            }]);
        }
        serde_json::Value::Array(blocks) => {
            if let Some(last) = blocks.last_mut() {
                if let Some(obj) = last.as_object_mut() {
                    obj.insert("cache_control".into(), cacheControlJson(ttl));
                }
            }
        }
        _ => {}
    }
}

/// Warn once per process if caching is enabled but the provider isn't
/// pinned to a single back-end — cross-provider routing thrashes the cache
/// because each back-end (Anthropic, Bedrock, Vertex) maintains its own.
fn warnIfProviderNotPinned(cfg: &ModelConfig) {
    if PROVIDER_PIN_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }

    // Only meaningful for OpenRouter — Fireworks/direct Anthropic don't
    // have multi-provider routing.
    if cfg.provider != "openrouter" {
        return;
    }

    if cfg.providerOrder.len() != 1 {
        tracing::warn!(
            providerOrder = ?cfg.providerOrder,
            "prompt caching enabled but providerOrder is not pinned to a single \
             back-end — cache hit rates will be degraded. Set providerOrder to \
             one provider (e.g. [\"Anthropic\"]) in config.toml."
        );
    }
}

/// Serialize messages with provider-specific field names.
///
/// Fireworks and DeepSeek use `reasoning_content` where OpenRouter uses
/// `reasoning` on assistant messages. This translates at the JSON boundary
/// so the internal Message type stays provider-agnostic.
fn adaptMessages(messages: &[Message], provider: &str) -> serde_json::Value {
    let mut value = serde_json::to_value(messages).unwrap_or_default();
    if provider == "fireworks" || provider == "deepseek" {
        if let serde_json::Value::Array(ref mut arr) = value {
            for msg in arr.iter_mut() {
                if let serde_json::Value::Object(map) = msg {
                    if map.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                        if let Some(reasoning) = map.remove("reasoning") {
                            map.insert("reasoning_content".to_string(), reasoning);
                        }
                    }
                }
            }
        }
    }
    value
}

/// Build the DeepSeek `thinking` request object. Effort strings `"disabled"`
/// or `"off"` produce `{"type":"disabled"}`; anything else maps to
/// `{"type":"enabled","reasoning_effort":<effort>}`. DeepSeek itself
/// silently maps `low`/`medium` → `high`, so callers can stay loose.
fn deepseekThinking(effort: &str) -> serde_json::Value {
    if effort.eq_ignore_ascii_case("disabled") || effort.eq_ignore_ascii_case("off") {
        serde_json::json!({ "type": "disabled" })
    } else {
        serde_json::json!({
            "type": "enabled",
            "reasoning_effort": effort,
        })
    }
}

/// Build an HTTP client for a specific model config, with provider-appropriate headers.
fn buildHttpClient(config: &ModelConfig) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse()?);
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", config.key).parse()?,
    );

    // OpenRouter-specific headers.
    if config.provider == "openrouter" {
        headers.insert("X-Title", "Flatline".parse()?);
    }

    reqwest::Client::builder()
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .context("Failed to build HTTP client")
}

/// Read an SSE stream and emit events.
///
/// Applies a 2-minute idle timeout — if no bytes arrive for that long,
/// the stream is treated as stalled and an error is emitted.
async fn readStream(
    response: reqwest::Response,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<()> {
    use futures::StreamExt;

    const IDLE_TIMEOUT: Duration = Duration::from_secs(120);

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    loop {
        let chunk = match tokio::time::timeout(IDLE_TIMEOUT, stream.next()).await {
            Ok(Some(chunk)) => chunk.context("Stream read error")?,
            Ok(None) => break,
            Err(_) => bail!("Stream stalled — no data received for {IDLE_TIMEOUT:?}"),
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE lines.
        while let Some(lineEnd) = buffer.find('\n') {
            let line = buffer[..lineEnd].trim().to_string();
            buffer = buffer[lineEnd + 1..].to_string();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    // NOTE: Usage arrives in the final content chunk, not in [DONE].
                    // The Done event with usage was already emitted by parseChunk.
                    return Ok(());
                }

                match serde_json::from_str::<StreamChunk>(data) {
                    Ok(chunk) => {
                        // Log full raw data for error chunks to preserve metadata.
                        if chunk.error.is_some() {
                            tracing::error!(raw = %data, "raw SSE error chunk");
                        } else {
                            tracing::trace!(
                                hasUsage = chunk.usage.is_some(),
                                hasChoices = chunk.choices.is_some(),
                                choiceCount = chunk.choices.as_ref().map(|c| c.len()).unwrap_or(0),
                                "parsed SSE chunk"
                            );
                        }
                        for event in parseChunk(chunk) {
                            if tx.send(event).await.is_err() {
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(data = %data, "failed to parse SSE chunk: {e}");
                    }
                }
            }
        }
    }

    Ok(())
}

/// State machine tag for prompt-injected thinking extraction.
enum ThinkingState {
    /// Buffering content to see if it starts with `<thinking>`.
    Pending(String),
    /// Inside a `<thinking>` block — content routes to ReasoningDelta.
    Thinking(String),
    /// Past the `</thinking>` close — content routes to ContentDelta.
    Content,
}

/// Extracts `<thinking>` blocks from content deltas and re-routes them
/// as ReasoningDelta events. Used for prompt-injected thinking where the
/// model is instructed to reason in `<thinking>` tags instead of using
/// the official thinking API.
pub(crate) struct ThinkingExtractor {
    state: ThinkingState,
}

const OPEN_TAG: &str = "<scratchpad>";
const CLOSE_TAG: &str = "</scratchpad>";

impl ThinkingExtractor {
    pub fn new() -> Self {
        Self {
            state: ThinkingState::Pending(String::new()),
        }
    }

    /// Feed a content delta through the extractor.
    /// Returns events that should be emitted (ContentDelta or ReasoningDelta).
    pub fn feed(&mut self, text: &str) -> Vec<StreamEvent> {
        match &mut self.state {
            ThinkingState::Pending(buf) => {
                buf.push_str(text);

                // Strip leading whitespace — models often emit newlines before <thinking>.
                let trimmed = buf.trim_start();
                if trimmed.is_empty() {
                    // All whitespace so far — keep buffering.
                    return vec![];
                }

                // Check if trimmed buffer is still a valid prefix of `<thinking>`.
                if OPEN_TAG.starts_with(trimmed) {
                    // Could still become the tag — keep buffering.
                    return vec![];
                }

                if trimmed.starts_with(OPEN_TAG) {
                    // Tag found. Everything after it is reasoning.
                    let after = trimmed[OPEN_TAG.len()..].to_string();
                    self.state = ThinkingState::Thinking(String::new());
                    if after.is_empty() {
                        return vec![];
                    }
                    return self.feed(&after);
                }

                // Not the tag — flush buffer as content and switch to passthrough.
                let content = std::mem::take(buf);
                self.state = ThinkingState::Content;
                vec![StreamEvent::ContentDelta(content)]
            }

            ThinkingState::Thinking(buf) => {
                buf.push_str(text);

                if let Some(pos) = buf.find(CLOSE_TAG) {
                    // Found end tag.
                    let reasoning = buf[..pos].to_string();
                    let after = buf[pos + CLOSE_TAG.len()..].to_string();
                    self.state = ThinkingState::Content;

                    let mut events = Vec::new();
                    if !reasoning.is_empty() {
                        events.push(StreamEvent::ReasoningDelta(reasoning));
                    }
                    let trimmed = after.trim_start();
                    if !trimmed.is_empty() {
                        events.push(StreamEvent::ContentDelta(trimmed.to_string()));
                    }
                    events
                } else {
                    // Emit everything safe — keep a tail that might be a partial close tag.
                    let safeLen = safeDrainPoint(buf, CLOSE_TAG);
                    if safeLen > 0 {
                        let reasoning = buf[..safeLen].to_string();
                        *buf = buf[safeLen..].to_string();
                        vec![StreamEvent::ReasoningDelta(reasoning)]
                    } else {
                        vec![]
                    }
                }
            }

            ThinkingState::Content => vec![StreamEvent::ContentDelta(text.to_string())],
        }
    }

    /// Flush any remaining buffered content at stream end.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        match std::mem::replace(&mut self.state, ThinkingState::Content) {
            ThinkingState::Pending(buf) if !buf.is_empty() => {
                vec![StreamEvent::ContentDelta(buf)]
            }
            ThinkingState::Thinking(buf) if !buf.is_empty() => {
                vec![StreamEvent::ReasoningDelta(buf)]
            }
            _ => vec![],
        }
    }
}

/// Find how many bytes from the front of `buf` are safe to emit,
/// keeping any suffix that could be the start of `tag`.
fn safeDrainPoint(buf: &str, tag: &str) -> usize {
    let maxOverlap = tag.len().min(buf.len());
    for i in (1..=maxOverlap).rev() {
        let pos = buf.len() - i;
        // Don't slice in the middle of a multi-byte character.
        if !buf.is_char_boundary(pos) {
            continue;
        }
        let suffix = &buf[pos..];
        if tag.starts_with(suffix) {
            return pos;
        }
    }
    buf.len()
}

/// Convert a deserialized SSE chunk into stream events.
fn parseChunk(chunk: StreamChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(error) = chunk.error {
        let msg = error.message.clone().unwrap_or_else(|| "Unknown error".into());
        tracing::error!(
            error = %msg,
            code = ?error.code,
            errorType = ?error.errorType,
            status = ?error.status,
            extra = ?error.extra,
            "stream error from API"
        );
        events.push(StreamEvent::Error(msg));
        return events;
    }

    // Parse usage from the chunk (arrives in the final chunk).
    let usage = chunk.usage.map(|u| {
        // Anthropic-direct places cache tokens at the top level; OpenRouter
        // nests them under prompt_tokens_details (OpenAI-compat shape).
        // Take whichever is non-zero — one or the other will always be zero
        // for a given route.
        let (nestedRead, nestedWrite) = u
            .prompt_tokens_details
            .as_ref()
            .map(|d| (d.cached_tokens.unwrap_or(0), d.cache_write_tokens.unwrap_or(0)))
            .unwrap_or((0, 0));
        let tu = TokenUsage {
            promptTokens: u.prompt_tokens.unwrap_or(0),
            completionTokens: u.completion_tokens.unwrap_or(0),
            totalTokens: u.total_tokens.unwrap_or(0),
            cost: u.cost,
            cacheReadTokens: u.cache_read_input_tokens.unwrap_or(0).max(nestedRead),
            cacheCreationTokens: u.cache_creation_input_tokens.unwrap_or(0).max(nestedWrite),
        };
        tracing::info!(
            promptTokens = tu.promptTokens,
            completionTokens = tu.completionTokens,
            totalTokens = tu.totalTokens,
            cacheReadTokens = tu.cacheReadTokens,
            cacheCreationTokens = tu.cacheCreationTokens,
            cost = ?tu.cost,
            "token usage"
        );
        tu
    });

    if let Some(choices) = chunk.choices {
        for choice in choices {
            if let Some(delta) = choice.delta {
                if let Some(content) = delta.content {
                    if !content.is_empty() {
                        tracing::trace!(len = content.len(), "content delta");
                        events.push(StreamEvent::ContentDelta(content));
                    }
                }

                // Reasoning tokens arrive under different field names per provider:
                //   - `reasoning` — OpenRouter (DeepSeek, Kimi)
                //   - `reasoning_content` — Fireworks (Kimi, DeepSeek, etc.)
                //   - `reasoning_details` — OpenRouter (Claude structured)
                // Check all three, but only emit once to avoid duplicates.
                let mut hadReasoning = false;

                // Simple reasoning field (OpenRouter).
                if let Some(reasoning) = delta.reasoning {
                    if !reasoning.is_empty() {
                        tracing::trace!(len = reasoning.len(), "reasoning delta (reasoning)");
                        events.push(StreamEvent::ReasoningDelta(reasoning));
                        hadReasoning = true;
                    }
                }

                // Fireworks reasoning_content field.
                if !hadReasoning {
                    if let Some(reasoning) = delta.reasoning_content {
                        if !reasoning.is_empty() {
                            tracing::trace!(len = reasoning.len(), "reasoning delta (reasoning_content)");
                            events.push(StreamEvent::ReasoningDelta(reasoning));
                            hadReasoning = true;
                        }
                    }
                }

                // Structured reasoning details (Claude via OpenRouter).
                if !hadReasoning {
                    if let Some(details) = delta.reasoning_details {
                        for detail in details {
                            if let Some(text) = detail.text {
                                if !text.is_empty() {
                                    tracing::trace!(len = text.len(), "reasoning delta (structured)");
                                    events.push(StreamEvent::ReasoningDelta(text));
                                }
                            }
                        }
                    }
                }

                if let Some(toolCalls) = delta.tool_calls {
                    for tc in &toolCalls {
                        tracing::debug!(
                            index = tc.index,
                            id = ?tc.id,
                            name = ?tc.function.as_ref().and_then(|f| f.name.as_ref()),
                            "tool call delta"
                        );
                    }
                    for tc in toolCalls {
                        events.push(StreamEvent::ToolCallDelta {
                            index: tc.index.unwrap_or(0),
                            id: tc.id,
                            name: tc.function.as_ref().and_then(|f| f.name.clone()),
                            arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                        });
                    }
                }
            }

            if let Some(reason) = choice.finish_reason {
                tracing::debug!(reason = %reason, "stream finished");
                events.push(StreamEvent::Done {
                    finishReason: Some(reason),
                    usage: usage.clone(),
                });
            }
        }
    }

    // Usage can arrive in a chunk with no choices (some providers).
    if events.is_empty() && usage.is_some() {
        events.push(StreamEvent::Done {
            finishReason: None,
            usage,
        });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Content, Message};

    fn serializedMessages(msgs: &[Message]) -> serde_json::Value {
        adaptMessages(msgs, "openrouter")
    }

    #[test]
    fn deepseekThinkingShapes() {
        assert_eq!(
            deepseekThinking("max"),
            serde_json::json!({ "type": "enabled", "reasoning_effort": "max" }),
        );
        assert_eq!(
            deepseekThinking("high"),
            serde_json::json!({ "type": "enabled", "reasoning_effort": "high" }),
        );
        assert_eq!(
            deepseekThinking("disabled"),
            serde_json::json!({ "type": "disabled" }),
        );
        assert_eq!(
            deepseekThinking("off"),
            serde_json::json!({ "type": "disabled" }),
        );
        assert_eq!(
            deepseekThinking("DISABLED"),
            serde_json::json!({ "type": "disabled" }),
        );
    }

    #[test]
    fn deepseekRenamesAssistantReasoning() {
        let msgs = vec![
            Message::Assistant {
                content: Some("answer".into()),
                tool_calls: None,
                reasoning: Some("ponder".into()),
            },
        ];
        let value = adaptMessages(&msgs, "deepseek");
        let arr = value.as_array().unwrap();
        let assistant = arr[0].as_object().unwrap();
        assert!(!assistant.contains_key("reasoning"));
        assert_eq!(assistant.get("reasoning_content").unwrap(), "ponder");
    }

    #[test]
    fn injectMarksSystemAndLastUser() {
        let msgs = vec![
            Message::System { content: "You are flatline.".into() },
            Message::User { content: Content::Text("hello".into()) },
            Message::Assistant {
                content: Some("hi".into()),
                tool_calls: None,
                reasoning: None,
            },
            Message::User { content: Content::Text("explain caching".into()) },
        ];
        let mut value = serializedMessages(&msgs);
        injectCacheControl(&mut value);

        let arr = value.as_array().unwrap();

        // System should have become a blocks array with cache_control.
        let sys = &arr[0];
        let sysBlocks = sys["content"].as_array().expect("system content blocks");
        let sysLast = sysBlocks.last().unwrap();
        assert_eq!(sysLast["cache_control"]["type"], "ephemeral");

        // First user should NOT have cache_control.
        let firstUser = &arr[1];
        assert!(firstUser["content"].is_string() || firstUser["content"]
            .as_array()
            .map(|a| a.iter().all(|b| b.get("cache_control").is_none()))
            .unwrap_or(true));

        // Last user (index 3) should have cache_control on its last block.
        let lastUser = &arr[3];
        let lastBlocks = lastUser["content"].as_array().unwrap();
        assert_eq!(
            lastBlocks.last().unwrap()["cache_control"]["type"],
            "ephemeral",
        );
    }

    #[test]
    fn injectIsIdempotentOnAlreadyBlocksContent() {
        let msgs = vec![
            Message::System { content: "sys".into() },
            Message::User {
                content: Content::withImages("look at this", vec!["data:image/png;base64,...".into()]),
            },
        ];
        let mut value = serializedMessages(&msgs);
        injectCacheControl(&mut value);

        // Last user content was already a blocks array — cache_control should be on the
        // trailing image block, not converted or lost.
        let arr = value.as_array().unwrap();
        let last = arr.last().unwrap();
        let blocks = last["content"].as_array().unwrap();
        let trailing = blocks.last().unwrap();
        assert_eq!(trailing["cache_control"]["type"], "ephemeral");
        assert_eq!(trailing["type"], "image_url");
    }

    #[test]
    fn injectIsNoOpForEmptyMessages() {
        let mut value = serde_json::json!([]);
        injectCacheControl(&mut value);
        assert_eq!(value, serde_json::json!([]));
    }

    #[test]
    fn injectHandlesSystemOnly() {
        let msgs = vec![Message::System { content: "sys only".into() }];
        let mut value = serializedMessages(&msgs);
        injectCacheControl(&mut value);
        let arr = value.as_array().unwrap();
        let sys = &arr[0];
        let sysBlocks = sys["content"].as_array().unwrap();
        assert_eq!(sysBlocks[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn systemSplitOnBoundaryYields1hPlusNoMarker() {
        let content = format!(
            "static persona block\n\n{}\n\n<runtime>cwd=/tmp</runtime>",
            crate::prompt::CACHE_BOUNDARY,
        );
        let msgs = vec![
            Message::System { content },
            Message::User { content: Content::Text("hi".into()) },
        ];
        let mut value = serializedMessages(&msgs);
        injectCacheControl(&mut value);

        let arr = value.as_array().unwrap();
        let sys = &arr[0];
        let blocks = sys["content"].as_array().expect("two blocks after split");
        assert_eq!(blocks.len(), 2);

        // First block (static) → 1h cache_control.
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "static persona block");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(blocks[0]["cache_control"]["ttl"], "1h");

        // Second block (dynamic) → no cache_control.
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "<runtime>cwd=/tmp</runtime>");
        assert!(blocks[1].get("cache_control").is_none());

        // Sentinel itself must not appear in either block's text.
        assert!(!blocks[0]["text"].as_str().unwrap()
            .contains(crate::prompt::CACHE_BOUNDARY));
        assert!(!blocks[1]["text"].as_str().unwrap()
            .contains(crate::prompt::CACHE_BOUNDARY));
    }

    #[test]
    fn systemWithoutBoundaryFallsBackTo5m() {
        let msgs = vec![Message::System {
            content: "legacy prompt without sentinel".into(),
        }];
        let mut value = serializedMessages(&msgs);
        injectCacheControl(&mut value);

        let arr = value.as_array().unwrap();
        let blocks = arr[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        // No explicit ttl = Anthropic default of 5 minutes.
        assert!(blocks[0]["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn lastUserTurnAlwaysGets5mEvenWithSplitSystem() {
        let content = format!(
            "static\n{}\ndynamic",
            crate::prompt::CACHE_BOUNDARY,
        );
        let msgs = vec![
            Message::System { content },
            Message::User { content: Content::Text("hello".into()) },
        ];
        let mut value = serializedMessages(&msgs);
        injectCacheControl(&mut value);

        let arr = value.as_array().unwrap();
        let user = &arr[1];
        let blocks = user["content"].as_array().unwrap();
        let last = blocks.last().unwrap();
        // User turn is 5m by design — ttl field absent, type ephemeral.
        assert_eq!(last["cache_control"]["type"], "ephemeral");
        assert!(last["cache_control"].get("ttl").is_none());
    }
}
