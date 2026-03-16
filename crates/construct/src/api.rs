//! OpenRouter API client with streaming and non-streaming support.
//!
//! Speaks the OpenAI-compatible chat completions format
//! with OpenRouter's reasoning extensions.
//!
//! # Public API
//! - [`Client`] — HTTP client for the LLM API
//! - [`Client::stream`] — send a prompt, get streaming events
//! - [`Client::complete`] — non-streaming completion for utility calls
//!
//! # Dependencies
//! `reqwest`, `tokio`, `serde_json`, `backon`

use std::time::Duration;

use anyhow::{Context, Result, bail};
use backon::{ExponentialBuilder, Retryable};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use tokio::sync::mpsc;

use crate::config::{Config, ModelConfig};
use crate::message::{
    Message, ReasoningConfig, StreamChunk, StreamEvent, TokenUsage, ToolDef,
};

/// Marker error for API failures that should not be retried (400, 401, 403, etc.).
#[derive(Debug)]
struct PermanentApiError(String);

impl std::fmt::Display for PermanentApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PermanentApiError {}

/// OpenRouter API client.
pub struct Client {
    http: reqwest::Client,
    main: ModelConfig,
    utility: ModelConfig,
}

impl Client {
    /// Create a new API client from config.
    pub fn new(config: &Config) -> Result<Self> {
        if config.main.key.is_empty() {
            bail!("API key not set. Edit config.toml (main.key or OPENROUTER_API_KEY)");
        }

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse()?);
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", config.main.key).parse()?,
        );
        // OpenRouter-specific headers.
        headers.insert("X-Title", "Flatline".parse()?);

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            http,
            main: config.main.clone(),
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
        let url = format!("{}/chat/completions", self.main.baseUrl);

        let mut body = serde_json::json!({
            "model": self.main.model,
            "messages": messages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });

        if let Some(max) = self.main.maxTokens {
            body["max_tokens"] = serde_json::json!(max);
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools)?;
            body["tool_choice"] = serde_json::json!("auto");
        }

        if let Some(r) = reasoning {
            body["reasoning"] = serde_json::to_value(r)?;
        }

        if !self.main.providerOrder.is_empty() {
            body["provider"] = serde_json::json!({
                "order": self.main.providerOrder,
                "allow_fallbacks": false,
            });
        }

        tracing::debug!(
            model = %self.main.model,
            messageCount = messages.len(),
            toolCount = tools.len(),
            hasReasoning = reasoning.is_some(),
            "sending API request"
        );
        tracing::trace!(body = %body, "request body");

        let response = (|| async {
            let response = self
                .http
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
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.utility.baseUrl);
        let modelId = model.unwrap_or(&self.utility.model);

        let mut body = serde_json::json!({
            "model": modelId,
            "messages": messages,
            "stream": false,
        });

        if !self.utility.providerOrder.is_empty() {
            body["provider"] = serde_json::json!({
                "order": self.utility.providerOrder,
                "allow_fallbacks": false,
            });
        }

        tracing::debug!(
            model = %modelId,
            messageCount = messages.len(),
            "sending utility completion request"
        );

        let response = (|| async {
            let response = self
                .http
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

        tracing::debug!(
            model = %modelId,
            responseLen = content.len(),
            "utility completion received"
        );

        Ok(content)
    }
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
                        tracing::trace!(
                            hasUsage = chunk.usage.is_some(),
                            hasChoices = chunk.choices.is_some(),
                            choiceCount = chunk.choices.as_ref().map(|c| c.len()).unwrap_or(0),
                            "parsed SSE chunk"
                        );
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
        let msg = error.message.unwrap_or_else(|| "Unknown error".into());
        tracing::error!(error = %msg, "stream error from API");
        events.push(StreamEvent::Error(msg));
        return events;
    }

    // Parse usage from the chunk (arrives in the final chunk).
    let usage = chunk.usage.map(|u| {
        let tu = TokenUsage {
            promptTokens: u.prompt_tokens.unwrap_or(0),
            completionTokens: u.completion_tokens.unwrap_or(0),
            totalTokens: u.total_tokens.unwrap_or(0),
        };
        tracing::info!(
            promptTokens = tu.promptTokens,
            completionTokens = tu.completionTokens,
            totalTokens = tu.totalTokens,
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

                // Prefer simple reasoning field (DeepSeek, Kimi).
                // Fall back to structured reasoning_details (Claude via OpenRouter).
                // Only use one to avoid duplicate output.
                let mut hadReasoning = false;
                if let Some(reasoning) = delta.reasoning {
                    if !reasoning.is_empty() {
                        tracing::trace!(len = reasoning.len(), "reasoning delta (simple)");
                        events.push(StreamEvent::ReasoningDelta(reasoning));
                        hadReasoning = true;
                    }
                }

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
