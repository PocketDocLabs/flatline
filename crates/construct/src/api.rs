//! OpenRouter API client with streaming support.
//!
//! Speaks the OpenAI-compatible chat completions format
//! with OpenRouter's reasoning extensions.
//!
//! # Public API
//! - [`Client`] — HTTP client for the LLM API
//! - [`Client::stream`] — send a prompt, get streaming events
//!
//! # Dependencies
//! `reqwest`, `tokio`, `serde_json`

use anyhow::{Context, Result, bail};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use tokio::sync::mpsc;

use crate::config::ApiConfig;
use crate::message::{
    Message, ReasoningConfig, StreamChunk, StreamEvent, ToolDef,
};

/// OpenRouter API client.
pub struct Client {
    http: reqwest::Client,
    config: ApiConfig,
}

impl Client {
    /// Create a new API client from config.
    pub fn new(config: &ApiConfig) -> Result<Self> {
        if config.key.is_empty() {
            bail!("API key not set. Edit ~/.config/flatline/config.toml");
        }

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse()?);
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", config.key).parse()?,
        );
        // OpenRouter-specific headers.
        headers.insert("X-Title", "Flatline".parse()?);

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            http,
            config: config.clone(),
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
        let url = format!("{}/chat/completions", self.config.baseUrl);

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools)?;
            body["tool_choice"] = serde_json::json!("auto");
        }

        if let Some(r) = reasoning {
            body["reasoning"] = serde_json::to_value(r)?;
        }

        if !self.config.providerOrder.is_empty() {
            body["provider"] = serde_json::json!({
                "order": self.config.providerOrder,
                "allow_fallbacks": false,
            });
        }

        tracing::debug!(
            model = %self.config.model,
            messageCount = messages.len(),
            toolCount = tools.len(),
            hasReasoning = reasoning.is_some(),
            "sending API request"
        );
        tracing::trace!(body = %body, "request body");

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send API request")?;

        if !response.status().is_success() {
            let status = response.status();
            let errorBody = response.text().await.unwrap_or_default();
            tracing::error!(%status, body = %errorBody, "API error");
            bail!("API error {status}: {errorBody}");
        }

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
}

/// Read an SSE stream and emit events.
async fn readStream(
    response: reqwest::Response,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<()> {
    use futures::StreamExt;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Stream read error")?;
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
                    let _ = tx
                        .send(StreamEvent::Done {
                            finishReason: Some("stop".into()),
                        })
                        .await;
                    return Ok(());
                }

                match serde_json::from_str::<StreamChunk>(data) {
                    Ok(chunk) => {
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

/// Convert a deserialized SSE chunk into stream events.
fn parseChunk(chunk: StreamChunk) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    if let Some(error) = chunk.error {
        let msg = error.message.unwrap_or_else(|| "Unknown error".into());
        tracing::error!(error = %msg, "stream error from API");
        events.push(StreamEvent::Error(msg));
        return events;
    }

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
                });
            }
        }
    }

    events
}
