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
            bail!("API error {status}: {errorBody}");
        }

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
                        tracing::warn!("Failed to parse SSE chunk: {e}");
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
        events.push(StreamEvent::Error(
            error.message.unwrap_or_else(|| "Unknown error".into()),
        ));
        return events;
    }

    if let Some(choices) = chunk.choices {
        for choice in choices {
            if let Some(delta) = choice.delta {
                if let Some(content) = delta.content {
                    if !content.is_empty() {
                        events.push(StreamEvent::ContentDelta(content));
                    }
                }

                if let Some(reasoning) = delta.reasoning {
                    if !reasoning.is_empty() {
                        events.push(StreamEvent::ReasoningDelta(reasoning));
                    }
                }

                if let Some(toolCalls) = delta.tool_calls {
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
                events.push(StreamEvent::Done {
                    finishReason: Some(reason),
                });
            }
        }
    }

    events
}
