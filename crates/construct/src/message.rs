//! Message types for LLM conversation and streaming events.
//!
//! Maps to the OpenAI-compatible chat completions format used by OpenRouter.
//!
//! # Public API
//! - [`Message`] — a conversation message (user, assistant, system, tool)
//! - [`Content`] — text or multimodal content (text + images)
//! - [`ContentBlock`] — a single block within multimodal content
//! - [`ToolCall`] — an assistant's request to invoke a tool
//! - [`ToolDef`] — a tool definition sent with the request
//! - [`StreamEvent`] — a single event from the streaming response
//! - [`ReasoningConfig`] — controls thinking/reasoning behavior
//!
//! # Dependencies
//! `serde`, `serde_json`

use serde::{Deserialize, Serialize};

/// Message content — either plain text or an array of typed blocks.
///
/// Serializes as a JSON string when text-only, or a JSON array when
/// multimodal. Matches the OpenAI content field format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// Plain text content — serializes as a JSON string.
    Text(String),
    /// Array of content blocks — serializes as a JSON array.
    Blocks(Vec<ContentBlock>),
}

/// A single block within multimodal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL data (typically a base64 data URI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl Content {
    /// Create text-only content.
    pub fn text(s: impl Into<String>) -> Self {
        Content::Text(s.into())
    }

    /// Extract the text portion, ignoring image blocks.
    pub fn textContent(&self) -> &str {
        match self {
            Content::Text(s) => s,
            Content::Blocks(blocks) => blocks
                .iter()
                .find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or(""),
        }
    }

    /// Rough byte count for context window estimation.
    pub fn charCount(&self) -> usize {
        match self {
            Content::Text(s) => s.len(),
            Content::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => text.len(),
                    ContentBlock::ImageUrl { image_url } => image_url.url.len(),
                })
                .sum(),
        }
    }

    /// Whether this content contains any image blocks.
    pub fn hasImages(&self) -> bool {
        matches!(
            self,
            Content::Blocks(blocks) if blocks.iter().any(|b| matches!(b, ContentBlock::ImageUrl { .. }))
        )
    }

    /// Build multimodal content from text + base64 image data URIs.
    pub fn withImages(text: &str, imageUris: Vec<String>) -> Self {
        let mut blocks = Vec::with_capacity(1 + imageUris.len());
        if !text.is_empty() {
            blocks.push(ContentBlock::Text { text: text.into() });
        }
        for url in imageUris {
            blocks.push(ContentBlock::ImageUrl {
                image_url: ImageUrl { url, detail: None },
            });
        }
        Content::Blocks(blocks)
    }

    /// Strip all image blocks, replacing them with "[image]" text.
    /// Returns the modified content, collapsing to Content::Text if
    /// only a single text block remains.
    pub fn stripImages(&self) -> Self {
        match self {
            Content::Text(_) => self.clone(),
            Content::Blocks(blocks) => {
                let stripped: Vec<ContentBlock> = blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::ImageUrl { .. } => ContentBlock::Text {
                            text: "[image]".into(),
                        },
                        other => other.clone(),
                    })
                    .collect();

                // Collapse to plain text if only one text block remains.
                if stripped.len() == 1 {
                    if let ContentBlock::Text { ref text } = stripped[0] {
                        return Content::Text(text.clone());
                    }
                }
                Content::Blocks(stripped)
            }
        }
    }

    /// Extract image data URIs from content blocks.
    pub fn imageUris(&self) -> Vec<&str> {
        match self {
            Content::Text(_) => Vec::new(),
            Content::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ImageUrl { image_url } => Some(image_url.url.as_str()),
                    _ => None,
                })
                .collect(),
        }
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text(s.to_string())
    }
}

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },

    #[serde(rename = "user")]
    User { content: Content },

    #[serde(rename = "assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,

        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,

        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },

    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: Content,
    },
}

/// An assistant's request to invoke a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,

    #[serde(rename = "type")]
    pub callType: String,

    pub function: FunctionCall,
}

/// The function name and arguments within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// A tool definition sent with the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub defType: String,

    pub function: FunctionDef,
}

/// Function metadata within a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Controls reasoning/thinking behavior for supported models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Reasoning effort level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,

    /// Summary style for reasoning output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// A single event from the streaming SSE response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of text content.
    ContentDelta(String),

    /// A chunk of reasoning/thinking content.
    ReasoningDelta(String),

    /// A tool call being constructed (may arrive in pieces).
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },

    /// Stream finished.
    Done {
        finishReason: Option<String>,
        usage: Option<TokenUsage>,
    },

    /// An error mid-stream.
    Error(String),
}

/// Token usage from an API response.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub promptTokens: usize,
    pub completionTokens: usize,
    pub totalTokens: usize,
}

/// Raw SSE chunk from the API (for deserialization).
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChunk {
    pub choices: Option<Vec<StreamChoice>>,
    pub usage: Option<ChunkUsage>,
    pub error: Option<StreamError>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkUsage {
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub total_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamChoice {
    pub delta: Option<ChunkDelta>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkDelta {
    pub content: Option<String>,
    /// Simple reasoning string (used by some providers like DeepSeek/Kimi).
    pub reasoning: Option<String>,
    /// Structured reasoning details (used by Claude via OpenRouter).
    pub reasoning_details: Option<Vec<ReasoningDetail>>,
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// A single reasoning detail entry from structured reasoning.
#[derive(Debug, Deserialize)]
pub(crate) struct ReasoningDetail {
    #[serde(rename = "type")]
    pub detailType: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkToolCall {
    pub index: Option<usize>,
    pub id: Option<String>,
    pub function: Option<ChunkFunction>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChunkFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamError {
    pub message: Option<String>,
    pub code: Option<serde_json::Value>,
    #[serde(rename = "type")]
    pub errorType: Option<String>,
    pub status: Option<serde_json::Value>,
    /// Catch-all for any other fields the provider sends.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contentTextSerializesAsString() {
        let c = Content::text("hello");
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, r#""hello""#);
    }

    #[test]
    fn contentBlocksSerializesAsArray() {
        let c = Content::withImages("describe this", vec![
            "data:image/png;base64,abc123".into(),
        ]);
        let json = serde_json::to_string(&c).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.is_array());
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "describe this");
        assert_eq!(arr[1]["type"], "image_url");
        assert_eq!(arr[1]["image_url"]["url"], "data:image/png;base64,abc123");
    }

    #[test]
    fn contentRoundTrip() {
        let original = Content::text("round trip");
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Content = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.textContent(), "round trip");
    }

    #[test]
    fn contentBlocksRoundTrip() {
        let original = Content::withImages("look", vec![
            "data:image/png;base64,xyz".into(),
        ]);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Content = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.textContent(), "look");
        assert!(deserialized.hasImages());
        assert_eq!(deserialized.imageUris(), vec!["data:image/png;base64,xyz"]);
    }

    #[test]
    fn userMessageSerializesCorrectly() {
        // Text-only user message.
        let msg = Message::User { content: Content::text("hi") };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hi");

        // Multimodal user message.
        let msg = Message::User {
            content: Content::withImages("what is this?", vec![
                "data:image/jpeg;base64,/9j/4A".into(),
            ]),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["role"], "user");
        assert!(v["content"].is_array());
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][1]["type"], "image_url");
    }

    #[test]
    fn toolMessageSerializesCorrectly() {
        let msg = Message::Tool {
            tool_call_id: "call_1".into(),
            content: Content::text("file contents here"),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_1");
        assert_eq!(v["content"], "file contents here");
    }

    #[test]
    fn stripImagesReplacesWithPlaceholder() {
        let c = Content::withImages("look at this", vec![
            "data:image/png;base64,abc".into(),
            "data:image/jpeg;base64,def".into(),
        ]);
        let stripped = c.stripImages();
        assert!(!stripped.hasImages());
        assert_eq!(stripped.textContent(), "look at this");
    }

    #[test]
    fn stripImagesCollapsesToText() {
        // Single text block + single image -> after stripping, two text blocks.
        // But a single image with no text -> collapses to text.
        let c = Content::Blocks(vec![
            ContentBlock::ImageUrl {
                image_url: ImageUrl { url: "data:image/png;base64,x".into(), detail: None },
            },
        ]);
        let stripped = c.stripImages();
        assert!(!stripped.hasImages());
        // Should collapse to Content::Text("[image]").
        assert!(matches!(stripped, Content::Text(_)));
        assert_eq!(stripped.textContent(), "[image]");
    }

    #[test]
    fn charCountIncludesImageUris() {
        let c = Content::withImages("hi", vec!["data:image/png;base64,abc123".into()]);
        // "hi" = 2, data URI = 28
        assert!(c.charCount() > 20);
    }

    #[test]
    fn fromStringAndStrConversions() {
        let c: Content = "hello".into();
        assert_eq!(c.textContent(), "hello");

        let c: Content = String::from("world").into();
        assert_eq!(c.textContent(), "world");
    }
}
