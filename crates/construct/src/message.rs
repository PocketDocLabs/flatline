//! Message types for LLM conversation and streaming events.
//!
//! Maps to the OpenAI-compatible chat completions format used by OpenRouter.
//!
//! # Public API
//! - [`Message`] — a conversation message (user, assistant, system, tool)
//! - [`ToolCall`] — an assistant's request to invoke a tool
//! - [`ToolDef`] — a tool definition sent with the request
//! - [`StreamEvent`] — a single event from the streaming response
//! - [`ReasoningConfig`] — controls thinking/reasoning behavior
//!
//! # Dependencies
//! `serde`, `serde_json`

use serde::{Deserialize, Serialize};

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },

    #[serde(rename = "user")]
    User { content: String },

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
        content: String,
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
    },

    /// An error mid-stream.
    Error(String),
}

/// Raw SSE chunk from the API (for deserialization).
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChunk {
    pub choices: Option<Vec<StreamChoice>>,
    pub error: Option<StreamError>,
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
}
