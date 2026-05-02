//! Control plane — typed request/reply and log event stream.
//!
//! Replaces the polymorphic `SessionEvent` enum with three shapes:
//! - [`LogEvent`] — monotone stream, session → consumer, fire and forget
//! - [`TuiRequest`] — consumer → session, request/reply with `oneshot` reply inline
//! - [`SessionRequest`] — session → consumer, request/reply (currently permits only)
//!
//! Every request carries its own `oneshot::Sender<Reply>` so the compiler
//! enforces exactly-one reply per request and each request/response pair
//! has a statically typed reply shape — no untyped string payloads, no
//! "emit event then wait for a matching variant" dance.
//!
//! # Public API
//! - [`LogEvent`], [`TuiRequest`], [`SessionRequest`]
//! - Reply payload types: [`CommandAck`], [`McpStatus`], [`LspStatus`],
//!   [`PermissionsStatus`], [`PermitOrigin`]

use tokio::sync::oneshot;

use crate::context::ContextState;
use crate::lsp;
use crate::permissions::{PermissionsSource, PermitMode, PermitResponse, Rule};
use crate::tool::ShellImpact;
use crate::transcript::{Fork, Turn};

/// Monotone log events emitted by the session during a turn.
///
/// Fire-and-forget: no reply is expected for any variant. Consumers drain
/// these at their own cadence. Same shape for TUI, `flatline exec`, and
/// subagent forwarders.
#[derive(Debug, Clone)]
pub enum LogEvent {
    /// Streaming text content from the assistant.
    ContentDelta(String),

    /// Streaming reasoning/thinking content.
    ReasoningDelta(String),

    /// A tool call is being assembled in the stream — its name just became
    /// known. Fires once per call index the first time a name is seen.
    ToolCallPending { index: usize, name: String },

    /// A tool call's arguments are accumulating. `bytes` is the running total
    /// of JSON arg bytes received for this index.
    ToolCallProgress { index: usize, bytes: usize },

    /// A tool call's preview string just changed — a key argument value has
    /// finished streaming and the short human-readable summary has been
    /// recomputed (e.g. `crates/deck/src/app.rs`). Re-emitted when a later
    /// field refines the preview.
    ToolCallPreview { index: usize, preview: String },

    /// A tool was auto-approved by the permission config.
    ToolAutoApproved { name: String, summary: String },

    /// A tool has started executing (after approval, before result).
    ToolStarted { name: String, summary: String },

    /// A tool was executed (after approval).
    ToolResult { name: String, output: String },

    /// A tool call was denied by a user action.
    ToolDenied { name: String },

    /// A tool call was auto-denied by a permission rule.
    ToolAutoDenied { name: String, summary: String },

    /// Turn aborted because a tool call was denied under Abort mode.
    TurnAborted { name: String },

    /// The full turn is complete.
    TurnComplete,

    /// The turn was cancelled by the user.
    TurnCancelled,

    /// Queued user messages were injected into the conversation.
    SteerInjected { texts: Vec<String> },

    /// Token usage update from the API.
    TokenUpdate {
        promptTokens: usize,
        completionTokens: usize,
        contextTokens: usize,
        turnCost: Option<f64>,
        sessionCost: f64,
        cacheReadTokens: usize,
        cacheCreationTokens: usize,
    },

    /// Session cost exceeded the configured warning threshold.
    BudgetWarning { sessionCost: f64, limit: f64 },

    /// A compaction stage started running.
    CompactionStarted { stage: String },

    /// A compaction stage finished.
    CompactionComplete {
        stage: String,
        reduction: String,
        markerBlock: Option<usize>,
    },

    /// Session was cleared — deck should wipe the display.
    Cleared,

    /// Session restored with transcript history for display replay.
    SessionRestored {
        turns: Vec<Turn>,
        markers: Vec<(String, usize)>,
    },

    /// The current topic label changed.
    TopicChanged { label: String },

    /// Conversation was rewound to a prior turn.
    Rewound { targetTurnId: String },

    /// An LSP server is not installed but could enhance the experience.
    LspHint { serverId: String, installHint: String },

    /// A subagent has started executing.
    SubagentStarted {
        sessionId: String,
        agentType: String,
        prompt: String,
    },

    /// An event from a running subagent (wraps a child LogEvent).
    SubagentEvent {
        sessionId: String,
        event: Box<LogEvent>,
    },

    /// Raw shell output bytes from a subagent's PTY.
    SubagentShellOutput { sessionId: String, data: Vec<u8> },

    /// A subagent has completed.
    SubagentComplete {
        sessionId: String,
        agentType: String,
        content: String,
        turns: usize,
    },

    /// A transient API error is being retried silently.
    Retrying { attempt: u32, maxAttempts: u32 },

    /// The model emitted a malformed `</scratchpad>` close (e.g. `</scratch>`,
    /// `</scratchpa>`) that the streaming extractor missed; the trailing
    /// content was retroactively re-classified as visible. Surfaced so the
    /// user can verify the recovery split looks right.
    ScratchpadRecovered {
        matchedTag: String,
        snippet: String,
        recoveredChars: usize,
    },

    /// An error occurred.
    Error(String),
}

/// Acknowledgement reply for mutations and side-effecting requests.
#[derive(Debug, Clone)]
pub struct CommandAck {
    pub ok: bool,
    pub message: String,
}

impl CommandAck {
    pub fn ok(message: impl Into<String>) -> Self {
        Self { ok: true, message: message.into() }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self { ok: false, message: message.into() }
    }
}

/// Snapshot of MCP server state for the `/mcp` panel.
#[derive(Debug, Clone)]
pub struct McpStatus {
    /// Vec of (name, state, toolCount, tools: Vec<(qualifiedName, description)>, transport).
    pub servers: Vec<(String, String, usize, Vec<(String, String)>, String)>,
    pub totalTools: usize,
    pub searchMode: bool,
    pub configPath: String,
}

/// Snapshot of LSP server state for the `/lsp` panel.
#[derive(Debug, Clone)]
pub struct LspStatus {
    pub servers: Vec<lsp::FullServerStatus>,
}

/// Snapshot of permissions state for the `/permissions` panel.
#[derive(Debug, Clone)]
pub struct PermissionsStatus {
    pub defaultMode: PermitMode,
    pub rules: Vec<Rule>,
    pub source: PermissionsSource,
    pub configPath: String,
}

/// Requests from the TUI (or any consumer) to the session. Each variant
/// carries its reply channel inline.
///
/// Not `Clone` (oneshot senders are single-use) and not `Debug` (reply
/// channels aren't debuggable). Print individual fields if needed.
pub enum TuiRequest {
    /// Get context usage stats for the `/context` panel.
    ShowContext { reply: oneshot::Sender<ContextState> },

    /// Restore project files to the last checkpoint.
    Undo { reply: oneshot::Sender<CommandAck> },

    /// Fetch rewindable turns for the picker.
    GetRewindOptions { reply: oneshot::Sender<Vec<Turn>> },

    /// Rewind conversation to a prior turn. Emits `Rewound` + `SessionRestored` log events.
    Rewind {
        target: String,
        saveFork: bool,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Fetch saved forks for the picker.
    GetForks { reply: oneshot::Sender<Vec<Fork>> },

    /// Switch to a saved fork. Emits log events similarly to Rewind.
    SwitchFork {
        forkId: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// List available sessions as a formatted text listing.
    ListSessions { reply: oneshot::Sender<String> },

    /// Resume a saved session. Consumes the current session's shell and rebuilds.
    /// Emits `SessionRestored` + `TopicChanged` log events on success.
    ResumeSession {
        sessionId: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Start a fresh session (keeps shell). Emits `Cleared` log event.
    Clear { reply: oneshot::Sender<CommandAck> },

    /// Show cost breakdown (formatted text).
    ShowCost { reply: oneshot::Sender<String> },

    /// Snapshot MCP server state.
    GetMcp { reply: oneshot::Sender<McpStatus> },

    /// Snapshot LSP server state.
    GetLsp { reply: oneshot::Sender<LspStatus> },

    /// Snapshot permissions state.
    GetPermissions { reply: oneshot::Sender<PermissionsStatus> },

    /// Persist a new permissions config and apply it in-session.
    SavePermissions {
        defaultMode: PermitMode,
        rules: Vec<Rule>,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Re-run the last user turn from scratch. Drives the turn loop.
    RetryLastTurn { reply: oneshot::Sender<CommandAck> },

    /// Resume streaming from where the last turn was cut off.
    ContinueLastTurn { reply: oneshot::Sender<CommandAck> },

    /// Gracefully shut down background services (LSP, MCP) and exit the session task.
    Shutdown,
}

/// Origin of a permit request. Tells the consumer how to frame the prompt.
#[derive(Debug, Clone)]
pub enum PermitOrigin {
    /// Top-level session tool call.
    Top,
    /// Tool call inside a subagent; `sessionId` identifies the child.
    Subagent { sessionId: String },
}

/// Requests from the session to the consumer. Currently only permit prompts —
/// both top-level and subagent tool calls funnel through the same shape,
/// retiring the embedded-mpsc-in-event-variant hack.
pub enum SessionRequest {
    Permit {
        origin: PermitOrigin,
        name: String,
        summary: String,
        args: String,
        diff: Option<String>,
        explanation: Option<String>,
        impact: ShellImpact,
        reply: oneshot::Sender<PermitResponse>,
    },
}
