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
use crate::model_catalog::ModelCatalogEntry;
use crate::permissions::{PermissionsSource, PermitMode, PermitResponse, Rule};
use crate::shells::TerminalInfo;
use crate::tool::ShellImpact;
use crate::transcript::{Fork, Turn};

/// Origin tag mirroring [`crate::shells::SpawnedBy`] but `Clone + Debug`
/// for use inside `LogEvent`s.
/// Why a wake fired. Used in `<wake kind="...">` so the model knows
/// what kind of event woke it without parsing the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeKind {
    /// A monitor's filter matched a stdout line.
    MonitorMatch,
    /// A visible terminal run or backgrounded subagent finished.
    TaskComplete,
    /// A one-shot delay timer (`scheduleWakeup`) elapsed.
    Delay,
    /// A cron schedule (`cronCreate`) fired.
    Cron,
    /// A filesystem watch (`fileWatch`) reported an event.
    FileWatch,
}

impl WakeKind {
    pub fn asStr(self) -> &'static str {
        match self {
            WakeKind::MonitorMatch => "MonitorMatch",
            WakeKind::TaskComplete => "TaskComplete",
            WakeKind::Delay => "Delay",
            WakeKind::Cron => "Cron",
            WakeKind::FileWatch => "FileWatch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalSpawnedBy {
    User,
    Agent,
}

impl From<crate::shells::SpawnedBy> for TerminalSpawnedBy {
    fn from(b: crate::shells::SpawnedBy) -> Self {
        match b {
            crate::shells::SpawnedBy::User => Self::User,
            crate::shells::SpawnedBy::Agent => Self::Agent,
        }
    }
}

/// Structured report returned by the automatic permission reviewer.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AutoReviewReport {
    pub decision: String,
    pub raiseToUser: String,
    pub risk: String,
    pub authorization: String,
    pub reason: String,
    pub messageToAgent: String,
}

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

    /// A tool is waiting on the automatic permission reviewer.
    ToolAutoReviewStarted {
        name: String,
        summary: String,
        diff: Option<String>,
    },

    /// A tool was auto-approved by the permission config or reviewer.
    ToolAutoApproved {
        name: String,
        summary: String,
        diff: Option<String>,
        review: Option<AutoReviewReport>,
    },

    /// A tool has started executing (after approval, before result).
    ToolStarted { name: String, summary: String },

    /// A tool was executed (after approval).
    ToolResult { name: String, output: String },

    /// A tool call was denied by a user action.
    ToolDenied { name: String },

    /// A tool call was auto-denied by a permission rule or reviewer.
    ToolAutoDenied {
        name: String,
        summary: String,
        diff: Option<String>,
        review: Option<AutoReviewReport>,
    },

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

    /// Runtime model settings changed without replacing the session.
    ModelConfigChanged {
        contextWindow: usize,
        cachingEnabled: bool,
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
    LspHint {
        serverId: String,
        installHint: String,
    },

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

    /// A new background job was spawned (phase 2 only fires for `bashSpawn`;
    /// future phases add Subagent and Monitor variants).
    JobSpawned {
        id: u64,
        kind: String,
        command: String,
    },

    /// A line of output arrived on a task's stdout/stderr. Lines arrive
    /// in stream order; deck buffers per-task to drive /tasks panel inspect.
    JobOutput { id: u64, line: String },

    /// A task finished on its own. `exitCode` is None if the wait
    /// itself failed (rare — the OS would have to be unhealthy).
    JobComplete { id: u64, exitCode: Option<i32> },

    /// A task was stopped — by `jobStop`, kill from the tasks panel, or an
    /// internal error. `reason` is "killed", a spawn error, etc.
    JobStopped { id: u64, reason: String },

    /// A foreground `shell` call was interrupted because either the
    /// auto-bg timer fired or the user pressed Ctrl+B. Deck uses this
    /// to render a notice; the model receives an `AUTO_BG_INTERRUPT`
    /// tool result describing what happened and asking it to decide.
    AutoBgWarning {
        command: String,
        elapsedSecs: u64,
        /// `true` if user-triggered (Ctrl+B), `false` if timer-triggered.
        userTriggered: bool,
    },

    /// A monitor was registered via `monitor(...)`. Monitors attach to
    /// an existing terminal output stream.
    MonitorRegistered {
        id: u64,
        description: String,
        terminal: String,
        filter: String,
    },

    /// A monitor line passed its filter. `eventCount` is the
    /// post-increment running total for the monitor.
    MonitorEvent {
        id: u64,
        line: String,
        eventCount: u64,
    },

    /// The monitor's rolling events/sec exceeded its threshold for the
    /// flood-guard window; the terminal listener is detached and the
    /// monitor transitions to `AutoStopped`.
    MonitorAutoStopped { id: u64, reason: String },

    /// The monitor was stopped by the agent or user (clean exit).
    MonitorStopped { id: u64 },

    /// A coalesced wake batch was injected into the session as a single
    /// synthetic `<wakes count="N">…</wakes>` user-shaped message. One
    /// event per batch — the deck renders a single notice chip, never
    /// per-fire. The session task drives this from the wake batcher; no
    /// other path emits it. (Replaces the old per-fire `WakeFired` that
    /// raced through `userInputTx` and produced one model turn per
    /// matching log line.)
    WakeBatchInjected { count: usize, summary: String },

    /// A scheduled wake source (Delay, Cron, FileWatch) was registered
    /// via `scheduleWakeup`/`cronCreate`/`fileWatch`. The deck uses
    /// this to populate the /tasks schedules section. Passive
    /// sources (MonitorMatch, TaskComplete) also emit this when
    /// their underlying monitor/task is registered.
    WakeRegistered {
        id: u64,
        kind: WakeKind,
        summary: String,
        prompt: Option<String>,
        nextFireAt: Option<std::time::Instant>,
    },

    /// A wake source was disarmed — either explicitly (`cronDelete`,
    /// `scheduleWakeup` one-shot fire, `monitorStop`) or its scheduler
    /// task exited. After this event, newly arriving fires from this id
    /// are ignored; any already-closed batch may still be delivered.
    WakeDisarmed { id: u64 },

    /// A new terminal was spawned in the session's shell registry.
    /// `name` is the resolved unique name. `spawnedBy` is User (Ctrl+T,
    /// Ctrl+T) or Agent (`terminalSpawn` tool call).
    TerminalSpawned {
        name: String,
        spawnedBy: TerminalSpawnedBy,
    },

    /// A terminal was killed.
    TerminalClosed { name: String },

    /// The agent's default target terminal changed (via `terminalSwitch`
    /// tool, agent-side only — user focus is handled in deck).
    TerminalActiveForAgent { name: String },

    /// A terminal was renamed (future).
    TerminalRenamed { from: String, to: String },

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
        Self {
            ok: true,
            message: message.into(),
        }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: message.into(),
        }
    }
}

/// Snapshot of MCP server state for the `/mcp` panel.
#[derive(Debug, Clone)]
pub struct McpStatus {
    pub servers: Vec<McpServerStatusEntry>,
    pub totalTools: usize,
    pub searchMode: bool,
    pub configPath: String,
}

/// (name, state, toolCount, tools, transport).
pub type McpServerStatusEntry = (String, String, usize, Vec<McpToolStatusEntry>, String);
/// (qualifiedName, description).
pub type McpToolStatusEntry = (String, String);

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

/// One named model profile in the `/model` panel.
#[derive(Debug, Clone)]
pub struct ModelProfileStatus {
    pub name: String,
    pub provider: String,
    pub model: String,
    pub contextWindow: usize,
    pub maxContextWindow: Option<usize>,
    pub promptThinking: bool,
    pub reasoningEffort: Option<String>,
    pub reasoningEfforts: Vec<String>,
    pub reasoningSummary: Option<String>,
    pub configured: bool,
}

/// One available config destination/source in the `/model` panel.
#[derive(Debug, Clone)]
pub struct ModelConfigScopeStatus {
    pub scope: crate::config::ConfigScope,
    pub label: String,
    pub path: String,
}

/// Snapshot of model profile state for the `/model` panel.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub heavyProfile: String,
    pub lightProfile: String,
    pub utilityProfile: String,
    pub profiles: Vec<ModelProfileStatus>,
    pub saveScope: crate::config::ConfigScope,
    pub scopes: Vec<ModelConfigScopeStatus>,
    pub configPath: String,
    pub openAiCodex: crate::auth::OpenAiCodexStatus,
}

/// Requests from the TUI (or any consumer) to the session. Each variant
/// carries its reply channel inline.
///
/// Not `Clone` (oneshot senders are single-use) and not `Debug` (reply
/// channels aren't debuggable). Print individual fields if needed.
pub enum TuiRequest {
    /// Get context usage stats for the `/context` panel.
    ShowContext {
        reply: oneshot::Sender<ContextState>,
    },

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
    GetPermissions {
        reply: oneshot::Sender<PermissionsStatus>,
    },

    /// Snapshot model profile state.
    GetModels { reply: oneshot::Sender<ModelStatus> },

    /// Persist a model profile selection to local project config.
    SaveModelSelection {
        scope: crate::config::ConfigScope,
        tier: crate::config::ModelTier,
        profile: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Discover provider models for the `/model` panel.
    DiscoverModels {
        provider: String,
        reply: oneshot::Sender<std::result::Result<Vec<ModelCatalogEntry>, String>>,
    },

    /// Persist a discovered model into an existing named profile.
    SaveDiscoveredModel {
        scope: crate::config::ConfigScope,
        profile: String,
        model: ModelCatalogEntry,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Create a new profile by copying an existing profile.
    CreateModelProfile {
        scope: crate::config::ConfigScope,
        profile: String,
        sourceProfile: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Rename a profile defined in the selected config scope.
    RenameModelProfile {
        scope: crate::config::ConfigScope,
        oldProfile: String,
        newProfile: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Delete a profile defined in the selected config scope.
    DeleteModelProfile {
        scope: crate::config::ConfigScope,
        profile: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Persist a profile's usable context window.
    SaveModelProfileContext {
        scope: crate::config::ConfigScope,
        profile: String,
        contextWindow: usize,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Persist a profile's thinking / reasoning behavior.
    SaveModelProfileThinking {
        scope: crate::config::ConfigScope,
        profile: String,
        promptThinking: bool,
        reasoningEffort: Option<String>,
        reasoningSummary: Option<String>,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Persist a new permissions config and apply it in-session.
    SavePermissions {
        defaultMode: PermitMode,
        rules: Vec<Rule>,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Change the live session permission mode without persisting config.
    SetPermitMode {
        mode: PermitMode,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Re-run the last user turn from scratch. Drives the turn loop.
    RetryLastTurn { reply: oneshot::Sender<CommandAck> },

    /// User spawned a new terminal (Ctrl+T / Ctrl+T).
    /// Reply carries the resolved name (registry may auto-generate).
    SpawnTerminal {
        name: Option<String>,
        reply: oneshot::Sender<std::result::Result<String, String>>,
    },

    /// User killed a terminal (the tab strip kill button).
    KillTerminal {
        name: String,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Snapshot of the shell registry for the tab strip.
    ListTerminals {
        reply: oneshot::Sender<Vec<TerminalInfo>>,
    },

    /// Snapshot of archived terminal-backed async runs for terminal history.
    ListTerminalRuns {
        reply: oneshot::Sender<Vec<crate::storage::TerminalRunRecord>>,
    },

    /// Snapshot of every JobPlane-backed background task for `/jobs`.
    ListJobs {
        reply: oneshot::Sender<Vec<crate::jobs::JobInfo>>,
    },

    /// Snapshot of every scheduled wake source (delay, cron, file-watch).
    /// Drives the schedules section of the `/jobs` panel.
    ListWakes {
        reply: oneshot::Sender<Vec<crate::wakes::WakeSourceInfo>>,
    },

    /// Kill a background task from the tasks panel.
    KillTask {
        id: u64,
        reply: oneshot::Sender<CommandAck>,
    },

    /// Fetch the buffered output of a task for the inspect popup. When
    /// `sinceLine` is `None` the response carries the latest tail
    /// (matches `taskOutput(None)`); `Some(N)` pages from that line so
    /// the user can scroll back through still-buffered earlier output.
    GetTaskOutput {
        id: u64,
        sinceLine: Option<u64>,
        reply: oneshot::Sender<Option<crate::jobs::JobOutputSnapshot>>,
    },

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
        review: Option<AutoReviewReport>,
        reply: oneshot::Sender<PermitResponse>,
    },
}
