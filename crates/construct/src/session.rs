//! Agent session — conversation loop with permission-gated tool execution.
//!
//! Manages the full turn cycle: user message → API stream →
//! accumulate response → check permissions → execute tool calls → repeat.
//!
//! The permission system has three layers:
//! 1. Pre-configured rules (allow/deny patterns per tool)
//! 2. Runtime approval via the permit channel (for `NeedsApproval` verdicts)
//! 3. A fallback mode (Ask, Deny, or Abort) when no rule matches
//!
//! # Public API
//! - [`Session`] — owns conversation state and drives the agent loop
//! - [`crate::control::LogEvent`] — events emitted during a turn
//!
//! # Dependencies
//! `tokio`, `serde_json`

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::sync::{mpsc, oneshot, watch};

use crate::api;
use crate::checkpoint::CheckpointManager;
use crate::compaction::CompactionLog;
use crate::compaction_trigger;
use crate::config::Config;
use crate::context;
use crate::control::{LogEvent, McpServerStatusEntry, PermitOrigin, SessionRequest};
use crate::jobs::JobPlane;
use crate::lsp;
use crate::mcp;
use crate::message::{
    Content, FunctionCall, Message, ReasoningConfig, StreamEvent, TokenUsage, ToolCall, ToolDef,
};
use crate::permissions::{Permissions, PermitMode, Verdict};
use crate::prompt::{self, DomainModule, InterfaceMode};
use crate::shell::Shell;
use crate::shells::{ShellRegistry, SpawnedBy};
use crate::tool;
use crate::topic::{TopicDecision, TopicTracker};
use crate::transcript::{self, SessionMeta, Transcript};
use crate::web;

/// A user input message with optional image attachments.
#[derive(Debug, Clone)]
pub struct UserInput {
    pub text: String,
    pub attachments: Vec<Attachment>,
}

/// A binary attachment to a user message.
///
/// For clipboard-pasted images, `data` contains raw RGBA pixels and
/// `rgbaDimensions` holds (width, height). PNG encoding is deferred to
/// submit time to avoid blocking the TUI event loop.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub mimeType: String,
    pub data: Vec<u8>,
    pub label: String,
    /// Raw RGBA dimensions — set when data is raw pixels, None when already encoded.
    pub rgbaDimensions: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ShellResolveError {
    MissingNamed {
        name: String,
        available: Vec<String>,
        target: String,
    },
    NoAgentTarget,
}

impl fmt::Display for ShellResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShellResolveError::MissingNamed {
                name,
                available,
                target,
            } => write!(
                f,
                "No terminal named '{name}'. \
                 Available terminals: [{}]. \
                 Agent's current target: '{target}'. \
                 Call terminalList to inspect, or terminalSpawn to create.",
                available.join(", "),
            ),
            ShellResolveError::NoAgentTarget => f.write_str("No agent target terminal available."),
        }
    }
}

impl std::error::Error for ShellResolveError {}

impl From<String> for UserInput {
    fn from(text: String) -> Self {
        UserInput {
            text,
            attachments: Vec::new(),
        }
    }
}

// Session events and control requests live in `crate::control` — this module
// only holds the `Session` struct and turn-loop logic.

/// Seconds a foreground `shell` call can run before the session
/// A scaffolding instruction prepended to the latest User message at API
/// call time. Riders never appear in `self.history`, the transcript, or
/// snapshots — they're ephemeral nudges to reinforce system-prompt behavior.
///
/// All active riders render into a single `<CRITICAL_INSTRUCTIONS>` block
/// placed above the user's text in the API request copy only.
#[derive(Debug, Clone)]
pub struct Rider {
    /// Tag name used inside the `<CRITICAL_INSTRUCTIONS>` wrapper
    /// (e.g. `"THINKING"`, `"MODE"`). Kept as a `&'static str` so each
    /// rider's identity is fixed at compile time.
    pub id: &'static str,
    /// Body text. Must not contain the outer wrapper tags.
    pub content: String,
}

/// Build child (subagent) permissions by inheriting from the parent and
/// layering the preset as a floor.
///
/// Rule order in the result (first-match-wins):
///   1. Preset floor (explicit denies for mutation tools, if `ReadOnly` toolSet)
///   2. Preset's own rules
///   3. Parent's rules
///
/// `defaultMode` is taken from the preset. This ensures a read-only preset
/// can't be widened by a permissive parent, while still inheriting the
/// parent's narrower allows (e.g. `readFile` patterns the user has already
/// approved in the top-level session).
/// Background subagent runner — owns the child session and all its
/// forwarders. Lives on its own `tokio::spawn` so the parent's tool
/// dispatch returns immediately.
///
/// Mirrors the structure of [`Session::executeTask`] but with three
/// extra wrinkles:
/// 1. Streamed content deltas are pushed into the [`SubagentJobHandle`]'s
///    ring buffer so `jobOutput` returns real-time progress.
/// 2. The child's `send` future races the handle's `cancelRx` so
///    `jobStop` aborts the run cooperatively.
/// 3. Completion calls `handle.complete(...)` / `handle.killed()` /
///    `handle.errored(...)` so the JobPlane's state machine and the
///    tasks panel both reflect the outcome.
#[allow(clippy::too_many_arguments)]
async fn runSubagentTaskInBackground(
    mut child: Session,
    childMainIo: crate::shell::ShellIo,
    mut childIoRx: mpsc::Receiver<(String, crate::shell::ShellIo, crate::shells::SpawnedBy)>,
    handle: crate::jobs::SubagentJobHandle,
    prompt: String,
    agentType: String,
    childSessionId: String,
    parentLogTx: mpsc::Sender<LogEvent>,
    parentSessionRequestTx: mpsc::Sender<SessionRequest>,
) {
    // Shell-output forwarder — same as foreground subagents.
    let shellForwardTx = parentLogTx.clone();
    let shellForwardId = childSessionId.clone();
    tokio::spawn(async move {
        let mut mainRx = childMainIo.outputRx;
        loop {
            tokio::select! {
                Some(data) = mainRx.recv() => {
                    let _ = shellForwardTx
                        .send(LogEvent::SubagentShellOutput {
                            sessionId: shellForwardId.clone(),
                            data,
                        })
                        .await;
                }
                Some((_n, _io, _by)) = childIoRx.recv() => {
                    // Subagents don't currently spawn extra shells. Drop io.
                }
                else => break,
            }
        }
    });

    // Notify parent that subagent has started — same event the foreground
    // path emits, so the deck's existing SubagentStarted handler renders it.
    let _ = parentLogTx
        .send(LogEvent::SubagentStarted {
            sessionId: childSessionId.clone(),
            agentType: agentType.clone(),
            prompt: prompt.clone(),
        })
        .await;

    tracing::info!(
        agent = %agentType,
        childSession = %childSessionId,
        taskId = handle.id,
        "background subagent spawned"
    );

    // Channels for the child session.
    let (childLogTx, mut childLogRx) = mpsc::channel::<LogEvent>(256);
    let (childRequestTx, mut childRequestRx) = mpsc::channel::<SessionRequest>(16);

    // Child cancel is driven by the task handle's watch — flipping the
    // cancel flag cancels both the child's tool-execution races and the
    // child.send future below.
    let (childCancelTx, mut childCancelRx) = watch::channel(false);
    let mut handleCancelRx = handle.cancelRx.clone();
    let cancelBridge = tokio::spawn(async move {
        loop {
            if handleCancelRx.changed().await.is_err() {
                break;
            }
            if *handleCancelRx.borrow() {
                let _ = childCancelTx.send(true);
                break;
            }
        }
    });

    // Log forwarder — push ContentDeltas into the BgJob ring buffer line
    // by line, and rewrap visible events as SubagentEvent for the parent.
    let logSessionId = childSessionId.clone();
    let logParentTx = parentLogTx.clone();
    let logHandleId = handle.id;
    let lineSender = handle.lineSender();
    let logHandle = tokio::spawn(async move {
        let mut content = String::new();
        let mut turns: usize = 0;
        // Buffer partial lines from streamed deltas so the ring stores
        // whole lines (matches bash semantics).
        let mut deltaCarry = String::new();
        while let Some(event) = childLogRx.recv().await {
            match &event {
                LogEvent::ContentDelta(text) => {
                    content.push_str(text);
                    deltaCarry.push_str(text);
                    while let Some(pos) = deltaCarry.find('\n') {
                        let line = deltaCarry[..pos].to_string();
                        deltaCarry.drain(..=pos);
                        lineSender.push(line);
                    }
                }
                LogEvent::TurnComplete => turns += 1,
                _ => {}
            }
            match &event {
                LogEvent::ContentDelta(_)
                | LogEvent::ReasoningDelta(_)
                | LogEvent::ToolStarted { .. }
                | LogEvent::ToolAutoApproved { .. }
                | LogEvent::ToolResult { .. }
                | LogEvent::ToolDenied { .. }
                | LogEvent::ToolAutoDenied { .. }
                | LogEvent::TurnAborted { .. }
                | LogEvent::TurnComplete
                | LogEvent::TurnCancelled
                | LogEvent::Error(_) => {
                    let _ = logParentTx
                        .send(LogEvent::SubagentEvent {
                            sessionId: logSessionId.clone(),
                            event: Box::new(event),
                        })
                        .await;
                }
                _ => {}
            }
        }
        // Flush any trailing partial line so the final assistant message
        // shows up in the ring buffer.
        if !deltaCarry.is_empty() {
            lineSender.push(deltaCarry);
        }
        let _ = logHandleId; // silence unused if logging gets stripped
        (content, turns)
    });

    // Permit forwarder — identical to executeTask.
    let permitSessionId = childSessionId.clone();
    let permitParentTx = parentSessionRequestTx.clone();
    let permitHandle = tokio::spawn(async move {
        while let Some(req) = childRequestRx.recv().await {
            match req {
                SessionRequest::Permit {
                    origin: _,
                    name,
                    summary,
                    args,
                    diff,
                    explanation,
                    impact,
                    reply: childReply,
                } => {
                    let (parentReplyTx, parentReplyRx) = oneshot::channel();
                    if permitParentTx
                        .send(SessionRequest::Permit {
                            origin: PermitOrigin::Subagent {
                                sessionId: permitSessionId.clone(),
                            },
                            name,
                            summary,
                            args,
                            diff,
                            explanation,
                            impact,
                            reply: parentReplyTx,
                        })
                        .await
                        .is_err()
                    {
                        let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                        continue;
                    }
                    match parentReplyRx.await {
                        Ok(response) => {
                            let _ = childReply.send(response);
                        }
                        Err(_) => {
                            let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                        }
                    }
                }
            }
        }
    });

    let childInput = UserInput::from(prompt.clone());
    let (_childSteerTx, mut childSteerRx) = mpsc::channel::<UserInput>(1);
    // Subagents have no user keybind — give them a closed channel.
    let (_childUserBgTx, mut childUserBgRx) = mpsc::channel::<()>(1);

    let sendResult = child
        .send(
            &childInput,
            &childLogTx,
            &childRequestTx,
            &mut childCancelRx,
            &mut childSteerRx,
            &mut childUserBgRx,
        )
        .await;

    drop(childLogTx);
    drop(childRequestTx);

    let (rawContent, turns) = logHandle.await.unwrap_or_default();
    let _ = permitHandle.await;
    cancelBridge.abort();

    // Determine outcome BEFORE notifying the parent so the conversation
    // block shows the right status. Decorate the content with a
    // visible marker so existing SubagentComplete handlers (which just
    // store `content` on the block) display the right outcome inline.
    enum Outcome {
        Completed,
        Killed,
        Errored(String),
    }
    let outcome = if handle.cancelRequested() {
        Outcome::Killed
    } else if let Err(e) = &sendResult {
        Outcome::Errored(e.to_string())
    } else {
        Outcome::Completed
    };
    let displayContent = match &outcome {
        Outcome::Completed => rawContent.clone(),
        Outcome::Killed => {
            if rawContent.is_empty() {
                "[subagent cancelled by user]".into()
            } else {
                format!("[subagent cancelled by user]\n\n{rawContent}")
            }
        }
        Outcome::Errored(e) => {
            if rawContent.is_empty() {
                format!("[subagent errored: {e}]")
            } else {
                format!("[subagent errored: {e}]\n\n{rawContent}")
            }
        }
    };

    // Notify parent — same SubagentComplete event the foreground path
    // emits so existing UI handlers fire. Content already carries the
    // outcome marker.
    let _ = parentLogTx
        .send(LogEvent::SubagentComplete {
            sessionId: childSessionId.clone(),
            agentType: agentType.clone(),
            content: displayContent,
            turns,
        })
        .await;

    // Final state on the task: kill > error > complete.
    match outcome {
        Outcome::Killed => handle.killed().await,
        Outcome::Errored(e) => handle.errored(format!("subagent failed: {e}")).await,
        Outcome::Completed => handle.complete(0).await,
    }
}

fn buildChildPermissions(
    parent: &crate::permissions::Permissions,
    preset: &crate::runner::AgentPreset,
) -> crate::permissions::Permissions {
    use crate::permissions::{Permissions, Rule};
    use crate::tool::ToolSet;

    let mut rules: Vec<Rule> = Vec::new();

    // Preset floor: explicit denies for every state-mutating tool when
    // the preset is read-only. These rules come BEFORE the parent's
    // rules so a parent `allow *` or broad `allow shell *` cannot widen
    // a read-only subagent's scope. `shell` is the key entry because
    // the model self-classifies impact and we don't want a parent
    // allow-all to let an "impact: read" claim through — and the same
    // denial covers `runInBackground: true` since both shapes use the
    // same tool name.
    if matches!(preset.toolSet, ToolSet::ReadOnly) {
        for tool in [
            "shell",
            "writeFile",
            "editFile",
            "multiEdit",
            "copyFile",
            "moveFile",
            "deleteFile",
            "makeDirs",
            "terminalSpawn",
            "terminalSwitch",
            "terminalKill",
            "monitor",
            "monitorStop",
            // Wake-scheduling tools schedule autonomous future LLM
            // calls — never auto-approve for a read-only explore
            // subagent. The parent agent can still call them in its
            // own context.
            "scheduleWakeup",
            "cronCreate",
            "cronDelete",
            "fileWatch",
        ] {
            rules.push(Rule {
                tool: tool.into(),
                pattern: None,
                allow: false,
            });
        }
    }

    // Preset rules next (e.g. allowReadOnly's readFile/grep/... allows).
    rules.extend(preset.permissions.rules.iter().cloned());

    // Parent rules last — matched only when no preset rule applies.
    rules.extend(parent.rules.iter().cloned());

    Permissions {
        defaultMode: preset.permissions.defaultMode.clone(),
        rules,
        source: parent.source,
    }
}

/// Build the active rider list for a session from its config.
fn buildRiders(config: &Config) -> Vec<Rider> {
    let mut riders = Vec::new();
    if config.heavy.promptThinking {
        riders.push(Rider {
            id: "THINKING",
            content: crate::prompt::THINKING_RIDER_BODY.to_string(),
        });
    }
    riders
}

/// Render active riders into the prefix that goes before the user's text.
/// Returns an empty string when no riders are active.
fn renderRiderPrefix(riders: &[Rider]) -> String {
    if riders.is_empty() {
        return String::new();
    }
    let mut out = String::from("<CRITICAL_INSTRUCTIONS>\n");
    for r in riders {
        out.push_str(&format!(
            "<{id}>\n{body}\n</{id}>\n",
            id = r.id,
            body = r.content
        ));
    }
    out.push_str("</CRITICAL_INSTRUCTIONS>\n\n");
    out
}

/// Build the messages array sent to the API from the clean `history`.
///
/// Two transforms happen here — neither touches `self.history`:
/// 1. The latest User message gets a `<CRITICAL_INSTRUCTIONS>` prefix when
///    any riders are active.
/// 2. When `promptThinking` is on, assistant messages' `reasoning` field
///    is baked into their `content` as `<scratchpad>...</scratchpad>` so
///    the model sees the pattern it's being asked to produce. Models in
///    this mode don't get a separate `reasoning` JSON key.
fn buildRequestMessages(
    history: &[Message],
    riders: &[Rider],
    promptThinking: bool,
) -> Vec<Message> {
    let prefix = renderRiderPrefix(riders);
    let lastUserIdx = history
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }));

    history
        .iter()
        .enumerate()
        .map(|(i, msg)| match msg {
            Message::User { content } if Some(i) == lastUserIdx && !prefix.is_empty() => {
                Message::User {
                    content: prependToContent(content, &prefix),
                }
            }
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
            } if promptThinking => {
                let merged = match (reasoning.as_ref(), content.as_ref()) {
                    (Some(r), Some(c)) => Some(format!("<scratchpad>\n{r}\n</scratchpad>\n{c}")),
                    (Some(r), None) => Some(format!("<scratchpad>\n{r}\n</scratchpad>")),
                    (None, c) => c.cloned(),
                };
                Message::Assistant {
                    content: merged,
                    tool_calls: tool_calls.clone(),
                    reasoning: None,
                }
            }
            other => other.clone(),
        })
        .collect()
}

/// Prepend `prefix` to a `Content`'s text portion. Preserves multimodal
/// structure — riders attach to the first text block, image blocks
/// keep their position.
fn prependToContent(content: &Content, prefix: &str) -> Content {
    use crate::message::ContentBlock;
    match content {
        Content::Text(s) => Content::Text(format!("{prefix}{s}")),
        Content::Blocks(blocks) => {
            let mut out = Vec::with_capacity(blocks.len() + 1);
            let mut attached = false;
            for b in blocks {
                match b {
                    ContentBlock::Text { text } if !attached => {
                        out.push(ContentBlock::Text {
                            text: format!("{prefix}{text}"),
                        });
                        attached = true;
                    }
                    other => out.push(other.clone()),
                }
            }
            if !attached {
                out.insert(
                    0,
                    ContentBlock::Text {
                        text: prefix.to_string(),
                    },
                );
            }
            Content::Blocks(out)
        }
    }
}

fn unixNow() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn shellImpactStorageName(impact: &crate::tool::ShellImpact) -> &'static str {
    match impact {
        crate::tool::ShellImpact::Read => "read",
        crate::tool::ShellImpact::MinorMod => "minorMod",
        crate::tool::ShellImpact::MajorMod => "majorMod",
        crate::tool::ShellImpact::Delete => "delete",
    }
}

/// Agent session — owns the conversation and drives the turn loop.
pub struct Session {
    client: api::Client,
    config: Config,
    history: Vec<Message>,
    tools: Vec<ToolDef>,
    reasoning: Option<ReasoningConfig>,
    permissions: Permissions,
    /// Shared via `Arc<tokio::sync::Mutex<>>` so the deck's terminal-
    /// management requests can run concurrently with `session.send`.
    /// `tokio::sync::Mutex` (not std) because `ShellRegistry::spawn` is
    /// async — holding a std Mutex guard across the spawn await would
    /// break Send.
    pub shells: std::sync::Arc<tokio::sync::Mutex<ShellRegistry>>,
    /// Background-job registry. Shared via `Arc<Mutex<>>` so the deck's
    /// tasks-panel request handler can run concurrently with `session.send`
    /// (the latter holds `&mut self` for the whole turn). Use
    /// `jobPlaneHandle()` to clone the handle.
    pub jobs: std::sync::Arc<std::sync::Mutex<JobPlane>>,
    /// Monitor registry. Same Arc<Mutex> pattern as `tasks` so the deck
    /// can list/inspect monitors concurrently with a turn.
    pub monitors: std::sync::Arc<std::sync::Mutex<crate::monitors::MonitorPlane>>,
    /// Wake-source registry — schedule/cron/file-watch sources. Uses
    /// tokio::sync::Mutex because the wake schedulers run as tokio
    /// tasks that need to lock across .await points.
    pub wakes: std::sync::Arc<tokio::sync::Mutex<crate::wakes::WakeRegistry>>,
    transcript: Transcript,
    snapshots: crate::snapshot::SnapshotStore,
    compactionLog: CompactionLog,
    compactionTracker: compaction_trigger::Tracker,
    costTracker: crate::cost::CostTracker,
    /// Hard budget limit (USD). None = no limit.
    maxBudgetUsd: Option<f64>,
    topicTracker: TopicTracker,
    checkpoint: Option<CheckpointManager>,
    filesRead: HashMap<String, [u8; 20]>,
    exaClient: Option<web::ExaClient>,
    urlCache: web::UrlCache,
    mcpManager: Option<mcp::McpManager>,
    mcpConfigs: HashMap<String, mcp::config::ServerConfig>,
    lspManager: lsp::LspManager,
    lspWarmedUp: bool,
    /// Active riders — prepended to the latest User message's text inside
    /// a `<CRITICAL_INSTRUCTIONS>` wrapper at API call time. Never stored
    /// in `history` or the transcript.
    riders: Vec<Rider>,
    /// Stored for rebuild on rewind/fork switch.
    systemPrompt: String,
    /// Active branch head turn ID. None for fresh sessions with no messages yet.
    headTurnId: Option<String>,
    /// Pending topic classification task from the previous turn.
    pendingTopicEval: Option<tokio::task::JoinHandle<TopicDecision>>,
    /// Block ID associated with the pending topic eval (needed to apply the result).
    pendingTopicBlockId: Option<String>,
    /// Pending checkpoint snapshot (must resolve before tool dispatch).
    pendingCheckpoint: Option<tokio::task::JoinHandle<Result<()>>>,
    /// Count of assistant turns that have returned usage, used to distinguish
    /// the first turn (expected cache miss) from later turns (expected hit)
    /// in the cache-watchdog trace.
    turnsWithUsage: usize,
    /// Receiver for coalesced `WakeBatch` values. The session task takes
    /// this once at startup and selects on it alongside user input; each
    /// batch becomes one synthetic user-shaped turn driving the model.
    /// `Option` so the host can pull it out via `takeWakeBatchRx()`.
    wakeBatchRx: Option<mpsc::Receiver<crate::wakes::WakeBatch>>,
}

impl Session {
    /// Clone of the task-plane handle for callers that need to query or
    /// mutate it independently of the session's `&mut self` borrow
    /// (e.g. a dedicated tasks-panel request handler running concurrently
    /// with `session.send`).
    pub fn jobPlaneHandle(&self) -> std::sync::Arc<std::sync::Mutex<JobPlane>> {
        self.jobs.clone()
    }

    /// Clone of the monitor-plane handle. Same rationale as
    /// `jobPlaneHandle()` — for off-thread queries during a turn.
    pub fn monitorPlaneHandle(
        &self,
    ) -> std::sync::Arc<std::sync::Mutex<crate::monitors::MonitorPlane>> {
        self.monitors.clone()
    }

    /// Clone of the wake-registry handle.
    pub fn wakeRegistryHandle(
        &self,
    ) -> std::sync::Arc<tokio::sync::Mutex<crate::wakes::WakeRegistry>> {
        self.wakes.clone()
    }

    /// Take ownership of the wake-batch receiver. Called once by the
    /// session host (e.g. the deck) so it can select on coalesced
    /// `WakeBatch` values alongside user input. Subsequent calls return
    /// `None`.
    pub fn takeWakeBatchRx(&mut self) -> Option<mpsc::Receiver<crate::wakes::WakeBatch>> {
        self.wakeBatchRx.take()
    }

    /// Consume this session and return its shell registry handle. The
    /// caller hands this to `Session::new` / `Session::resume` so the
    /// resumed session shares the same live PTYs and the deck's
    /// dedicated terminal-request handler stays pointed at a valid
    /// registry without needing a hot-swap.
    pub fn intoShells(self) -> std::sync::Arc<tokio::sync::Mutex<ShellRegistry>> {
        self.shells
    }

    /// Clone of the shell-registry handle. Used by the deck's dedicated
    /// terminal-request handler so spawn/kill/list operations don't
    /// have to wait for `session.send` to release `&mut session`.
    pub fn shellsHandle(&self) -> std::sync::Arc<tokio::sync::Mutex<ShellRegistry>> {
        self.shells.clone()
    }

    /// Create a new session.
    ///
    /// Args:
    ///     config: Application config (API settings, etc).
    ///     permissions: Permission rules for tool execution.
    ///     shells: Named registry of PTYs for command execution.
    ///     interface: How the agent is being driven.
    ///     domains: Task-specific skill modules to include.
    pub fn new(
        config: &Config,
        permissions: Permissions,
        shells: std::sync::Arc<tokio::sync::Mutex<ShellRegistry>>,
        interface: InterfaceMode,
        domains: &[DomainModule],
    ) -> Result<Self> {
        let client = api::Client::new(config)?;
        let tools = tool::builtinDefs();

        let reasoning = config.heavy.reasoning.as_ref().map(|r| ReasoningConfig {
            effort: r.effort.clone(),
            summary: r.summary.clone(),
        });

        let systemPrompt = prompt::build(interface, domains, config.heavy.promptThinking);

        let history = vec![Message::System {
            content: systemPrompt.clone(),
        }];

        let sessionId = transcript::newSessionId();
        let transcript = Transcript::create(&sessionId)?;
        let snapshots = crate::snapshot::SnapshotStore::open(transcript.sessionDir())?;
        let compactionLog = CompactionLog::open(transcript.sessionDir())?;
        let compactionTracker =
            compaction_trigger::Tracker::new(config.heavy.contextWindow, config.compactRatio);
        // System prompt is ephemeral — never recorded in transcript.
        tracing::info!(sessionId = %sessionId, "session created");

        let exaClient = web::ExaClient::new(&config.web.searchKey);
        let projectLsp = lsp::config::loadProjectLsp(
            config
                .projectRoot
                .as_deref()
                .unwrap_or(&std::env::current_dir().unwrap_or_default()),
        )
        .unwrap_or_default();
        let lspManager = lsp::LspManager::new(&config.lsp, &projectLsp);

        let (wakeArc, wakeBatchRx) = crate::wakes::WakeRegistry::new();

        Ok(Self {
            client,
            config: config.clone(),
            history,
            tools,
            reasoning,
            permissions,
            shells,
            jobs: std::sync::Arc::new(std::sync::Mutex::new(JobPlane::new(
                config.projectRoot.clone(),
            ))),
            monitors: std::sync::Arc::new(std::sync::Mutex::new(
                crate::monitors::MonitorPlane::new(),
            )),
            wakes: wakeArc,
            transcript,
            snapshots,
            compactionLog,
            compactionTracker,
            costTracker: crate::cost::CostTracker::new(),
            maxBudgetUsd: None,
            topicTracker: TopicTracker::new(),
            checkpoint: None,
            filesRead: HashMap::new(),
            exaClient,
            urlCache: web::UrlCache::new(),
            mcpManager: None,
            mcpConfigs: HashMap::new(),
            lspManager,
            lspWarmedUp: false,
            riders: buildRiders(config),
            systemPrompt,
            headTurnId: None,
            pendingTopicEval: None,
            pendingTopicBlockId: None,
            pendingCheckpoint: None,
            turnsWithUsage: 0,
            wakeBatchRx: Some(wakeBatchRx),
        })
    }

    /// Get the session ID.
    pub fn sessionId(&self) -> &str {
        &self.transcript.sessionId
    }

    /// Replace the tool definitions for this session.
    ///
    /// Used by subagent execution to restrict tools.
    pub fn setTools(&mut self, tools: Vec<ToolDef>) {
        self.tools = tools;
    }

    /// Set a hard budget limit (USD). The session will stop when exceeded.
    pub fn setMaxBudget(&mut self, limit: f64) {
        self.maxBudgetUsd = Some(limit);
    }

    /// Resume an existing session.
    ///
    /// Reconstructs conversation history from the transcript and compaction log,
    /// restores topic tracker state, and opens existing log files for append.
    ///
    /// Args:
    ///     config: Application config.
    ///     permissions: Permission rules for tool execution.
    ///     shell: Stateful shell session.
    ///     interface: How the agent is being driven.
    ///     domains: Task-specific skill modules.
    ///     sessionId: The session to resume.
    pub async fn resume(
        config: &Config,
        permissions: Permissions,
        shells: std::sync::Arc<tokio::sync::Mutex<ShellRegistry>>,
        interface: InterfaceMode,
        domains: &[DomainModule],
        sessionId: &str,
    ) -> std::result::Result<
        Self,
        (
            anyhow::Error,
            std::sync::Arc<tokio::sync::Mutex<ShellRegistry>>,
        ),
    > {
        Self::resumeInner(config, permissions, shells, interface, domains, sessionId).await
    }

    async fn resumeInner(
        config: &Config,
        permissions: Permissions,
        shells: std::sync::Arc<tokio::sync::Mutex<ShellRegistry>>,
        interface: InterfaceMode,
        domains: &[DomainModule],
        sessionId: &str,
    ) -> std::result::Result<
        Self,
        (
            anyhow::Error,
            std::sync::Arc<tokio::sync::Mutex<ShellRegistry>>,
        ),
    > {
        let client = match api::Client::new(config) {
            Ok(c) => c,
            Err(e) => return Err((e, shells)),
        };
        let tools = tool::builtinDefs();

        let reasoning = config.heavy.reasoning.as_ref().map(|r| ReasoningConfig {
            effort: r.effort.clone(),
            summary: r.summary.clone(),
        });

        // System prompt is rebuilt from current config, not from transcript.
        let systemPrompt = prompt::build(interface, domains, config.heavy.promptThinking);

        let mut transcript = match Transcript::open(sessionId) {
            Ok(t) => t,
            Err(e) => return Err((e, shells)),
        };
        let compactionLog = match CompactionLog::open(transcript.sessionDir()) {
            Ok(c) => c,
            Err(e) => return Err((e, shells)),
        };
        let snapshots = match crate::snapshot::SnapshotStore::open(transcript.sessionDir()) {
            Ok(s) => s,
            Err(e) => return Err((e, shells)),
        };

        // Load headTurn from meta to determine active branch.
        let meta = Transcript::loadMeta(transcript.sessionDir()).ok();
        let headTurnId = meta
            .as_ref()
            .and_then(|m| m.headTurn.clone())
            .or_else(|| transcript.lastTurnId());

        // Set the transcript's append point to the active branch head.
        if let Some(ref head) = headTurnId
            && let Ok(allTurns) = transcript.loadAll()
            && let Some(headTurn) = allTurns.iter().find(|t| t.id == *head)
        {
            transcript.setHead(head, &headTurn.blockId);
        }

        // Reconstruct conversation from the active branch.
        let reconstructed = match &headTurnId {
            Some(head) => match context::reconstruct(&transcript, &compactionLog, head) {
                Ok(h) => h,
                Err(e) => return Err((e, shells)),
            },
            None => Vec::new(),
        };

        // Prepend system prompt (ephemeral, not from transcript).
        let mut history = vec![Message::System {
            content: systemPrompt.clone(),
        }];
        history.extend(reconstructed);

        let compactionTracker =
            compaction_trigger::Tracker::new(config.heavy.contextWindow, config.compactRatio);

        // Restore topic tracker state from meta.json.
        let mut topicTracker = TopicTracker::new();
        if let Some(ref m) = meta {
            if !m.topics.is_empty() {
                // New format: full TopicInfo persisted.
                topicTracker.restoreState(m.topics.clone());
            } else if !m.topicLabels.is_empty() {
                // Backward compat: old sessions with only labels.
                let topicInfos: Vec<crate::topic::TopicInfo> = m
                    .topicLabels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| crate::topic::TopicInfo {
                        topicId: format!("topic-{:02}", i + 1),
                        label: label.clone(),
                        startBlock: String::new(),
                        blockCount: 0,
                    })
                    .collect();
                topicTracker.restoreState(topicInfos);
            }
        }

        // Rebuild filesRead from history — hash current disk content for staleness detection.
        let mut filesRead: HashMap<String, [u8; 20]> = HashMap::new();
        for msg in &history {
            if let Message::Assistant {
                tool_calls: Some(calls),
                ..
            } = msg
            {
                for call in calls {
                    if call.function.name == "readFile"
                        && let Ok(args) =
                            serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                        && let Some(path) = args["path"].as_str()
                    {
                        let norm = normalizePath(path);
                        if let Ok(bytes) = std::fs::read(&norm) {
                            let digest = sha1_smol::Sha1::from(&bytes).digest().bytes();
                            filesRead.insert(norm, digest);
                        }
                    }
                }
            }
        }

        tracing::info!(
            sessionId = %sessionId,
            historyLen = history.len(),
            filesTracked = filesRead.len(),
            "session resumed"
        );

        let exaClient = web::ExaClient::new(&config.web.searchKey);
        let projectLsp = lsp::config::loadProjectLsp(
            config
                .projectRoot
                .as_deref()
                .unwrap_or(&std::env::current_dir().unwrap_or_default()),
        )
        .unwrap_or_default();
        let lspManager = lsp::LspManager::new(&config.lsp, &projectLsp);

        let (wakeArc, wakeBatchRx) = crate::wakes::WakeRegistry::new();

        Ok(Self {
            client,
            config: config.clone(),
            history,
            tools,
            reasoning,
            permissions,
            shells,
            jobs: std::sync::Arc::new(std::sync::Mutex::new(JobPlane::new(
                config.projectRoot.clone(),
            ))),
            monitors: std::sync::Arc::new(std::sync::Mutex::new(
                crate::monitors::MonitorPlane::new(),
            )),
            wakes: wakeArc,
            transcript,
            snapshots,
            compactionLog,
            compactionTracker,
            costTracker: {
                let mut ct = crate::cost::CostTracker::new();
                if let Some(ref m) = meta {
                    ct.seed(m.totalCost);
                }
                ct
            },
            maxBudgetUsd: None,
            topicTracker,
            checkpoint: None,
            filesRead,
            exaClient,
            urlCache: web::UrlCache::new(),
            mcpManager: None,
            mcpConfigs: HashMap::new(),
            lspManager,
            lspWarmedUp: false,
            riders: buildRiders(config),
            systemPrompt,
            headTurnId,
            pendingTopicEval: None,
            pendingTopicBlockId: None,
            pendingCheckpoint: None,
            turnsWithUsage: 0,
            wakeBatchRx: Some(wakeBatchRx),
        })
    }

    /// Current topic label (for title bar display on resume).
    pub fn currentTopicLabel(&self) -> &str {
        self.topicTracker.currentLabel()
    }

    /// Collect the pending topic classification and emit TopicChanged.
    ///
    /// Called at the end of sendInner so the title updates within the
    /// same turn, not deferred to the next user message.
    async fn collectTopicEval(&mut self, logTx: &mpsc::Sender<LogEvent>) {
        if let Some(handle) = self.pendingTopicEval.take() {
            let prevBlockId = self.pendingTopicBlockId.take().unwrap_or_default();
            match handle.await {
                Ok(decision) => {
                    let result = self.topicTracker.applyDecision(decision, &prevBlockId);
                    self.transcript.setTopicId(&result.topicId);
                    if result.isNewTopic {
                        tracing::info!(
                            topicId = %result.topicId,
                            label = %result.label,
                            "new topic segment"
                        );
                    }
                    let _ = logTx
                        .send(LogEvent::TopicChanged {
                            label: result.label.clone(),
                        })
                        .await;
                }
                Err(e) => {
                    tracing::warn!("pending topic eval panicked: {e}");
                }
            }
        }
    }

    /// Initialize the checkpoint system for a project directory.
    ///
    /// Args:
    ///     projectDir: Path to the project root.
    pub async fn initCheckpoint(&mut self, projectDir: &Path) -> Result<()> {
        let dirStr = projectDir.to_str().unwrap_or("");
        let manager = CheckpointManager::init(dirStr).await?;
        self.checkpoint = Some(manager);
        tracing::info!("checkpoint system initialized");
        Ok(())
    }

    /// Load all transcript turns for this session.
    pub fn loadTranscript(&self) -> Result<Vec<crate::transcript::Turn>> {
        self.transcript.loadAll()
    }

    /// Load turns on the active branch by walking the parent-child chain.
    pub fn loadBranchTurns(&self) -> Result<Vec<crate::transcript::Turn>> {
        let headId = match &self.headTurnId {
            Some(id) => id.clone(),
            None => return Ok(Vec::new()),
        };
        let allTurns = self.transcript.loadAll()?;
        let turnMap: std::collections::HashMap<&str, &crate::transcript::Turn> =
            allTurns.iter().map(|t| (t.id.as_str(), t)).collect();

        let mut chain = Vec::new();
        let mut current: Option<&str> = Some(&headId);
        while let Some(id) = current {
            if let Some(turn) = turnMap.get(id) {
                // Skip system turns (ephemeral).
                if !matches!(turn.role, crate::transcript::TurnRole::System) {
                    chain.push((*turn).clone());
                }
                current = turn.parentId.as_deref();
            } else {
                break;
            }
        }
        chain.reverse();
        Ok(chain)
    }

    /// Rebuild the topic tracker from the active branch turns.
    ///
    /// Called after rewind or fork-switch so topic state reflects only the
    /// active branch. Labels are sourced from the union of the live tracker
    /// and the on-disk `meta.topics`. The disk fallback matters when a
    /// prior rewind landed on a pre-classification point and transiently
    /// emptied the tracker — without it, any subsequent rebuild would lose
    /// every label and bake `"Unknown"` into persisted meta. `setTopicId` is
    /// called unconditionally so the transcript's stamp-on-write state
    /// cannot drift from the tracker's `currentTopicId`.
    fn rebuildTopicTracker(&mut self) {
        let branchTurns = self.loadBranchTurns().unwrap_or_default();

        let mut labelSources: Vec<crate::topic::TopicInfo> = self.topicTracker.topics().to_vec();
        if let Ok(meta) = Transcript::loadMeta(self.transcript.sessionDir()) {
            for t in meta.topics {
                if !labelSources.iter().any(|x| x.topicId == t.topicId) {
                    labelSources.push(t);
                }
            }
        }

        let rebuilt = crate::topic::rebuildTopicInfos(&branchTurns, &labelSources);
        self.topicTracker.restoreState(rebuilt);
        self.transcript
            .setTopicId(self.topicTracker.currentTopicId());
    }

    /// Load turns for display — extends past the current head through any
    /// un-branched continuation. Once the user sends a new message (creating
    /// a second child at the head), this collapses to match `loadBranchTurns`.
    pub fn loadDisplayTurns(&self) -> Result<Vec<crate::transcript::Turn>> {
        let tipId = match self.findChainTip() {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };
        let allTurns = self.transcript.loadAll()?;
        let turnMap: std::collections::HashMap<&str, &crate::transcript::Turn> =
            allTurns.iter().map(|t| (t.id.as_str(), t)).collect();

        let mut chain = Vec::new();
        let mut current: Option<&str> = Some(&tipId);
        while let Some(id) = current {
            if let Some(turn) = turnMap.get(id) {
                if !matches!(turn.role, crate::transcript::TurnRole::System) {
                    chain.push((*turn).clone());
                }
                current = turn.parentId.as_deref();
            } else {
                break;
            }
        }
        chain.reverse();
        Ok(chain)
    }

    /// Walk forward from the current head through single-child continuations.
    /// Returns the tip of the un-branched chain, or the head itself if it has
    /// 0 or 2+ children.
    fn findChainTip(&self) -> Option<String> {
        let headId = self.headTurnId.as_ref()?;
        let allTurns = self.transcript.loadAll().ok()?;

        let mut children: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        for turn in &allTurns {
            if let Some(ref pid) = turn.parentId {
                children.entry(pid.as_str()).or_default().push(&turn.id);
            }
        }

        let mut current = headId.as_str();
        loop {
            match children.get(current) {
                Some(kids) if kids.len() == 1 => current = kids[0],
                _ => break,
            }
        }

        Some(current.to_string())
    }

    /// Derive compaction markers from the compaction log.
    ///
    /// Returns `(stage, blockIdx)` pairs for each stage that has replaced
    /// whole blocks. The block index is relative to the transcript's block
    /// sequence (0 = first block).
    pub fn compactionMarkers(&self) -> Vec<(String, usize)> {
        let ops = match self.compactionLog.loadAll() {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        let mut markers: Vec<(String, usize)> = Vec::new();
        let mut hasS2 = false;
        let mut hasS3 = false;
        let mut hasS4 = false;

        for op in &ops {
            match op {
                crate::compaction::CompactionOp::BlockCompact { .. } => hasS2 = true,
                crate::compaction::CompactionOp::TopicCompact { .. } => hasS3 = true,
                crate::compaction::CompactionOp::FullCompact { .. } => hasS4 = true,
                _ => {}
            }
        }

        // S2/S3/S4 zones all start at the oldest block.
        if hasS4 {
            markers.push(("S4".into(), 0));
        } else if hasS3 {
            markers.push(("S3".into(), 0));
        } else if hasS2 {
            markers.push(("S2".into(), 0));
        }

        markers
    }

    /// List available sessions, optionally filtered by project directory.
    pub fn listSessions(projectDir: Option<&Path>) -> Result<Vec<SessionMeta>> {
        transcript::listSessions(projectDir.and_then(|p| p.to_str()))
    }

    /// Send a user message and run the full turn loop.
    ///
    /// When a tool call verdict is `NeedsApproval` and the permit mode is
    /// `Ask`, the session emits `SessionRequest::Permit` on
    /// `sessionRequestTx` and awaits the `oneshot` reply carried inside it.
    ///
    /// Args:
    ///     input: The user's input (text + optional image attachments).
    ///     logTx: Channel for monotone log events.
    ///     sessionRequestTx: Channel for session → consumer requests (permits).
    pub fn send<'a>(
        &'a mut self,
        input: &'a UserInput,
        logTx: &'a mpsc::Sender<LogEvent>,
        sessionRequestTx: &'a mpsc::Sender<SessionRequest>,
        cancelRx: &'a mut watch::Receiver<bool>,
        steerRx: &'a mut mpsc::Receiver<UserInput>,
        userBgRx: &'a mut mpsc::Receiver<()>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Drain stale steer messages from a previous cancelled turn.
            while steerRx.try_recv().is_ok() {}
            // Drain stale user-bg requests from a previous turn.
            while userBgRx.try_recv().is_ok() {}

            let userMessage = &input.text;
            tracing::info!(
                len = userMessage.len(),
                attachments = input.attachments.len(),
                "user message received"
            );

            // Warm up LSP servers on first send (scans project, starts matching servers).
            if !self.lspWarmedUp {
                self.lspWarmedUp = true;
                let projectDir = std::env::current_dir().unwrap_or_default();
                self.lspManager.warmUp(&projectDir).await;
            }

            // Build content — multimodal if attachments present. Riders are
            // applied later against the API-call copy; `self.history` and the
            // transcript get the clean user text.
            let (content, turnAttachments) = buildUserContent(userMessage, &input.attachments);
            self.history.push(Message::User { content });
            match self.transcript.recordUser(
                userMessage,
                self.headTurnId.as_deref(),
                turnAttachments,
            ) {
                Ok(turnId) => self.headTurnId = Some(turnId),
                Err(e) => tracing::warn!("transcript write failed: {e}"),
            }

            self.sendInner(logTx, sessionRequestTx, cancelRx, steerRx, userBgRx)
                .await
        })
    }

    /// Inject a coalesced `WakeBatch` and run one turn. Formats the
    /// batch as a single `<wakes count="N">…</wakes>` envelope, pushes
    /// it as user-shaped content to the model, and records it as a
    /// `TurnRole::Wake` transcript entry so resume can render the turn
    /// as a notice rather than a real user bubble.
    ///
    /// Wakes that arrive while a turn is running are queued in the
    /// batcher's receiver — the deck's session task selects on that
    /// receiver alongside `userInputRx`, so this only runs while idle.
    pub fn injectWakeBatch<'a>(
        &'a mut self,
        batch: crate::wakes::WakeBatch,
        logTx: &'a mpsc::Sender<LogEvent>,
        sessionRequestTx: &'a mpsc::Sender<SessionRequest>,
        cancelRx: &'a mut watch::Receiver<bool>,
        steerRx: &'a mut mpsc::Receiver<UserInput>,
        userBgRx: &'a mut mpsc::Receiver<()>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // User input that landed since the last in-turn drain beats
            // the wake. If anything is queued in steerRx, requeue the
            // batch's fires (they re-batch on the next idle tick) and
            // process the queued input as a real user turn instead.
            let mut queued: Vec<UserInput> = Vec::new();
            while let Ok(input) = steerRx.try_recv() {
                queued.push(input);
            }
            if !queued.is_empty() {
                let fireTx = self.wakes.lock().await.fireSender();
                for fire in batch.fires {
                    let _ = fireTx.send(fire);
                }
                while userBgRx.try_recv().is_ok() {}
                let combined = UserInput {
                    text: queued
                        .iter()
                        .map(|i| i.text.clone())
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                    attachments: queued.into_iter().flat_map(|i| i.attachments).collect(),
                };
                return self
                    .send(
                        &combined,
                        logTx,
                        sessionRequestTx,
                        cancelRx,
                        steerRx,
                        userBgRx,
                    )
                    .await;
            }
            while userBgRx.try_recv().is_ok() {}

            if batch.fires.is_empty() {
                return Ok(());
            }

            let envelope = formatWakeBatch(&batch);
            tracing::info!(count = batch.fires.len(), "injecting coalesced wake batch",);

            // Warm up LSP servers on first send.
            if !self.lspWarmedUp {
                self.lspWarmedUp = true;
                let projectDir = std::env::current_dir().unwrap_or_default();
                self.lspManager.warmUp(&projectDir).await;
            }

            // The model sees user-shaped content; the transcript stores
            // it under TurnRole::Wake so resume can render it as a
            // notice instead of a real user bubble.
            let (content, _atts) = buildUserContent(&envelope, &[]);
            self.history.push(Message::User { content });
            match self
                .transcript
                .recordWake(&envelope, self.headTurnId.as_deref())
            {
                Ok(turnId) => self.headTurnId = Some(turnId),
                Err(e) => tracing::warn!("wake transcript write failed: {e}"),
            }

            // Surface the batch to the deck for UI rendering. One
            // WakeBatchInjected per batch — no per-fire chips.
            let _ = logTx
                .send(LogEvent::WakeBatchInjected {
                    count: batch.fires.len(),
                    summary: wakeBatchSummary(&batch),
                })
                .await;

            self.sendInner(logTx, sessionRequestTx, cancelRx, steerRx, userBgRx)
                .await
        })
    }

    /// Re-run the last user turn. Drops a trailing errored-or-cancelled
    /// assistant message from history if present so the model responds
    /// fresh. Returns early with a no-op if history doesn't end with a
    /// user message (i.e. there's nothing to retry).
    ///
    /// In-memory only — the committed Errored transcript entry stays as a
    /// dead branch, and the new assistant turn becomes a sibling child of
    /// the user turn. This mirrors how `rewind` + new-user-send would look,
    /// minus the fork bookkeeping.
    pub fn retryLastTurn<'a>(
        &'a mut self,
        logTx: &'a mpsc::Sender<LogEvent>,
        sessionRequestTx: &'a mpsc::Sender<SessionRequest>,
        cancelRx: &'a mut watch::Receiver<bool>,
        steerRx: &'a mut mpsc::Receiver<UserInput>,
        userBgRx: &'a mut mpsc::Receiver<()>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            while steerRx.try_recv().is_ok() {}
            while userBgRx.try_recv().is_ok() {}

            // Pop the trailing partial assistant, if any.
            if matches!(self.history.last(), Some(Message::Assistant { .. })) {
                self.history.pop();
            }

            // If history doesn't end with a user message, there's nothing to
            // retry — bail out cleanly.
            if !matches!(self.history.last(), Some(Message::User { .. })) {
                tracing::warn!("retryLastTurn called but history has no trailing user message");
                return Ok(());
            }

            // Walk headTurnId back to the user turn so the new assistant's
            // transcript entry parents onto the user, not the errored turn.
            if let Some(erroredId) = self.headTurnId.clone()
                && let Ok(turns) = self.transcript.loadAll()
                && let Some(t) = turns.iter().find(|t| t.id == erroredId)
                && matches!(t.role, crate::transcript::TurnRole::Assistant)
                && let Some(parentId) = t.parentId.clone()
                && let Some(parent) = turns.iter().find(|t| t.id == parentId)
            {
                self.transcript.setHead(&parentId, &parent.blockId);
                self.headTurnId = Some(parentId);
            }

            tracing::info!("retrying last turn");
            self.sendInner(logTx, sessionRequestTx, cancelRx, steerRx, userBgRx)
                .await
        })
    }

    /// Resume streaming from where the last turn was cut off. The errored
    /// assistant (with its partial content/reasoning) stays in history and
    /// acts as a prefill for the model to continue from.
    ///
    /// Anthropic treats a trailing assistant message as a continuation
    /// point; the model picks up where the previous one left off.
    pub fn continueLastTurn<'a>(
        &'a mut self,
        logTx: &'a mpsc::Sender<LogEvent>,
        sessionRequestTx: &'a mpsc::Sender<SessionRequest>,
        cancelRx: &'a mut watch::Receiver<bool>,
        steerRx: &'a mut mpsc::Receiver<UserInput>,
        userBgRx: &'a mut mpsc::Receiver<()>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            while steerRx.try_recv().is_ok() {}
            while userBgRx.try_recv().is_ok() {}

            // For continue we need a trailing assistant (the prefill).
            if !matches!(self.history.last(), Some(Message::Assistant { .. })) {
                tracing::warn!("continueLastTurn called with no trailing assistant to continue");
                return Ok(());
            }

            tracing::info!("continuing from partial assistant response");
            self.sendInner(logTx, sessionRequestTx, cancelRx, steerRx, userBgRx)
                .await
        })
    }

    /// Inner turn loop.
    async fn sendInner(
        &mut self,
        logTx: &mpsc::Sender<LogEvent>,
        sessionRequestTx: &mpsc::Sender<SessionRequest>,
        cancelRx: &mut watch::Receiver<bool>,
        steerRx: &mut mpsc::Receiver<UserInput>,
        userBgRx: &mut mpsc::Receiver<()>,
    ) -> Result<()> {
        let blockId = self.transcript.currentBlock().to_string();

        // Spawn topic classification — runs concurrently with the main model
        // stream. Collected at the end of this sendInner call (not deferred).
        let topicMessages = self.topicTracker.prepareClassification(&self.history);
        let topicClient = self.client.clone();
        let topicModel = self.config.utility.model.clone();
        self.pendingTopicBlockId = Some(blockId.clone());
        self.pendingTopicEval = Some(tokio::spawn(async move {
            crate::topic::classifyPrepared(topicMessages, topicClient, topicModel).await
        }));

        // Persist session metadata so /resume can find this session.
        self.updateMeta();

        // Spawn checkpoint snapshot — must resolve before tool dispatch but
        // can overlap with the main model's text streaming.
        let checkpointClone = self.checkpoint.clone();
        if let Some(cp) = checkpointClone {
            let turnId = self.transcript.currentBlock().to_string();
            self.pendingCheckpoint = Some(tokio::spawn(async move { cp.snapshot(&turnId).await }));
        }

        const MAX_RETRIES: u32 = 5;
        let mut retryCount: u32 = 0;

        // NOTE: All exit paths use `break` (not `return`) so the topic eval
        // collection at the bottom of this block always runs.
        let result: Result<()> = 'turns: {
            loop {
                // Check for cancellation between turns.
                if *cancelRx.borrow() {
                    tracing::info!("turn cancelled before streaming");
                    let _ = logTx.send(LogEvent::TurnCancelled).await;
                    break 'turns Ok(());
                }

                tracing::debug!(historyLen = self.history.len(), "starting turn");
                // NOTE: Err from streamOneTurn means either a permanent API error
                // or a transient one that already exhausted the API client's own
                // 8-attempt retry loop. Don't retry again here — only retry
                // mid-stream SSE errors (returned as TurnResult::TransientError).
                let turnResult = self.streamOneTurn(logTx, cancelRx).await?;

                match turnResult {
                    TurnResult::TransientError(msg) => {
                        retryCount += 1;
                        if retryCount > MAX_RETRIES {
                            let _ = logTx.send(LogEvent::Error(msg)).await;
                            break 'turns Ok(());
                        }
                        tracing::warn!(
                            attempt = retryCount,
                            max = MAX_RETRIES,
                            error = %msg,
                            "transient API error, retrying"
                        );
                        let _ = logTx
                            .send(LogEvent::Retrying {
                                attempt: retryCount,
                                maxAttempts: MAX_RETRIES,
                            })
                            .await;
                        // Exponential backoff: 1s, 2s, 4s, 8s, 16s.
                        let delay = Duration::from_secs(1 << (retryCount - 1));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    TurnResult::Done { promptTokens } => {
                        retryCount = 0;
                        if let Some(tokens) = promptTokens {
                            self.compactionTracker.updateTokens(tokens);
                            self.checkCompactionTrigger(logTx).await;
                        }
                        // Extend turn if user queued messages during streaming.
                        if self.drainSteer(steerRx, logTx).await {
                            tracing::info!("extending turn with queued user messages");
                            continue;
                        }
                        tracing::info!("turn complete (no tool calls)");
                        let _ = logTx.send(LogEvent::TurnComplete).await;
                        break 'turns Ok(());
                    }
                    TurnResult::Cancelled => {
                        tracing::info!("turn cancelled during streaming");
                        let _ = logTx.send(LogEvent::TurnCancelled).await;
                        break 'turns Ok(());
                    }
                    TurnResult::ToolCalls {
                        calls,
                        content,
                        reasoning,
                        promptTokens,
                    } => {
                        retryCount = 0;
                        // Update token count but don't trigger compaction mid-loop.
                        if let Some(tokens) = promptTokens {
                            self.compactionTracker.updateTokens(tokens);
                        }
                        // Hard budget enforcement.
                        if let Some(limit) = self.maxBudgetUsd
                            && self.costTracker.sessionCost() >= limit
                        {
                            let msg = format!(
                                "Budget limit reached ({} / {}). Stopping.",
                                crate::cost::formatCost(self.costTracker.sessionCost()),
                                crate::cost::formatCost(limit),
                            );
                            tracing::warn!(%msg, "hard budget limit hit");
                            let _ = logTx.send(LogEvent::Error(msg)).await;
                            break 'turns Ok(());
                        }
                        tracing::info!(
                            callCount = calls.len(),
                            hasContent = content.is_some(),
                            hasReasoning = reasoning.is_some(),
                            "turn produced tool calls"
                        );
                        // Record each tool call to transcript.
                        for call in &calls {
                            let args: serde_json::Value =
                                serde_json::from_str(&call.function.arguments)
                                    .unwrap_or(serde_json::Value::Null);
                            match self.transcript.recordToolCall(
                                &call.id,
                                &call.function.name,
                                &args,
                            ) {
                                Ok(turnId) => self.headTurnId = Some(turnId),
                                Err(e) => tracing::warn!("transcript write failed: {e}"),
                            }
                        }

                        self.history.push(buildAssistantMessage(
                            content,
                            Some(calls.clone()),
                            reasoning,
                        ));

                        // Checkpoint must complete before tools modify files.
                        if let Some(handle) = self.pendingCheckpoint.take() {
                            match handle.await {
                                Ok(Ok(())) => {}
                                Ok(Err(e)) => tracing::warn!("checkpoint snapshot failed: {e}"),
                                Err(e) => tracing::warn!("checkpoint task panicked: {e}"),
                            }
                        }

                        let mut aborted = false;

                        for (callIdx, call) in calls.iter().enumerate() {
                            // Check for cancellation between tool calls.
                            if *cancelRx.borrow() {
                                tracing::info!("cancelled between tool calls");
                                for remaining in &calls[callIdx..] {
                                    self.pushToolResult(
                                        &remaining.id,
                                        crate::message::Content::text("Cancelled by user."),
                                    );
                                }
                                let _ = logTx.send(LogEvent::TurnCancelled).await;
                                break 'turns Ok(());
                            }

                            let action =
                                match tool::parse(&call.function.name, &call.function.arguments) {
                                    Ok(a) => a,
                                    Err(err) => {
                                        self.pushToolResult(&call.id, err.to_string().into());
                                        continue;
                                    }
                                };
                            let summary = tool::summarize(&action);

                            // Pre-emptive LSP notification: send didChange with proposed
                            // content so RA starts analyzing while the user reviews.
                            // Stores (path, original_content) for revert on denial.
                            let lspPreemptive: Option<(String, String)> =
                                if let Some((path, proposed)) = tool::proposedContent(&action) {
                                    let original = std::fs::read_to_string(&path).ok();
                                    self.lspManager.touchFile(&path, &proposed).await;
                                    original.map(|orig| (path, orig))
                                } else {
                                    None
                                };

                            let verdict = self.permissions.check(&action);

                            tracing::debug!(
                                tool = %call.function.name,
                                verdict = ?verdict,
                                "checking tool permission"
                            );

                            let approved = match verdict {
                                Verdict::Allow => {
                                    let _ = logTx
                                        .send(LogEvent::ToolAutoApproved {
                                            name: call.function.name.clone(),
                                            summary: summary.clone(),
                                        })
                                        .await;
                                    true
                                }
                                Verdict::Deny => {
                                    let _ = logTx
                                        .send(LogEvent::ToolAutoDenied {
                                            name: call.function.name.clone(),
                                            summary: summary.clone(),
                                        })
                                        .await;
                                    false
                                }
                                Verdict::NeedsApproval => {
                                    match self.permissions.defaultMode {
                                        PermitMode::Ask => {
                                            let diff = tool::diffPreview(&action);
                                            let explanation =
                                                crate::permissions::toolExplanation(&action)
                                                    .map(|s| s.to_string());
                                            let impact = crate::permissions::toolImpact(&action);
                                            let (replyTx, replyRx) = oneshot::channel();
                                            let _ = sessionRequestTx
                                                .send(SessionRequest::Permit {
                                                    origin: PermitOrigin::Top,
                                                    name: call.function.name.clone(),
                                                    summary,
                                                    args: call.function.arguments.clone(),
                                                    diff,
                                                    explanation,
                                                    impact,
                                                    reply: replyTx,
                                                })
                                                .await;

                                            // Wait for supervisor response or cancellation.
                                            tokio::select! {
                                                permit = replyRx => {
                                                    use crate::permissions::PermitResponse;
                                                    match permit {
                                                        Ok(PermitResponse::Allow) => true,
                                                        Ok(PermitResponse::AlwaysAllow { pattern }) => {
                                                            let (toolName, _) = crate::permissions::actionKey(&action);
                                                            let rulePattern = crate::permissions::normalizeRulePattern(
                                                                &action, &pattern,
                                                            );
                                                            let persistPattern = rulePattern
                                                                .clone()
                                                                .unwrap_or_default();
                                                            self.permissions.addRule(crate::permissions::Rule {
                                                                tool: toolName.into(),
                                                                pattern: rulePattern,
                                                                allow: true,
                                                            });
                                                            // Persist to .flatline/config.toml if we have a project root.
                                                            if let Some(ref root) = self.config.projectRoot
                                                                && let Err(e) = crate::config::persistPermissionRule(
                                                                    root,
                                                                    &self.permissions,
                                                                    toolName,
                                                                    &persistPattern,
                                                                    true,
                                                                ) {
                                                                    tracing::warn!("failed to persist permission rule: {e}");
                                                                }
                                                            true
                                                        }
                                                        Ok(PermitResponse::AlwaysDeny { pattern }) => {
                                                            let (toolName, _) = crate::permissions::actionKey(&action);
                                                            let rulePattern = crate::permissions::normalizeRulePattern(
                                                                &action, &pattern,
                                                            );
                                                            let persistPattern = rulePattern
                                                                .clone()
                                                                .unwrap_or_default();
                                                            self.permissions.addRule(crate::permissions::Rule {
                                                                tool: toolName.into(),
                                                                pattern: rulePattern,
                                                                allow: false,
                                                            });
                                                            if let Some(ref root) = self.config.projectRoot
                                                                && let Err(e) = crate::config::persistPermissionRule(
                                                                    root,
                                                                    &self.permissions,
                                                                    toolName,
                                                                    &persistPattern,
                                                                    false,
                                                                ) {
                                                                    tracing::warn!("failed to persist deny rule: {e}");
                                                                }
                                                            false
                                                        }
                                                        // Deny or disconnected reply → reject.
                                                        Ok(PermitResponse::Deny) | Err(_) => false,
                                                    }
                                                }
                                                _ = cancelRx.changed() => {
                                                    tracing::info!("cancelled during permission wait");
                                                    for remaining in &calls[callIdx..] {
                                                        self.pushToolResult(&remaining.id, crate::message::Content::text("Cancelled by user."));
                                                    }
                                                    let _ = logTx.send(LogEvent::TurnCancelled).await;
                                                    break 'turns Ok(());
                                                }
                                            }
                                        }
                                        PermitMode::Deny => {
                                            let _ = logTx
                                                .send(LogEvent::ToolDenied {
                                                    name: call.function.name.clone(),
                                                })
                                                .await;
                                            false
                                        }
                                        PermitMode::Abort => {
                                            let _ = logTx
                                                .send(LogEvent::TurnAborted {
                                                    name: call.function.name.clone(),
                                                })
                                                .await;
                                            aborted = true;
                                            false
                                        }
                                    }
                                }
                            };

                            // Revert pre-emptive LSP notification if denied/aborted.
                            if !approved && let Some((ref path, ref original)) = lspPreemptive {
                                self.lspManager.touchFile(path, original).await;
                            }

                            if aborted {
                                self.pushToolResult(
                                    &call.id,
                                    "Turn aborted: tool call not permitted.".into(),
                                );
                                for remaining in &calls[callIdx + 1..] {
                                    self.pushToolResult(
                                        &remaining.id,
                                        "Turn aborted: tool call not permitted.".into(),
                                    );
                                }
                                break;
                            }

                            // Guard: editFile/writeFile require a prior readFile of the same path.
                            if approved
                                && let Some(ref rejection) = self.checkReadBeforeWrite(&action)
                            {
                                tracing::info!(
                                    tool = %call.function.name,
                                    "rejected: file not read first"
                                );
                                // Revert pre-emptive LSP notification.
                                if let Some((ref path, ref original)) = lspPreemptive {
                                    self.lspManager.touchFile(path, original).await;
                                }
                                let _ = logTx
                                    .send(LogEvent::ToolResult {
                                        name: call.function.name.clone(),
                                        output: rejection.clone(),
                                    })
                                    .await;
                                self.pushToolResult(&call.id, rejection.clone().into());
                                continue;
                            }

                            let output = if approved {
                                tracing::info!(tool = %call.function.name, "executing tool");

                                if tool::needsTask(&action) {
                                    // Subagent events handle all TUI rendering — no ToolStarted needed.
                                    let (taskPrompt, taskAgent, runInBackground) = match &action {
                                        tool::ToolAction::Task {
                                            prompt,
                                            agent,
                                            runInBackground,
                                        } => (
                                            prompt.clone(),
                                            agent.as_deref().unwrap_or("general").to_string(),
                                            *runInBackground,
                                        ),
                                        _ => unreachable!(),
                                    };
                                    let result = if runInBackground {
                                        self.executeTaskBackground(
                                            &taskPrompt,
                                            &taskAgent,
                                            logTx,
                                            sessionRequestTx,
                                            cancelRx,
                                        )
                                        .await
                                    } else {
                                        self.executeTask(
                                            &taskPrompt,
                                            &taskAgent,
                                            logTx,
                                            sessionRequestTx,
                                            cancelRx,
                                        )
                                        .await
                                    };
                                    // NOTE: No ToolResult event — SubagentComplete already notified the TUI.
                                    crate::message::Content::text(result)
                                } else {
                                    // Emit ToolStarted for non-task tools.
                                    let _ = logTx
                                        .send(LogEvent::ToolStarted {
                                            name: call.function.name.clone(),
                                            summary: tool::summarize(&action),
                                        })
                                        .await;

                                    if tool::needsMcp(&action) {
                                        crate::message::Content::text(
                                            self.executeMcpTool(&action).await,
                                        )
                                    } else if tool::needsTranscript(&action) {
                                        crate::message::Content::text(
                                            self.executeTranscriptTool(&action),
                                        )
                                    } else if tool::needsWeb(&action) {
                                        crate::message::Content::text(
                                            self.executeWebTool(&action).await,
                                        )
                                    } else if tool::needsLsp(&action) {
                                        crate::message::Content::text(
                                            self.executeLspTool(&action).await,
                                        )
                                    } else if tool::needsRegistry(&action) {
                                        crate::message::Content::text(
                                            self.executeTerminalTool(&action, logTx).await,
                                        )
                                    } else if tool::needsJobPlane(&action) {
                                        crate::message::Content::text(
                                            self.executeJobTool(&action, logTx).await,
                                        )
                                    } else if tool::needsMonitor(&action) {
                                        crate::message::Content::text(
                                            self.executeMonitorTool(&action, logTx).await,
                                        )
                                    } else if tool::needsWakes(&action) {
                                        crate::message::Content::text(
                                            self.executeWakeTool(&action, logTx).await,
                                        )
                                    }
                                    // Resolve target shell for shell-using tools. The
                                    // shell handle is also used for cancel-side
                                    // interrupt() if a Shell call is racing cancel.
                                    else {
                                        let resolvedShell = self.resolveShell(&action).await;
                                        // Display name for output labeling — what the
                                        // model called the terminal, or the agent's
                                        // current target if it omitted the field.
                                        let targetName = match action.terminal() {
                                            Some(s) => s.to_string(),
                                            None => self
                                                .shells
                                                .lock()
                                                .await
                                                .activeForAgent()
                                                .to_string(),
                                        };
                                        match resolvedShell {
                                            Err(e) => crate::message::Content::text(e.to_string()),
                                            Ok(shell) => {
                                                // Race tool execution against cancel + user-triggered
                                                // bg (Ctrl+B) for shell commands. Auto-bg on a long
                                                // run is not a separate timer — the shell's own
                                                // timeout (default + model override) does the work;
                                                // when it fires the result carries a sentinel suffix
                                                // and we wrap with structured guidance below.
                                                if matches!(action, tool::ToolAction::Shell { .. })
                                                {
                                                    let (shellCommand, modelTimeout) = match &action
                                                    {
                                                        tool::ToolAction::Shell {
                                                            command,
                                                            timeout,
                                                            ..
                                                        } => (command.clone(), *timeout),
                                                        _ => (String::new(), None),
                                                    };
                                                    let effectiveTimeout = modelTimeout.unwrap_or(
                                                        crate::shell::SHELL_DEFAULT_TIMEOUT_SECS,
                                                    );
                                                    let startedAt = unixNow();
                                                    let shellForExec = shell.clone();
                                                    let commandForExec = shellCommand.clone();
                                                    let mut execTask = tokio::spawn(async move {
                                                        shellForExec
                                                            .executeDetailedNoTimeout(
                                                                &commandForExec,
                                                            )
                                                            .await
                                                    });
                                                    let timeoutSleep = tokio::time::sleep(
                                                        std::time::Duration::from_secs(
                                                            effectiveTimeout,
                                                        ),
                                                    );
                                                    tokio::pin!(timeoutSleep);
                                                    loop {
                                                        tokio::select! {
                                                            result = &mut execTask => {
                                                                let exec = match result {
                                                                    Ok(exec) => exec,
                                                                    Err(e) => crate::shell::CommandExecution {
                                                                        command: shellCommand.clone(),
                                                                        output: format!("Terminal command task failed to join: {e}"),
                                                                        exitCode: None,
                                                                        lineCount: 1,
                                                                        replayBytes: Vec::new(),
                                                                        timedOut: false,
                                                                    },
                                                                };
                                                                let raw = exec.responseText();
                                                                let index = shell.historyLen().saturating_sub(1);
                                                                let result = crate::message::Content::text(
                                                                    crate::tool::truncateOutput(&raw, index, &targetName),
                                                                );
                                                                tracing::debug!(
                                                                    tool = %call.function.name,
                                                                    outputLen = result.charCount(),
                                                                    "tool execution complete"
                                                                );
                                                                break result;
                                                            }
                                                            _ = &mut timeoutSleep => {
                                                                let _ = logTx
                                                                    .send(LogEvent::AutoBgWarning {
                                                                        command: shellCommand.clone(),
                                                                        elapsedSecs: effectiveTimeout,
                                                                        userTriggered: false,
                                                                    })
                                                                    .await;
                                                                let trigger = format!("the shell exceeded its {effectiveTimeout}s timeout");
                                                                let message = self
                                                                    .detachTerminalRunJoin(
                                                                        shellCommand.clone(),
                                                                        match &action {
                                                                            tool::ToolAction::Shell { explanation, .. } => explanation.clone(),
                                                                            _ => String::new(),
                                                                        },
                                                                        match &action {
                                                                            tool::ToolAction::Shell { impact, .. } => impact.clone(),
                                                                            _ => crate::tool::ShellImpact::Read,
                                                                        },
                                                                        targetName.clone(),
                                                                        startedAt,
                                                                        execTask,
                                                                        logTx,
                                                                        trigger,
                                                                    )
                                                                    .await;
                                                                break crate::message::Content::text(message);
                                                            }
                                                            _ = userBgRx.recv() => {
                                                                let _ = logTx
                                                                    .send(LogEvent::AutoBgWarning {
                                                                        command: shellCommand.clone(),
                                                                        elapsedSecs: 0,
                                                                        userTriggered: true,
                                                                    })
                                                                    .await;
                                                                let message = self
                                                                    .detachTerminalRunJoin(
                                                                        shellCommand.clone(),
                                                                        match &action {
                                                                            tool::ToolAction::Shell { explanation, .. } => explanation.clone(),
                                                                            _ => String::new(),
                                                                        },
                                                                        match &action {
                                                                            tool::ToolAction::Shell { impact, .. } => impact.clone(),
                                                                            _ => crate::tool::ShellImpact::Read,
                                                                        },
                                                                        targetName.clone(),
                                                                        startedAt,
                                                                        execTask,
                                                                        logTx,
                                                                        "you pressed Ctrl+B".to_string(),
                                                                    )
                                                                    .await;
                                                                break crate::message::Content::text(message);
                                                            }
                                                            _ = cancelRx.changed() => {
                                                                if !*cancelRx.borrow() {
                                                                    // Spurious wakeup — retry select.
                                                                    continue;
                                                                }
                                                                tracing::info!(tool = %call.function.name, "cancelled during shell execution");
                                                                shell.interrupt();
                                                                let _ = (&mut execTask).await;
                                                                self.pushToolResult(&call.id, crate::message::Content::text("Cancelled by user."));
                                                                for remaining in &calls[callIdx + 1..] {
                                                                    self.pushToolResult(&remaining.id, crate::message::Content::text("Cancelled by user."));
                                                                }
                                                                let _ = logTx.send(LogEvent::TurnCancelled).await;
                                                                break 'turns Ok(());
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    // File operations are fast — no cancel race needed.
                                                    let result =
                                                        tool::execute(&action, &shell, &targetName)
                                                            .await;
                                                    tracing::debug!(
                                                        tool = %call.function.name,
                                                        outputLen = result.charCount(),
                                                        "tool execution complete"
                                                    );
                                                    result
                                                }
                                            }
                                        }
                                    }
                                } // Close the else block for non-task tools.
                            } else {
                                let _ = logTx
                                    .send(LogEvent::ToolDenied {
                                        name: call.function.name.clone(),
                                    })
                                    .await;
                                crate::message::Content::text("User denied this action.")
                            };

                            // Track file reads for the edit gate (hash for staleness detection).
                            if call.function.name == "readFile"
                                && let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                    &call.function.arguments,
                                )
                                && let Some(path) = args["path"].as_str()
                            {
                                let norm = normalizePath(path);
                                if let Ok(bytes) = std::fs::read(&norm) {
                                    let digest = sha1_smol::Sha1::from(&bytes).digest().bytes();
                                    self.filesRead.insert(norm.clone(), digest);
                                }
                                // Sync file with LSP server (lazy spawn if needed).
                                if let Ok(content) = std::fs::read_to_string(&norm)
                                    && let Some(hint) =
                                        self.lspManager.touchFile(&norm, &content).await
                                {
                                    let _ = logTx
                                        .send(LogEvent::LspHint {
                                            serverId: hint.serverId,
                                            installHint: hint.installHint,
                                        })
                                        .await;
                                }
                            }

                            // Update hash after successful file mutations.
                            if matches!(
                                call.function.name.as_str(),
                                "editFile" | "writeFile" | "multiEdit"
                            ) && let Ok(args) =
                                serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                                && let Some(path) = args["path"].as_str()
                            {
                                let norm = normalizePath(path);
                                if let Ok(bytes) = std::fs::read(&norm) {
                                    let digest = sha1_smol::Sha1::from(&bytes).digest().bytes();
                                    self.filesRead.insert(norm, digest);
                                }
                            }

                            // Collect LSP diagnostics after file mutations.
                            // Diff against baseline to only show errors introduced by the edit.
                            let mut output = output;
                            if matches!(
                                call.function.name.as_str(),
                                "editFile" | "writeFile" | "multiEdit"
                            ) && let Ok(args) =
                                serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                                && let Some(path) = args["path"].as_str()
                            {
                                // Baseline: cached diagnostics from before the edit
                                // (populated by the pre-emptive touchFile or prior reads).
                                let baseline = self.lspManager.getRawCachedDiagnostics(path);

                                let content = std::fs::read_to_string(path).unwrap_or_default();
                                let (postEdit, hint) = self
                                    .lspManager
                                    .getRawDiagnostics(
                                        path,
                                        &content,
                                        std::time::Duration::from_secs(10),
                                    )
                                    .await;

                                // Multiset diff: only new errors survive.
                                let newErrors =
                                    lsp::diagnostics::diffDiagnostics(&baseline, &postEdit);
                                if !newErrors.is_empty() {
                                    let formatted = lsp::diagnostics::formatDiagnostics(
                                        path,
                                        &newErrors,
                                        async_lsp::lsp_types::DiagnosticSeverity::ERROR,
                                    );
                                    if !formatted.is_empty() {
                                        // Append diagnostics to text content.
                                        let mut text = output.textContent().to_string();
                                        text.push_str("\n\nNew LSP errors after edit:\n");
                                        text.push_str(&formatted);
                                        output = crate::message::Content::text(text);
                                    }
                                }
                                if let Some(hint) = hint {
                                    let _ = logTx
                                        .send(LogEvent::LspHint {
                                            serverId: hint.serverId,
                                            installHint: hint.installHint,
                                        })
                                        .await;
                                }
                            }

                            // Emit ToolResult AFTER diagnostics injection so the TUI
                            // shows the same content the model sees. Skip the emit
                            // when the tool was denied — `ToolDenied` already told
                            // the TUI, and re-rendering "User denied this action."
                            // as a tool-result block duplicates that signal. The
                            // transcript/history still gets the content below so
                            // the model sees the denial.
                            if approved {
                                let _ = logTx
                                    .send(LogEvent::ToolResult {
                                        name: call.function.name.clone(),
                                        output: output.textContent().to_string(),
                                    })
                                    .await;
                            }

                            self.pushToolResult(&call.id, output);
                        }

                        if aborted {
                            let _ = logTx.send(LogEvent::TurnComplete).await;
                            break 'turns Ok(());
                        }
                        // Inject queued user messages before the next API call.
                        self.drainSteer(steerRx, logTx).await;
                    }
                }
            }
        };

        // Collect topic classification before returning — the eval ran
        // concurrently with the turn loop and should be done by now.
        self.collectTopicEval(logTx).await;

        // Always persist meta after the turn completes so headTurn reflects
        // the latest assistant/tool turn, not the stale user turn written
        // at the start of sendInner.
        self.updateMeta();

        // Fire compaction on every exit path, not just Done. Cancelled,
        // Error, BudgetHit, MaxRetries, and tool-permission-denied otherwise
        // leave the session parked at high context until the next message.
        // Idempotent when Done already ran it during the loop.
        self.checkCompactionTrigger(logTx).await;

        result
    }

    /// Drain pending steer messages and inject as a User message.
    ///
    /// Collects all buffered `UserInput` payloads from the steer channel,
    /// combines their text and attachments, pushes a single User message to
    /// history, records to transcript, and emits `SteerInjected`.
    ///
    /// Returns true if anything was injected (caller should continue the turn).
    async fn drainSteer(
        &mut self,
        steerRx: &mut mpsc::Receiver<UserInput>,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> bool {
        let mut allTexts: Vec<String> = Vec::new();
        let mut allAttachments: Vec<Attachment> = Vec::new();

        while let Ok(input) = steerRx.try_recv() {
            allTexts.push(input.text);
            allAttachments.extend(input.attachments);
        }

        if allTexts.is_empty() {
            return false;
        }

        let combined = allTexts.join("\n\n");
        tracing::info!(count = allTexts.len(), "injecting queued user messages");

        let (content, turnAttachments) = buildUserContent(&combined, &allAttachments);
        self.history.push(Message::User { content });

        match self
            .transcript
            .recordUser(&combined, self.headTurnId.as_deref(), turnAttachments)
        {
            Ok(turnId) => self.headTurnId = Some(turnId),
            Err(e) => tracing::warn!("steer transcript write failed: {e}"),
        }

        let _ = logTx
            .send(LogEvent::SteerInjected { texts: allTexts })
            .await;
        true
    }

    /// Stream one API call and return what happened.
    async fn streamOneTurn(
        &mut self,
        tx: &mpsc::Sender<LogEvent>,
        cancelRx: &mut watch::Receiver<bool>,
    ) -> Result<TurnResult> {
        // When prompt-injected thinking is active, don't send the reasoning
        // config (we're faking it via prompt) and set up the content extractor.
        let reasoning = if self.config.heavy.promptThinking {
            None
        } else {
            self.reasoning.as_ref()
        };
        let mut thinkingExtractor = if self.config.heavy.promptThinking {
            Some(crate::api::ThinkingExtractor::new())
        } else {
            None
        };

        // Capture snapshot of the exact request before it goes over the wire.
        // NOTE: If a future change introduces history-mutating behavior inside
        // Client::stream, this capture point must move into a body-building
        // stage shared with the send.
        let snapshotHash = crate::snapshot::captureSnapshot(
            &mut self.snapshots,
            crate::snapshot::BuildCtx {
                history: &self.history,
                tools: &self.tools,
                reasoning,
                cfg: &self.config.heavy,
            },
        )
        .map_err(|e| {
            tracing::warn!(error = %e, "snapshot capture failed; turn will be recorded without snapshotHash");
            e
        })
        .ok();

        // Build the API-call copy. Riders and promptThinking-mode scratchpad
        // baking apply here, not to `self.history`.
        let requestMessages = buildRequestMessages(
            &self.history,
            &self.riders,
            self.config.heavy.promptThinking,
        );

        let mut rx = self
            .client
            .stream(&requestMessages, &self.tools, reasoning)
            .await?;

        let mut contentBuf = String::new();
        let mut reasoningBuf = String::new();
        let mut toolAccum = ToolCallAccumulator::new();
        let mut toolCallPreviews: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();
        let mut lastUsage: Option<TokenUsage> = None;
        let mut lastFinishReason: Option<String> = None;

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(StreamEvent::ContentDelta(text)) => {
                            // Route through thinking extractor if active.
                            if let Some(ref mut extractor) = thinkingExtractor {
                                for extracted in extractor.feed(&text) {
                                    match extracted {
                                        StreamEvent::ContentDelta(t) => {
                                            contentBuf.push_str(&t);
                                            let _ = tx.send(LogEvent::ContentDelta(t)).await;
                                        }
                                        StreamEvent::ReasoningDelta(t) => {
                                            reasoningBuf.push_str(&t);
                                            let _ = tx.send(LogEvent::ReasoningDelta(t)).await;
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                contentBuf.push_str(&text);
                                let _ = tx.send(LogEvent::ContentDelta(text)).await;
                            }
                        }
                        Some(StreamEvent::ReasoningDelta(text)) => {
                            reasoningBuf.push_str(&text);
                            let _ = tx.send(LogEvent::ReasoningDelta(text)).await;
                        }
                        Some(StreamEvent::ToolCallDelta {
                            index,
                            id,
                            name,
                            arguments,
                        }) => {
                            let (newName, totalBytes) =
                                toolAccum.accumulate(index, id, name, arguments);
                            if let Some(n) = newName {
                                let _ = tx
                                    .send(LogEvent::ToolCallPending {
                                        index,
                                        name: n,
                                    })
                                    .await;
                            }
                            if let Some(bytes) = totalBytes {
                                let _ = tx
                                    .send(LogEvent::ToolCallProgress { index, bytes })
                                    .await;
                            }
                            if let Some((toolName, argsStr)) =
                                toolAccum.pendingCall(index)
                                && let Some(preview) =
                                    crate::tool_preview::previewForTool(
                                        toolName, argsStr,
                                    )
                                {
                                    let slot = toolCallPreviews.entry(index).or_default();
                                    if *slot != preview {
                                        *slot = preview.clone();
                                        let _ = tx
                                            .send(LogEvent::ToolCallPreview {
                                                index,
                                                preview,
                                            })
                                            .await;
                                    }
                                }
                        }
                        Some(StreamEvent::Done { usage, finishReason }) => {
                            if let Some(u) = usage {
                                lastUsage = Some(u);
                            }
                            if let Some(reason) = finishReason {
                                lastFinishReason = Some(reason);
                            }
                            // Don't break — usage may arrive in a subsequent chunk.
                        }
                        None => {
                            break;
                        }
                        Some(StreamEvent::Error(msg)) => {
                            // If nothing was streamed yet and the error looks
                            // transient, signal the caller to retry silently.
                            let hadContent = !contentBuf.is_empty()
                                || !reasoningBuf.is_empty()
                                || toolAccum.hasContent();
                            if !hadContent && isTransientError(&msg) {
                                return Ok(TurnResult::TransientError(msg));
                            }

                            // Fatal path — commit any partial content to
                            // history/transcript so the user can resume
                            // from where we were cut off. Mirrors the
                            // cancel path but with Errored status.
                            if !contentBuf.is_empty() || !reasoningBuf.is_empty() {
                                let reasonRef = if reasoningBuf.is_empty() {
                                    None
                                } else {
                                    Some(reasoningBuf.as_str())
                                };
                                if !contentBuf.is_empty() {
                                    let meta = transcript::AssistantMeta {
                                        reasoning: reasonRef,
                                        cost: lastUsage.as_ref().and_then(|u| u.cost),
                                        promptTokens: lastUsage.as_ref().map(|u| u.promptTokens),
                                        completionTokens: lastUsage.as_ref().map(|u| u.completionTokens),
                                        model: Some(self.config.heavy.model.as_str()),
                                        finishReason: lastFinishReason.as_deref(),
                                        snapshotHash: snapshotHash.as_deref(),
                                        status: transcript::TurnStatus::Errored,
                                    };
                                    match self.transcript.recordAssistant(&contentBuf, meta) {
                                        Ok(turnId) => self.headTurnId = Some(turnId),
                                        Err(e) => tracing::warn!("transcript write failed: {e}"),
                                    }
                                }
                                let content = if contentBuf.is_empty() {
                                    None
                                } else {
                                    Some(std::mem::take(&mut contentBuf))
                                };
                                let reasoning = if reasoningBuf.is_empty() {
                                    None
                                } else {
                                    Some(std::mem::take(&mut reasoningBuf))
                                };
                                self.history.push(buildAssistantMessage(
                                    content, None, reasoning,
                                ));
                            }

                            let _ = tx.send(LogEvent::Error(msg.clone())).await;
                            bail!("Stream error: {msg}");
                        }
                    }
                }
                _ = cancelRx.changed() => {
                    if *cancelRx.borrow() {
                        tracing::info!("stream cancelled, committing partial content");
                        // Drop rx — kills the SSE background job.
                        drop(rx);
                        // Commit partial content to history (skip if nothing was streamed).
                        if !contentBuf.is_empty() || !reasoningBuf.is_empty() {
                            if !contentBuf.is_empty() {
                                let reasonRef = if reasoningBuf.is_empty() { None } else { Some(reasoningBuf.as_str()) };
                                let meta = transcript::AssistantMeta {
                                    reasoning: reasonRef,
                                    cost: lastUsage.as_ref().and_then(|u| u.cost),
                                    promptTokens: lastUsage.as_ref().map(|u| u.promptTokens),
                                    completionTokens: lastUsage.as_ref().map(|u| u.completionTokens),
                                    model: Some(self.config.heavy.model.as_str()),
                                    finishReason: lastFinishReason.as_deref(),
                                    snapshotHash: snapshotHash.as_deref(),
                                    status: transcript::TurnStatus::Cancelled,
                                };
                                match self.transcript.recordAssistant(&contentBuf, meta) {
                                    Ok(turnId) => self.headTurnId = Some(turnId),
                                    Err(e) => tracing::warn!("transcript write failed: {e}"),
                                }
                            }
                            let content = if contentBuf.is_empty() { None } else { Some(contentBuf) };
                            let reasoning = if reasoningBuf.is_empty() { None } else { Some(reasoningBuf) };
                            self.history.push(buildAssistantMessage(
                                content, None, reasoning
                            ));
                        }
                        return Ok(TurnResult::Cancelled);
                    }
                }
            }
        }

        // Flush any remaining buffered content from the thinking extractor.
        if let Some(ref mut extractor) = thinkingExtractor {
            for extracted in extractor.finish() {
                match extracted {
                    StreamEvent::ContentDelta(t) => {
                        contentBuf.push_str(&t);
                        let _ = tx.send(LogEvent::ContentDelta(t)).await;
                    }
                    StreamEvent::ReasoningDelta(t) => {
                        reasoningBuf.push_str(&t);
                        let _ = tx.send(LogEvent::ReasoningDelta(t)).await;
                    }
                    _ => {}
                }
            }
        }

        // Strip variation selectors from emoji-only codepoints.
        contentBuf = crate::text::sanitizeVariationSelectors(&contentBuf);
        reasoningBuf = crate::text::sanitizeVariationSelectors(&reasoningBuf);

        // Retroactive scratchpad-close recovery. When promptThinking is on
        // and the model botches the close (`</scratch>`, `</scratchpa>`,
        // `</scratchpad` w/ no `>`), the streaming extractor flushes the
        // entire buffer as reasoning and the visible answer is lost. Scan
        // the reasoning tail for a malformed close and split it back out.
        if self.config.heavy.promptThinking
            && contentBuf.is_empty()
            && !reasoningBuf.is_empty()
            && let Some(recovery) = crate::text::recoverScratchpadClose(&reasoningBuf)
        {
            let recoveredChars = recovery.content.chars().count();
            let snippet: String = recovery.content.chars().take(80).collect::<String>();
            let snippet = if recoveredChars > 80 {
                format!("{snippet}…")
            } else {
                snippet
            };
            tracing::info!(
                matchedTag = %recovery.matchedTag,
                recoveredChars,
                "recovered malformed scratchpad close"
            );
            contentBuf = recovery.content;
            reasoningBuf = recovery.reasoning;
            let _ = tx
                .send(LogEvent::ScratchpadRecovered {
                    matchedTag: recovery.matchedTag,
                    snippet,
                    recoveredChars,
                })
                .await;
        }

        let calls = toolAccum.finish();

        // Emit token usage from the API response.
        if let Some(ref u) = lastUsage {
            // Record cost before emitting the event.
            if let Some(cost) = u.cost {
                self.costTracker.record(cost, &self.config.heavy.model);
            }
            let contextTokens = u.promptTokens + u.completionTokens;
            let _ = tx
                .send(LogEvent::TokenUpdate {
                    promptTokens: u.promptTokens,
                    completionTokens: u.completionTokens,
                    contextTokens,
                    turnCost: u.cost,
                    sessionCost: self.costTracker.sessionCost(),
                    cacheReadTokens: u.cacheReadTokens,
                    cacheCreationTokens: u.cacheCreationTokens,
                })
                .await;
            // Check budget warning threshold.
            if let Some(limit) = self.config.budget.sessionLimit
                && self.costTracker.checkWarning(limit)
            {
                let _ = tx
                    .send(LogEvent::BudgetWarning {
                        sessionCost: self.costTracker.sessionCost(),
                        limit,
                    })
                    .await;
            }

            // Cache watchdog: if we requested caching and this isn't the
            // first usage-reporting turn, we expect to see activity. Zero on
            // both counters means the provider silently ignored our markers
            // (regression like the March 2026 1h→5m TTL drop in Claude Code).
            if self.config.heavy.cachingActive()
                && self.turnsWithUsage >= 1
                && u.cacheReadTokens == 0
                && u.cacheCreationTokens == 0
            {
                tracing::warn!(
                    model = %self.config.heavy.model,
                    provider = %self.config.heavy.provider,
                    providerOrder = ?self.config.heavy.providerOrder,
                    "requested cache_control but provider returned zero cache activity \
                     — caching may be silently disabled"
                );
            }
            self.turnsWithUsage = self.turnsWithUsage.saturating_add(1);
        } else {
            tracing::warn!(
                "no usage data received from API — provider may not support stream_options.include_usage"
            );
        }

        tracing::debug!(
            contentLen = contentBuf.len(),
            reasoningLen = reasoningBuf.len(),
            toolCalls = calls.len(),
            "turn stream complete"
        );

        let reportedTokens = lastUsage.as_ref().map(|u| u.promptTokens);

        if !calls.is_empty() {
            // Record any text content or reasoning that accompanied the tool calls.
            let reasonRef = if reasoningBuf.is_empty() {
                None
            } else {
                Some(reasoningBuf.as_str())
            };
            if !contentBuf.is_empty() || reasonRef.is_some() {
                let meta = transcript::AssistantMeta {
                    reasoning: reasonRef,
                    cost: lastUsage.as_ref().and_then(|u| u.cost),
                    promptTokens: lastUsage.as_ref().map(|u| u.promptTokens),
                    completionTokens: lastUsage.as_ref().map(|u| u.completionTokens),
                    model: Some(self.config.heavy.model.as_str()),
                    finishReason: lastFinishReason.as_deref(),
                    snapshotHash: snapshotHash.as_deref(),
                    status: transcript::TurnStatus::Completed,
                };
                match self.transcript.recordAssistant(&contentBuf, meta) {
                    Ok(turnId) => self.headTurnId = Some(turnId),
                    Err(e) => tracing::warn!("transcript write failed: {e}"),
                }
            }
            let content = if contentBuf.is_empty() {
                None
            } else {
                Some(contentBuf)
            };
            let reasoning = if reasoningBuf.is_empty() {
                None
            } else {
                Some(reasoningBuf)
            };
            Ok(TurnResult::ToolCalls {
                calls,
                content,
                reasoning,
                promptTokens: reportedTokens,
            })
        } else {
            let content = if contentBuf.is_empty() {
                None
            } else {
                Some(contentBuf)
            };
            let reasoning = if reasoningBuf.is_empty() {
                None
            } else {
                Some(reasoningBuf)
            };

            // Record assistant content + reasoning to transcript.
            if content.is_some() || reasoning.is_some() {
                let textRef = content.as_deref().unwrap_or("");
                let reasonRef = reasoning.as_deref();
                let meta = transcript::AssistantMeta {
                    reasoning: reasonRef,
                    cost: lastUsage.as_ref().and_then(|u| u.cost),
                    promptTokens: lastUsage.as_ref().map(|u| u.promptTokens),
                    completionTokens: lastUsage.as_ref().map(|u| u.completionTokens),
                    model: Some(self.config.heavy.model.as_str()),
                    finishReason: lastFinishReason.as_deref(),
                    snapshotHash: snapshotHash.as_deref(),
                    status: transcript::TurnStatus::Completed,
                };
                match self.transcript.recordAssistant(textRef, meta) {
                    Ok(turnId) => self.headTurnId = Some(turnId),
                    Err(e) => tracing::warn!("transcript write failed: {e}"),
                }
            }

            self.history
                .push(buildAssistantMessage(content, None, reasoning));

            Ok(TurnResult::Done {
                promptTokens: reportedTokens,
            })
        }
    }

    /// Check compaction trigger and run the appropriate stage.
    ///
    /// Loops on exhaustion: if a stage exhausts without reducing context,
    /// re-evaluates and tries the next cheapest stage. Stops when a stage
    /// does work, nothing is returned, or all stages are exhausted.
    async fn checkCompactionTrigger(&mut self, logTx: &mpsc::Sender<LogEvent>) {
        loop {
            let tokens = self.compactionTracker.lastTokens();
            let stage = match self.compactionTracker.evaluate(tokens) {
                Some(s) => s,
                None => return,
            };

            let ratio = self.compactionTracker.usageRatio();
            tracing::info!(
                stage = ?stage,
                tokens,
                ratio = format!("{:.1}%", ratio * 100.0),
                "compaction trigger fired"
            );

            let stageStr = format!("{stage}");
            let _ = logTx
                .send(LogEvent::CompactionStarted {
                    stage: stageStr.clone(),
                })
                .await;

            // didWork tracks whether this stage reduced context.
            // If false, we loop to try the next stage.
            let didWork = match stage {
                compaction_trigger::StagePick::S1 => self.runS1(&stageStr, logTx).await,
                compaction_trigger::StagePick::S2 => self.runS2(&stageStr, logTx).await,
                compaction_trigger::StagePick::S3 => self.runS3(&stageStr, logTx).await,
                compaction_trigger::StagePick::S4 => self.runS4Trigger(&stageStr, logTx).await,
            };

            if didWork || self.compactionTracker.allExhausted() {
                return;
            }
            // Stage exhausted without reducing — loop to try next.
        }
    }

    /// Run S1 mechanical pruning. Returns true if context was reduced.
    async fn runS1(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        // Build toolCallId → blockId map so truncation markers
        // can point the model to historyFetch for full content.
        let blockHints = match self.transcript.loadAll() {
            Ok(turns) => {
                let mut map = std::collections::HashMap::new();
                for t in &turns {
                    if let Some(tcid) = &t.toolCallId {
                        map.insert(tcid.clone(), t.blockId.clone());
                    }
                }
                map
            }
            Err(e) => {
                tracing::warn!("failed to load transcript for block hints: {e}");
                std::collections::HashMap::new()
            }
        };

        // Build set of tool_call_ids already in the compaction log
        // so S1 doesn't produce duplicate entries on re-runs.
        let alreadyProcessed = match self.compactionLog.loadAll() {
            Ok(ops) => {
                let mut set = std::collections::HashSet::new();
                for op in &ops {
                    match op {
                        crate::compaction::CompactionOp::FileDedup { targetIds, .. } => {
                            set.extend(targetIds.iter().cloned());
                        }
                        crate::compaction::CompactionOp::MiddleOut { targetIds, .. } => {
                            set.extend(targetIds.iter().cloned());
                        }
                        _ => {}
                    }
                }
                set
            }
            Err(_) => std::collections::HashSet::new(),
        };

        let s1Result = crate::s1::run(
            &mut self.history,
            crate::s1::DEFAULT_MIDDLE_OUT_THRESHOLD,
            &blockHints,
            &alreadyProcessed,
        );
        if s1Result.didWork {
            let afterTurn = self.headTurnId.clone().unwrap_or_default();
            if !s1Result.dedupedCallIds.is_empty()
                && let Err(e) = self
                    .compactionLog
                    .recordFileDedup(s1Result.dedupedCallIds.clone(), &afterTurn)
            {
                tracing::warn!("compaction log write failed: {e}");
            }
            if !s1Result.middleOutCallIds.is_empty()
                && let Err(e) = self.compactionLog.recordMiddleOut(
                    s1Result.middleOutCallIds.clone(),
                    &afterTurn,
                    s1Result.middleOutThreshold,
                )
            {
                tracing::warn!("compaction log write failed: {e}");
            }
            for path in &s1Result.invalidatedFiles {
                self.filesRead.remove(path);
            }
            self.compactionTracker.clearExhaustion();
            let reduction = format!(
                "deduped {} reads, truncated {} outputs",
                s1Result.dedupedCallIds.len(),
                s1Result.middleOutCallIds.len()
            );
            let _ = logTx
                .send(LogEvent::CompactionComplete {
                    stage: stageStr.to_string(),
                    reduction,
                    markerBlock: None,
                })
                .await;
            true
        } else {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S1);
            tracing::debug!("S1 exhausted \u{2014} nothing to prune");
            false
        }
    }

    /// Run S2 block compaction. Returns true if context was reduced.
    async fn runS2(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        let headTurn = self.headTurnId.clone().unwrap_or_default();
        let s2Result = match crate::s2::run(
            &self.transcript,
            &self.compactionLog,
            &headTurn,
            &self.client,
            &self.config.utility.model,
            self.config.heavy.contextWindow,
            self.config.compactRatio,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S2 compaction failed: {e}");
                self.compactionTracker
                    .markExhausted(compaction_trigger::StagePick::S2);
                return false;
            }
        };
        // Record utility model cost from S2 compaction.
        if let Some(cost) = s2Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s2Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S2);
            tracing::debug!("S2 exhausted \u{2014} no blocks to compact");
            return false;
        }
        let afterTurn = self.headTurnId.clone().unwrap_or_default();
        let blockCount = s2Result.compacted.len();
        for block in &s2Result.compacted {
            if let Err(e) = self.compactionLog.recordBlockCompact(
                &block.blockId,
                &block.summary,
                block.sourceIds.clone(),
                &afterTurn,
            ) {
                tracing::warn!("compaction log write failed for {}: {e}", block.blockId);
            }
            for path in &block.invalidatedFiles {
                self.filesRead.remove(path);
            }
        }
        // Reconstruct live history from transcript + updated compaction log.
        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S2: {e}");
                return false;
            }
        }
        self.compactionTracker.clearExhaustion();
        let reduction = format!("compressed {blockCount} blocks");
        // S2 zone always starts at the oldest block (index 0).
        let _ = logTx
            .send(LogEvent::CompactionComplete {
                stage: stageStr.to_string(),
                reduction,
                markerBlock: Some(0),
            })
            .await;
        true
    }

    /// Run S3 topic compaction. Returns true if context was reduced.
    async fn runS3(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        let s3Result = match crate::s3::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            self.topicTracker.topics(),
            &self.client,
            &self.config.utility.model,
            self.config.heavy.contextWindow,
            self.config.compactRatio,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S3 compaction failed: {e}");
                self.compactionTracker
                    .markExhausted(compaction_trigger::StagePick::S3);
                return false;
            }
        };
        // Record utility model cost from S3 compaction.
        if let Some(cost) = s3Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s3Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S3);
            tracing::debug!("S3 exhausted \u{2014} no topics to compact");
            return false;
        }
        let afterTurn = self.headTurnId.clone().unwrap_or_default();
        let topicCount = s3Result.compacted.len();
        for topic in &s3Result.compacted {
            if let Err(e) = self.compactionLog.recordTopicCompact(
                &topic.topicLabel,
                &topic.summary,
                topic.sourceBlockIds.clone(),
                &afterTurn,
            ) {
                tracing::warn!("compaction log write failed for {}: {e}", topic.topicId);
            }
            for path in &topic.invalidatedFiles {
                self.filesRead.remove(path);
            }
        }
        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S3: {e}");
                return false;
            }
        }
        self.compactionTracker.clearExhaustion();
        let reduction = format!("compressed {topicCount} topics");
        let _ = logTx
            .send(LogEvent::CompactionComplete {
                stage: stageStr.to_string(),
                reduction,
                markerBlock: Some(0),
            })
            .await;
        true
    }

    /// Run S4 deep recompaction. Merges the latest active S4 briefing,
    /// fresh S3 topic summaries, and orphan S2 summaries from outside
    /// the protected recent band into a single handoff briefing.
    /// Returns true if context was reduced.
    async fn runS4Trigger(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        let s4Result = match crate::s4::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            &self.client,
            &self.config.utility.model,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("S4 compaction failed: {e}");
                self.compactionTracker
                    .markExhausted(compaction_trigger::StagePick::S4);
                return false;
            }
        };

        // Record utility model cost from S4 compaction.
        if let Some(cost) = s4Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s4Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S4);
            tracing::debug!("S4 exhausted \u{2014} no S3/S4 content to merge");
            return false;
        }

        let afterTurn = self.headTurnId.clone().unwrap_or_default();
        let blockCount = s4Result.sourceBlockIds.len();
        let summaryLen = s4Result.summary.len();
        if let Err(e) = self.compactionLog.recordFullCompact(
            &s4Result.summary,
            s4Result.sourceBlockIds,
            &afterTurn,
        ) {
            tracing::warn!("compaction log write failed: {e}");
        }

        // S4 replaces older read evidence with a handoff briefing, so clear
        // the conservative edit gate and let future reads rebuild it.
        self.filesRead.clear();

        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S4: {e}");
                return false;
            }
        }

        self.compactionTracker.clearExhaustion();
        let reduction =
            format!("merged {blockCount} source blocks into briefing ({summaryLen} chars)");
        let _ = logTx
            .send(LogEvent::CompactionComplete {
                stage: stageStr.to_string(),
                reduction,
                markerBlock: Some(0),
            })
            .await;
        true
    }

    /// Execute a transcript-dependent tool (historyFetch, historySearch).
    /// Resolve the target shell for a shell-using action. Returns an
    /// error message — including the agent's current target and the
    /// list of available terminals — if the named terminal doesn't
    /// exist. The verbose error is the model's recovery path; without
    /// it, a single typo can derail several turns.
    async fn resolveShell(
        &self,
        action: &tool::ToolAction,
    ) -> std::result::Result<Shell, ShellResolveError> {
        let guard = self.shells.lock().await;
        match action.terminal() {
            Some(name) => {
                guard
                    .shellFor(Some(name))
                    .ok_or_else(|| ShellResolveError::MissingNamed {
                        name: name.to_string(),
                        available: guard.names().to_vec(),
                        target: guard.activeForAgent().to_string(),
                    })
            }
            None => guard.shellFor(None).ok_or(ShellResolveError::NoAgentTarget),
        }
    }

    /// Handle the terminal-management tools (Spawn/Switch/Kill/List).
    /// These mutate `self.shells` and emit terminal lifecycle events on
    /// the log channel so the deck can update its tab strip.
    async fn executeTerminalTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        match action {
            tool::ToolAction::TerminalSpawn { name } => {
                // NOTE: agent-spawned terminals do NOT emit TerminalSpawned
                // log events — the surrounding ToolStarted/ToolResult pair
                // already represents the action in the panel. Emitting a
                // separate notice between Started and Result causes the
                // panel to fail to pop the ToolActive entry (it pops only
                // when the trailing entry is still ToolActive), leaving
                // an orphan throbber. User-initiated spawns DO emit the
                // event (they have no surrounding tool lifecycle).
                // Hold the lock across `spawn`'s await — fine with
                // tokio::sync::Mutex; the user-side terminal handler
                // simply waits for this to finish.
                let spawnResult = {
                    let mut guard = self.shells.lock().await;
                    guard.spawn(name.clone(), SpawnedBy::Agent).await
                };
                match spawnResult {
                    Ok(resolved) => format!(
                        "Spawned terminal '{resolved}'. Use shell with \
                         terminal:'{resolved}' to run commands there."
                    ),
                    Err(e) => format!("Failed to spawn terminal: {e}"),
                }
            }
            tool::ToolAction::TerminalSwitch { name } => {
                // Same reasoning as TerminalSpawn — no separate notice.
                let switchResult = {
                    let mut guard = self.shells.lock().await;
                    guard.setActiveForAgent(name)
                };
                match switchResult {
                    Ok(()) => format!("Agent target terminal is now '{name}'."),
                    Err(e) => format!("Failed to switch terminal: {e}"),
                }
            }
            tool::ToolAction::TerminalKill { name } => {
                let killResult = {
                    let mut guard = self.shells.lock().await;
                    guard.kill(name)
                };
                match killResult {
                    Ok(()) => {
                        self.stopMonitorsForTerminal(name, logTx).await;
                        // TerminalClosed is needed regardless — the deck must
                        // know to drop the tab.
                        let _ = logTx
                            .send(LogEvent::TerminalClosed { name: name.clone() })
                            .await;
                        format!("Terminal '{name}' killed.")
                    }
                    Err(e) => format!("Failed to kill terminal: {e}"),
                }
            }
            tool::ToolAction::TerminalList => {
                let infos = self.shells.lock().await.list();
                if infos.is_empty() {
                    return "No terminals.".into();
                }
                let mut out = String::from("Terminals:\n");
                for info in infos {
                    let active = if info.activeForAgent { " (active)" } else { "" };
                    let busy = if info.busy { " busy" } else { "" };
                    let by = match info.spawnedBy {
                        SpawnedBy::User => "user",
                        SpawnedBy::Agent => "agent",
                    };
                    out.push_str(&format!(
                        "  {} — by {}, age {}s{}{}\n",
                        info.name, by, info.ageSecs, active, busy
                    ));
                }
                out
            }
            tool::ToolAction::TerminalRunList => {
                let runs = match self.listTerminalRuns() {
                    Ok(runs) => runs,
                    Err(e) => return format!("Failed to list terminal runs: {e}"),
                };
                if runs.is_empty() {
                    return "No terminal runs archived yet.".into();
                }
                let mut out = String::from("Terminal runs:\n");
                for run in runs.into_iter().take(50) {
                    let exit = run
                        .exitCode
                        .map(|c| format!(" exit {c}"))
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "  {} [{}{}] {} · terminal={} · impact={} · lines={}\n",
                        run.runId,
                        run.status,
                        exit,
                        run.purpose,
                        run.terminalName,
                        run.impact,
                        run.lineCount,
                    ));
                }
                out
            }
            tool::ToolAction::TerminalRunStop { runId } => {
                let conn = match crate::storage::openSessionDb(self.transcript.sessionDir()) {
                    Ok(conn) => conn,
                    Err(e) => return format!("Failed to open terminal-run archive: {e}"),
                };
                let Some(run) = (match crate::storage::getTerminalRun(&conn, runId) {
                    Ok(run) => run,
                    Err(e) => return format!("Failed to read terminal run {runId}: {e}"),
                }) else {
                    return format!("No terminal run with id {runId}.");
                };
                if run.status != "running" {
                    return format!(
                        "Terminal run {runId} is already {} — no signal sent.",
                        run.status
                    );
                }
                let shell = {
                    let guard = self.shells.lock().await;
                    guard.shellFor(Some(&run.terminalName))
                };
                let Some(shell) = shell else {
                    return format!(
                        "Terminal run {runId} is marked running, but terminal '{}' is no longer live.",
                        run.terminalName
                    );
                };
                shell.interrupt();
                format!(
                    "Sent interrupt to terminal run {runId} in terminal '{}'.",
                    run.terminalName
                )
            }
            _ => unreachable!("non-registry action passed to executeTerminalTool"),
        }
    }

    /// Spawn a visible terminal-backed async shell run. This replaces the
    /// old `JobPlane` bash path for `shell(runInBackground: true)`.
    async fn spawnTerminalRun(
        &mut self,
        command: String,
        purpose: String,
        impact: crate::tool::ShellImpact,
        timeout: Option<u64>,
        requestedTerminal: Option<String>,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        let runId = crate::transcript::randomHexId("run");
        let startedAt = unixNow();
        let sessionDir = self.transcript.sessionDir().to_path_buf();

        let (terminalName, shell, ephemeral) = match requestedTerminal {
            Some(name) => {
                let guard = self.shells.lock().await;
                let Some(shell) = guard.shellFor(Some(&name)) else {
                    return format!(
                        "No terminal named '{name}'. Use terminalList to see available terminals."
                    );
                };
                if shell.isBusy() {
                    return format!(
                        "Terminal '{name}' is busy; wait for it to finish or choose another terminal."
                    );
                }
                (name, shell, false)
            }
            None => {
                let spawnResult = {
                    let mut guard = self.shells.lock().await;
                    guard.spawn(None, SpawnedBy::Agent).await
                };
                let name = match spawnResult {
                    Ok(name) => name,
                    Err(e) => return format!("Failed to spawn ephemeral terminal: {e}"),
                };
                let shell = {
                    let guard = self.shells.lock().await;
                    guard.shellFor(Some(&name))
                };
                let Some(shell) = shell else {
                    return format!("Failed to resolve ephemeral terminal '{name}' after spawn.");
                };
                (name, shell, true)
            }
        };

        let impactString = shellImpactStorageName(&impact).to_string();
        let initialRecord = crate::storage::TerminalRunRecord {
            runId: runId.clone(),
            terminalName: terminalName.clone(),
            command: command.clone(),
            purpose: if purpose.trim().is_empty() {
                command.clone()
            } else {
                purpose.clone()
            },
            impact: impactString.clone(),
            ephemeral,
            startedAt,
            endedAt: None,
            status: "running".into(),
            exitCode: None,
            lineCount: 0,
            replayBlob: Vec::new(),
        };
        if let Ok(conn) = crate::storage::openSessionDb(&sessionDir) {
            if let Err(e) = crate::storage::upsertTerminalRun(&conn, &initialRecord) {
                tracing::warn!("failed to record terminal run start: {e}");
            }
        }

        let (wakeId, fireTx) = {
            let mut g = self.wakes.lock().await;
            let wid = g.registerTerminalRun(&runId, logTx);
            (wid, g.fireSender())
        };

        let wakes = self.wakes.clone();
        let shells = self.shells.clone();
        let monitors = self.monitors.clone();
        let logTxClone = logTx.clone();
        let runIdForTask = runId.clone();
        let terminalForTask = terminalName.clone();
        let commandForTask = command.clone();
        let purposeForTask = initialRecord.purpose.clone();
        tokio::spawn(async move {
            let dur = timeout.map(std::time::Duration::from_secs);
            let exec = shell.executeDetailed(&commandForTask, dur).await;
            let status = if exec.output.starts_with("Terminal is busy") {
                "rejected"
            } else if exec.timedOut && exec.exitCode.is_none() {
                "timed_out"
            } else if exec.exitCode.unwrap_or(0) == 0 {
                "completed"
            } else {
                "failed"
            };
            let completed = crate::storage::TerminalRunRecord {
                runId: runIdForTask.clone(),
                terminalName: terminalForTask.clone(),
                command: commandForTask.clone(),
                purpose: purposeForTask.clone(),
                impact: impactString,
                ephemeral,
                startedAt,
                endedAt: Some(unixNow()),
                status: status.into(),
                exitCode: exec.exitCode,
                lineCount: exec.lineCount,
                replayBlob: exec.replayBytes.clone(),
            };
            if let Ok(conn) = crate::storage::openSessionDb(&sessionDir) {
                if let Err(e) = crate::storage::upsertTerminalRun(&conn, &completed) {
                    tracing::warn!("failed to record terminal run completion: {e}");
                }
            }

            let payload = format!(
                "terminal run {runIdForTask} in {terminalForTask} finished with status {status}{}.\n{}",
                exec.exitCode
                    .map(|c| format!(" (exit code {c})"))
                    .unwrap_or_default(),
                if exec.output.trim().is_empty() {
                    "(no output)".to_string()
                } else {
                    exec.output.lines().take(20).collect::<Vec<_>>().join("\n")
                },
            );
            let _ = fireTx.send(crate::wakes::WakeFire {
                wakeId,
                source: format!("terminalRun#{runIdForTask}"),
                kind: crate::control::WakeKind::TaskComplete,
                payload,
                firedAt: std::time::Instant::now(),
            });
            wakes.lock().await.unregisterPassive(wakeId, &logTxClone);

            if ephemeral {
                let killed = {
                    let mut guard = shells.lock().await;
                    guard.kill(&terminalForTask)
                };
                if killed.is_ok() {
                    let stopped = {
                        let plane = monitors.lock().unwrap();
                        plane.stopForTerminal(&terminalForTask)
                    };
                    for (id, wakeId) in stopped {
                        if let Some(wid) = wakeId {
                            wakes.lock().await.unregisterPassive(wid, &logTxClone);
                        }
                        let _ = logTxClone.send(LogEvent::MonitorStopped { id }).await;
                    }
                    let _ = logTxClone
                        .send(LogEvent::TerminalClosed {
                            name: terminalForTask,
                        })
                        .await;
                }
            }
        });

        format!(
            "Started terminal run {runId} in terminal '{terminalName}': {command}\n\n\
             It is running asynchronously in a visible terminal. You'll be notified when it completes; do not poll. Use /runs or terminalRunList to inspect archived output."
        )
    }

    /// Terminal-owned monitor cleanup: attach-only monitors subscribe to a
    /// visible terminal's output stream, so closing that terminal stops and
    /// disarms every monitor attached to it.
    async fn stopMonitorsForTerminal(&self, terminalName: &str, logTx: &mpsc::Sender<LogEvent>) {
        let stopped = {
            let plane = self.monitors.lock().unwrap();
            plane.stopForTerminal(terminalName)
        };
        for (id, wakeId) in stopped {
            if let Some(wid) = wakeId {
                self.wakes.lock().await.unregisterPassive(wid, logTx);
            }
            let _ = logTx.send(LogEvent::MonitorStopped { id }).await;
        }
    }

    /// Detach an already-running visible terminal command into the terminal
    /// run archive. This is the timeout/Ctrl+B path: the command keeps
    /// running in the same terminal and the model turn gets the run id
    /// immediately, with no hidden respawn.
    async fn detachTerminalRunJoin(
        &mut self,
        command: String,
        purpose: String,
        impact: crate::tool::ShellImpact,
        terminalName: String,
        startedAt: u64,
        execTask: tokio::task::JoinHandle<crate::shell::CommandExecution>,
        logTx: &mpsc::Sender<LogEvent>,
        trigger: String,
    ) -> String {
        let runId = crate::transcript::randomHexId("run");
        let sessionDir = self.transcript.sessionDir().to_path_buf();
        let impactString = shellImpactStorageName(&impact).to_string();
        let purpose = if purpose.trim().is_empty() {
            command.clone()
        } else {
            purpose
        };

        let initialRecord = crate::storage::TerminalRunRecord {
            runId: runId.clone(),
            terminalName: terminalName.clone(),
            command: command.clone(),
            purpose: purpose.clone(),
            impact: impactString.clone(),
            ephemeral: false,
            startedAt,
            endedAt: None,
            status: "running".into(),
            exitCode: None,
            lineCount: 0,
            replayBlob: Vec::new(),
        };
        if let Ok(conn) = crate::storage::openSessionDb(&sessionDir) {
            if let Err(e) = crate::storage::upsertTerminalRun(&conn, &initialRecord) {
                tracing::warn!("failed to record detached terminal run start: {e}");
            }
        }

        let (wakeId, fireTx) = {
            let mut g = self.wakes.lock().await;
            let wid = g.registerTerminalRun(&runId, logTx);
            (wid, g.fireSender())
        };

        let wakes = self.wakes.clone();
        let logTxClone = logTx.clone();
        let runIdForTask = runId.clone();
        let terminalForTask = terminalName.clone();
        let commandForTask = command.clone();
        let purposeForTask = purpose.clone();
        tokio::spawn(async move {
            let exec = match execTask.await {
                Ok(exec) => exec,
                Err(e) => crate::shell::CommandExecution {
                    command: commandForTask.clone(),
                    output: format!("Terminal run task failed to join: {e}"),
                    exitCode: None,
                    lineCount: 1,
                    replayBytes: Vec::new(),
                    timedOut: false,
                },
            };
            let status = if exec.output.starts_with("Terminal is busy") {
                "rejected"
            } else if exec.timedOut && exec.exitCode.is_none() {
                "timed_out"
            } else if exec.exitCode.unwrap_or(0) == 0 {
                "completed"
            } else {
                "failed"
            };
            let completed = crate::storage::TerminalRunRecord {
                runId: runIdForTask.clone(),
                terminalName: terminalForTask.clone(),
                command: commandForTask.clone(),
                purpose: purposeForTask,
                impact: impactString,
                ephemeral: false,
                startedAt,
                endedAt: Some(unixNow()),
                status: status.into(),
                exitCode: exec.exitCode,
                lineCount: exec.lineCount,
                replayBlob: exec.replayBytes.clone(),
            };
            if let Ok(conn) = crate::storage::openSessionDb(&sessionDir) {
                if let Err(e) = crate::storage::upsertTerminalRun(&conn, &completed) {
                    tracing::warn!("failed to record detached terminal run completion: {e}");
                }
            }

            let payload = format!(
                "terminal run {runIdForTask} in {terminalForTask} finished with status {status}{}.\n{}",
                exec.exitCode
                    .map(|c| format!(" (exit code {c})"))
                    .unwrap_or_default(),
                if exec.output.trim().is_empty() {
                    "(no output)".to_string()
                } else {
                    exec.output.lines().take(20).collect::<Vec<_>>().join("\n")
                },
            );
            let _ = fireTx.send(crate::wakes::WakeFire {
                wakeId,
                source: format!("terminalRun#{runIdForTask}"),
                kind: crate::control::WakeKind::TaskComplete,
                payload,
                firedAt: std::time::Instant::now(),
            });
            wakes.lock().await.unregisterPassive(wakeId, &logTxClone);
        });

        format!(
            "DETACHED_TERMINAL_RUN: {trigger}. The command is still running in terminal '{terminalName}' as run {runId}.\n\n\
             You will be notified when it completes; do not poll. Use /runs or terminalRunList to inspect archived output."
        )
    }

    /// Handle the task-plane tools: backgrounded `shell` calls
    /// (`runInBackground: true`) and the lifecycle tools TaskOutput /
    /// TaskStop / TaskList. These mutate or query `self.jobs`; output
    /// for the agent is text suitable for an LLM tool result.
    async fn executeJobTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        match action {
            tool::ToolAction::Shell {
                command,
                explanation,
                impact,
                timeout,
                terminal,
                runInBackground: true,
            } => {
                self.spawnTerminalRun(
                    command.clone(),
                    explanation.clone(),
                    impact.clone(),
                    *timeout,
                    terminal.clone(),
                    logTx,
                )
                .await
            }
            tool::ToolAction::JobOutput {
                jobId,
                sinceLine,
                maxLines,
            } => {
                let cap = maxLines.unwrap_or(200);
                match self.jobs.lock().unwrap().output(*jobId, *sinceLine, cap) {
                    Ok(snap) => formatJobOutput(*jobId, &snap, *sinceLine),
                    Err(e) => format!("{e}. Use jobList to see available job ids."),
                }
            }
            tool::ToolAction::JobStop { jobId } => {
                // Distinguish the no-op case ("already terminal") from
                // an active kill so the agent doesn't think both
                // statuses mean "the kill signal was just delivered".
                let preState = self
                    .jobs
                    .lock()
                    .unwrap()
                    .list()
                    .into_iter()
                    .find(|t| t.id == *jobId)
                    .map(|t| t.state);
                match self.jobs.lock().unwrap().stop(*jobId) {
                    Ok(()) => match preState {
                        Some(s) if s.isTerminal() => {
                            format!("Job #{jobId} was already {:?} \u{2014} no signal sent.", s,)
                        }
                        _ => format!("Sent kill signal to job #{jobId}."),
                    },
                    Err(e) => format!("Failed to stop job: {e}"),
                }
            }
            tool::ToolAction::JobList => formatJobList(&self.jobs.lock().unwrap().list()),
            _ => unreachable!("non-job action passed to executeJobTool"),
        }
    }

    /// Handle the monitor-plane tools (Monitor / MonitorStop /
    /// MonitorList). Monitors attach to existing terminal output streams.
    async fn executeMonitorTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        match action {
            tool::ToolAction::Monitor {
                description,
                terminal,
                filter,
            } => {
                let (terminalName, shell) = {
                    let guard = self.shells.lock().await;
                    let name = terminal
                        .clone()
                        .unwrap_or_else(|| guard.activeForAgent().to_string());
                    let Some(shell) = guard.shellFor(Some(&name)) else {
                        return format!(
                            "No terminal named '{name}'. Use terminalList to see available terminals."
                        );
                    };
                    (name, shell)
                };
                // Reserve monitor id, register the passive wake
                // against it, THEN call registerWithId so the per-line
                // callback sees wakeCtx from match #1. Eliminates the
                // attach-after-spawn race for monitor fires.
                let id = self.monitors.lock().unwrap().reserveMonitorId();
                let (wakeId, fireTx) = {
                    let mut g = self.wakes.lock().await;
                    let wid = g.registerMonitor(id, logTx);
                    (wid, g.fireSender())
                };
                let result = {
                    let mut plane = self.monitors.lock().unwrap();
                    plane.registerWithId(
                        id,
                        description.clone(),
                        terminalName.clone(),
                        filter.clone(),
                        crate::monitors::DEFAULT_AUTOSTOP_EPS,
                        shell,
                        logTx.clone(),
                        Some(crate::monitors::MonitorWakeCtx {
                            wakeId,
                            registry: self.wakes.clone(),
                            fireTx,
                        }),
                    )
                };
                match result {
                    Ok(_) => format!(
                        "Registered monitor #{id} \"{description}\" with filter /{filter}/.\n\n\
                         Watching terminal: {terminalName}\n\n\
                         You'll be notified when matches arrive (do not poll). \
                         Use monitorList to check event counts, monitorStop({id}) to stop."
                    ),
                    Err(e) => {
                        // Register failed — roll back the passive wake.
                        self.wakes.lock().await.unregisterPassive(wakeId, logTx);
                        format!("Failed to register monitor: {e}")
                    }
                }
            }
            tool::ToolAction::MonitorStop { monitorId } => {
                // Take the wake id BEFORE stopping so we can unregister
                // the passive source — otherwise it would linger in
                // /jobs schedules after the monitor is dead.
                let wakeId = self.monitors.lock().unwrap().takeWakeId(*monitorId);
                let stopResult = {
                    let plane = self.monitors.lock().unwrap();
                    plane.stop(*monitorId)
                };
                match stopResult {
                    Ok(()) => {
                        if let Some(wid) = wakeId {
                            self.wakes.lock().await.unregisterPassive(wid, logTx);
                        }
                        let _ = logTx
                            .send(LogEvent::MonitorStopped { id: *monitorId })
                            .await;
                        format!("Stopped monitor #{monitorId}.")
                    }
                    Err(e) => format!("Failed to stop monitor: {e}"),
                }
            }
            tool::ToolAction::MonitorList => {
                let snapshot = self.monitors.lock().unwrap().list();
                formatMonitorList(&snapshot)
            }
            _ => unreachable!("non-monitor action passed to executeMonitorTool"),
        }
    }

    /// Handle the wake-registry tools: scheduleWakeup, cronCreate,
    /// cronList, cronDelete, fileWatch. These mutate `self.wakes`;
    /// schedulers are spawned as tokio tasks and outlive the turn.
    async fn executeWakeTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        match action {
            tool::ToolAction::ScheduleWakeup {
                delaySeconds,
                prompt,
            } => {
                if *delaySeconds == 0 {
                    return "delaySeconds must be at least 1".into();
                }
                // armDelay locks the registry with blocking_lock — call
                // it from spawn_blocking so we don't block the runtime.
                let regArc = self.wakes.clone();
                let promptOwned = prompt.clone();
                let logTxClone = logTx.clone();
                let secs = *delaySeconds;
                let id = tokio::task::spawn_blocking(move || {
                    crate::wakes::WakeRegistry::armDelay(
                        &regArc,
                        std::time::Duration::from_secs(secs),
                        promptOwned,
                        logTxClone,
                    )
                })
                .await
                .unwrap_or(0);
                format!(
                    "Armed wake #{id} \u{2014} will fire in {secs}s with prompt: {prompt}.\n\
                     You'll receive a <wake source=\"delay#{id}\" kind=\"Delay\"> message at that time. \
                     Cancel with cronDelete({id})."
                )
            }
            tool::ToolAction::CronCreate {
                spec,
                prompt,
                recurring,
            } => {
                let regArc = self.wakes.clone();
                let specOwned = spec.clone();
                let promptOwned = prompt.clone();
                let logTxClone = logTx.clone();
                let recurringFlag = *recurring;
                let result = tokio::task::spawn_blocking(move || {
                    crate::wakes::WakeRegistry::armCron(
                        &regArc,
                        specOwned,
                        recurringFlag,
                        promptOwned,
                        logTxClone,
                    )
                })
                .await;
                match result {
                    Ok(Ok(id)) => format!(
                        "Armed wake #{id} \u{2014} cron `{spec}`{}.\n\
                         You'll receive a <wake source=\"cron#{id}\" kind=\"Cron\"> message on each fire. \
                         Cancel with cronDelete({id}).",
                        if *recurring {
                            " (recurring)"
                        } else {
                            " (one-shot)"
                        },
                    ),
                    Ok(Err(e)) => format!("Failed to arm cron: {e}"),
                    Err(e) => format!("Failed to arm cron: join error: {e}"),
                }
            }
            tool::ToolAction::CronList => {
                let sources = self.wakes.lock().await.list();
                formatWakeList(&sources)
            }
            tool::ToolAction::CronDelete { wakeId } => {
                let removed = self.wakes.lock().await.disarm(*wakeId, logTx);
                if removed {
                    format!("Disarmed wake #{wakeId}.")
                } else {
                    format!("No wake source #{wakeId} (use cronList to see active ids).")
                }
            }
            tool::ToolAction::FileWatch { path, prompt } => {
                let regArc = self.wakes.clone();
                let pathBuf = std::path::PathBuf::from(path);
                let promptOwned = prompt.clone();
                let logTxClone = logTx.clone();
                let result = tokio::task::spawn_blocking(move || {
                    crate::wakes::WakeRegistry::armFileWatch(
                        &regArc,
                        pathBuf,
                        promptOwned,
                        logTxClone,
                    )
                })
                .await;
                match result {
                    Ok(Ok(id)) => format!(
                        "Armed wake #{id} \u{2014} watching {path}.\n\
                         Each fs event under that path fires a <wake source=\"fileWatch#{id}\"> message. \
                         Cancel with cronDelete({id})."
                    ),
                    Ok(Err(e)) => format!("Failed to arm fileWatch: {e}"),
                    Err(e) => format!("Failed to arm fileWatch: join error: {e}"),
                }
            }
            _ => unreachable!("non-wake action passed to executeWakeTool"),
        }
    }

    fn executeTranscriptTool(&self, action: &tool::ToolAction) -> String {
        match action {
            tool::ToolAction::HistoryFetch { blockId } => {
                match self.transcript.loadAll() {
                    Ok(turns) => {
                        let blockTurns: Vec<_> =
                            turns.iter().filter(|t| t.blockId == *blockId).collect();

                        if blockTurns.is_empty() {
                            return format!("No block found with ID \"{blockId}\".");
                        }

                        let mut output = format!("## Block {blockId}\n\n");
                        for turn in &blockTurns {
                            let roleLabel = match turn.role {
                                crate::transcript::TurnRole::User => "User",
                                crate::transcript::TurnRole::Assistant => "Assistant",
                                crate::transcript::TurnRole::ToolCall => "Tool Call",
                                crate::transcript::TurnRole::ToolResult => "Tool Result",
                                crate::transcript::TurnRole::System => "System",
                                crate::transcript::TurnRole::Wake => "Wake",
                            };

                            output.push_str(&format!("### [{roleLabel}] {}\n", turn.id));

                            if let Some(ref toolName) = turn.tool {
                                output.push_str(&format!("Tool: {toolName}\n"));
                            }
                            if let Some(ref args) = turn.args {
                                output.push_str(&format!("Args: {args}\n"));
                            }

                            if !turn.content.is_empty() {
                                output.push_str(&turn.content);
                                output.push('\n');
                            }

                            // Indicate attachments.
                            if let Some(ref atts) = turn.attachments
                                && !atts.is_empty()
                            {
                                output.push_str(&format!("[+{} image(s) attached]\n", atts.len()));
                            }
                            output.push('\n');
                        }
                        output
                    }
                    Err(e) => format!("Failed to load transcript: {e}"),
                }
            }
            tool::ToolAction::HistorySearch { query, mediaType } => {
                match self.transcript.loadAll() {
                    Ok(turns) => {
                        let queryLower = query.to_lowercase();
                        let mut matches: Vec<(String, String, String)> = Vec::new();

                        for turn in &turns {
                            // Filter by mediaType if specified.
                            if let Some(mt) = mediaType {
                                let hasMatchingMedia =
                                    turn.attachments.as_ref().is_some_and(|atts| {
                                        atts.iter().any(|a| a.mimeType.starts_with(mt.as_str()))
                                    });
                                if !hasMatchingMedia {
                                    continue;
                                }
                            }

                            if turn.content.to_lowercase().contains(&queryLower) {
                                let snippet = extractSnippet(&turn.content, &queryLower);
                                let roleLabel = match turn.role {
                                    crate::transcript::TurnRole::User => "user",
                                    crate::transcript::TurnRole::Assistant => "assistant",
                                    crate::transcript::TurnRole::ToolCall => "tool_call",
                                    crate::transcript::TurnRole::ToolResult => "tool_result",
                                    crate::transcript::TurnRole::System => "system",
                                    crate::transcript::TurnRole::Wake => "wake",
                                };
                                // Annotate if turn has attachments.
                                let imageNote = turn
                                    .attachments
                                    .as_ref()
                                    .filter(|a| !a.is_empty())
                                    .map(|a| format!(" [+{} image(s)]", a.len()))
                                    .unwrap_or_default();
                                matches.push((
                                    turn.blockId.clone(),
                                    format!("{} ({}){imageNote}", turn.id, roleLabel),
                                    snippet,
                                ));
                            }
                        }

                        if matches.is_empty() {
                            return format!("No matches found for \"{query}\".");
                        }

                        let totalMatches = matches.len();
                        // Limit output to first 20 matches.
                        let shown = matches.len().min(20);
                        let mut output =
                            format!("Found {totalMatches} matches for \"{query}\":\n\n");
                        for (blockId, turnInfo, snippet) in &matches[..shown] {
                            output.push_str(&format!(
                                "- **{blockId}** {turnInfo}: ...{snippet}...\n"
                            ));
                        }
                        if totalMatches > shown {
                            output.push_str(&format!(
                                "\n({} more matches not shown)\n",
                                totalMatches - shown
                            ));
                        }
                        output
                    }
                    Err(e) => format!("Failed to load transcript: {e}"),
                }
            }
            _ => "Not a transcript tool.".into(),
        }
    }

    /// Execute a web tool (webSearch, webFetch, webSimilar).
    async fn executeWebTool(&mut self, action: &tool::ToolAction) -> String {
        let exa = match &self.exaClient {
            Some(c) => c,
            None => return web::notConfiguredError(),
        };

        match action {
            tool::ToolAction::WebSearch {
                query,
                allowedDomains,
                blockedDomains,
                maxResults,
            } => {
                web::executeSearch(
                    exa,
                    query,
                    allowedDomains.as_deref(),
                    blockedDomains.as_deref(),
                    *maxResults,
                )
                .await
            }
            tool::ToolAction::WebFetch {
                url,
                prompt,
                subpages,
            } => {
                web::executeFetch(
                    exa,
                    &mut self.urlCache,
                    &self.client,
                    &self.config,
                    url,
                    prompt.as_deref(),
                    *subpages,
                )
                .await
            }
            tool::ToolAction::WebSimilar {
                url,
                allowedDomains,
                blockedDomains,
                maxResults,
            } => {
                web::executeSimilar(
                    exa,
                    url,
                    allowedDomains.as_deref(),
                    blockedDomains.as_deref(),
                    *maxResults,
                )
                .await
            }
            _ => "Not a web tool.".into(),
        }
    }

    /// Initialize MCP server connections.
    ///
    /// Starts all configured servers in parallel and merges their tool
    /// definitions into the session's tool list. Failures are logged
    /// but not fatal — the session continues without the failed servers.
    ///
    /// Args:
    ///     servers: Server name → config map from the config file.
    pub async fn initMcp(
        &mut self,
        servers: std::collections::HashMap<String, mcp::config::ServerConfig>,
    ) {
        if servers.is_empty() {
            return;
        }

        let (elicitationTx, _elicitationRx) = mpsc::channel(8);
        let mut mgr = mcp::McpManager::new(elicitationTx);
        self.mcpConfigs = servers.clone();
        let statuses = mgr.startAll(servers).await;

        for status in &statuses {
            let stateStr = format!("{:?}", status.state);
            tracing::info!(
                server = %status.name,
                state = %stateStr,
                "MCP server status"
            );
        }

        // Merge MCP tool defs with builtins.
        let contextBudget = self.config.heavy.contextWindow;
        let mcpDefs = mgr.toolDefs(contextBudget).await;
        if !mcpDefs.is_empty() {
            self.tools.extend(mcpDefs);
            let mcpToolCount = mgr.toolCount().await;
            tracing::info!(
                totalTools = self.tools.len(),
                mcpTools = mcpToolCount,
                "merged MCP tools"
            );
        }

        // Inject MCP section into the system prompt.
        let searchMode = mgr.isSearchMode(contextBudget).await;
        let serverInfos: Vec<prompt::McpServerInfo> = statuses
            .iter()
            .map(|s| prompt::McpServerInfo {
                name: s.name.clone(),
                toolCount: 0,
                status: format!("{:?}", s.state),
            })
            .collect();

        let mcpPrompt = prompt::mcpSection(&serverInfos, searchMode);
        if !mcpPrompt.is_empty()
            && let Some(Message::System { content }) = self.history.first_mut()
        {
            content.push_str("\n\n");
            content.push_str(&mcpPrompt);
        }

        self.mcpManager = Some(mgr);
    }

    /// Execute a subagent task.
    ///
    /// Spawns a child session with its own context, shell, and tool set,
    /// runs the task to completion, and returns the child's final text.
    /// Child log events are wrapped as `SubagentEvent` and forwarded on the
    /// parent log channel; child permit requests are rewrapped with
    /// `PermitOrigin::Subagent` and forwarded on the parent request channel.
    async fn executeTask(
        &mut self,
        prompt: &str,
        agentType: &str,
        parentLogTx: &mpsc::Sender<LogEvent>,
        parentSessionRequestTx: &mpsc::Sender<SessionRequest>,
        parentCancelRx: &mut watch::Receiver<bool>,
    ) -> String {
        use crate::runner;

        let preset = runner::agentPreset(agentType);

        // Clone config. Swap the child's heavy slot to the chosen tier so
        // Client code stays tier-agnostic (it always streams from heavy).
        let mut childConfig = self.config.clone();
        childConfig.heavy = match preset.tier {
            runner::AgentTier::Heavy => childConfig.heavy.clone(),
            runner::AgentTier::Light => childConfig.light.clone(),
            runner::AgentTier::Utility => childConfig.utility.clone(),
        };

        // Spawn an isolated shell registry for the subagent. Phase 1
        // child sessions only ever have a `main` shell — terminal-mgmt
        // tools are filtered out of subagent toolsets in `filterDefs`.
        let (childIoTx, mut childIoRx) =
            mpsc::channel::<(String, crate::shell::ShellIo, crate::shells::SpawnedBy)>(8);
        let (childRegistry, childMainIo) =
            match crate::shells::ShellRegistry::newWithMain(120, 40, childIoTx) {
                Ok(r) => r,
                Err(e) => return format!("Failed to spawn subagent shell: {e}"),
            };

        // Inherit parent permissions, with the preset's rules layered on top
        // as a floor. Rule order is: [preset floor (read-only denies)] →
        // [preset rules] → [parent rules]. First-match-wins ensures the
        // preset's restrictions hold even when the parent is broadly permissive.
        let childPermissions = buildChildPermissions(&self.permissions, &preset);

        // Create child session with inherited+floored permissions.
        let childRegistryArc = std::sync::Arc::new(tokio::sync::Mutex::new(childRegistry));
        let mut child = match Session::new(
            &childConfig,
            childPermissions,
            childRegistryArc,
            preset.interface,
            &[crate::prompt::DomainModule::Swe],
        ) {
            Ok(s) => s,
            Err(e) => return format!("Failed to create subagent session: {e}"),
        };

        let filtered = tool::filterDefs(&tool::builtinDefs(), &preset.toolSet);
        child.setTools(filtered);

        let childSessionId = child.sessionId().to_string();

        // Forward child shell output to parent as SubagentShellOutput events.
        // Drain main shell's output and any future per-child spawn ioRx
        // deliveries (phase 1: only main, but the structure is in place).
        let shellForwardTx = parentLogTx.clone();
        let shellForwardId = childSessionId.clone();
        tokio::spawn(async move {
            let mut mainRx = childMainIo.outputRx;
            loop {
                tokio::select! {
                    Some(data) = mainRx.recv() => {
                        let _ = shellForwardTx
                            .send(LogEvent::SubagentShellOutput {
                                sessionId: shellForwardId.clone(),
                                data,
                            })
                            .await;
                    }
                    Some((_name, _io, _by)) = childIoRx.recv() => {
                        // Phase 1: subagents don't spawn additional shells.
                        // Drop the io to avoid leaking the channel; phase 2+
                        // will multiplex these into shellForwardTx by name.
                    }
                    else => break,
                }
            }
        });

        // Notify parent that subagent has started.
        let _ = parentLogTx
            .send(LogEvent::SubagentStarted {
                sessionId: childSessionId.clone(),
                agentType: agentType.into(),
                prompt: prompt.into(),
            })
            .await;

        tracing::info!(
            agent = %agentType,
            childSession = %childSessionId,
            "subagent spawned"
        );

        // Set up channels for the child session.
        let (childLogTx, mut childLogRx) = mpsc::channel::<LogEvent>(256);
        let (childRequestTx, mut childRequestRx) = mpsc::channel::<SessionRequest>(16);

        // Clone cancel receiver for the child — parent cancel propagates.
        let mut childCancelRx = parentCancelRx.clone();

        // Log forwarder: wrap child log events as SubagentEvent on the parent log channel,
        // accumulate assistant content, and count TurnComplete to report final turn count.
        let logSessionId = childSessionId.clone();
        let logParentTx = parentLogTx.clone();
        let logHandle = tokio::spawn(async move {
            let mut content = String::new();
            let mut turns: usize = 0;
            while let Some(event) = childLogRx.recv().await {
                match &event {
                    LogEvent::ContentDelta(text) => content.push_str(text),
                    LogEvent::TurnComplete => turns += 1,
                    _ => {}
                }
                // Forward visible events; skip noisy internals. The outer match
                // only reads for accounting — all variants flow through the
                // wrap below unless explicitly filtered.
                match &event {
                    LogEvent::ContentDelta(_)
                    | LogEvent::ReasoningDelta(_)
                    | LogEvent::ToolStarted { .. }
                    | LogEvent::ToolAutoApproved { .. }
                    | LogEvent::ToolResult { .. }
                    | LogEvent::ToolDenied { .. }
                    | LogEvent::ToolAutoDenied { .. }
                    | LogEvent::TurnAborted { .. }
                    | LogEvent::TurnComplete
                    | LogEvent::TurnCancelled
                    | LogEvent::Error(_) => {
                        let _ = logParentTx
                            .send(LogEvent::SubagentEvent {
                                sessionId: logSessionId.clone(),
                                event: Box::new(event),
                            })
                            .await;
                    }
                    _ => {}
                }
            }
            (content, turns)
        });

        // Permit forwarder: rewrap each child permit request with origin=Subagent and
        // plumb the parent's reply back into the child's original oneshot.
        let permitSessionId = childSessionId.clone();
        let permitParentTx = parentSessionRequestTx.clone();
        let permitHandle = tokio::spawn(async move {
            while let Some(req) = childRequestRx.recv().await {
                match req {
                    SessionRequest::Permit {
                        origin: _,
                        name,
                        summary,
                        args,
                        diff,
                        explanation,
                        impact,
                        reply: childReply,
                    } => {
                        let (parentReplyTx, parentReplyRx) = oneshot::channel();
                        if permitParentTx
                            .send(SessionRequest::Permit {
                                origin: PermitOrigin::Subagent {
                                    sessionId: permitSessionId.clone(),
                                },
                                name,
                                summary,
                                args,
                                diff,
                                explanation,
                                impact,
                                reply: parentReplyTx,
                            })
                            .await
                            .is_err()
                        {
                            // Parent closed — deny.
                            let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                            continue;
                        }
                        match parentReplyRx.await {
                            Ok(response) => {
                                let _ = childReply.send(response);
                            }
                            Err(_) => {
                                let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                            }
                        }
                    }
                }
            }
        });

        // Run the child session. Subagents don't support mid-turn steering.
        let childInput = UserInput::from(prompt.to_string());
        let (_childSteerTx, mut childSteerRx) = mpsc::channel::<UserInput>(1);
        let (_childUserBgTx, mut childUserBgRx) = mpsc::channel::<()>(1);
        let sendResult = child
            .send(
                &childInput,
                &childLogTx,
                &childRequestTx,
                &mut childCancelRx,
                &mut childSteerRx,
                &mut childUserBgRx,
            )
            .await;

        // Drop senders so forwarding tasks exit.
        drop(childLogTx);
        drop(childRequestTx);

        let (content, turns) = match logHandle.await {
            Ok(r) => r,
            Err(e) => {
                let _ = parentLogTx
                    .send(LogEvent::SubagentComplete {
                        sessionId: childSessionId.clone(),
                        agentType: agentType.into(),
                        content: String::new(),
                        turns: 0,
                    })
                    .await;
                return format!("Subagent forwarding failed: {e}");
            }
        };
        // Permit forwarder has nothing to return; await to clean up.
        let _ = permitHandle.await;

        // Notify parent that subagent completed.
        let _ = parentLogTx
            .send(LogEvent::SubagentComplete {
                sessionId: childSessionId.clone(),
                agentType: agentType.into(),
                content: content.clone(),
                turns,
            })
            .await;

        if let Err(e) = sendResult {
            return format!("Subagent failed: {e}");
        }

        tracing::info!(
            agent = %agentType,
            childSession = %childSessionId,
            turns = turns,
            "subagent completed"
        );

        if content.is_empty() {
            format!("[subagent session: {childSessionId}]\n\nTask completed (no text output).")
        } else {
            format!("[subagent session: {childSessionId}]\n\n{content}")
        }
    }

    /// Background variant of [`Self::executeTask`]. Registers a subagent
    /// task in [`JobPlane`] and `tokio::spawn`s the entire child-session
    /// runner so the parent's tool-call returns immediately with the new
    /// task id. The parent retrieves streaming progress via
    /// `jobOutput(jobId)` and the final content from the ring buffer.
    ///
    /// Cancellation is cooperative: `jobStop` flips the handle's
    /// `cancelRx` watch, which races the child's `send` future inside the
    /// runner.
    async fn executeTaskBackground(
        &mut self,
        prompt: &str,
        agentType: &str,
        parentLogTx: &mpsc::Sender<LogEvent>,
        parentSessionRequestTx: &mpsc::Sender<SessionRequest>,
        _parentCancelRx: &mut watch::Receiver<bool>,
    ) -> String {
        use crate::runner;

        let preset = runner::agentPreset(agentType);
        let mut childConfig = self.config.clone();
        childConfig.heavy = match preset.tier {
            runner::AgentTier::Heavy => childConfig.heavy.clone(),
            runner::AgentTier::Light => childConfig.light.clone(),
            runner::AgentTier::Utility => childConfig.utility.clone(),
        };

        let (childIoTx, childIoRx) =
            mpsc::channel::<(String, crate::shell::ShellIo, crate::shells::SpawnedBy)>(8);
        let (childRegistry, childMainIo) =
            match crate::shells::ShellRegistry::newWithMain(120, 40, childIoTx) {
                Ok(r) => r,
                Err(e) => return format!("Failed to spawn subagent shell: {e}"),
            };

        let childPermissions = buildChildPermissions(&self.permissions, &preset);

        let childRegistryArc = std::sync::Arc::new(tokio::sync::Mutex::new(childRegistry));
        let mut child = match Session::new(
            &childConfig,
            childPermissions,
            childRegistryArc,
            preset.interface,
            &[crate::prompt::DomainModule::Swe],
        ) {
            Ok(s) => s,
            Err(e) => return format!("Failed to create subagent session: {e}"),
        };

        let filtered = tool::filterDefs(&tool::builtinDefs(), &preset.toolSet);
        child.setTools(filtered);

        // Register the task BEFORE spawning the runner so the caller's
        // return string can quote the real task id. Reserve the job id
        // first so the TaskComplete wake source can be registered against
        // it before the runner starts feeding output.
        let taskId = self.jobs.lock().unwrap().reserveJobId();
        let (wakeId, fireTx) = {
            let mut g = self.wakes.lock().await;
            let wid = g.registerTaskComplete(taskId, parentLogTx);
            (wid, g.fireSender())
        };
        let wakeCtx = crate::jobs::TaskWakeCtx {
            wakeId,
            registry: self.wakes.clone(),
            fireTx,
        };
        let handle = self.jobs.lock().unwrap().spawnSubagentWithId(
            taskId,
            agentType.to_string(),
            prompt.to_string(),
            parentLogTx.clone(),
            Some(wakeCtx),
        );

        let childSessionId = child.sessionId().to_string();
        let promptOwned = prompt.to_string();
        let agentTypeOwned = agentType.to_string();
        let parentLogTxOwned = parentLogTx.clone();
        let parentRequestTxOwned = parentSessionRequestTx.clone();

        // Move everything into the runner task. From here on the parent's
        // tool-call thread is free.
        tokio::spawn(async move {
            runSubagentTaskInBackground(
                child,
                childMainIo,
                childIoRx,
                handle,
                promptOwned,
                agentTypeOwned,
                childSessionId.clone(),
                parentLogTxOwned,
                parentRequestTxOwned,
            )
            .await;
        });

        format!(
            "Spawned subagent #{taskId} ({agentType}). The agent is running in the \
             background \u{2014} call jobOutput(jobId: {taskId}) for streaming \
             progress or to read its final answer, jobList to see status, and \
             jobStop(jobId: {taskId}) to cancel."
        )
    }

    /// Execute an MCP tool action.
    async fn executeMcpTool(&self, action: &tool::ToolAction) -> String {
        let mgr = match &self.mcpManager {
            Some(m) => m,
            None => return "MCP not configured.".into(),
        };

        match action {
            tool::ToolAction::Mcp {
                qualifiedName,
                args,
            } => {
                if qualifiedName == "mcpToolSearch" {
                    mgr.executeSearch(args).await
                } else {
                    mgr.routeToolCall(qualifiedName, args).await
                }
            }
            _ => "Not an MCP tool.".into(),
        }
    }

    /// Execute an LSP diagnostics tool call.
    async fn executeLspTool(&mut self, action: &tool::ToolAction) -> String {
        let tool::ToolAction::Diagnostics { path, severity } = action else {
            return "Not an LSP tool.".into();
        };

        let minSeverity = match severity.as_str() {
            "warning" => async_lsp::lsp_types::DiagnosticSeverity::WARNING,
            _ => async_lsp::lsp_types::DiagnosticSeverity::ERROR,
        };

        self.lspManager
            .getDiagnosticsForTool(path, minSeverity, std::time::Duration::from_secs(15))
            .await
    }

    /// Gather structured MCP status data for the TUI panel.
    /// Replace the current permission set.
    pub fn setPermissions(&mut self, permissions: crate::permissions::Permissions) {
        self.permissions = permissions;
    }

    /// Get permissions data for the /permissions panel.
    pub fn permissionsStatusData(
        &self,
    ) -> (
        crate::permissions::PermitMode,
        Vec<crate::permissions::Rule>,
        crate::permissions::PermissionsSource,
        String,
    ) {
        let configPath = self
            .config
            .projectRoot
            .as_ref()
            .map(|r| r.join(".flatline/config.toml").display().to_string())
            .unwrap_or_else(|| "~/.config/flatline/config.toml".into());
        (
            self.permissions.defaultMode.clone(),
            self.permissions.rules.clone(),
            self.permissions.source,
            configPath,
        )
    }

    pub async fn mcpStatusData(
        &self,
    ) -> (
        Vec<(String, String, usize, Vec<(String, String)>, String)>,
        usize,
        bool,
        String,
    ) {
        let configPath = ".mcp.json".to_string();

        let mgr = match &self.mcpManager {
            Some(m) => m,
            None => return (Vec::new(), 0, false, configPath),
        };

        let statuses = mgr.serverStatuses();
        let totalTools = mgr.toolCount().await;
        let searchMode = mgr.isSearchMode(self.config.heavy.contextWindow).await;

        let registry = mgr.registry().read().await;

        let servers: Vec<McpServerStatusEntry> = statuses
            .iter()
            .map(|s| {
                let stateStr = format!("{:?}", s.state);

                // Get tools for this server from registry search.
                let tools: Vec<(String, String)> = registry
                    .search("", Some(&s.name))
                    .iter()
                    .map(|r| (r.qualifiedName.clone(), r.description.clone()))
                    .collect();

                let toolCount = tools.len();

                // Build transport description from config.
                let transport = self
                    .mcpConfigs
                    .get(&s.name)
                    .map(|cfg| {
                        if let Some(ref cmd) = cfg.command {
                            let args = if cfg.args.is_empty() {
                                String::new()
                            } else {
                                format!(" {}", cfg.args.join(" "))
                            };
                            format!("stdio: {cmd}{args}")
                        } else if let Some(ref url) = cfg.url {
                            format!("http: {url}")
                        } else {
                            "unknown".into()
                        }
                    })
                    .unwrap_or_else(|| "unknown".into());

                (s.name.clone(), stateStr, toolCount, tools, transport)
            })
            .collect();

        (servers, totalTools, searchMode, configPath)
    }

    /// Gracefully shut down all MCP server connections.
    pub async fn shutdownMcp(&mut self) {
        if let Some(ref mut mgr) = self.mcpManager {
            mgr.shutdown().await;
        }
    }

    /// Gracefully shut down all LSP server connections.
    pub async fn shutdownLsp(&mut self) {
        self.lspManager.shutdown().await;
    }

    /// Get LSP server status data for the /lsp panel.
    pub fn lspStatusData(&self) -> Vec<lsp::FullServerStatus> {
        self.lspManager.allServerStatuses()
    }

    /// Push a tool result to history and record to transcript.
    fn pushToolResult(&mut self, callId: &str, content: crate::message::Content) {
        // Extract image attachments for transcript persistence.
        let turnAttachments = if content.hasImages() {
            let atts: Vec<crate::transcript::TurnAttachment> = content
                .imageUris()
                .iter()
                .map(|uri| {
                    // Parse data URI: "data:image/png;base64,..."
                    let (mime, data) = if let Some(rest) = uri.strip_prefix("data:") {
                        if let Some((header, b64)) = rest.split_once(",") {
                            let mime = header.strip_suffix(";base64").unwrap_or(header);
                            (mime.to_string(), b64.to_string())
                        } else {
                            ("image/png".into(), String::new())
                        }
                    } else {
                        ("image/png".into(), String::new())
                    };
                    crate::transcript::TurnAttachment {
                        mimeType: mime,
                        data,
                    }
                })
                .collect();
            if atts.is_empty() { None } else { Some(atts) }
        } else {
            None
        };
        match self
            .transcript
            .recordToolResult(callId, content.textContent(), turnAttachments)
        {
            Ok(turnId) => self.headTurnId = Some(turnId),
            Err(e) => tracing::warn!("transcript write failed: {e}"),
        }
        self.history.push(Message::Tool {
            tool_call_id: callId.into(),
            content,
        });
    }

    /// Update session metadata on disk.
    /// Persist session metadata to disk. Called after each user message
    /// so that `/resume` can discover and list sessions.
    pub fn updateMeta(&self) {
        // Try to load existing meta to preserve createdAt.
        let existingMeta = Transcript::loadMeta(self.transcript.sessionDir()).ok();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let projectDir = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let meta = SessionMeta {
            sessionId: self.transcript.sessionId.clone(),
            projectDir,
            createdAt: existingMeta.as_ref().map(|m| m.createdAt).unwrap_or(now),
            updatedAt: now,
            name: existingMeta.as_ref().and_then(|m| m.name.clone()),
            topicLabels: self.topicTracker.topicLabels(),
            topics: self.topicTracker.topics().to_vec(),
            headTurn: self.headTurnId.clone(),
            forks: existingMeta.map(|m| m.forks).unwrap_or_default(),
            totalCost: self.costTracker.sessionCost(),
        };
        if let Err(e) = self.transcript.writeMeta(&meta) {
            tracing::warn!("meta write failed: {e}");
        }
    }

    /// Build context state for the /context display.
    pub fn buildContextState(&self) -> context::ContextState {
        let input = context::BuildStateInput {
            contextWindow: self.config.heavy.contextWindow,
            compactionLog: &self.compactionLog,
            reportedTokens: self.compactionTracker.lastTokens(),
            transcript: &self.transcript,
            headTurnId: self.headTurnId.as_deref().unwrap_or(""),
        };
        context::buildState(&input)
    }

    /// Restore project files to the last checkpoint.
    pub async fn undoCheckpoint(&self) -> crate::control::CommandAck {
        match &self.checkpoint {
            Some(cp) => match cp.undo().await {
                Ok(turnId) => {
                    crate::control::CommandAck::ok(format!("Restored to checkpoint: {turnId}"))
                }
                Err(e) => crate::control::CommandAck::err(format!("Undo failed: {e}")),
            },
            None => crate::control::CommandAck::err("Checkpoint system not initialized."),
        }
    }

    /// Format the list of saved sessions as a text listing (for `/resume` without id).
    pub fn listSessionsText(&self) -> String {
        match transcript::listSessions(None) {
            Ok(sessions) => {
                if sessions.is_empty() {
                    return "No saved sessions found.".to_string();
                }
                let mut output = String::from("Available sessions:\n\n");
                for (i, meta) in sessions.iter().take(20).enumerate() {
                    let name = meta.name.as_deref().unwrap_or("unnamed");
                    let topics = if meta.topicLabels.is_empty() {
                        String::new()
                    } else {
                        format!(" \u{2014} {}", meta.topicLabels.join(", "))
                    };
                    output.push_str(&format!(
                        "{}. {} [{}]{}\n   {}\n\n",
                        i + 1,
                        meta.sessionId,
                        name,
                        topics,
                        meta.projectDir,
                    ));
                }
                output
            }
            Err(e) => format!("Failed to list sessions: {e}"),
        }
    }

    /// List archived visible terminal runs for `/runs`.
    pub fn listTerminalRuns(&self) -> Result<Vec<crate::storage::TerminalRunRecord>> {
        let conn = crate::storage::openSessionDb(self.transcript.sessionDir())?;
        crate::storage::listTerminalRuns(&conn)
    }

    /// Format the cost breakdown as a text report.
    pub fn formatCostBreakdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Session              {}\n",
            crate::cost::formatCost(self.costTracker.sessionCost()),
        ));

        let perModel = self.costTracker.perModel();
        if !perModel.is_empty() {
            let mut models: Vec<_> = perModel.iter().collect();
            models.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            let last = models.len() - 1;
            for (i, (model, cost)) in models.iter().enumerate() {
                let branch = if i == last {
                    "\u{2514}\u{2500}"
                } else {
                    "\u{251C}\u{2500}"
                };
                let short = model.rsplit('/').next().unwrap_or(model);
                out.push_str(&format!(
                    "{branch} {:<20} {}\n",
                    short,
                    crate::cost::formatCost(**cost),
                ));
            }
        }

        let projectDir = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let rolling = crate::cost::rollingWindowCost(Some(&projectDir));
        out.push_str(&format!(
            "\n16h rolling          {}",
            crate::cost::formatCost(rolling),
        ));

        out
    }

    /// Rewind conversation to a prior turn.
    ///
    /// If the user has sent messages on the current branch, the current
    /// state is saved as a fork before rewinding.
    pub async fn rewind(
        &mut self,
        targetTurnId: &str,
        saveFork: bool,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        let mut meta = match Transcript::loadMeta(self.transcript.sessionDir()) {
            Ok(m) => m,
            Err(e) => return format!("Failed to load session metadata: {e}"),
        };

        // Only save a fork if explicitly requested.
        if saveFork {
            self.maybeSaveFork(&mut meta);
        }

        // Switch to the new head.
        self.headTurnId = Some(targetTurnId.to_string());
        meta.headTurn = self.headTurnId.clone();

        if let Err(e) = self.transcript.writeMeta(&meta) {
            return format!("Failed to save rewind: {e}");
        }

        // Set transcript append point to the rewind target.
        if let Ok(allTurns) = self.transcript.loadAll()
            && let Some(turn) = allTurns.iter().find(|t| t.id == targetTurnId)
        {
            self.transcript.setHead(targetTurnId, &turn.blockId);
        }

        // Rebuild history.
        match context::reconstruct(&self.transcript, &self.compactionLog, targetTurnId) {
            Ok(h) => {
                self.history = vec![Message::System {
                    content: self.systemPrompt.clone(),
                }];
                self.history.extend(h);
            }
            Err(e) => return format!("Failed to reconstruct history after rewind: {e}"),
        }

        self.filesRead.clear();
        self.compactionTracker.clearExhaustion();

        // Cancel stale topic classification and rebuild from active branch.
        if let Some(handle) = self.pendingTopicEval.take() {
            handle.abort();
        }
        self.pendingTopicBlockId = None;
        self.rebuildTopicTracker();

        // Emit events for panel replay.
        let branchTurns = self.loadBranchTurns().unwrap_or_default();
        let markers = self.compactionMarkers();
        let _ = logTx
            .send(LogEvent::Rewound {
                targetTurnId: targetTurnId.to_string(),
            })
            .await;
        let _ = logTx
            .send(LogEvent::SessionRestored {
                turns: branchTurns,
                markers,
            })
            .await;
        // Update window title to the topic at the rewind point.
        let label = self.topicTracker.currentLabel();
        if !label.is_empty() {
            let _ = logTx
                .send(LogEvent::TopicChanged {
                    label: label.to_string(),
                })
                .await;
        }

        format!("Rewound to {targetTurnId}")
    }

    /// Switch to a previously saved fork.
    pub async fn switchFork(&mut self, forkId: &str, logTx: &mpsc::Sender<LogEvent>) -> String {
        let mut meta = match Transcript::loadMeta(self.transcript.sessionDir()) {
            Ok(m) => m,
            Err(e) => return format!("Failed to load session metadata: {e}"),
        };

        let forkIdx = match meta.forks.iter().position(|f| f.id == forkId) {
            Some(i) => i,
            None => return format!("Fork {forkId} not found."),
        };

        // Save the current branch as a fork before switching away.
        self.maybeSaveFork(&mut meta);

        // Remove selected fork and restore its head.
        let fork = meta.forks.remove(forkIdx);
        self.headTurnId = Some(fork.headTurn.clone());
        meta.headTurn = self.headTurnId.clone();

        if let Err(e) = self.transcript.writeMeta(&meta) {
            return format!("Failed to save fork switch: {e}");
        }

        // Set transcript append point.
        if let Ok(allTurns) = self.transcript.loadAll()
            && let Some(turn) = allTurns.iter().find(|t| t.id == fork.headTurn)
        {
            self.transcript.setHead(&fork.headTurn, &turn.blockId);
        }

        // Rebuild.
        match context::reconstruct(&self.transcript, &self.compactionLog, &fork.headTurn) {
            Ok(h) => {
                self.history = vec![Message::System {
                    content: self.systemPrompt.clone(),
                }];
                self.history.extend(h);
            }
            Err(e) => return format!("Failed to reconstruct after fork switch: {e}"),
        }

        self.filesRead.clear();
        self.compactionTracker.clearExhaustion();

        // Cancel stale topic classification and rebuild from active branch.
        if let Some(handle) = self.pendingTopicEval.take() {
            handle.abort();
        }
        self.pendingTopicBlockId = None;
        self.rebuildTopicTracker();

        let branchTurns = self.loadBranchTurns().unwrap_or_default();
        let markers = self.compactionMarkers();
        let _ = logTx
            .send(LogEvent::Rewound {
                targetTurnId: fork.headTurn,
            })
            .await;
        let _ = logTx
            .send(LogEvent::SessionRestored {
                turns: branchTurns,
                markers,
            })
            .await;
        // Update window title to the topic on this fork.
        let label = self.topicTracker.currentLabel();
        if !label.is_empty() {
            let _ = logTx
                .send(LogEvent::TopicChanged {
                    label: label.to_string(),
                })
                .await;
        }

        format!("Switched to fork: {}", fork.label)
    }

    /// Save the current branch as a fork if the user sent messages.
    fn maybeSaveFork(&self, meta: &mut SessionMeta) {
        let branchTurns = match self.loadBranchTurns() {
            Ok(t) => t,
            Err(_) => return,
        };

        let hasUserTurns = branchTurns
            .iter()
            .any(|t| matches!(t.role, crate::transcript::TurnRole::User));

        if !hasUserTurns {
            return;
        }

        let currentHead = match &self.headTurnId {
            Some(id) => id.clone(),
            None => return,
        };

        // Build label from the first user message on this branch.
        let label = branchTurns
            .iter()
            .find(|t| matches!(t.role, crate::transcript::TurnRole::User))
            .map(|t| {
                let first = t
                    .content
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("");
                let trimmed = first.trim();
                if trimmed.len() > 60 {
                    format!("{}\u{2026}", &trimmed[..trimmed.floor_char_boundary(59)])
                } else {
                    trimmed.to_string()
                }
            })
            .unwrap_or_else(|| "unnamed fork".to_string());

        let forkId = crate::transcript::randomHexId("fork");
        meta.forks.push(crate::transcript::Fork {
            id: forkId,
            label,
            headTurn: currentHead,
            createdAt: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
    }

    /// List available forks.
    pub fn listForks(&self) -> Vec<crate::transcript::Fork> {
        Transcript::loadMeta(self.transcript.sessionDir())
            .map(|m| m.forks)
            .unwrap_or_default()
    }

    /// Format forks for inline display.
    pub fn formatForksListing(&self) -> String {
        let forks = self.listForks();
        if forks.is_empty() {
            return "No saved forks.".to_string();
        }

        let mut out = format!("**Saved forks** ({})\n", forks.len());
        for fork in &forks {
            let age = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .saturating_sub(fork.createdAt);
            let agoStr = if age < 60 {
                "just now".to_string()
            } else if age < 3600 {
                format!("{}m ago", age / 60)
            } else if age < 86400 {
                format!("{}h ago", age / 3600)
            } else {
                format!("{}d ago", age / 86400)
            };
            out.push_str(&format!(
                "  `{}` \u{2014} {} ({})\n",
                fork.id, fork.label, agoStr
            ));
        }
        out.push_str("\nSwitch with `/forks <id>`");
        out
    }

    /// Check if a write/edit action targets a file that was previously read.
    /// Returns an error message if the file hasn't been read or has changed on disk, None if OK.
    fn checkReadBeforeWrite(&self, action: &tool::ToolAction) -> Option<String> {
        let targetPath = match action {
            tool::ToolAction::EditFile { path, .. } | tool::ToolAction::MultiEdit { path, .. } => {
                path
            }
            tool::ToolAction::WriteFile { path, .. } => {
                // writeFile to a new file is fine — no read needed.
                if !std::path::Path::new(path).exists() {
                    return None;
                }
                path
            }
            _ => return None,
        };

        let targetNorm = normalizePath(targetPath);

        // Stage 1: has the file been read at all?
        let storedHash = match self.filesRead.get(&targetNorm) {
            Some(h) => h,
            None => {
                return Some(format!(
                    "You must read \"{targetPath}\" with readFile before editing or overwriting it."
                ));
            }
        };

        // Stage 2: staleness — has the file changed since the last read?
        match std::fs::read(&targetNorm) {
            Ok(bytes) => {
                let currentHash = sha1_smol::Sha1::from(&bytes).digest().bytes();
                if &currentHash != storedHash {
                    return Some(format!(
                        "File \"{targetPath}\" has changed on disk since you last read it. \
                         Read it again before editing."
                    ));
                }
            }
            Err(_) => {
                return Some(format!(
                    "File \"{targetPath}\" could not be read from disk."
                ));
            }
        }

        None
    }
}

/// Best-effort path normalization for read-before-write comparison.
fn normalizePath(path: &str) -> String {
    std::path::Path::new(path)
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string())
}

/// Extract a short snippet around the first occurrence of a query in text.
fn extractSnippet(text: &str, queryLower: &str) -> String {
    let textLower = text.to_lowercase();
    let pos = match textLower.find(queryLower) {
        Some(p) => p,
        None => return String::new(),
    };

    let contextChars = 80;
    let start = pos.saturating_sub(contextChars);
    let end = (pos + queryLower.len() + contextChars).min(text.len());

    // Snap to char boundaries.
    let start = text.floor_char_boundary(start);
    let end = text.ceil_char_boundary(end);

    text[start..end].replace('\n', " ")
}

/// Build an `Assistant` message for history.
fn buildAssistantMessage(
    content: Option<String>,
    toolCalls: Option<Vec<ToolCall>>,
    reasoning: Option<String>,
) -> Message {
    Message::Assistant {
        content,
        tool_calls: toolCalls,
        reasoning,
    }
}

/// Outcome of a single API call.
enum TurnResult {
    Done {
        promptTokens: Option<usize>,
    },
    ToolCalls {
        calls: Vec<ToolCall>,
        content: Option<String>,
        reasoning: Option<String>,
        promptTokens: Option<usize>,
    },
    Cancelled,
    /// A transient API error that can be retried (e.g. 500, 502, timeout).
    TransientError(String),
}

/// Check whether an API error message looks transient (worth retrying).
fn isTransientError(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("internal server error")
        || lower.contains("bad gateway")
        || lower.contains("service unavailable")
        || lower.contains("gateway timeout")
        || lower.contains("stream stalled")
        || lower.contains("overloaded")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("temporarily unavailable")
        || lower.contains("server error")
        || lower.contains("timeout")
        || lower.contains("connection reset")
        || lower.contains("stream read error")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
}

/// Encode attachments and build multimodal user content for the conversation history.
///
/// Format a coalesced `WakeBatch` as a single user-shaped envelope. The
/// model sees one user message containing N nested `<wake>` elements,
/// one per fire, in arrival order. Single-fire batches still go through
/// this path so the on-the-wire shape is uniform.
fn formatWakeBatch(batch: &crate::wakes::WakeBatch) -> String {
    use std::fmt::Write;
    let count = batch.fires.len();
    let mut buf = String::with_capacity(64 + count * 96);
    let _ = write!(buf, "<wakes count=\"{count}\">");
    for fire in &batch.fires {
        let firedAtSecs = fire.firedAt.elapsed().as_secs();
        let kindStr = fire.kind.asStr();
        let source = escapeWakeXml(&fire.source);
        let payload = escapeWakeXml(&fire.payload);
        let _ = write!(
            buf,
            "\n<wake source=\"{}\" kind=\"{kindStr}\" ageSecs=\"{firedAtSecs}\">\n{}\n</wake>",
            source, payload,
        );
    }
    buf.push_str("\n</wakes>");
    buf
}

fn escapeWakeXml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// One-line summary for `LogEvent::WakeBatchInjected` — drives the deck
/// notice chip without exposing the full envelope text.
fn wakeBatchSummary(batch: &crate::wakes::WakeBatch) -> String {
    let count = batch.fires.len();
    let first = batch.fires.first();
    match (count, first) {
        (1, Some(f)) => format!("{} \u{00B7} {}", f.source, snippet(&f.payload, 80)),
        (n, Some(f)) => format!("{n} wakes (first: {})", f.source),
        _ => "wake".to_string(),
    }
}

fn snippet(s: &str, n: usize) -> String {
    let first = s.lines().next().unwrap_or("");
    if first.len() <= n {
        first.to_string()
    } else {
        // Find a char-boundary <= n so we don't slice a multi-byte char.
        let mut cut = n;
        while cut > 0 && !first.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}\u{2026}", &first[..cut])
    }
}

/// Returns the `Content` for `Message::User` and optional `TurnAttachment` list for transcript.
fn buildUserContent(
    text: &str,
    attachments: &[Attachment],
) -> (
    crate::message::Content,
    Option<Vec<crate::transcript::TurnAttachment>>,
) {
    use base64::Engine;

    let encoded: Vec<(String, Vec<u8>)> = attachments
        .iter()
        .map(|att| {
            if let Some((w, h)) = att.rgbaDimensions {
                let png = encodeRgbaToPng(&att.data, w, h);
                ("image/png".to_string(), png)
            } else {
                (att.mimeType.clone(), att.data.clone())
            }
        })
        .collect();

    let content = if encoded.is_empty() {
        crate::message::Content::text(text)
    } else {
        let imageUris: Vec<String> = encoded
            .iter()
            .map(|(mime, data)| {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                format!("data:{mime};base64,{b64}")
            })
            .collect();
        crate::message::Content::withImages(text, imageUris)
    };

    let turnAttachments = if encoded.is_empty() {
        None
    } else {
        Some(
            encoded
                .iter()
                .map(|(mime, data)| crate::transcript::TurnAttachment {
                    mimeType: mime.clone(),
                    data: base64::engine::general_purpose::STANDARD.encode(data),
                })
                .collect(),
        )
    };

    (content, turnAttachments)
}

/// Encode raw RGBA pixel data to PNG bytes.
/// Resizes images larger than 2048px on either side.
fn encodeRgbaToPng(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .expect("RGBA buffer size mismatch");
    let dynamic = image::DynamicImage::ImageRgba8(img);

    // Resize oversized images with a fast filter.
    let final_img = if width > 2048 || height > 2048 {
        dynamic.resize(2048, 2048, image::imageops::FilterType::Triangle)
    } else {
        dynamic
    };

    let mut buf = Vec::new();
    final_img
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("PNG encoding failed");
    buf
}

/// Accumulates streaming tool call deltas into complete tool calls.
struct ToolCallAccumulator {
    pending: Vec<PendingCall>,
}

struct PendingCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Whether any tool call deltas have been accumulated.
    fn hasContent(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Peek at the current (name, arguments) for an index. Used to refresh
    /// tool-call previews after each arg delta.
    fn pendingCall(&self, index: usize) -> Option<(&str, &str)> {
        self.pending
            .get(index)
            .map(|p| (p.name.as_str(), p.arguments.as_str()))
    }

    /// Returns `(newName, totalArgBytes)`:
    /// - `newName` is `Some(name)` only on the delta that first sets the name
    ///   for this index (used to emit ToolCallPending exactly once).
    /// - `totalArgBytes` is the running total of accumulated argument bytes
    ///   for this index, or `None` if this delta didn't carry any args.
    fn accumulate(
        &mut self,
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) -> (Option<String>, Option<usize>) {
        while self.pending.len() <= index {
            self.pending.push(PendingCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
        }

        let entry = &mut self.pending[index];
        if let Some(id) = id {
            entry.id = id;
        }
        let mut firstName = None;
        if let Some(name) = name {
            if !name.is_empty() && entry.name.is_empty() {
                firstName = Some(name.clone());
            }
            entry.name = name;
        }
        let bytes = if let Some(args) = arguments {
            entry.arguments.push_str(&args);
            Some(entry.arguments.len())
        } else {
            None
        };
        (firstName, bytes)
    }

    fn finish(self) -> Vec<ToolCall> {
        self.pending
            .into_iter()
            .filter(|p| !p.name.is_empty())
            .map(|p| ToolCall {
                id: p.id,
                callType: "function".into(),
                function: FunctionCall {
                    name: p.name,
                    arguments: p.arguments,
                },
            })
            .collect()
    }
}

/// Format a JobOutputSnapshot for the agent. Includes the command,
/// state, line range, and the buffered lines themselves with a hint
/// about how to page if `totalLines` exceeds what we returned.
fn formatJobOutput(
    taskId: u64,
    snap: &crate::jobs::JobOutputSnapshot,
    sinceLine: Option<u64>,
) -> String {
    use crate::jobs::JobState;
    let stateLabel = match &snap.state {
        JobState::Running => "running".to_string(),
        JobState::Completed { exitCode } => format!("completed exit {exitCode}"),
        JobState::Killed => "killed".into(),
        JobState::Errored(msg) => format!("errored: {msg}"),
    };
    let returned = snap.lines.len() as u64;
    let nextLine = snap.firstLine + returned;
    let header = format!(
        "Task #{} \u{2014} {}\nState: {} \u{00B7} {} total lines \u{00B7} \
         showing lines {}..{}\n\n",
        taskId,
        snap.command,
        stateLabel,
        snap.totalLines,
        snap.firstLine,
        snap.firstLine + returned.saturating_sub(1),
    );
    let mut body = String::new();
    // Two cases for "you're missing earlier lines":
    //
    // 1. The caller asked for `sinceLine = N` but the ring evicted past
    //    that point. firstLine > N AND firstLine > earliestBuffered →
    //    those lines are gone for good.
    //
    // 2. The caller passed `sinceLine = None` (default tail) and there
    //    are buffered lines older than what we returned. firstLine >
    //    earliestBuffered → the lines are still recoverable via an
    //    explicit `sinceLine`. We must NOT say they "fell off the ring
    //    buffer" — they're still there. Just hint at how to fetch them.
    let askedFor = sinceLine.unwrap_or(0);
    if snap.firstLine > askedFor && askedFor < snap.earliestBuffered {
        let lost = snap.earliestBuffered - askedFor;
        body.push_str(&format!(
            "[earlier {lost} lines fell off the ring buffer; oldest buffered is line {}]\n",
            snap.earliestBuffered,
        ));
    } else if sinceLine.is_none() && snap.firstLine > snap.earliestBuffered {
        let recoverable = snap.firstLine - snap.earliestBuffered;
        body.push_str(&format!(
            "[{recoverable} earlier lines still buffered \u{2014} call \
             jobOutput(jobId: {taskId}, sinceLine: {}) to read from the start]\n",
            snap.earliestBuffered,
        ));
    }
    for line in &snap.lines {
        body.push_str(line);
        body.push('\n');
    }
    if nextLine < snap.totalLines {
        body.push_str(&format!(
            "\n[{} more lines \u{2014} call jobOutput(jobId: {}, sinceLine: {}) to continue]",
            snap.totalLines - nextLine,
            taskId,
            nextLine,
        ));
    } else if matches!(snap.state, JobState::Running) {
        body.push_str(&format!(
            "\n[task is still running \u{2014} next sinceLine: {}]",
            nextLine,
        ));
    }
    format!("{header}{body}")
}

/// Format a TaskList snapshot.
fn formatJobList(tasks: &[crate::jobs::JobInfo]) -> String {
    use crate::jobs::JobState;
    if tasks.is_empty() {
        return "No background jobs.".into();
    }
    let mut out = String::from("Background tasks:\n");
    for info in tasks {
        let stateLabel = match &info.state {
            JobState::Running => "running".to_string(),
            JobState::Completed { exitCode } => format!("completed exit {exitCode}"),
            JobState::Killed => "killed".into(),
            JobState::Errored(msg) => format!("errored: {msg}"),
        };
        let age = info.spawnedAt.elapsed().as_secs();
        let cmdPreview = if info.command.len() > 80 {
            format!(
                "{}\u{2026}",
                &info.command[..info.command.floor_char_boundary(80)]
            )
        } else {
            info.command.clone()
        };
        out.push_str(&format!(
            "  #{} {} \u{2014} {} \u{00B7} {}s \u{00B7} {} lines\n",
            info.id, cmdPreview, stateLabel, age, info.totalLines,
        ));
    }
    out
}

fn formatMonitorList(monitors: &[crate::monitors::MonitorInfo]) -> String {
    use crate::monitors::MonitorState;
    if monitors.is_empty() {
        return "No monitors.".into();
    }
    let mut out = String::from("Monitors:\n");
    for info in monitors {
        let stateLabel = match &info.state {
            MonitorState::Running => "running".to_string(),
            MonitorState::Stopped => "stopped".into(),
            MonitorState::AutoStopped(reason) => format!("auto-stopped ({reason})"),
        };
        let lastEvent = match info.lastEventAt {
            Some(t) => format!("{}s ago", t.elapsed().as_secs()),
            None => "never".into(),
        };
        out.push_str(&format!(
            "  #{} \"{}\" terminal {} | /{}/ \u{2014} {} \u{00B7} {} events \u{00B7} last {}\n",
            info.id,
            info.description,
            info.terminal,
            info.filter,
            stateLabel,
            info.eventCount,
            lastEvent,
        ));
    }
    out
}

fn formatWakeList(sources: &[crate::wakes::WakeSourceInfo]) -> String {
    if sources.is_empty() {
        return "No wake sources.".into();
    }
    let mut out = String::from("Wake sources:\n");
    for info in sources {
        let promptPreview = info
            .prompt
            .as_deref()
            .filter(|p| !p.is_empty())
            .map(|p| {
                if p.len() > 40 {
                    format!(" \u{2014} {}\u{2026}", &p[..p.floor_char_boundary(40)])
                } else {
                    format!(" \u{2014} {p}")
                }
            })
            .unwrap_or_default();
        let age = info.createdAt.elapsed().as_secs();
        out.push_str(&format!(
            "  #{} [{}] {} \u{00B7} {} fires \u{00B7} {age}s ago{promptPreview}\n",
            info.id,
            info.kind.asStr(),
            info.summary,
            info.firesSoFar,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Content;

    #[test]
    fn shellResolveErrorFormatsRecoveryHint() {
        let err = ShellResolveError::MissingNamed {
            name: "build".into(),
            available: vec!["main".into(), "logs".into()],
            target: "main".into(),
        };
        let text = err.to_string();
        assert!(text.contains("No terminal named 'build'"));
        assert!(text.contains("Available terminals: [main, logs]"));
        assert!(text.contains("Agent's current target: 'main'"));
    }

    #[test]
    fn formatWakeBatchEscapesMarkupPayload() {
        let batch = crate::wakes::WakeBatch {
            fires: vec![crate::wakes::WakeFire {
                wakeId: 1,
                source: "monitor#1".into(),
                kind: crate::control::WakeKind::MonitorMatch,
                payload: "line </wake><wake source=\"evil\"> & <tag>".into(),
                firedAt: std::time::Instant::now(),
            }],
            closedAt: std::time::Instant::now(),
        };

        let envelope = formatWakeBatch(&batch);
        assert_eq!(
            envelope.matches("<wake ").count(),
            1,
            "payload must not create extra wake tags: {envelope}",
        );
        assert!(envelope.contains("&lt;/wake&gt;"));
        assert!(envelope.contains("&lt;tag&gt;"));
        assert!(envelope.contains("&amp;"));
    }

    #[test]
    fn noRidersLeavesHistoryUnchanged() {
        let history = vec![Message::User {
            content: Content::Text("hello".into()),
        }];
        let out = buildRequestMessages(&history, &[], false);
        assert_eq!(out.len(), 1);
        if let Message::User { content } = &out[0] {
            assert_eq!(content.textContent(), "hello");
        } else {
            panic!("expected user");
        }
    }

    #[test]
    fn singleRiderWrapsInCriticalInstructions() {
        let history = vec![Message::User {
            content: Content::Text("fix the bug".into()),
        }];
        let riders = vec![Rider {
            id: "THINKING",
            content: "Body text.".into(),
        }];
        let out = buildRequestMessages(&history, &riders, false);
        let text = if let Message::User { content } = &out[0] {
            content.textContent().to_string()
        } else {
            panic!("expected user");
        };
        let expected = "<CRITICAL_INSTRUCTIONS>\n<THINKING>\nBody text.\n</THINKING>\n</CRITICAL_INSTRUCTIONS>\n\nfix the bug";
        assert_eq!(text, expected);
    }

    #[test]
    fn multipleRidersStackInOneWrapper() {
        let history = vec![Message::User {
            content: Content::Text("do stuff".into()),
        }];
        let riders = vec![
            Rider {
                id: "THINKING",
                content: "Think first.".into(),
            },
            Rider {
                id: "MODE",
                content: "Review only.".into(),
            },
        ];
        let out = buildRequestMessages(&history, &riders, false);
        let text = if let Message::User { content } = &out[0] {
            content.textContent().to_string()
        } else {
            panic!();
        };
        assert!(text.starts_with("<CRITICAL_INSTRUCTIONS>\n"));
        assert!(text.contains("<THINKING>\nThink first.\n</THINKING>"));
        assert!(text.contains("<MODE>\nReview only.\n</MODE>"));
        assert!(text.ends_with("do stuff"));
        // One outer wrapper only.
        assert_eq!(text.matches("<CRITICAL_INSTRUCTIONS>").count(), 1);
    }

    #[test]
    fn ridersApplyOnlyToLatestUserMessage() {
        let history = vec![
            Message::User {
                content: Content::Text("first".into()),
            },
            Message::Assistant {
                content: Some("ok".into()),
                tool_calls: None,
                reasoning: None,
            },
            Message::User {
                content: Content::Text("second".into()),
            },
        ];
        let riders = vec![Rider {
            id: "X",
            content: "body".into(),
        }];
        let out = buildRequestMessages(&history, &riders, false);
        assert_eq!(out.len(), 3);
        if let Message::User { content } = &out[0] {
            assert_eq!(content.textContent(), "first", "first user untouched");
        }
        if let Message::User { content } = &out[2] {
            assert!(content.textContent().contains("<CRITICAL_INSTRUCTIONS>"));
            assert!(content.textContent().ends_with("second"));
        }
    }

    #[test]
    fn promptThinkingBakesScratchpadAtApiBoundary() {
        let history = vec![
            Message::User {
                content: Content::Text("q".into()),
            },
            Message::Assistant {
                content: Some("answer".into()),
                tool_calls: None,
                reasoning: Some("thought".into()),
            },
            Message::User {
                content: Content::Text("followup".into()),
            },
        ];
        let out = buildRequestMessages(&history, &[], true);
        if let Message::Assistant {
            content, reasoning, ..
        } = &out[1]
        {
            assert_eq!(
                content.as_deref(),
                Some("<scratchpad>\nthought\n</scratchpad>\nanswer"),
            );
            assert!(
                reasoning.is_none(),
                "reasoning field cleared in promptThinking mode"
            );
        } else {
            panic!("expected assistant at idx 1");
        }
    }

    #[test]
    fn promptThinkingOffLeavesReasoningSeparate() {
        let history = vec![Message::Assistant {
            content: Some("answer".into()),
            tool_calls: None,
            reasoning: Some("thought".into()),
        }];
        let out = buildRequestMessages(&history, &[], false);
        if let Message::Assistant {
            content, reasoning, ..
        } = &out[0]
        {
            assert_eq!(content.as_deref(), Some("answer"));
            assert_eq!(reasoning.as_deref(), Some("thought"));
        } else {
            panic!();
        }
    }

    #[test]
    fn accumulatorReportsFirstNameOnce() {
        let mut acc = ToolCallAccumulator::new();
        // Name first arrives alongside an id and empty args.
        let (firstName, bytes) = acc.accumulate(
            0,
            Some("call_1".into()),
            Some("editFile".into()),
            Some(String::new()),
        );
        assert_eq!(firstName.as_deref(), Some("editFile"));
        assert_eq!(bytes, Some(0));

        // Same name echoed in a later chunk must NOT fire firstName again.
        let (firstName, _) = acc.accumulate(0, None, Some("editFile".into()), None);
        assert!(firstName.is_none());
    }

    #[test]
    fn accumulatorReportsRunningByteTotal() {
        let mut acc = ToolCallAccumulator::new();
        let c1 = r#"{"path""#;
        let c2 = r#": "crates/deck/src/app.rs""#;
        acc.accumulate(0, None, Some("editFile".into()), Some(c1.into()));
        let (_, bytes) = acc.accumulate(0, None, None, Some(c2.into()));
        assert_eq!(bytes, Some(c1.len() + c2.len()));
        let (_, bytes) = acc.accumulate(0, None, None, Some("}".into()));
        assert_eq!(bytes, Some(c1.len() + c2.len() + 1));
    }

    #[test]
    fn accumulatorTracksParallelCallsIndependently() {
        let mut acc = ToolCallAccumulator::new();
        let (n0, _) = acc.accumulate(0, None, Some("editFile".into()), Some("{".into()));
        let (n1, _) = acc.accumulate(1, None, Some("grep".into()), Some("{".into()));
        assert_eq!(n0.as_deref(), Some("editFile"));
        assert_eq!(n1.as_deref(), Some("grep"));

        let (n0, bytes0) = acc.accumulate(0, None, None, Some("}".into()));
        assert!(n0.is_none());
        assert_eq!(bytes0, Some(2));
        assert_eq!(acc.pendingCall(0), Some(("editFile", "{}")));
        assert_eq!(acc.pendingCall(1), Some(("grep", "{")));
    }

    #[test]
    fn previewResolvesOncePathValueCloses() {
        let mut acc = ToolCallAccumulator::new();
        acc.accumulate(
            0,
            None,
            Some("editFile".into()),
            Some(r#"{"path": "crates"#.into()),
        );
        // Value still open — preview should be None.
        let (name, args) = acc.pendingCall(0).unwrap();
        assert_eq!(crate::tool_preview::previewForTool(name, args), None);

        // Closing quote arrives — preview resolves.
        acc.accumulate(0, None, None, Some(r#"/deck/src/app.rs""#.into()));
        let (name, args) = acc.pendingCall(0).unwrap();
        assert_eq!(
            crate::tool_preview::previewForTool(name, args).as_deref(),
            Some("crates/deck/src/app.rs"),
        );
    }

    #[test]
    fn previewGrepRefinesWhenPathArrivesAfterPattern() {
        let mut acc = ToolCallAccumulator::new();
        acc.accumulate(
            0,
            None,
            Some("grep".into()),
            Some(r#"{"pattern": "ToolCall""#.into()),
        );
        let (name, args) = acc.pendingCall(0).unwrap();
        assert_eq!(
            crate::tool_preview::previewForTool(name, args).as_deref(),
            Some("\"ToolCall\""),
        );

        acc.accumulate(0, None, None, Some(r#", "path": "crates/""#.into()));
        let (name, args) = acc.pendingCall(0).unwrap();
        assert_eq!(
            crate::tool_preview::previewForTool(name, args).as_deref(),
            Some("\"ToolCall\" in crates/"),
        );
    }
}
