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
//! - [`SessionEvent`] — events emitted during a turn
//!
//! # Dependencies
//! `tokio`, `serde_json`

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::sync::{mpsc, watch};

use crate::api;
use crate::checkpoint::CheckpointManager;
use crate::compaction::CompactionLog;
use crate::compaction_trigger;
use crate::config::Config;
use crate::context;
use crate::message::{
    FunctionCall, Message, ReasoningConfig, StreamEvent, TokenUsage, ToolCall, ToolDef,
};
use crate::permissions::{PermitMode, Permissions, Verdict};
use crate::prompt::{self, DomainModule, InterfaceMode};
use crate::shell::Shell;
use crate::tool;
use crate::topic::TopicTracker;
use crate::transcript::{self, Transcript, SessionMeta};
use crate::lsp;
use crate::mcp;
use crate::web;

/// Events emitted by the session during a turn.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Streaming text content from the assistant.
    ContentDelta(String),

    /// Streaming reasoning/thinking content.
    ReasoningDelta(String),

    /// A tool call needs permission before execution.
    /// The consumer must send `true`/`false` on the permit channel.
    ToolRequest {
        name: String,
        summary: String,
        args: String,
        diff: Option<String>,
    },

    /// A tool was auto-approved by the permission config.
    ToolAutoApproved { name: String, summary: String },

    /// A tool has started executing (after approval, before result).
    ToolStarted { name: String, summary: String },

    /// A tool was executed (after approval).
    ToolResult { name: String, output: String },

    /// A tool call was denied.
    ToolDenied { name: String },

    /// Turn aborted because a tool call was denied under Abort mode.
    TurnAborted { name: String },

    /// The full turn is complete.
    TurnComplete,

    /// The turn was cancelled by the user.
    TurnCancelled,

    /// Token usage update from the API.
    /// contextTokens = prompt + completion (what the next call will see as input).
    TokenUpdate {
        promptTokens: usize,
        completionTokens: usize,
        contextTokens: usize,
    },

    /// Result of a slash command execution.
    CommandResult(String),

    /// A compaction stage started running.
    CompactionStarted { stage: String },

    /// A compaction stage finished.
    CompactionComplete {
        stage: String,
        reduction: String,
        /// Block index where the marker should be inserted in the panel.
        /// None for stages that don't replace whole blocks (S1).
        markerBlock: Option<usize>,
    },

    /// Session resume finished (success or failure).
    ResumeComplete { success: bool, message: String },

    /// Session was cleared — deck should wipe the display.
    Cleared,

    /// Session restored with transcript history for display replay.
    SessionRestored {
        turns: Vec<crate::transcript::Turn>,
        /// Compaction markers to insert: `(stage, blockIdx)`.
        markers: Vec<(String, usize)>,
    },

    /// MCP server status response.
    McpStatus {
        /// Vec of (name, state, toolCount, tools: Vec<(qualifiedName, description)>, transport).
        servers: Vec<(String, String, usize, Vec<(String, String)>, String)>,
        totalTools: usize,
        searchMode: bool,
        configPath: String,
    },

    /// An LSP server is not installed but could enhance the experience.
    LspHint { serverId: String, installHint: String },

    /// LSP server status for the /lsp panel.
    LspStatus {
        servers: Vec<crate::lsp::FullServerStatus>,
    },

    /// The current topic label changed (for title bar updates).
    TopicChanged { label: String },

    /// Conversation was rewound to a prior turn.
    Rewound { targetTurnId: String },

    /// Transcript turns for the rewind picker.
    RewindPickerData {
        turns: Vec<crate::transcript::Turn>,
    },

    /// Saved forks for the interactive fork picker.
    ForkPickerData {
        forks: Vec<crate::transcript::Fork>,
    },

    /// A subagent has started executing.
    SubagentStarted {
        sessionId: String,
        agentType: String,
        prompt: String,
    },

    /// An event from a running subagent (wraps a child SessionEvent).
    SubagentEvent {
        sessionId: String,
        event: Box<SessionEvent>,
    },

    /// A subagent needs permission — the TUI should show a prompt and
    /// send the response on the escalation channel.
    SubagentPermitRequest {
        sessionId: String,
        name: String,
        summary: String,
        args: String,
        diff: Option<String>,
        /// One-shot channel for the TUI to send the response back.
        responseTx: mpsc::Sender<crate::permissions::PermitResponse>,
    },

    /// Raw shell output bytes from a subagent's PTY.
    SubagentShellOutput {
        sessionId: String,
        data: Vec<u8>,
    },

    /// A subagent has completed.
    SubagentComplete {
        sessionId: String,
        agentType: String,
        content: String,
        turns: usize,
    },

    /// A transient API error is being retried silently.
    Retrying { attempt: u32, maxAttempts: u32 },

    /// An error occurred.
    Error(String),
}

/// Actions that require session state to execute (from slash commands).
#[derive(Debug)]
pub enum CommandAction {
    /// Show context usage stats.
    ShowContext,
    /// Restore project to before the last file-modifying tool.
    Undo,
    /// Rewind conversation to a prior turn (destructive).
    Rewind { target: String },
    /// Fork current branch, then rewind.
    ForkAndRewind { target: String },
    /// List saved forks or switch to one.
    Forks { forkId: Option<String> },
    /// List sessions or get info about a specific one.
    Resume { sessionId: Option<String> },
    /// Start a fresh session (keep the shell).
    Clear,
    /// Show MCP server status.
    Mcp,
    /// Show LSP server status.
    Lsp,
}

/// Agent session — owns the conversation and drives the turn loop.
pub struct Session {
    client: api::Client,
    config: Config,
    history: Vec<Message>,
    tools: Vec<ToolDef>,
    reasoning: Option<ReasoningConfig>,
    permissions: Permissions,
    shell: Shell,
    transcript: Transcript,
    compactionLog: CompactionLog,
    compactionTracker: compaction_trigger::Tracker,
    topicTracker: TopicTracker,
    checkpoint: Option<CheckpointManager>,
    filesRead: HashMap<String, [u8; 20]>,
    exaClient: Option<web::ExaClient>,
    urlCache: web::UrlCache,
    mcpManager: Option<mcp::McpManager>,
    mcpConfigs: HashMap<String, mcp::config::ServerConfig>,
    lspManager: lsp::LspManager,
    lspWarmedUp: bool,
    /// One-shot system message injected on the first API call after resume, then cleared.
    resumeNotice: Option<String>,
    /// Stored for rebuild on rewind/fork switch.
    systemPrompt: String,
    /// Active branch head turn ID. None for fresh sessions with no messages yet.
    headTurnId: Option<String>,
}

impl Session {
    /// Consume this session and return its shell.
    /// Used when switching to a resumed session — the shell persists.
    pub fn intoShell(self) -> Shell {
        self.shell
    }

    /// Create a new session.
    ///
    /// Args:
    ///     config: Application config (API settings, etc).
    ///     permissions: Permission rules for tool execution.
    ///     shell: Stateful shell session for command execution.
    ///     interface: How the agent is being driven.
    ///     domains: Task-specific skill modules to include.
    pub fn new(
        config: &Config,
        permissions: Permissions,
        shell: Shell,
        interface: InterfaceMode,
        domains: &[DomainModule],
    ) -> Result<Self> {
        let client = api::Client::new(&config)?;
        let tools = tool::builtinDefs();

        let reasoning = config.main.reasoning.as_ref().map(|r| ReasoningConfig {
            effort: r.effort.clone(),
            summary: r.summary.clone(),
        });

        let systemPrompt = prompt::build(interface, domains, config.main.promptThinking);

        let history = vec![Message::System {
            content: systemPrompt.clone(),
        }];

        let sessionId = transcript::newSessionId();
        let transcript = Transcript::create(&sessionId)?;
        let compactionLog = CompactionLog::open(transcript.sessionDir())?;
        let compactionTracker = compaction_trigger::Tracker::new(
            config.main.contextWindow,
            config.compactRatio,
        );
        // System prompt is ephemeral — never recorded in transcript.
        tracing::info!(sessionId = %sessionId, "session created");

        let exaClient = web::ExaClient::new(&config.web.searchKey);
        let projectLsp = lsp::config::loadProjectLsp(
            &std::env::current_dir().unwrap_or_default(),
        ).unwrap_or_default();
        let lspManager = lsp::LspManager::new(&config.lsp, &projectLsp);

        Ok(Self {
            client,
            config: config.clone(),
            history,
            tools,
            reasoning,
            permissions,
            shell,
            transcript,
            compactionLog,
            compactionTracker,
            topicTracker: TopicTracker::new(),
            checkpoint: None,
            filesRead: HashMap::new(),
            exaClient,
            urlCache: web::UrlCache::new(),
            mcpManager: None,
            mcpConfigs: HashMap::new(),
            lspManager,
            lspWarmedUp: false,
            resumeNotice: None,
            systemPrompt,
            headTurnId: None,
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
        shell: Shell,
        interface: InterfaceMode,
        domains: &[DomainModule],
        sessionId: &str,
    ) -> std::result::Result<Self, (anyhow::Error, Shell)> {
        Self::resumeInner(config, permissions, shell, interface, domains, sessionId).await
    }

    async fn resumeInner(
        config: &Config,
        permissions: Permissions,
        shell: Shell,
        interface: InterfaceMode,
        domains: &[DomainModule],
        sessionId: &str,
    ) -> std::result::Result<Self, (anyhow::Error, Shell)> {
        let client = match api::Client::new(&config) {
            Ok(c) => c,
            Err(e) => return Err((e, shell)),
        };
        let tools = tool::builtinDefs();

        let reasoning = config.main.reasoning.as_ref().map(|r| ReasoningConfig {
            effort: r.effort.clone(),
            summary: r.summary.clone(),
        });

        // System prompt is rebuilt from current config, not from transcript.
        let systemPrompt = prompt::build(interface, domains, config.main.promptThinking);

        let mut transcript = match Transcript::open(sessionId) {
            Ok(t) => t,
            Err(e) => return Err((e, shell)),
        };
        let compactionLog = match CompactionLog::open(transcript.sessionDir()) {
            Ok(c) => c,
            Err(e) => return Err((e, shell)),
        };

        // Load headTurn from meta to determine active branch.
        let meta = Transcript::loadMeta(transcript.sessionDir()).ok();
        let headTurnId = meta
            .as_ref()
            .and_then(|m| m.headTurn.clone())
            .or_else(|| transcript.lastTurnId());

        // Set the transcript's append point to the active branch head.
        if let Some(ref head) = headTurnId {
            if let Ok(allTurns) = transcript.loadAll() {
                if let Some(headTurn) = allTurns.iter().find(|t| t.id == *head) {
                    transcript.setHead(head, &headTurn.blockId);
                }
            }
        }

        // Reconstruct conversation from the active branch.
        let reconstructed = match &headTurnId {
            Some(head) => match context::reconstruct(&transcript, &compactionLog, head, config.main.promptThinking) {
                Ok(h) => h,
                Err(e) => return Err((e, shell)),
            },
            None => Vec::new(),
        };

        // Prepend system prompt (ephemeral, not from transcript).
        let mut history = vec![Message::System {
            content: systemPrompt.clone(),
        }];
        history.extend(reconstructed);

        let compactionTracker = compaction_trigger::Tracker::new(
            config.main.contextWindow,
            config.compactRatio,
        );

        // Restore topic tracker state from meta.json.
        let mut topicTracker = TopicTracker::new();
        if let Some(ref m) = meta {
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

        // Rebuild filesRead from history — hash current disk content for staleness detection.
        let mut filesRead: HashMap<String, [u8; 20]> = HashMap::new();
        for msg in &history {
            if let Message::Assistant { tool_calls: Some(calls), .. } = msg {
                for call in calls {
                    if call.function.name == "readFile" {
                        if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                            &call.function.arguments,
                        ) {
                            if let Some(path) = args["path"].as_str() {
                                let norm = normalizePath(path);
                                if let Ok(bytes) = std::fs::read(&norm) {
                                    let digest = sha1_smol::Sha1::from(&bytes).digest().bytes();
                                    filesRead.insert(norm, digest);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Ephemeral resume notice — injected into API calls but never persisted.
        // Cleared after the model's first shell command.
        let resumeNotice = Some("[Session resumed] This conversation was restored from a saved session. \
            The shell environment is fresh \u{2014} working directory, environment variables, \
            and running processes from the prior session are not preserved.".to_string());

        tracing::info!(
            sessionId = %sessionId,
            historyLen = history.len(),
            filesTracked = filesRead.len(),
            "session resumed"
        );

        let exaClient = web::ExaClient::new(&config.web.searchKey);
        let projectLsp = lsp::config::loadProjectLsp(
            &std::env::current_dir().unwrap_or_default(),
        ).unwrap_or_default();
        let lspManager = lsp::LspManager::new(&config.lsp, &projectLsp);

        Ok(Self {
            client,
            config: config.clone(),
            history,
            tools,
            reasoning,
            permissions,
            shell,
            transcript,
            compactionLog,
            compactionTracker,
            topicTracker,
            checkpoint: None,
            filesRead,
            exaClient,
            urlCache: web::UrlCache::new(),
            mcpManager: None,
            mcpConfigs: HashMap::new(),
            lspManager,
            lspWarmedUp: false,
            resumeNotice,
            systemPrompt,
            headTurnId,
        })
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
    /// The `permitRx` channel is used when a tool call verdict is `NeedsApproval`
    /// and the permit mode is `Ask`. If the mode is `Deny` or `Abort`, the
    /// permit channel is not consulted.
    ///
    /// Args:
    ///     userMessage: The user's input text.
    ///     eventTx: Channel for session events.
    ///     permitRx: Channel for permission responses.
    pub fn send<'a>(
        &'a mut self,
        userMessage: &'a str,
        eventTx: &'a mpsc::Sender<SessionEvent>,
        permitRx: &'a mut mpsc::Receiver<crate::permissions::PermitResponse>,
        cancelRx: &'a mut watch::Receiver<bool>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
        tracing::info!(len = userMessage.len(), "user message received");

        // Warm up LSP servers on first send (scans project, starts matching servers).
        if !self.lspWarmedUp {
            self.lspWarmedUp = true;
            let projectDir = std::env::current_dir().unwrap_or_default();
            self.lspManager.warmUp(&projectDir).await;
        }

        // Inject ephemeral resume notice before the user message so the model
        // sees it in context. Stripped at end of send(); re-injected next call
        // until cleared by a shell command.
        if self.resumeNotice.is_some() {
            self.history.push(Message::System {
                content: self.resumeNotice.as_ref().unwrap().clone(),
            });
        }

        // When prompt thinking is active, prefix the user message with a
        // short rider reminding the model to use its scratchpad.
        let historyContent = if self.config.main.promptThinking {
            format!("{}{}", prompt::THINKING_RIDER, userMessage)
        } else {
            userMessage.to_string()
        };

        self.history.push(Message::User {
            content: historyContent,
        });
        match self.transcript.recordUser(userMessage, self.headTurnId.as_deref()) {
            Ok(turnId) => self.headTurnId = Some(turnId),
            Err(e) => tracing::warn!("transcript write failed: {e}"),
        }

        let result = self.sendInner(eventTx, permitRx, cancelRx).await;

        // Always strip injected messages from history so they don't accumulate.
        self.removeResumeInjection();

        result
        })
    }

    /// Inner turn loop — separated from `send()` so the ephemeral resume
    /// notice cleanup runs unconditionally regardless of how the loop exits.
    async fn sendInner(
        &mut self,
        eventTx: &mpsc::Sender<SessionEvent>,
        permitRx: &mut mpsc::Receiver<crate::permissions::PermitResponse>,
        cancelRx: &mut watch::Receiver<bool>,
    ) -> Result<()> {
        // Evaluate topic tracker — sends full history to utility model for
        // prefix cache hits, then classifies the latest user message.
        let blockId = self.transcript.currentBlock().to_string();
        match self.topicTracker.evaluate(
            &self.history,
            &blockId,
            &self.client,
            &self.config.utility.model,
        ).await {
            Ok(result) => {
                self.transcript.setTopicId(&result.topicId);
                if result.isNewTopic {
                    tracing::info!(
                        topicId = %result.topicId,
                        label = %result.label,
                        "new topic segment"
                    );
                }
                let _ = eventTx
                    .send(SessionEvent::TopicChanged {
                        label: result.label.clone(),
                    })
                    .await;
            }
            Err(e) => {
                tracing::warn!("topic evaluation failed: {e}");
            }
        }

        // Persist session metadata so /resume can find this session.
        self.updateMeta();

        // Take a checkpoint snapshot before the model turn.
        if let Some(ref checkpoint) = self.checkpoint {
            let turnId = self.transcript.currentBlock().to_string();
            if let Err(e) = checkpoint.snapshot(&turnId).await {
                tracing::warn!("checkpoint snapshot failed: {e}");
            }
        }

        const MAX_RETRIES: u32 = 5;
        let mut retryCount: u32 = 0;

        loop {
            // Check for cancellation between turns.
            if *cancelRx.borrow() {
                tracing::info!("turn cancelled before streaming");
                let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                return Ok(());
            }

            tracing::debug!(historyLen = self.history.len(), "starting turn");
            // NOTE: Err from streamOneTurn means either a permanent API error
            // or a transient one that already exhausted the API client's own
            // 8-attempt retry loop. Don't retry again here — only retry
            // mid-stream SSE errors (returned as TurnResult::TransientError).
            let turnResult = self.streamOneTurn(eventTx, cancelRx).await?;

            match turnResult {
                TurnResult::TransientError(msg) => {
                    retryCount += 1;
                    if retryCount > MAX_RETRIES {
                        let _ = eventTx.send(SessionEvent::Error(msg)).await;
                        return Ok(());
                    }
                    tracing::warn!(
                        attempt = retryCount,
                        max = MAX_RETRIES,
                        error = %msg,
                        "transient API error, retrying"
                    );
                    let _ = eventTx.send(SessionEvent::Retrying {
                        attempt: retryCount,
                        maxAttempts: MAX_RETRIES,
                    }).await;
                    // Exponential backoff: 1s, 2s, 4s, 8s, 16s.
                    let delay = Duration::from_secs(1 << (retryCount - 1));
                    tokio::time::sleep(delay).await;
                    continue;
                }
                TurnResult::Done { promptTokens } => {
                    retryCount = 0;
                    if let Some(tokens) = promptTokens {
                        self.compactionTracker.updateTokens(tokens);
                        self.checkCompactionTrigger(eventTx).await;
                    }
                    tracing::info!("turn complete (no tool calls)");
                    let _ = eventTx.send(SessionEvent::TurnComplete).await;
                    return Ok(());
                }
                TurnResult::Cancelled => {
                    tracing::info!("turn cancelled during streaming");
                    let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                    return Ok(());
                }
                TurnResult::ToolCalls { calls, content, reasoning, promptTokens } => {
                    retryCount = 0;
                    if let Some(tokens) = promptTokens {
                        self.compactionTracker.updateTokens(tokens);
                        self.checkCompactionTrigger(eventTx).await;
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
                        content, Some(calls.clone()), reasoning,
                        self.config.main.promptThinking,
                    ));

                    let mut aborted = false;

                    for (callIdx, call) in calls.iter().enumerate() {
                        // Check for cancellation between tool calls.
                        if *cancelRx.borrow() {
                            tracing::info!("cancelled between tool calls");
                            for remaining in &calls[callIdx..] {
                                self.pushToolResult(&remaining.id, "Cancelled by user.".into());
                            }
                            let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                            return Ok(());
                        }

                        let action =
                            tool::parse(&call.function.name, &call.function.arguments);
                        let summary = tool::summarize(&action);

                        // Pre-emptive LSP notification: send didChange with proposed
                        // content so RA starts analyzing while the user reviews.
                        // Stores (path, original_content) for revert on denial.
                        let lspPreemptive: Option<(String, String)> = if let Some((path, proposed)) = tool::proposedContent(&action) {
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
                                let _ = eventTx
                                    .send(SessionEvent::ToolAutoApproved {
                                        name: call.function.name.clone(),
                                        summary: summary.clone(),
                                    })
                                    .await;
                                true
                            }
                            Verdict::Deny => {
                                let _ = eventTx
                                    .send(SessionEvent::ToolDenied {
                                        name: call.function.name.clone(),
                                    })
                                    .await;
                                false
                            }
                            Verdict::NeedsApproval => {
                                match self.permissions.defaultMode {
                                    PermitMode::Ask => {
                                        let diff = tool::diffPreview(&action);
                                        let _ = eventTx
                                            .send(SessionEvent::ToolRequest {
                                                name: call.function.name.clone(),
                                                summary,
                                                args: call.function.arguments.clone(),
                                                diff,
                                            })
                                            .await;

                                        // Wait for supervisor response or cancellation.
                                        tokio::select! {
                                            permit = permitRx.recv() => {
                                                use crate::permissions::PermitResponse;
                                                match permit {
                                                    Some(PermitResponse::Allow) => true,
                                                    Some(PermitResponse::AlwaysAllow { pattern }) => {
                                                        let (toolName, _) = crate::permissions::actionKey(&action);
                                                        self.permissions.addRule(crate::permissions::Rule {
                                                            tool: toolName.into(),
                                                            pattern: Some(pattern),
                                                            allow: true,
                                                        });
                                                        true
                                                    }
                                                    Some(PermitResponse::Deny) | None => false,
                                                }
                                            }
                                            _ = cancelRx.changed() => {
                                                tracing::info!("cancelled during permission wait");
                                                for remaining in &calls[callIdx..] {
                                                    self.pushToolResult(&remaining.id, "Cancelled by user.".into());
                                                }
                                                let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                                                return Ok(());
                                            }
                                        }
                                    }
                                    PermitMode::Deny => {
                                        let _ = eventTx
                                            .send(SessionEvent::ToolDenied {
                                                name: call.function.name.clone(),
                                            })
                                            .await;
                                        false
                                    }
                                    PermitMode::Abort => {
                                        let _ = eventTx
                                            .send(SessionEvent::TurnAborted {
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
                        if !approved {
                            if let Some((ref path, ref original)) = lspPreemptive {
                                self.lspManager.touchFile(path, original).await;
                            }
                        }

                        if aborted {
                            self.pushToolResult(&call.id, "Turn aborted: tool call not permitted.".into());
                            for remaining in &calls[callIdx + 1..] {
                                self.pushToolResult(&remaining.id, "Turn aborted: tool call not permitted.".into());
                            }
                            break;
                        }

                        // Guard: editFile/writeFile require a prior readFile of the same path.
                        if approved {
                            if let Some(ref rejection) = self.checkReadBeforeWrite(&action) {
                                tracing::info!(
                                    tool = %call.function.name,
                                    "rejected: file not read first"
                                );
                                // Revert pre-emptive LSP notification.
                                if let Some((ref path, ref original)) = lspPreemptive {
                                    self.lspManager.touchFile(path, original).await;
                                }
                                let _ = eventTx.send(SessionEvent::ToolResult {
                                    name: call.function.name.clone(),
                                    output: rejection.clone(),
                                }).await;
                                self.pushToolResult(&call.id, rejection.clone());
                                continue;
                            }
                        }

                        let output = if approved {
                            tracing::info!(tool = %call.function.name, "executing tool");

                            if tool::needsTask(&action) {
                                // Subagent events handle all TUI rendering — no ToolStarted needed.
                                let (taskPrompt, taskAgent) = match &action {
                                    tool::ToolAction::Task { prompt, agent } => {
                                        (prompt.clone(), agent.as_deref().unwrap_or("general").to_string())
                                    }
                                    _ => unreachable!(),
                                };
                                let result = self.executeTask(&taskPrompt, &taskAgent, eventTx, cancelRx).await;
                                // NOTE: No ToolResult event — SubagentComplete already notified the TUI.
                                result
                            } else {
                            // Emit ToolStarted for non-task tools.
                            let _ = eventTx
                                .send(SessionEvent::ToolStarted {
                                    name: call.function.name.clone(),
                                    summary: tool::summarize(&action),
                                })
                                .await;

                            if tool::needsMcp(&action) {
                                self.executeMcpTool(&action).await
                            } else if tool::needsTranscript(&action) {
                                self.executeTranscriptTool(&action)
                            } else if tool::needsWeb(&action) {
                                self.executeWebTool(&action).await
                            } else if tool::needsLsp(&action) {
                                self.executeLspTool(&action).await
                            }
                            // Race tool execution against cancellation for shell commands.
                            else if matches!(action, tool::ToolAction::Shell { .. }) {
                                // Shell is now warm — drop the ephemeral resume notice.
                                self.resumeNotice = None;

                                loop {
                                tokio::select! {
                                    result = tool::execute(&action, &self.shell) => {
                                        tracing::debug!(
                                            tool = %call.function.name,
                                            outputLen = result.len(),
                                            "tool execution complete"
                                        );
                                        break result;
                                    }
                                    _ = cancelRx.changed() => {
                                        if !*cancelRx.borrow() {
                                            // Spurious wakeup — retry select.
                                            continue;
                                        }
                                        tracing::info!(tool = %call.function.name, "cancelled during shell execution");
                                        self.shell.interrupt();
                                        self.pushToolResult(&call.id, "Cancelled by user.".into());
                                        for remaining in &calls[callIdx + 1..] {
                                            self.pushToolResult(&remaining.id, "Cancelled by user.".into());
                                        }
                                        let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                                        return Ok(());
                                    }
                                }
                                }
                            } else {
                                // File operations are fast — no cancel race needed.
                                let result = tool::execute(&action, &self.shell).await;
                                tracing::debug!(
                                    tool = %call.function.name,
                                    outputLen = result.len(),
                                    "tool execution complete"
                                );
                                result
                            }
                            } // Close the else block for non-task tools.
                        } else {
                            let _ = eventTx
                                .send(SessionEvent::ToolDenied {
                                    name: call.function.name.clone(),
                                })
                                .await;
                            "User denied this action.".into()
                        };

                        // Track file reads for the edit gate (hash for staleness detection).
                        if call.function.name == "readFile" {
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                &call.function.arguments,
                            ) {
                                if let Some(path) = args["path"].as_str() {
                                    let norm = normalizePath(path);
                                    if let Ok(bytes) = std::fs::read(&norm) {
                                        let digest = sha1_smol::Sha1::from(&bytes).digest().bytes();
                                        self.filesRead.insert(norm.clone(), digest);
                                    }
                                    // Sync file with LSP server (lazy spawn if needed).
                                    if let Ok(content) = std::fs::read_to_string(&norm) {
                                        if let Some(hint) = self.lspManager.touchFile(&norm, &content).await {
                                            let _ = eventTx.send(SessionEvent::LspHint {
                                                serverId: hint.serverId,
                                                installHint: hint.installHint,
                                            }).await;
                                        }
                                    }
                                }
                            }
                        }

                        // Update hash after successful file mutations.
                        if matches!(call.function.name.as_str(), "editFile" | "writeFile" | "multiEdit") {
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                &call.function.arguments,
                            ) {
                                if let Some(path) = args["path"].as_str() {
                                    let norm = normalizePath(path);
                                    if let Ok(bytes) = std::fs::read(&norm) {
                                        let digest = sha1_smol::Sha1::from(&bytes).digest().bytes();
                                        self.filesRead.insert(norm, digest);
                                    }
                                }
                            }
                        }

                        // Collect LSP diagnostics after file mutations.
                        // Diff against baseline to only show errors introduced by the edit.
                        let mut output = output;
                        if matches!(call.function.name.as_str(), "editFile" | "writeFile" | "multiEdit") {
                            if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                                &call.function.arguments,
                            ) {
                                if let Some(path) = args["path"].as_str() {
                                    // Baseline: cached diagnostics from before the edit
                                    // (populated by the pre-emptive touchFile or prior reads).
                                    let baseline = self.lspManager.getRawCachedDiagnostics(path);

                                    let content = std::fs::read_to_string(path).unwrap_or_default();
                                    let (postEdit, hint) = self.lspManager.getRawDiagnostics(
                                        path,
                                        &content,
                                        std::time::Duration::from_secs(10),
                                    ).await;

                                    // Multiset diff: only new errors survive.
                                    let newErrors = lsp::diagnostics::diffDiagnostics(&baseline, &postEdit);
                                    if !newErrors.is_empty() {
                                        let formatted = lsp::diagnostics::formatDiagnostics(
                                            path,
                                            &newErrors,
                                            async_lsp::lsp_types::DiagnosticSeverity::ERROR,
                                        );
                                        if !formatted.is_empty() {
                                            output.push_str("\n\nNew LSP errors after edit:\n");
                                            output.push_str(&formatted);
                                        }
                                    }
                                    if let Some(hint) = hint {
                                        let _ = eventTx.send(SessionEvent::LspHint {
                                            serverId: hint.serverId,
                                            installHint: hint.installHint,
                                        }).await;
                                    }
                                }
                            }
                        }

                        // Emit ToolResult AFTER diagnostics injection so the TUI
                        // shows the same content the model sees.
                        let _ = eventTx
                            .send(SessionEvent::ToolResult {
                                name: call.function.name.clone(),
                                output: output.clone(),
                            })
                            .await;

                        self.pushToolResult(&call.id, output);
                    }

                    if aborted {
                        let _ = eventTx.send(SessionEvent::TurnComplete).await;
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Stream one API call and return what happened.
    async fn streamOneTurn(
        &mut self,
        tx: &mpsc::Sender<SessionEvent>,
        cancelRx: &mut watch::Receiver<bool>,
    ) -> Result<TurnResult> {
        // When prompt-injected thinking is active, don't send the reasoning
        // config (we're faking it via prompt) and set up the content extractor.
        let reasoning = if self.config.main.promptThinking {
            None
        } else {
            self.reasoning.as_ref()
        };
        let mut thinkingExtractor = if self.config.main.promptThinking {
            Some(crate::api::ThinkingExtractor::new())
        } else {
            None
        };

        let mut rx = self
            .client
            .stream(&self.history, &self.tools, reasoning)
            .await?;

        let mut contentBuf = String::new();
        let mut reasoningBuf = String::new();
        let mut toolAccum = ToolCallAccumulator::new();
        let mut lastUsage: Option<TokenUsage> = None;

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
                                            let _ = tx.send(SessionEvent::ContentDelta(t)).await;
                                        }
                                        StreamEvent::ReasoningDelta(t) => {
                                            reasoningBuf.push_str(&t);
                                            let _ = tx.send(SessionEvent::ReasoningDelta(t)).await;
                                        }
                                        _ => {}
                                    }
                                }
                            } else {
                                contentBuf.push_str(&text);
                                let _ = tx.send(SessionEvent::ContentDelta(text)).await;
                            }
                        }
                        Some(StreamEvent::ReasoningDelta(text)) => {
                            reasoningBuf.push_str(&text);
                            let _ = tx.send(SessionEvent::ReasoningDelta(text)).await;
                        }
                        Some(StreamEvent::ToolCallDelta {
                            index,
                            id,
                            name,
                            arguments,
                        }) => {
                            toolAccum.accumulate(index, id, name, arguments);
                        }
                        Some(StreamEvent::Done { usage, .. }) => {
                            if let Some(u) = usage {
                                lastUsage = Some(u);
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
                            let _ = tx.send(SessionEvent::Error(msg.clone())).await;
                            bail!("Stream error: {msg}");
                        }
                    }
                }
                _ = cancelRx.changed() => {
                    if *cancelRx.borrow() {
                        tracing::info!("stream cancelled, committing partial content");
                        // Drop rx — kills the SSE background task.
                        drop(rx);
                        // Commit partial content to history (skip if nothing was streamed).
                        if !contentBuf.is_empty() || !reasoningBuf.is_empty() {
                            if !contentBuf.is_empty() {
                                let reasonRef = if reasoningBuf.is_empty() { None } else { Some(reasoningBuf.as_str()) };
                                match self.transcript.recordAssistant(&contentBuf, reasonRef) {
                                    Ok(turnId) => self.headTurnId = Some(turnId),
                                    Err(e) => tracing::warn!("transcript write failed: {e}"),
                                }
                            }
                            let content = if contentBuf.is_empty() { None } else { Some(contentBuf) };
                            let reasoning = if reasoningBuf.is_empty() { None } else { Some(reasoningBuf) };
                            self.history.push(buildAssistantMessage(
                                content, None, reasoning,
                                self.config.main.promptThinking,
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
                        let _ = tx.send(SessionEvent::ContentDelta(t)).await;
                    }
                    StreamEvent::ReasoningDelta(t) => {
                        reasoningBuf.push_str(&t);
                        let _ = tx.send(SessionEvent::ReasoningDelta(t)).await;
                    }
                    _ => {}
                }
            }
        }

        // Strip variation selectors from emoji-only codepoints.
        contentBuf = crate::text::sanitizeVariationSelectors(&contentBuf);
        reasoningBuf = crate::text::sanitizeVariationSelectors(&reasoningBuf);

        let calls = toolAccum.finish();

        // Emit token usage from the API response.
        if let Some(ref u) = lastUsage {
            let contextTokens = u.promptTokens + u.completionTokens;
            let _ = tx
                .send(SessionEvent::TokenUpdate {
                    promptTokens: u.promptTokens,
                    completionTokens: u.completionTokens,
                    contextTokens,
                })
                .await;
        } else {
            tracing::warn!("no usage data received from API — provider may not support stream_options.include_usage");
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
            let reasonRef = if reasoningBuf.is_empty() { None } else { Some(reasoningBuf.as_str()) };
            if !contentBuf.is_empty() || reasonRef.is_some() {
                match self.transcript.recordAssistant(&contentBuf, reasonRef) {
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
            Ok(TurnResult::ToolCalls { calls, content, reasoning, promptTokens: reportedTokens })
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
                match self.transcript.recordAssistant(textRef, reasonRef) {
                    Ok(turnId) => self.headTurnId = Some(turnId),
                    Err(e) => tracing::warn!("transcript write failed: {e}"),
                }
            }

            self.history.push(buildAssistantMessage(
                content, None, reasoning,
                self.config.main.promptThinking,
            ));

            Ok(TurnResult::Done { promptTokens: reportedTokens })
        }
    }

    /// Check compaction trigger and run the appropriate stage.
    ///
    /// Loops on exhaustion: if a stage exhausts without reducing context,
    /// re-evaluates and tries the next cheapest stage. Stops when a stage
    /// does work, nothing is returned, or all stages are exhausted.
    async fn checkCompactionTrigger(&mut self, eventTx: &mpsc::Sender<SessionEvent>) {
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
            let _ = eventTx
                .send(SessionEvent::CompactionStarted {
                    stage: stageStr.clone(),
                })
                .await;

            // didWork tracks whether this stage reduced context.
            // If false, we loop to try the next stage.
            let didWork = match stage {
                compaction_trigger::StagePick::S1 => {
                    self.runS1(&stageStr, eventTx).await
                }
                compaction_trigger::StagePick::S2 => {
                    self.runS2(&stageStr, eventTx).await
                }
                compaction_trigger::StagePick::S3 => {
                    self.runS3(&stageStr, eventTx).await
                }
                compaction_trigger::StagePick::S4 => {
                    self.runS4Trigger(&stageStr, eventTx).await
                }
            };

            if didWork || self.compactionTracker.allExhausted() {
                return;
            }
            // Stage exhausted without reducing — loop to try next.
        }
    }

    /// Run S1 mechanical pruning. Returns true if context was reduced.
    async fn runS1(
        &mut self,
        stageStr: &str,
        eventTx: &mpsc::Sender<SessionEvent>,
    ) -> bool {
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
            let afterTurn = self.transcript.currentBlock().to_string();
            if !s1Result.dedupedCallIds.is_empty() {
                if let Err(e) = self.compactionLog.recordFileDedup(
                    s1Result.dedupedCallIds.clone(),
                    &afterTurn,
                ) {
                    tracing::warn!("compaction log write failed: {e}");
                }
            }
            if !s1Result.middleOutCallIds.is_empty() {
                if let Err(e) = self.compactionLog.recordMiddleOut(
                    s1Result.middleOutCallIds.clone(),
                    &afterTurn,
                    s1Result.middleOutThreshold,
                ) {
                    tracing::warn!("compaction log write failed: {e}");
                }
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
            let _ = eventTx
                .send(SessionEvent::CompactionComplete {
                    stage: stageStr.to_string(),
                    reduction,
                    markerBlock: None,
                })
                .await;
            true
        } else {
            self.compactionTracker.markExhausted(compaction_trigger::StagePick::S1);
            tracing::debug!("S1 exhausted \u{2014} nothing to prune");
            false
        }
    }

    /// Run S2 block compaction. Returns true if context was reduced.
    async fn runS2(
        &mut self,
        stageStr: &str,
        eventTx: &mpsc::Sender<SessionEvent>,
    ) -> bool {
        let s2Result = match crate::s2::run(
            &self.transcript,
            &self.compactionLog,
            &self.client,
            &self.config.utility.model,
            self.config.main.contextWindow,
            self.config.compactRatio,
        ).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S2 compaction failed: {e}");
                self.compactionTracker.markExhausted(compaction_trigger::StagePick::S2);
                return false;
            }
        };
        if !s2Result.didWork {
            self.compactionTracker.markExhausted(compaction_trigger::StagePick::S2);
            tracing::debug!("S2 exhausted \u{2014} no blocks to compact");
            return false;
        }
        let afterTurn = self.transcript.currentBlock().to_string();
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
        match context::reconstruct(&self.transcript, &self.compactionLog, headId, self.config.main.promptThinking) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S2: {e}");
                return false;
            }
        }
        self.compactionTracker.clearExhaustion();
        let reduction = format!("compressed {blockCount} blocks");
        // S2 zone always starts at the oldest block (index 0).
        let _ = eventTx.send(SessionEvent::CompactionComplete {
            stage: stageStr.to_string(),
            reduction,
            markerBlock: Some(0),
        }).await;
        true
    }

    /// Run S3 topic compaction. Returns true if context was reduced.
    async fn runS3(
        &mut self,
        stageStr: &str,
        eventTx: &mpsc::Sender<SessionEvent>,
    ) -> bool {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        let s3Result = match crate::s3::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            self.topicTracker.topics(),
            &self.client,
            &self.config.utility.model,
            self.config.main.contextWindow,
            self.config.compactRatio,
        ).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S3 compaction failed: {e}");
                self.compactionTracker.markExhausted(compaction_trigger::StagePick::S3);
                return false;
            }
        };
        if !s3Result.didWork {
            self.compactionTracker.markExhausted(compaction_trigger::StagePick::S3);
            tracing::debug!("S3 exhausted \u{2014} no topics to compact");
            return false;
        }
        let afterTurn = self.transcript.currentBlock().to_string();
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
        match context::reconstruct(&self.transcript, &self.compactionLog, headId, self.config.main.promptThinking) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S3: {e}");
                return false;
            }
        }
        self.compactionTracker.clearExhaustion();
        let reduction = format!("compressed {topicCount} topics");
        let _ = eventTx.send(SessionEvent::CompactionComplete {
            stage: stageStr.to_string(),
            reduction,
            markerBlock: Some(0),
        }).await;
        true
    }

    /// Run S4 full compaction. Merges S3 topic summaries and any prior
    /// S4 briefings into a single handoff briefing. Returns true if
    /// context was reduced.
    async fn runS4Trigger(
        &mut self,
        stageStr: &str,
        eventTx: &mpsc::Sender<SessionEvent>,
    ) -> bool {
        let s4Result = match crate::s4::run(
            &self.compactionLog,
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

        if !s4Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S4);
            tracing::debug!("S4 exhausted \u{2014} no S3/S4 content to merge");
            return false;
        }

        let afterTurn = self.transcript.currentBlock().to_string();
        let blockCount = s4Result.sourceBlockIds.len();
        let summaryLen = s4Result.summary.len();
        if let Err(e) = self.compactionLog.recordFullCompact(
            &s4Result.summary,
            s4Result.sourceBlockIds,
            &afterTurn,
        ) {
            tracing::warn!("compaction log write failed: {e}");
        }

        // S4 covers everything — clear the edit gate entirely.
        self.filesRead.clear();

        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId, self.config.main.promptThinking) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S4: {e}");
                return false;
            }
        }

        self.compactionTracker.clearExhaustion();
        let reduction = format!(
            "merged {blockCount} source blocks into briefing ({summaryLen} chars)"
        );
        let _ = eventTx.send(SessionEvent::CompactionComplete {
            stage: stageStr.to_string(),
            reduction,
            markerBlock: Some(0),
        }).await;
        true
    }

    /// Execute a transcript-dependent tool (historyFetch, historySearch).
    fn executeTranscriptTool(&self, action: &tool::ToolAction) -> String {
        match action {
            tool::ToolAction::HistoryFetch { blockId } => {
                match self.transcript.loadAll() {
                    Ok(turns) => {
                        let blockTurns: Vec<_> = turns
                            .iter()
                            .filter(|t| t.blockId == *blockId)
                            .collect();

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
                            output.push('\n');
                        }
                        output
                    }
                    Err(e) => format!("Failed to load transcript: {e}"),
                }
            }
            tool::ToolAction::HistorySearch { query } => {
                match self.transcript.loadAll() {
                    Ok(turns) => {
                        let queryLower = query.to_lowercase();
                        let mut matches: Vec<(String, String, String)> = Vec::new();

                        for turn in &turns {
                            if turn.content.to_lowercase().contains(&queryLower) {
                                // Extract a snippet around the match.
                                let snippet = extractSnippet(&turn.content, &queryLower);
                                let roleLabel = match turn.role {
                                    crate::transcript::TurnRole::User => "user",
                                    crate::transcript::TurnRole::Assistant => "assistant",
                                    crate::transcript::TurnRole::ToolCall => "tool_call",
                                    crate::transcript::TurnRole::ToolResult => "tool_result",
                                    crate::transcript::TurnRole::System => "system",
                                };
                                matches.push((
                                    turn.blockId.clone(),
                                    format!("{} ({})", turn.id, roleLabel),
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
                        let mut output = format!("Found {totalMatches} matches for \"{query}\":\n\n");
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
        let contextBudget = self.config.main.contextWindow;
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
        if !mcpPrompt.is_empty() {
            if let Some(Message::System { content }) = self.history.first_mut() {
                content.push_str("\n\n");
                content.push_str(&mcpPrompt);
            }
        }

        self.mcpManager = Some(mgr);
    }

    /// Execute a subagent task.
    ///
    /// Spawns a child session with its own context, shell, and tool set,
    /// runs the task to completion, and returns the child's final text.
    async fn executeTask(
        &mut self,
        prompt: &str,
        agentType: &str,
        parentEventTx: &mpsc::Sender<SessionEvent>,
        parentCancelRx: &mut watch::Receiver<bool>,
    ) -> String {
        use crate::runner;

        let preset = runner::agentPreset(agentType);

        // Clone config. For cheap agents, swap to utility model config.
        let mut childConfig = self.config.clone();
        if preset.useUtilityModel {
            childConfig.main = childConfig.utility.clone();
        }

        // Spawn an isolated shell for the subagent.
        let (childShell, childIo) = match crate::shell::spawnShell(120, 40) {
            Ok(s) => s,
            Err(e) => return format!("Failed to spawn subagent shell: {e}"),
        };

        // Create child session with restricted tools.
        let mut child = match Session::new(
            &childConfig,
            preset.permissions,
            childShell,
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
        let shellForwardTx = parentEventTx.clone();
        let shellForwardId = childSessionId.clone();
        tokio::spawn(async move {
            let mut rx: tokio::sync::mpsc::Receiver<Vec<u8>> = childIo.outputRx;
            while let Some(data) = rx.recv().await {
                let _ = shellForwardTx
                    .send(SessionEvent::SubagentShellOutput {
                        sessionId: shellForwardId.clone(),
                        data,
                    })
                    .await;
            }
        });

        // Notify parent that subagent has started.
        let _ = parentEventTx
            .send(SessionEvent::SubagentStarted {
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
        let (childEventTx, mut childEventRx) = mpsc::channel::<SessionEvent>(256);
        let (childPermitTx, mut childPermitRx) =
            mpsc::channel::<crate::permissions::PermitResponse>(1);

        // Clone cancel receiver for the child — parent cancel propagates.
        let mut childCancelRx = parentCancelRx.clone();

        // Forwarding task: relay child events to parent, handle permission escalation.
        let forwardSessionId = childSessionId.clone();
        let forwardParentTx = parentEventTx.clone();
        let forwardHandle = tokio::spawn(async move {
            let mut content = String::new();
            let mut turns: usize = 0;
            while let Some(event) = childEventRx.recv().await {
                match &event {
                    SessionEvent::ContentDelta(text) => {
                        content.push_str(text);
                        let _ = forwardParentTx
                            .send(SessionEvent::SubagentEvent {
                                sessionId: forwardSessionId.clone(),
                                event: Box::new(event),
                            })
                            .await;
                    }
                    SessionEvent::TurnComplete => turns += 1,

                    // Permission escalation: forward to parent TUI and relay response.
                    SessionEvent::ToolRequest { name, summary, args, diff } => {
                        let (responseTx, mut responseRx) =
                            mpsc::channel::<crate::permissions::PermitResponse>(1);
                        let _ = forwardParentTx
                            .send(SessionEvent::SubagentPermitRequest {
                                sessionId: forwardSessionId.clone(),
                                name: name.clone(),
                                summary: summary.clone(),
                                args: args.clone(),
                                diff: diff.clone(),
                                responseTx,
                            })
                            .await;
                        // Wait for the parent TUI's response.
                        if let Some(response) = responseRx.recv().await {
                            let _ = childPermitTx.send(response).await;
                        } else {
                            // Parent closed — deny.
                            let _ = childPermitTx
                                .send(crate::permissions::PermitResponse::Deny)
                                .await;
                        }
                    }

                    // Forward other relevant events.
                    SessionEvent::ToolStarted { .. }
                    | SessionEvent::ToolAutoApproved { .. }
                    | SessionEvent::ToolResult { .. }
                    | SessionEvent::ToolDenied { .. }
                    | SessionEvent::Error(_) => {
                        let _ = forwardParentTx
                            .send(SessionEvent::SubagentEvent {
                                sessionId: forwardSessionId.clone(),
                                event: Box::new(event),
                            })
                            .await;
                    }
                    _ => {}
                }
            }
            (content, turns)
        });

        // Run the child session.
        let sendResult = child
            .send(prompt, &childEventTx, &mut childPermitRx, &mut childCancelRx)
            .await;

        // Drop sender so forwarding task exits.
        drop(childEventTx);

        let (content, turns) = match forwardHandle.await {
            Ok(r) => r,
            Err(e) => {
                let _ = parentEventTx
                    .send(SessionEvent::SubagentComplete {
                        sessionId: childSessionId.clone(),
                        agentType: agentType.into(),
                        content: String::new(),
                        turns: 0,
                    })
                    .await;
                return format!("Subagent forwarding failed: {e}");
            }
        };

        // Notify parent that subagent completed.
        let _ = parentEventTx
            .send(SessionEvent::SubagentComplete {
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

    /// Execute an MCP tool action.
    async fn executeMcpTool(&self, action: &tool::ToolAction) -> String {
        let mgr = match &self.mcpManager {
            Some(m) => m,
            None => return "MCP not configured.".into(),
        };

        match action {
            tool::ToolAction::Mcp { qualifiedName, args } => {
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

        self.lspManager.getDiagnosticsForTool(path, minSeverity, std::time::Duration::from_secs(15))
            .await
    }

    /// Gather structured MCP status data for the TUI panel.
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
        let searchMode = mgr.isSearchMode(self.config.main.contextWindow).await;

        let registry = mgr.registry().read().await;

        let servers: Vec<(String, String, usize, Vec<(String, String)>, String)> = statuses
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
    /// Remove the ephemeral resume notice from history (if present).
    /// Called at the end of `send()` so the injected System message doesn't
    /// persist across calls — it's re-injected at the start of the next
    /// `send()` if `self.resumeNotice` is still `Some`.
    fn removeResumeInjection(&mut self) {
        let notice = match &self.resumeNotice {
            Some(n) => n.clone(),
            None => return,
        };
        self.history.retain(|msg| {
            !matches!(msg, Message::System { content } if content == &notice)
        });
    }

    fn pushToolResult(&mut self, callId: &str, content: String) {
        match self.transcript.recordToolResult(callId, &content) {
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
            headTurn: self.headTurnId.clone(),
            forks: existingMeta.map(|m| m.forks).unwrap_or_default(),
        };
        if let Err(e) = self.transcript.writeMeta(&meta) {
            tracing::warn!("meta write failed: {e}");
        }
    }

    /// Execute a command action (from slash commands).
    ///
    /// Args:
    ///     action: The command to execute.
    ///
    /// Returns:
    ///     String: The result text to display to the user.
    pub async fn executeCommand(&mut self, action: &CommandAction) -> String {
        match action {
            CommandAction::ShowContext => {
                let state = context::buildState(
                    &self.history,
                    self.config.main.contextWindow,
                    self.config.compactRatio,
                    &self.compactionLog,
                );
                context::formatState(&state)
            }
            CommandAction::Undo => {
                match &self.checkpoint {
                    Some(cp) => match cp.undo().await {
                        Ok(turnId) => format!("Restored to checkpoint: {turnId}"),
                        Err(e) => format!("Undo failed: {e}"),
                    },
                    None => "Checkpoint system not initialized.".to_string(),
                }
            }
            CommandAction::Resume { sessionId } => {
                match sessionId {
                    Some(id) => {
                        // Return info about the session for the caller to act on.
                        match Transcript::loadMeta(
                            &crate::transcript::sessionsDir().join(id),
                        ) {
                            Ok(meta) => format!(
                                "Session {}: project={}, topics={}, updated={}",
                                id,
                                meta.projectDir,
                                meta.topicLabels.join(", "),
                                meta.updatedAt,
                            ),
                            Err(e) => format!("Failed to load session {id}: {e}"),
                        }
                    }
                    None => {
                        // List available sessions.
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
                }
            }
            // Handled by the spawned task before reaching executeCommand.
            CommandAction::Clear => "Session cleared.".to_string(),
            // Handled specially by the session task — shouldn't reach here.
            CommandAction::Mcp => "Use the /mcp command in the TUI.".to_string(),
            CommandAction::Lsp => "Use the /lsp command in the TUI.".to_string(),
            // Dispatched as special cases — need eventTx, so not handled here.
            CommandAction::Rewind { .. }
            | CommandAction::ForkAndRewind { .. }
            | CommandAction::Forks { .. } => {
                "Use /rewind or /forks in the TUI.".to_string()
            }
        }
    }

    /// Rewind conversation to a prior turn.
    ///
    /// If the user has sent messages on the current branch, the current
    /// state is saved as a fork before rewinding.
    pub async fn rewind(
        &mut self,
        targetTurnId: &str,
        saveFork: bool,
        eventTx: &mpsc::Sender<SessionEvent>,
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
        if let Ok(allTurns) = self.transcript.loadAll() {
            if let Some(turn) = allTurns.iter().find(|t| t.id == targetTurnId) {
                self.transcript.setHead(targetTurnId, &turn.blockId);
            }
        }

        // Rebuild history.
        match context::reconstruct(&self.transcript, &self.compactionLog, targetTurnId, self.config.main.promptThinking) {
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

        // Emit events for panel replay.
        let branchTurns = self.loadBranchTurns().unwrap_or_default();
        let markers = self.compactionMarkers();
        let _ = eventTx
            .send(SessionEvent::Rewound {
                targetTurnId: targetTurnId.to_string(),
            })
            .await;
        let _ = eventTx
            .send(SessionEvent::SessionRestored {
                turns: branchTurns,
                markers,
            })
            .await;

        format!("Rewound to {targetTurnId}")
    }

    /// Switch to a previously saved fork.
    pub async fn switchFork(
        &mut self,
        forkId: &str,
        eventTx: &mpsc::Sender<SessionEvent>,
    ) -> String {
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
        if let Ok(allTurns) = self.transcript.loadAll() {
            if let Some(turn) = allTurns.iter().find(|t| t.id == fork.headTurn) {
                self.transcript.setHead(&fork.headTurn, &turn.blockId);
            }
        }

        // Rebuild.
        match context::reconstruct(&self.transcript, &self.compactionLog, &fork.headTurn, self.config.main.promptThinking) {
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

        let branchTurns = self.loadBranchTurns().unwrap_or_default();
        let markers = self.compactionMarkers();
        let _ = eventTx
            .send(SessionEvent::Rewound {
                targetTurnId: fork.headTurn,
            })
            .await;
        let _ = eventTx
            .send(SessionEvent::SessionRestored {
                turns: branchTurns,
                markers,
            })
            .await;

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
                let first = t.content.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
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
            tool::ToolAction::EditFile { path, .. }
            | tool::ToolAction::MultiEdit { path, .. } => path,
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

/// Build an `Assistant` message for history. When `promptThinking` is true,
/// reasoning is baked into content as `<thinking>` blocks so the model sees
/// a consistent pattern across turns (instead of the `reasoning` JSON key).
fn buildAssistantMessage(
    content: Option<String>,
    toolCalls: Option<Vec<ToolCall>>,
    reasoning: Option<String>,
    promptThinking: bool,
) -> Message {
    if promptThinking {
        let merged = match (reasoning, content) {
            (Some(r), Some(c)) => Some(format!("<scratchpad>\n{r}\n</scratchpad>\n{c}")),
            (Some(r), None) => Some(format!("<scratchpad>\n{r}\n</scratchpad>")),
            (None, c) => c,
        };
        Message::Assistant {
            content: merged,
            tool_calls: toolCalls,
            reasoning: None,
        }
    } else {
        Message::Assistant {
            content,
            tool_calls: toolCalls,
            reasoning,
        }
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

    fn accumulate(
        &mut self,
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) {
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
        if let Some(name) = name {
            entry.name = name;
        }
        if let Some(args) = arguments {
            entry.arguments.push_str(&args);
        }
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

