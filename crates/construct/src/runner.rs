#![allow(non_snake_case)]

//! Headless execution engine.
//!
//! Shared runner for `flatline exec` CLI mode and subagent `task` tool execution.
//! Creates a session, runs the agentic turn loop to completion, and returns the result.
//!
//! # Public API
//! - [`RunConfig`] — execution parameters
//! - [`RunResult`] — outcome of a headless run
//! - [`run`] — create a fresh session and run to completion
//! - [`runSession`] — run the turn loop on an existing session
//!
//! # Dependencies
//! `session`, `config`, `permissions`, `shell`, `prompt`

use anyhow::Result;
use tokio::sync::{mpsc, watch};

use crate::config::Config;
use crate::control::{LogEvent, SessionRequest};
use crate::permissions::Permissions;
use crate::prompt::{DomainModule, InterfaceMode};
use crate::session::Session;
use crate::shell;

use crate::tool::ToolSet;

/// Spawn a task that auto-denies every permit request arriving on the given
/// channel. Used by headless runners with no human in the loop — tools that
/// hit `NeedsApproval` under the configured permissions are rejected.
fn spawnPermitAutoDeny(mut rx: mpsc::Receiver<SessionRequest>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            match req {
                SessionRequest::Permit { reply, .. } => {
                    let _ = reply.send(crate::permissions::PermitResponse::Deny);
                }
            }
        }
    })
}

/// Which tier a subagent runs on. Drives which profile its session uses in
/// `executeTask` — the tier's `ModelConfig` gets swapped into the child's
/// `heavy` slot so Client code stays tier-agnostic.
#[derive(Debug, Clone, Copy)]
pub enum AgentTier {
    Heavy,
    Light,
    Utility,
}

/// Agent preset for subagent execution.
pub struct AgentPreset {
    pub toolSet: ToolSet,
    pub permissions: Permissions,
    pub interface: InterfaceMode,
    pub maxTurns: usize,
    /// Which tier this agent runs on.
    pub tier: AgentTier,
    /// Appended to the system prompt for role-specific instructions.
    pub systemSuffix: &'static str,
}

/// Get the preset for a named agent type.
pub fn agentPreset(name: &str) -> AgentPreset {
    match name {
        "explore" => AgentPreset {
            toolSet: ToolSet::ReadOnly,
            permissions: Permissions::allowReadOnly(),
            interface: InterfaceMode::MultiAgent,
            maxTurns: 20,
            tier: AgentTier::Utility,
            systemSuffix: "You are an exploration agent. \
                Investigate the codebase and report findings. \
                You cannot modify files — use read-only tools only.",
        },
        _ => AgentPreset {
            toolSet: ToolSet::All,
            permissions: Permissions::askForEverything(),
            interface: InterfaceMode::MultiAgent,
            maxTurns: 50,
            tier: AgentTier::Heavy,
            systemSuffix: "You are a subtask agent. \
                Complete the assigned task and return your result. \
                Be thorough but focused — do the work, report what you did.",
        },
    }
}

/// How to override the system prompt.
pub enum SystemPromptOverride {
    /// Replace the entire system prompt.
    Replace(String),
    /// Append to the default system prompt.
    Append(String),
}

/// Execution parameters for a headless run.
pub struct RunConfig {
    pub maxTurns: usize,
    /// Profile name for the heavy tier (overrides `heavyProfile`). Applied
    /// before `load()` via the `FLATLINE_HEAVY_PROFILE` env var by the caller.
    pub heavyProfile: Option<String>,

    /// Profile name for the light tier (overrides `lightProfile`).
    /// Defaults to heavy when unset.
    pub lightProfile: Option<String>,

    /// Profile name for the utility tier (overrides `utilityProfile`).
    /// Defaults to light when unset.
    pub utilityProfile: Option<String>,
    pub model: Option<String>,
    pub systemPrompt: Option<SystemPromptOverride>,
    pub allowedTools: Option<Vec<String>>,
    pub tools: Option<Vec<String>>,
    pub agent: Option<String>,
    pub mcpConfigPath: Option<std::path::PathBuf>,
    pub strictMcp: bool,
    pub ephemeral: bool,
    /// Maximum budget in USD (hard stop).
    pub maxBudgetUsd: Option<f64>,
}

/// Outcome of a headless run.
pub struct RunResult {
    /// Final assistant text content.
    pub content: String,
    /// Session ID (for transcript lookup).
    pub sessionId: String,
    /// Number of agentic turns executed.
    pub turns: usize,
    /// Whether the run hit the max turns limit.
    pub maxTurnsHit: bool,
}

/// Create a fresh headless session and run to completion.
///
/// Args:
///     config: Application config.
///     prompt: User prompt to execute.
///     runConfig: Execution parameters.
pub async fn run(config: &Config, prompt: &str, runConfig: &RunConfig) -> Result<RunResult> {
    let (agentShell, shellIo) = shell::spawnShell(120, 40)?;

    // Drain PTY output in the background — headless mode has no terminal display.
    tokio::spawn(async move {
        let mut rx = shellIo.outputRx;
        while rx.recv().await.is_some() {}
    });

    // Apply model override. Clear provider routing when switching models
    // since pinned providers are model-specific.
    let mut config = config.clone();
    if let Some(ref model) = runConfig.model {
        config.heavy.model = model.clone();
        config.heavy.providerOrder = Vec::new();
    }

    // Build permissions based on --allowed-tools, config, or defaults.
    let permissions = if let Some(ref allowed) = runConfig.allowedTools {
        let mut perms = Permissions::askForEverything();
        for toolName in allowed {
            perms.addRule(crate::permissions::Rule {
                tool: toolName.clone(),
                pattern: None,
                allow: true,
            });
        }
        perms
    } else if let Some(ref perms) = config.permissions {
        perms.clone()
    } else {
        // Headless default: ask for everything. With no TUI to answer,
        // unapproved tools are denied. User must use --allowed-tools.
        Permissions::askForEverything()
    };

    let mut session = Session::new(
        &config,
        permissions,
        agentShell,
        InterfaceMode::Headless,
        &[DomainModule::Swe],
    )?;

    // Apply hard budget limit.
    if let Some(limit) = runConfig.maxBudgetUsd {
        session.setMaxBudget(limit);
    }

    // Apply tool restriction (--tools flag).
    if let Some(ref toolNames) = runConfig.tools {
        let allDefs = crate::tool::builtinDefs();
        let filtered: Vec<_> = allDefs
            .into_iter()
            .filter(|d| toolNames.contains(&d.function.name))
            .collect();
        session.setTools(filtered);
    }

    // Initialize MCP servers from .mcp.json files.
    if !runConfig.strictMcp {
        match crate::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
            Ok(servers) if !servers.is_empty() => {
                session.initMcp(servers).await;
            }
            Err(e) => tracing::warn!("failed to load MCP config: {e}"),
            _ => {}
        }
    }

    runSession(&mut session, prompt, runConfig.maxTurns).await
}

/// Create a fresh headless session and return the event receiver for streaming output.
///
/// The session turn loop runs in a spawned task. The caller drains the event
/// receiver for output formatting while the task runs concurrently.
///
/// Args:
///     config: Application config.
///     prompt: User prompt to execute.
///     runConfig: Execution parameters.
pub async fn runStreaming(
    config: &Config,
    prompt: &str,
    runConfig: &RunConfig,
) -> Result<(tokio::task::JoinHandle<Result<RunResult>>, mpsc::Receiver<LogEvent>)> {
    let (agentShell, shellIo) = shell::spawnShell(120, 40)?;

    // Drain PTY output in the background.
    tokio::spawn(async move {
        let mut rx = shellIo.outputRx;
        while rx.recv().await.is_some() {}
    });

    let permissions = Permissions::allowAll();
    let mut session = Session::new(
        config,
        permissions,
        agentShell,
        InterfaceMode::Headless,
        &[DomainModule::Swe],
    )?;

    // Apply hard budget limit.
    if let Some(limit) = runConfig.maxBudgetUsd {
        session.setMaxBudget(limit);
    }

    match crate::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
        Ok(servers) if !servers.is_empty() => {
            session.initMcp(servers).await;
        }
        Err(e) => tracing::warn!("failed to load MCP config: {e}"),
        _ => {}
    }

    let sessionId = session.sessionId().to_string();
    let _maxTurns = runConfig.maxTurns;
    let input = crate::session::UserInput::from(prompt.to_string());

    let (logTx, logRx) = mpsc::channel::<LogEvent>(256);
    let (sessionRequestTx, sessionRequestRx) = mpsc::channel::<SessionRequest>(16);
    let (cancelTx, cancelRx) = watch::channel(false);
    let mut cancelRx = cancelRx;

    // Headless: auto-deny any permit request.
    let _permitAutoDeny = spawnPermitAutoDeny(sessionRequestRx);

    let handle = tokio::spawn(async move {
        // Keep the sender alive so cancelRx.changed() doesn't resolve immediately.
        let _cancel = cancelTx;

        // Subagents don't support mid-turn steering.
        let (_steerTx, mut steerRx) = mpsc::channel::<crate::session::UserInput>(1);

        let sendResult = session
            .send(&input, &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx)
            .await;

        // Shut down background LSP/MCP cleanly so their senders drop before
        // the tokio runtime unwinds.
        session.shutdownLsp().await;
        session.shutdownMcp().await;

        // Drop senders so the receivers know we're done.
        drop(logTx);
        drop(sessionRequestTx);

        sendResult?;

        Ok(RunResult {
            content: String::new(), // Caller accumulates from events.
            sessionId,
            turns: 0,
            maxTurnsHit: false,
        })
    });

    Ok((handle, logRx))
}

/// Run the turn loop on an existing session.
///
/// Used by both the CLI `exec` command and subagent `task` tool execution.
/// Calls `session.send()` which handles the full agentic loop internally
/// (streams one API response, executes tool calls, loops until the model
/// produces a text-only response).
///
/// Args:
///     session: An initialized session.
///     prompt: User prompt to send.
///     maxTurns: Safety limit (currently one send() = one full agentic run).
pub fn runSession<'a>(
    session: &'a mut Session,
    prompt: &'a str,
    _maxTurns: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<RunResult>> + Send + 'a>> {
    Box::pin(async move {
    let (logTx, mut logRx) = mpsc::channel::<LogEvent>(256);
    let (sessionRequestTx, sessionRequestRx) = mpsc::channel::<SessionRequest>(16);
    let (_cancelTx, cancelRx) = watch::channel(false);
    let mut cancelRx = cancelRx;

    let sessionId = session.sessionId().to_string();

    // Headless: auto-deny any permit request.
    let _permitAutoDeny = spawnPermitAutoDeny(sessionRequestRx);

    // Spawn log drain in background. session.send() emits events as it runs,
    // so we need to consume them concurrently to avoid blocking on a full channel.
    let drainHandle = tokio::spawn(async move {
        let mut content = String::new();
        let mut turns: usize = 0;
        while let Some(event) = logRx.recv().await {
            match event {
                LogEvent::ContentDelta(text) => content.push_str(&text),
                LogEvent::TurnComplete => turns += 1,
                LogEvent::ToolStarted { name, summary } => {
                    tracing::info!(tool = %name, summary = %summary, "tool started");
                }
                LogEvent::ToolAutoApproved { name, summary } => {
                    tracing::info!(tool = %name, summary = %summary, "tool auto-approved");
                }
                LogEvent::ToolResult { name, .. } => {
                    tracing::debug!(tool = %name, "tool result received");
                }
                LogEvent::Error(msg) => tracing::error!("session error: {msg}"),
                _ => {}
            }
        }
        (content, turns)
    });

    // Run the agentic turn loop.
    let input = crate::session::UserInput::from(prompt.to_string());
    // Headless runners don't support mid-turn steering.
    let (_steerTx, mut steerRx) = mpsc::channel::<crate::session::UserInput>(1);

    let sendResult = session
        .send(&input, &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx)
        .await;

    // Shut down background LSP/MCP cleanly so their senders drop before
    // the tokio runtime unwinds (async-lsp panics if servers outlive the
    // runtime shutdown).
    session.shutdownLsp().await;
    session.shutdownMcp().await;

    // Drop senders so drain tasks exit.
    drop(logTx);
    drop(sessionRequestTx);

    let (content, turns) = drainHandle.await?;

    // Propagate errors from the turn loop.
    sendResult?;

    Ok(RunResult {
        content,
        sessionId,
        turns,
        maxTurnsHit: false,
    })
    })
}
