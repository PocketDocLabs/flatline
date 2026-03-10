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

use anyhow::{Result, bail};
use tokio::sync::{mpsc, watch};

use crate::api;
use crate::config::Config;
use crate::message::{
    FunctionCall, Message, ReasoningConfig, StreamEvent, ToolCall, ToolDef,
};
use crate::permissions::{PermitMode, Permissions, Verdict};
use crate::prompt::{self, DomainModule, InterfaceMode};
use crate::shell::Shell;
use crate::tool;

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
    },

    /// A tool was auto-approved by the permission config.
    ToolAutoApproved { name: String, summary: String },

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

    /// An error occurred.
    Error(String),
}

/// Agent session — owns the conversation and drives the turn loop.
pub struct Session {
    client: api::Client,
    history: Vec<Message>,
    tools: Vec<ToolDef>,
    reasoning: Option<ReasoningConfig>,
    permissions: Permissions,
    shell: Shell,
}

impl Session {
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
        let client = api::Client::new(&config.api)?;
        let tools = tool::builtinDefs();

        let reasoning = config.api.reasoning.as_ref().map(|r| ReasoningConfig {
            effort: r.effort.clone(),
            summary: r.summary.clone(),
        });

        let systemPrompt = prompt::build(interface, domains);

        let history = vec![Message::System {
            content: systemPrompt,
        }];

        Ok(Self {
            client,
            history,
            tools,
            reasoning,
            permissions,
            shell,
        })
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
    ///     permitRx: Channel for permission responses (true = approve).
    pub async fn send(
        &mut self,
        userMessage: &str,
        eventTx: &mpsc::Sender<SessionEvent>,
        permitRx: &mut mpsc::Receiver<bool>,
        cancelRx: &mut watch::Receiver<bool>,
    ) -> Result<()> {
        tracing::info!(len = userMessage.len(), "user message received");
        self.history.push(Message::User {
            content: userMessage.into(),
        });

        loop {
            // Check for cancellation between turns.
            if *cancelRx.borrow() {
                tracing::info!("turn cancelled before streaming");
                let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                return Ok(());
            }

            tracing::debug!(historyLen = self.history.len(), "starting turn");
            let turnResult = self.streamOneTurn(eventTx, cancelRx).await?;

            match turnResult {
                TurnResult::Done => {
                    tracing::info!("turn complete (no tool calls)");
                    let _ = eventTx.send(SessionEvent::TurnComplete).await;
                    return Ok(());
                }
                TurnResult::Cancelled => {
                    tracing::info!("turn cancelled during streaming");
                    let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                    return Ok(());
                }
                TurnResult::ToolCalls { calls, reasoning } => {
                    tracing::info!(
                        callCount = calls.len(),
                        hasReasoning = reasoning.is_some(),
                        "turn produced tool calls"
                    );
                    self.history.push(Message::Assistant {
                        content: None,
                        tool_calls: Some(calls.clone()),
                        reasoning,
                    });

                    let mut aborted = false;

                    for (callIdx, call) in calls.iter().enumerate() {
                        // Check for cancellation between tool calls.
                        if *cancelRx.borrow() {
                            tracing::info!("cancelled between tool calls");
                            for remaining in &calls[callIdx..] {
                                self.history.push(Message::Tool {
                                    tool_call_id: remaining.id.clone(),
                                    content: "Cancelled by user.".into(),
                                });
                            }
                            let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                            return Ok(());
                        }

                        let action =
                            tool::parse(&call.function.name, &call.function.arguments);
                        let summary = tool::summarize(&action);
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
                                        let _ = eventTx
                                            .send(SessionEvent::ToolRequest {
                                                name: call.function.name.clone(),
                                                summary,
                                                args: call.function.arguments.clone(),
                                            })
                                            .await;

                                        // Wait for supervisor response or cancellation.
                                        tokio::select! {
                                            permit = permitRx.recv() => {
                                                permit.unwrap_or(false)
                                            }
                                            _ = cancelRx.changed() => {
                                                tracing::info!("cancelled during permission wait");
                                                for remaining in &calls[callIdx..] {
                                                    self.history.push(Message::Tool {
                                                        tool_call_id: remaining.id.clone(),
                                                        content: "Cancelled by user.".into(),
                                                    });
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

                        if aborted {
                            self.history.push(Message::Tool {
                                tool_call_id: call.id.clone(),
                                content: "Turn aborted: tool call not permitted.".into(),
                            });
                            break;
                        }

                        // Guard: editFile/writeFile require a prior readFile of the same path.
                        if approved {
                            if let Some(rejection) = self.checkReadBeforeWrite(&action) {
                                tracing::info!(
                                    tool = %call.function.name,
                                    "rejected: file not read first"
                                );
                                self.history.push(Message::Tool {
                                    tool_call_id: call.id.clone(),
                                    content: rejection,
                                });
                                continue;
                            }
                        }

                        let output = if approved {
                            tracing::info!(tool = %call.function.name, "executing tool");

                            // Race tool execution against cancellation for shell commands.
                            if matches!(action, tool::ToolAction::Shell { .. }) {
                                tokio::select! {
                                    result = tool::execute(&action, &self.shell) => {
                                        tracing::debug!(
                                            tool = %call.function.name,
                                            outputLen = result.len(),
                                            "tool execution complete"
                                        );
                                        let _ = eventTx
                                            .send(SessionEvent::ToolResult {
                                                name: call.function.name.clone(),
                                                output: result.clone(),
                                            })
                                            .await;
                                        result
                                    }
                                    _ = cancelRx.changed() => {
                                        tracing::info!(tool = %call.function.name, "cancelled during shell execution");
                                        self.shell.interrupt();
                                        self.history.push(Message::Tool {
                                            tool_call_id: call.id.clone(),
                                            content: "Cancelled by user.".into(),
                                        });
                                        for remaining in &calls[callIdx + 1..] {
                                            self.history.push(Message::Tool {
                                                tool_call_id: remaining.id.clone(),
                                                content: "Cancelled by user.".into(),
                                            });
                                        }
                                        let _ = eventTx.send(SessionEvent::TurnCancelled).await;
                                        return Ok(());
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
                                let _ = eventTx
                                    .send(SessionEvent::ToolResult {
                                        name: call.function.name.clone(),
                                        output: result.clone(),
                                    })
                                    .await;
                                result
                            }
                        } else {
                            let _ = eventTx
                                .send(SessionEvent::ToolDenied {
                                    name: call.function.name.clone(),
                                })
                                .await;
                            "User denied this action.".into()
                        };

                        self.history.push(Message::Tool {
                            tool_call_id: call.id.clone(),
                            content: output,
                        });
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
        let mut rx = self
            .client
            .stream(&self.history, &self.tools, self.reasoning.as_ref())
            .await?;

        let mut contentBuf = String::new();
        let mut reasoningBuf = String::new();
        let mut toolAccum = ToolCallAccumulator::new();

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(StreamEvent::ContentDelta(text)) => {
                            contentBuf.push_str(&text);
                            let _ = tx.send(SessionEvent::ContentDelta(text)).await;
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
                        Some(StreamEvent::Done { .. }) | None => {
                            break;
                        }
                        Some(StreamEvent::Error(msg)) => {
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
                            let content = if contentBuf.is_empty() { None } else { Some(contentBuf) };
                            let reasoning = if reasoningBuf.is_empty() { None } else { Some(reasoningBuf) };
                            self.history.push(Message::Assistant {
                                content,
                                tool_calls: None,
                                reasoning,
                            });
                        }
                        return Ok(TurnResult::Cancelled);
                    }
                }
            }
        }

        let calls = toolAccum.finish();

        tracing::debug!(
            contentLen = contentBuf.len(),
            reasoningLen = reasoningBuf.len(),
            toolCalls = calls.len(),
            "turn stream complete"
        );

        if !calls.is_empty() {
            let reasoning = if reasoningBuf.is_empty() {
                None
            } else {
                Some(reasoningBuf)
            };
            Ok(TurnResult::ToolCalls { calls, reasoning })
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

            self.history.push(Message::Assistant {
                content,
                tool_calls: None,
                reasoning,
            });

            Ok(TurnResult::Done)
        }
    }

    /// Check if a write/edit action targets a file that was previously read.
    /// Returns an error message if the file hasn't been read, None if OK.
    fn checkReadBeforeWrite(&self, action: &tool::ToolAction) -> Option<String> {
        let targetPath = match action {
            tool::ToolAction::EditFile { path, .. } => path,
            tool::ToolAction::WriteFile { path, .. } => {
                // writeFile to a new file is fine — no read needed.
                if !std::path::Path::new(path).exists() {
                    return None;
                }
                path
            }
            _ => return None,
        };

        // Normalize to absolute for comparison.
        let targetNorm = normalizePath(targetPath);

        // Scan history for a readFile tool call with a matching path.
        for msg in &self.history {
            if let Message::Assistant { tool_calls: Some(calls), .. } = msg {
                for call in calls {
                    if call.function.name == "readFile" {
                        if let Ok(args) = serde_json::from_str::<serde_json::Value>(
                            &call.function.arguments,
                        ) {
                            if let Some(readPath) = args["path"].as_str() {
                                if normalizePath(readPath) == targetNorm {
                                    return None;
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(format!(
            "You must read \"{targetPath}\" with readFile before editing or overwriting it."
        ))
    }
}

/// Best-effort path normalization for read-before-write comparison.
fn normalizePath(path: &str) -> String {
    std::path::Path::new(path)
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string())
}

/// Outcome of a single API call.
enum TurnResult {
    Done,
    ToolCalls {
        calls: Vec<ToolCall>,
        reasoning: Option<String>,
    },
    Cancelled,
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

