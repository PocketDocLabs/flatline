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
use tokio::sync::mpsc;

use crate::api;
use crate::config::Config;
use crate::message::{
    FunctionCall, Message, ReasoningConfig, StreamEvent, ToolCall, ToolDef,
};
use crate::permissions::{PermitMode, Permissions, Verdict};
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
    pub fn new(config: &Config, permissions: Permissions, shell: Shell) -> Result<Self> {
        let client = api::Client::new(&config.api)?;
        let tools = tool::builtinDefs();

        let reasoning = config.api.reasoning.as_ref().map(|r| ReasoningConfig {
            effort: r.effort.clone(),
            summary: r.summary.clone(),
        });

        let systemPrompt = buildSystemPrompt();

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
    ) -> Result<()> {
        self.history.push(Message::User {
            content: userMessage.into(),
        });

        loop {
            let turnResult = self.streamOneTurn(eventTx).await?;

            match turnResult {
                TurnResult::Done => {
                    let _ = eventTx.send(SessionEvent::TurnComplete).await;
                    return Ok(());
                }
                TurnResult::ToolCalls(calls) => {
                    self.history.push(Message::Assistant {
                        content: None,
                        tool_calls: Some(calls.clone()),
                        reasoning: None,
                    });

                    let mut aborted = false;

                    for call in &calls {
                        let action =
                            tool::parse(&call.function.name, &call.function.arguments);
                        let summary = tool::summarize(&action);
                        let verdict = self.permissions.check(&action);

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

                                        // Wait for supervisor response.
                                        permitRx.recv().await.unwrap_or(false)
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
                            // Push denial for this call and stop processing.
                            self.history.push(Message::Tool {
                                tool_call_id: call.id.clone(),
                                content: "Turn aborted: tool call not permitted.".into(),
                            });
                            break;
                        }

                        let output = if approved {
                            let result = tool::execute(&action, &self.shell).await;
                            let _ = eventTx
                                .send(SessionEvent::ToolResult {
                                    name: call.function.name.clone(),
                                    output: result.clone(),
                                })
                                .await;
                            result
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
    ) -> Result<TurnResult> {
        let mut rx = self
            .client
            .stream(&self.history, &self.tools, self.reasoning.as_ref())
            .await?;

        let mut contentBuf = String::new();
        let mut reasoningBuf = String::new();
        let mut toolAccum = ToolCallAccumulator::new();

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::ContentDelta(text) => {
                    contentBuf.push_str(&text);
                    let _ = tx.send(SessionEvent::ContentDelta(text)).await;
                }
                StreamEvent::ReasoningDelta(text) => {
                    reasoningBuf.push_str(&text);
                    let _ = tx.send(SessionEvent::ReasoningDelta(text)).await;
                }
                StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                } => {
                    toolAccum.accumulate(index, id, name, arguments);
                }
                StreamEvent::Done { .. } => {
                    break;
                }
                StreamEvent::Error(msg) => {
                    let _ = tx.send(SessionEvent::Error(msg.clone())).await;
                    bail!("Stream error: {msg}");
                }
            }
        }

        let calls = toolAccum.finish();

        if !calls.is_empty() {
            Ok(TurnResult::ToolCalls(calls))
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
}

/// Outcome of a single API call.
enum TurnResult {
    Done,
    ToolCalls(Vec<ToolCall>),
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

fn buildSystemPrompt() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".into());

    // NOTE: Tool descriptions are sent via the native `tools` API parameter.
    // Don't duplicate them here — models work better with native tool schemas.
    format!(
        "You are Flatline, a general-purpose agent running in a terminal.\n\
         \n\
         Working directory: {cwd}\n\
         \n\
         Be direct and concise. Execute tasks rather than explaining how to do them. \
         When you need information, use your tools to get it rather than guessing."
    )
}
