#![allow(non_snake_case)]

//! Deck — TUI harness for flatline agents.
//!
//! Boots a ratatui TUI with an embedded terminal emulator
//! and agent interface.
//!
//! # Public API
//! Binary entry point only.
//!
//! # Dependencies
//! `ratatui`, `crossterm`, `tokio`, `alacritty_terminal`

mod agent_panel;
mod app;
mod command;
mod fork_picker;
mod history;
mod markdown;
mod lsp_panel;
mod mcp_panel;
mod rewind_picker;
mod selection;
mod session_picker;
mod subagent_panel;
mod terminal;
mod text_area;
mod throbber;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let subcommand = args.first().map(|s| s.as_str());

    match subcommand {
        Some("mcp-serve") => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            construct::mcp::serve::run().await
        }

        Some("exec") => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            let execArgs = parseExecArgs(&args[1..])?;
            let config = construct::config::load()?;
            let runConfig = construct::runner::RunConfig {
                maxTurns: execArgs.maxTurns,
                model: execArgs.model,
                systemPrompt: execArgs.systemPrompt,
                allowedTools: execArgs.allowedTools,
                tools: execArgs.tools,
                agent: execArgs.agent,
                mcpConfigPath: None, // TODO: --mcp-config flag
                strictMcp: execArgs.strictMcp,
                ephemeral: execArgs.ephemeral,
            };

            match execArgs.output.as_str() {
                "json" => {
                    // Buffer everything, print structured JSON at the end.
                    let result = construct::runner::run(
                        &config, &execArgs.prompt, &runConfig,
                    ).await?;
                    let out = serde_json::json!({
                        "sessionId": result.sessionId,
                        "content": result.content,
                        "turns": result.turns,
                        "maxTurnsHit": result.maxTurnsHit,
                    });
                    println!("{}", serde_json::to_string_pretty(&out)?);
                    if result.maxTurnsHit {
                        std::process::exit(2);
                    }
                }
                "events" => {
                    // Stream each event as NDJSON.
                    let (handle, mut eventRx) = construct::runner::runStreaming(
                        &config, &execArgs.prompt, &runConfig,
                    ).await?;
                    while let Some(event) = eventRx.recv().await {
                        let line = formatEventJson(&event);
                        println!("{line}");
                    }
                    handle.await??;
                }
                _ => {
                    // Text mode: stream content to stdout, tool activity to stderr.
                    use std::io::Write;
                    let (handle, mut eventRx) = construct::runner::runStreaming(
                        &config, &execArgs.prompt, &runConfig,
                    ).await?;
                    while let Some(event) = eventRx.recv().await {
                        use construct::session::SessionEvent;
                        match event {
                            SessionEvent::ContentDelta(text) => {
                                print!("{text}");
                                std::io::stdout().flush().ok();
                            }
                            SessionEvent::ToolStarted { name, summary } => {
                                eprint!("\x1b[2m\u{25b8} {name}: {summary}...\x1b[0m");
                            }
                            SessionEvent::ToolAutoApproved { name, summary } => {
                                eprintln!("\x1b[2m\u{25b8} {name}: {summary}\x1b[0m");
                            }
                            SessionEvent::ToolResult { name, .. } => {
                                // Clear the "started" line and show completion.
                                eprint!("\r\x1b[K");
                                tracing::debug!(tool = %name, "tool result");
                            }
                            SessionEvent::Error(msg) => {
                                eprintln!("\x1b[31merror: {msg}\x1b[0m");
                            }
                            _ => {}
                        }
                    }
                    // Ensure final newline.
                    println!();
                    handle.await??;
                }
            }
            Ok(())
        }

        // Legacy support for --serve.
        Some("--serve") => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            construct::mcp::serve::run().await
        }

        _ => {
            // TUI mode: file-based logging so it doesn't collide with the TUI.
            let logDir = construct::config::configDir();
            std::fs::create_dir_all(&logDir)?;
            let logFile = std::fs::File::create(logDir.join("flatline.log"))?;
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug"));
            tracing_subscriber::fmt()
                .with_writer(logFile)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            app::run().await
        }
    }
}

struct ExecArgs {
    prompt: String,
    output: String,
    maxTurns: usize,
    model: Option<String>,
    systemPrompt: Option<construct::runner::SystemPromptOverride>,
    allowedTools: Option<Vec<String>>,
    tools: Option<Vec<String>>,
    agent: Option<String>,
    strictMcp: bool,
    ephemeral: bool,
}

fn parseExecArgs(args: &[String]) -> Result<ExecArgs> {
    let mut prompt: Option<String> = None;
    let mut output = "text".to_string();
    let mut maxTurns: usize = 50;
    let mut model: Option<String> = None;
    let mut systemPrompt: Option<construct::runner::SystemPromptOverride> = None;
    let mut allowedTools: Option<Vec<String>> = None;
    let mut tools: Option<Vec<String>> = None;
    let mut agent: Option<String> = None;
    let mut strictMcp = false;
    let mut ephemeral = false;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = args[i].clone();
                } else {
                    anyhow::bail!("--output requires a value (text, json, events)");
                }
            }
            "--max-turns" => {
                i += 1;
                if i < args.len() {
                    maxTurns = args[i].parse().map_err(|_| {
                        anyhow::anyhow!("--max-turns requires a number")
                    })?;
                } else {
                    anyhow::bail!("--max-turns requires a value");
                }
            }
            "--model" => {
                i += 1;
                if i < args.len() {
                    model = Some(args[i].clone());
                } else {
                    anyhow::bail!("--model requires a value");
                }
            }
            "--system-prompt" => {
                i += 1;
                if i < args.len() {
                    systemPrompt = Some(construct::runner::SystemPromptOverride::Replace(args[i].clone()));
                } else {
                    anyhow::bail!("--system-prompt requires a value");
                }
            }
            "--system-prompt-file" => {
                i += 1;
                if i < args.len() {
                    let content = std::fs::read_to_string(&args[i])
                        .map_err(|e| anyhow::anyhow!("failed to read system prompt file: {e}"))?;
                    systemPrompt = Some(construct::runner::SystemPromptOverride::Replace(content));
                } else {
                    anyhow::bail!("--system-prompt-file requires a path");
                }
            }
            "--append-system-prompt" => {
                i += 1;
                if i < args.len() {
                    systemPrompt = Some(construct::runner::SystemPromptOverride::Append(args[i].clone()));
                } else {
                    anyhow::bail!("--append-system-prompt requires a value");
                }
            }
            "--allowed-tools" => {
                i += 1;
                if i < args.len() {
                    allowedTools = Some(
                        args[i].split(',').map(|s| s.trim().to_string()).collect()
                    );
                } else {
                    anyhow::bail!("--allowed-tools requires comma-separated tool names");
                }
            }
            "--tools" => {
                i += 1;
                if i < args.len() {
                    if args[i].is_empty() {
                        tools = Some(Vec::new());
                    } else {
                        tools = Some(
                            args[i].split(',').map(|s| s.trim().to_string()).collect()
                        );
                    }
                } else {
                    anyhow::bail!("--tools requires comma-separated tool names (or empty string)");
                }
            }
            "--agent" => {
                i += 1;
                if i < args.len() {
                    agent = Some(args[i].clone());
                } else {
                    anyhow::bail!("--agent requires a name");
                }
            }
            "--strict-mcp" => { strictMcp = true; }
            "--ephemeral" => { ephemeral = true; }
            arg if !arg.starts_with('-') && prompt.is_none() => {
                prompt = Some(arg.to_string());
            }
            other => {
                anyhow::bail!("unknown flag: {other}");
            }
        }
        i += 1;
    }

    // If no prompt arg, try reading stdin.
    let prompt = match prompt {
        Some(p) => p,
        None => {
            use std::io::IsTerminal;
            if std::io::stdin().is_terminal() {
                anyhow::bail!("usage: flatline exec \"prompt\" [flags]\n\nFlags:\n  --output text|json|events\n  --max-turns N\n  --model MODEL\n  --system-prompt TEXT\n  --system-prompt-file PATH\n  --append-system-prompt TEXT\n  --allowed-tools TOOLS\n  --tools TOOLS\n  --agent NAME\n  --strict-mcp\n  --ephemeral");
            }
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            if buf.trim().is_empty() {
                anyhow::bail!("no prompt provided");
            }
            buf
        }
    };

    Ok(ExecArgs {
        prompt, output, maxTurns, model, systemPrompt,
        allowedTools, tools, agent, strictMcp, ephemeral,
    })
}

/// Serialize a SessionEvent to a JSON line for NDJSON output.
fn formatEventJson(event: &construct::session::SessionEvent) -> String {
    use construct::session::SessionEvent;
    let val = match event {
        SessionEvent::ContentDelta(text) => serde_json::json!({
            "type": "contentDelta", "text": text,
        }),
        SessionEvent::ReasoningDelta(text) => serde_json::json!({
            "type": "reasoningDelta", "text": text,
        }),
        SessionEvent::ToolRequest { name, summary, args, diff } => serde_json::json!({
            "type": "toolRequest", "name": name, "summary": summary,
            "args": args, "diff": diff,
        }),
        SessionEvent::ToolStarted { name, summary } => serde_json::json!({
            "type": "toolStarted", "name": name, "summary": summary,
        }),
        SessionEvent::ToolAutoApproved { name, summary } => serde_json::json!({
            "type": "toolAutoApproved", "name": name, "summary": summary,
        }),
        SessionEvent::ToolResult { name, output } => serde_json::json!({
            "type": "toolResult", "name": name, "output": output,
        }),
        SessionEvent::ToolDenied { name } => serde_json::json!({
            "type": "toolDenied", "name": name,
        }),
        SessionEvent::TurnAborted { name } => serde_json::json!({
            "type": "turnAborted", "name": name,
        }),
        SessionEvent::TurnComplete => serde_json::json!({
            "type": "turnComplete",
        }),
        SessionEvent::TurnCancelled => serde_json::json!({
            "type": "turnCancelled",
        }),
        SessionEvent::TokenUpdate { promptTokens, completionTokens, contextTokens } => {
            serde_json::json!({
                "type": "tokenUpdate",
                "promptTokens": promptTokens,
                "completionTokens": completionTokens,
                "contextTokens": contextTokens,
            })
        }
        SessionEvent::CompactionStarted { stage } => serde_json::json!({
            "type": "compactionStarted", "stage": stage,
        }),
        SessionEvent::CompactionComplete { stage, reduction, markerBlock } => serde_json::json!({
            "type": "compactionComplete", "stage": stage,
            "reduction": reduction, "markerBlock": markerBlock,
        }),
        SessionEvent::SubagentStarted { sessionId, agentType, prompt } => serde_json::json!({
            "type": "subagentStarted", "sessionId": sessionId,
            "agentType": agentType, "prompt": prompt,
        }),
        SessionEvent::SubagentEvent { sessionId, event } => serde_json::json!({
            "type": "subagentEvent", "sessionId": sessionId,
            "event": formatEventJson(event),
        }),
        SessionEvent::SubagentShellOutput { sessionId, .. } => serde_json::json!({
            "type": "subagentShellOutput", "sessionId": sessionId,
        }),
        SessionEvent::SubagentPermitRequest { sessionId, name, summary, .. } => serde_json::json!({
            "type": "subagentPermitRequest", "sessionId": sessionId,
            "name": name, "summary": summary,
        }),
        SessionEvent::SubagentComplete { sessionId, agentType, content, turns } => serde_json::json!({
            "type": "subagentComplete", "sessionId": sessionId,
            "agentType": agentType, "content": content, "turns": turns,
        }),
        SessionEvent::Error(msg) => serde_json::json!({
            "type": "error", "message": msg,
        }),
        // TUI-specific events — emit type only for completeness.
        _ => serde_json::json!({ "type": "other" }),
    };
    serde_json::to_string(&val).unwrap_or_else(|_| r#"{"type":"serializationError"}"#.into())
}
