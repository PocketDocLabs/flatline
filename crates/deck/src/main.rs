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
mod permissions_panel;
mod rewind_picker;
mod selection;
mod session_picker;
mod subagent_panel;
mod terminal;
mod text_area;
mod throbber;

use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "flatline", version, about = "General-purpose agentic terminal tool")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a prompt headlessly.
    Exec(ExecArgs),

    /// Start MCP server.
    McpServe,

    /// Generate shell completions.
    #[command(hide = true)]
    Completions {
        /// Shell to generate for (bash, zsh, fish, powershell, elvish).
        shell: clap_complete::Shell,
    },
}

#[derive(Args)]
struct ExecArgs {
    /// The prompt to execute (reads stdin if omitted).
    prompt: Option<String>,

    /// Output format.
    #[arg(long, default_value = "text", value_parser = ["text", "json", "events"])]
    output: String,

    /// Maximum agent turns.
    #[arg(long, default_value_t = 50)]
    maxTurns: usize,

    /// Model override.
    #[arg(long)]
    model: Option<String>,

    /// Replace the system prompt.
    #[arg(long, group = "sysprompt")]
    systemPrompt: Option<String>,

    /// Read system prompt from file.
    #[arg(long, group = "sysprompt")]
    systemPromptFile: Option<std::path::PathBuf>,

    /// Append to the default system prompt.
    #[arg(long)]
    appendSystemPrompt: Option<String>,

    /// Comma-separated list of allowed tools.
    #[arg(long, value_delimiter = ',')]
    allowedTools: Option<Vec<String>>,

    /// Comma-separated list of tools (empty string = none).
    #[arg(long, value_delimiter = ',')]
    tools: Option<Vec<String>>,

    /// Agent name.
    #[arg(long)]
    agent: Option<String>,

    /// Strict MCP mode.
    #[arg(long)]
    strictMcp: bool,

    /// Ephemeral session.
    #[arg(long)]
    ephemeral: bool,

    /// Maximum budget in USD (hard stop).
    #[arg(long)]
    maxBudgetUsd: Option<f64>,
}

/// Resolve ExecArgs into a prompt string and RunConfig.
fn resolveExecArgs(args: ExecArgs) -> Result<(String, construct::runner::RunConfig)> {
    // Resolve prompt: positional arg or stdin fallback.
    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            use std::io::IsTerminal;
            if std::io::stdin().is_terminal() {
                Cli::command()
                    .find_subcommand_mut("exec")
                    .expect("exec subcommand exists")
                    .print_help()?;
                std::process::exit(2);
            }
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            if buf.trim().is_empty() {
                anyhow::bail!("no prompt provided");
            }
            buf
        }
    };

    // Resolve system prompt override.
    let systemPrompt = if let Some(text) = args.systemPrompt {
        Some(construct::runner::SystemPromptOverride::Replace(text))
    } else if let Some(path) = args.systemPromptFile {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("failed to read system prompt file: {e}"))?;
        Some(construct::runner::SystemPromptOverride::Replace(content))
    } else {
        args.appendSystemPrompt.map(construct::runner::SystemPromptOverride::Append)
    };

    // Filter empty strings from tools (handles `--tools ""`).
    let tools = args.tools.map(|v| {
        let filtered: Vec<String> = v.into_iter().filter(|s| !s.is_empty()).collect();
        filtered
    });

    let config = construct::runner::RunConfig {
        maxTurns: args.maxTurns,
        model: args.model,
        systemPrompt,
        allowedTools: args.allowedTools,
        tools,
        agent: args.agent,
        mcpConfigPath: None,
        strictMcp: args.strictMcp,
        ephemeral: args.ephemeral,
        maxBudgetUsd: args.maxBudgetUsd,
    };

    Ok((prompt, config))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::McpServe) => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            construct::mcp::serve::run().await
        }

        Some(Commands::Exec(args)) => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            let outputMode = args.output.clone();
            let (prompt, runConfig) = resolveExecArgs(args)?;
            let config = construct::config::load()?;

            match outputMode.as_str() {
                "json" => {
                    let result = construct::runner::run(
                        &config, &prompt, &runConfig,
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
                    let (handle, mut eventRx) = construct::runner::runStreaming(
                        &config, &prompt, &runConfig,
                    ).await?;
                    while let Some(event) = eventRx.recv().await {
                        let line = formatEventJson(&event);
                        println!("{line}");
                    }
                    handle.await??;
                }
                _ => {
                    use std::io::Write;
                    let (handle, mut eventRx) = construct::runner::runStreaming(
                        &config, &prompt, &runConfig,
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
                                eprint!("\r\x1b[K");
                                tracing::debug!(tool = %name, "tool result");
                            }
                            SessionEvent::Error(msg) => {
                                eprintln!("\x1b[31merror: {msg}\x1b[0m");
                            }
                            _ => {}
                        }
                    }
                    println!();
                    handle.await??;
                }
            }
            Ok(())
        }

        Some(Commands::Completions { shell }) => {
            clap_complete::generate(
                shell,
                &mut Cli::command(),
                "flatline",
                &mut std::io::stdout(),
            );
            Ok(())
        }

        None => {
            // TUI mode: file-based logging so it doesn't collide with the TUI.
            let logDir = construct::config::configDir().join("logs");
            std::fs::create_dir_all(&logDir)?;
            let logPath = logDir.join(format!("flatline-{}.log", std::process::id()));
            let logFile = std::fs::File::create(&logPath)?;
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug"));
            tracing_subscriber::fmt()
                .with_writer(logFile)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            // Redirect panics to the log file instead of stderr, which would
            // corrupt the TUI. On the main thread, also restore the terminal
            // so the panic is readable after exit.
            std::panic::set_hook(Box::new(|info| {
                tracing::error!("{info}");

                if std::thread::current().name() == Some("main") {
                    let _ = crossterm::terminal::disable_raw_mode();
                    let _ = crossterm::execute!(
                        std::io::stdout(),
                        crossterm::terminal::LeaveAlternateScreen,
                    );
                    eprintln!("{info}");
                }
            }));

            app::run().await
        }
    }
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
        SessionEvent::ToolRequest { name, summary, args, diff, .. } => serde_json::json!({
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
        SessionEvent::ToolAutoDenied { name, summary } => serde_json::json!({
            "type": "toolAutoDenied", "name": name, "summary": summary,
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
        SessionEvent::TokenUpdate { promptTokens, completionTokens, contextTokens, turnCost, sessionCost } => {
            serde_json::json!({
                "type": "tokenUpdate",
                "promptTokens": promptTokens,
                "completionTokens": completionTokens,
                "contextTokens": contextTokens,
                "turnCost": turnCost,
                "sessionCost": sessionCost,
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
        SessionEvent::BudgetWarning { sessionCost, limit } => serde_json::json!({
            "type": "budgetWarning", "sessionCost": sessionCost, "limit": limit,
        }),
        SessionEvent::Error(msg) => serde_json::json!({
            "type": "error", "message": msg,
        }),
        // TUI-specific events — emit type only for completeness.
        _ => serde_json::json!({ "type": "other" }),
    };
    serde_json::to_string(&val).unwrap_or_else(|_| r#"{"type":"serializationError"}"#.into())
}
