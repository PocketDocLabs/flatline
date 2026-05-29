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
mod export;
mod fork_picker;
mod history;
mod impact;
mod jobs_panel;
#[allow(dead_code)]
mod layout;
mod log_panel;
mod lsp_panel;
mod markdown;
mod mcp_panel;
mod model_panel;
mod permissions_panel;
mod rewind_picker;
mod runs_panel;
mod selection;
mod session_picker;
mod subagent_panel;
mod terminal;
mod terminal_pane;
mod text_area;
mod throbber;
mod toast;

use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "flatline",
    version,
    about = "General-purpose agentic terminal tool"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Run a prompt headlessly.
    Exec(ExecArgs),

    /// Manage provider authentication.
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },

    /// Start MCP server.
    McpServe,

    /// Generate shell completions.
    #[command(hide = true)]
    Completions {
        /// Shell to generate for (bash, zsh, fish, powershell, elvish).
        shell: clap_complete::Shell,
    },

    /// Export session transcripts as SFT training data (OpenAI-format JSON).
    Export {
        /// Session ID (e.g. ses_abc123). Omit when using --all.
        sessionId: Option<String>,
        /// Export every session that has snapshots, merged into one array.
        #[arg(long, conflicts_with = "sessionId")]
        all: bool,
        /// With --all, only include sessions whose projectDir matches.
        #[arg(long, requires = "all")]
        project: Option<std::path::PathBuf>,
        /// Output file. Stdout if omitted.
        #[arg(long, short = 'o')]
        output: Option<std::path::PathBuf>,
        /// Drop the `reasoning` field from emitted examples.
        #[arg(long)]
        noReasoning: bool,
        /// Minimum messages required per emitted example.
        #[arg(long, default_value_t = 2)]
        minMessages: usize,
        /// Print stats without writing any examples.
        #[arg(long)]
        dryRun: bool,
        /// Include cancelled turns as training targets (default: skip).
        #[arg(long)]
        includeCancelled: bool,
    },
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Sign in to ChatGPT/Codex OAuth.
    Login {
        /// Provider to authenticate.
        #[arg(value_parser = ["openai-codex"])]
        provider: String,
    },

    /// Show authentication status.
    Status,

    /// Remove stored ChatGPT/Codex OAuth credentials.
    Logout {
        /// Provider to clear.
        #[arg(value_parser = ["openai-codex"])]
        provider: String,
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

    /// Profile to use for the heavy tier (overrides `heavyProfile` in config).
    #[arg(long, alias = "profile")]
    heavyProfile: Option<String>,

    /// Profile to use for the light tier (overrides `lightProfile`).
    /// Defaults to the heavy profile when unset.
    #[arg(long)]
    lightProfile: Option<String>,

    /// Profile to use for the utility tier (overrides `utilityProfile`).
    /// Defaults to the light profile when unset.
    #[arg(long)]
    utilityProfile: Option<String>,

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
        args.appendSystemPrompt
            .map(construct::runner::SystemPromptOverride::Append)
    };

    // Filter empty strings from tools (handles `--tools ""`).
    let tools = args.tools.map(|v| {
        let filtered: Vec<String> = v.into_iter().filter(|s| !s.is_empty()).collect();
        filtered
    });

    let config = construct::runner::RunConfig {
        maxTurns: args.maxTurns,
        heavyProfile: args.heavyProfile,
        lightProfile: args.lightProfile,
        utilityProfile: args.utilityProfile,
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

async fn runAuthCommand(command: AuthCommands) -> Result<()> {
    match command {
        AuthCommands::Login { provider } if provider == "openai-codex" => {
            let device = construct::auth::requestOpenAiCodexDeviceCode().await?;
            println!("Open this URL and enter the code:\n");
            println!("  {}", device.verificationUrl);
            println!("  code: {}", device.userCode);
            println!();
            println!("Waiting for OpenAI sign-in...");

            let auth = construct::auth::completeOpenAiCodexDeviceLogin(device).await?;
            let who = auth
                .email
                .as_deref()
                .or(auth.accountId.as_deref())
                .unwrap_or("OpenAI account");
            if let Some(plan) = auth.planType.as_deref() {
                println!("Signed in as {who} ({plan}).");
            } else {
                println!("Signed in as {who}.");
            }
            println!(
                "Credentials saved to {}",
                construct::auth::authPath().display()
            );
            Ok(())
        }
        AuthCommands::Login { provider } => {
            anyhow::bail!("unsupported auth provider: {provider}");
        }
        AuthCommands::Status => {
            let status = construct::auth::openAiCodexStatus();
            println!("openai-codex:");
            println!("  configured: {}", status.configured);
            println!("  path: {}", status.storagePath.display());
            if let Some(email) = status.email.as_deref() {
                println!("  account: {email}");
            } else if let Some(accountId) = status.accountId.as_deref() {
                println!("  account: {accountId}");
            }
            if let Some(plan) = status.planType.as_deref() {
                println!("  plan: {plan}");
            }
            if let Some(expiresAt) = status.expiresAt {
                println!(
                    "  token: {}",
                    if status.expired { "expired" } else { "valid" }
                );
                println!("  expiresAt: {expiresAt}");
            }
            Ok(())
        }
        AuthCommands::Logout { provider } if provider == "openai-codex" => {
            construct::auth::clearOpenAiCodexAuth()?;
            println!("Removed openai-codex credentials.");
            Ok(())
        }
        AuthCommands::Logout { provider } => {
            anyhow::bail!("unsupported auth provider: {provider}");
        }
    }
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

        Some(Commands::Auth { command }) => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            runAuthCommand(command).await
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
            // Apply tier overrides before load so the resolver picks the right profiles.
            // SAFETY: single-threaded CLI startup; nothing else reads these yet.
            if let Some(ref profile) = runConfig.heavyProfile {
                unsafe { std::env::set_var("FLATLINE_HEAVY_PROFILE", profile) };
            }
            if let Some(ref profile) = runConfig.lightProfile {
                unsafe { std::env::set_var("FLATLINE_LIGHT_PROFILE", profile) };
            }
            if let Some(ref profile) = runConfig.utilityProfile {
                unsafe { std::env::set_var("FLATLINE_UTILITY_PROFILE", profile) };
            }
            let config = construct::config::load()?;

            match outputMode.as_str() {
                "json" => {
                    let result = construct::runner::run(&config, &prompt, &runConfig).await?;
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
                    let (handle, mut eventRx) =
                        construct::runner::runStreaming(&config, &prompt, &runConfig).await?;
                    while let Some(event) = eventRx.recv().await {
                        let line = formatEventJson(&event);
                        println!("{line}");
                    }
                    handle.await??;
                }
                _ => {
                    use std::io::Write;
                    let (handle, mut logRx) =
                        construct::runner::runStreaming(&config, &prompt, &runConfig).await?;
                    while let Some(event) = logRx.recv().await {
                        use construct::control::LogEvent;
                        match event {
                            LogEvent::ContentDelta(text) => {
                                print!("{text}");
                                std::io::stdout().flush().ok();
                            }
                            LogEvent::ToolStarted { name, summary } => {
                                eprint!("\x1b[2m\u{25b8} {name}: {summary}...\x1b[0m");
                            }
                            LogEvent::ToolAutoApproved { name, summary } => {
                                eprintln!("\x1b[2m\u{25b8} {name}: {summary}\x1b[0m");
                            }
                            LogEvent::ToolResult { name, .. } => {
                                eprint!("\r\x1b[K");
                                tracing::debug!(tool = %name, "tool result");
                            }
                            LogEvent::Error(msg) => {
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

        Some(Commands::Export {
            sessionId,
            all,
            project,
            output,
            noReasoning,
            minMessages,
            dryRun,
            includeCancelled,
        }) => {
            let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .with_env_filter(envFilter)
                .init();

            if sessionId.is_none() && !all {
                anyhow::bail!("provide a sessionId or pass --all");
            }

            export::run(export::ExportArgs {
                sessionId,
                all,
                project,
                output,
                noReasoning,
                minMessages,
                dryRun,
                includeCancelled,
            })
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

/// Serialize a LogEvent to a JSON line for NDJSON output.
fn formatEventJson(event: &construct::control::LogEvent) -> String {
    use construct::control::LogEvent;
    let val = match event {
        LogEvent::ContentDelta(text) => serde_json::json!({
            "type": "contentDelta", "text": text,
        }),
        LogEvent::ReasoningDelta(text) => serde_json::json!({
            "type": "reasoningDelta", "text": text,
        }),
        LogEvent::ToolStarted { name, summary } => serde_json::json!({
            "type": "toolStarted", "name": name, "summary": summary,
        }),
        LogEvent::ToolCallPending { index, name } => serde_json::json!({
            "type": "toolCallPending", "index": index, "name": name,
        }),
        LogEvent::ToolCallProgress { index, bytes } => serde_json::json!({
            "type": "toolCallProgress", "index": index, "bytes": bytes,
        }),
        LogEvent::ToolCallPreview { index, preview } => serde_json::json!({
            "type": "toolCallPreview", "index": index, "preview": preview,
        }),
        LogEvent::ToolAutoApproved { name, summary } => serde_json::json!({
            "type": "toolAutoApproved", "name": name, "summary": summary,
        }),
        LogEvent::ToolResult { name, output } => serde_json::json!({
            "type": "toolResult", "name": name, "output": output,
        }),
        LogEvent::ToolDenied { name } => serde_json::json!({
            "type": "toolDenied", "name": name,
        }),
        LogEvent::ToolAutoDenied { name, summary } => serde_json::json!({
            "type": "toolAutoDenied", "name": name, "summary": summary,
        }),
        LogEvent::TurnAborted { name } => serde_json::json!({
            "type": "turnAborted", "name": name,
        }),
        LogEvent::TurnComplete => serde_json::json!({
            "type": "turnComplete",
        }),
        LogEvent::TurnCancelled => serde_json::json!({
            "type": "turnCancelled",
        }),
        LogEvent::SteerInjected { texts } => serde_json::json!({
            "type": "steerInjected", "texts": texts,
        }),
        LogEvent::TokenUpdate {
            promptTokens,
            completionTokens,
            contextTokens,
            turnCost,
            sessionCost,
            cacheReadTokens,
            cacheCreationTokens,
        } => {
            serde_json::json!({
                "type": "tokenUpdate",
                "promptTokens": promptTokens,
                "completionTokens": completionTokens,
                "contextTokens": contextTokens,
                "turnCost": turnCost,
                "sessionCost": sessionCost,
                "cacheReadTokens": cacheReadTokens,
                "cacheCreationTokens": cacheCreationTokens,
            })
        }
        LogEvent::ModelConfigChanged {
            contextWindow,
            cachingEnabled,
        } => serde_json::json!({
            "type": "modelConfigChanged",
            "contextWindow": contextWindow,
            "cachingEnabled": cachingEnabled,
        }),
        LogEvent::CompactionStarted { stage } => serde_json::json!({
            "type": "compactionStarted", "stage": stage,
        }),
        LogEvent::CompactionComplete {
            stage,
            reduction,
            markerBlock,
        } => serde_json::json!({
            "type": "compactionComplete", "stage": stage,
            "reduction": reduction, "markerBlock": markerBlock,
        }),
        LogEvent::Cleared => serde_json::json!({ "type": "cleared" }),
        LogEvent::SessionRestored { turns, markers } => serde_json::json!({
            "type": "sessionRestored",
            "turnCount": turns.len(),
            "markers": markers,
        }),
        LogEvent::TopicChanged { label } => serde_json::json!({
            "type": "topicChanged", "label": label,
        }),
        LogEvent::Rewound { targetTurnId } => serde_json::json!({
            "type": "rewound", "targetTurnId": targetTurnId,
        }),
        LogEvent::LspHint {
            serverId,
            installHint,
        } => serde_json::json!({
            "type": "lspHint", "serverId": serverId, "installHint": installHint,
        }),
        LogEvent::SubagentStarted {
            sessionId,
            agentType,
            prompt,
        } => serde_json::json!({
            "type": "subagentStarted", "sessionId": sessionId,
            "agentType": agentType, "prompt": prompt,
        }),
        LogEvent::SubagentEvent { sessionId, event } => serde_json::json!({
            "type": "subagentEvent", "sessionId": sessionId,
            "event": formatEventJson(event),
        }),
        LogEvent::SubagentShellOutput { sessionId, .. } => serde_json::json!({
            "type": "subagentShellOutput", "sessionId": sessionId,
        }),
        LogEvent::SubagentComplete {
            sessionId,
            agentType,
            content,
            turns,
        } => serde_json::json!({
            "type": "subagentComplete", "sessionId": sessionId,
            "agentType": agentType, "content": content, "turns": turns,
        }),
        LogEvent::Retrying {
            attempt,
            maxAttempts,
        } => serde_json::json!({
            "type": "retrying", "attempt": attempt, "maxAttempts": maxAttempts,
        }),
        LogEvent::BudgetWarning { sessionCost, limit } => serde_json::json!({
            "type": "budgetWarning", "sessionCost": sessionCost, "limit": limit,
        }),
        LogEvent::ScratchpadRecovered {
            matchedTag,
            snippet,
            recoveredChars,
        } => serde_json::json!({
            "type": "scratchpadRecovered",
            "matchedTag": matchedTag,
            "snippet": snippet,
            "recoveredChars": recoveredChars,
        }),
        LogEvent::JobSpawned { id, kind, command } => serde_json::json!({
            "type": "taskSpawned", "id": id, "kind": kind, "command": command,
        }),
        LogEvent::JobOutput { id, line } => serde_json::json!({
            "type": "taskOutput", "id": id, "line": line,
        }),
        LogEvent::JobComplete { id, exitCode } => serde_json::json!({
            "type": "taskComplete", "id": id, "exitCode": exitCode,
        }),
        LogEvent::JobStopped { id, reason } => serde_json::json!({
            "type": "taskStopped", "id": id, "reason": reason,
        }),
        LogEvent::MonitorRegistered {
            id,
            description,
            terminal,
            filter,
        } => serde_json::json!({
            "type": "monitorRegistered",
            "id": id,
            "description": description, "terminal": terminal, "filter": filter,
        }),
        LogEvent::MonitorEvent {
            id,
            line,
            eventCount,
        } => serde_json::json!({
            "type": "monitorEvent", "id": id, "line": line, "eventCount": eventCount,
        }),
        LogEvent::MonitorAutoStopped { id, reason } => serde_json::json!({
            "type": "monitorAutoStopped", "id": id, "reason": reason,
        }),
        LogEvent::MonitorStopped { id } => serde_json::json!({
            "type": "monitorStopped", "id": id,
        }),
        LogEvent::TerminalSpawned { name, spawnedBy } => serde_json::json!({
            "type": "terminalSpawned", "name": name,
            "spawnedBy": format!("{:?}", spawnedBy),
        }),
        LogEvent::TerminalClosed { name } => serde_json::json!({
            "type": "terminalClosed", "name": name,
        }),
        LogEvent::TerminalActiveForAgent { name } => serde_json::json!({
            "type": "terminalActiveForAgent", "name": name,
        }),
        LogEvent::TerminalRenamed { from, to } => serde_json::json!({
            "type": "terminalRenamed", "from": from, "to": to,
        }),
        LogEvent::WakeBatchInjected { count, summary } => serde_json::json!({
            "type": "wakeBatchInjected", "count": count, "summary": summary,
        }),
        LogEvent::WakeRegistered {
            id,
            kind,
            summary,
            prompt,
            nextFireAt: _,
        } => serde_json::json!({
            "type": "wakeRegistered", "id": id, "kind": kind.asStr(), "summary": summary, "prompt": prompt,
        }),
        LogEvent::WakeDisarmed { id } => serde_json::json!({
            "type": "wakeDisarmed", "id": id,
        }),
        LogEvent::AutoBgWarning {
            command,
            elapsedSecs,
            userTriggered,
        } => serde_json::json!({
            "type": "autoBgWarning",
            "command": command,
            "elapsedSecs": elapsedSecs,
            "userTriggered": userTriggered,
        }),
        LogEvent::Error(msg) => serde_json::json!({
            "type": "error", "message": msg,
        }),
    };
    serde_json::to_string(&val).unwrap_or_else(|_| r#"{"type":"serializationError"}"#.into())
}
