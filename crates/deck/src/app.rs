//! Application loop — ratatui event loop with terminal and agent panel.
//!
//! Boots a fullscreen TUI with a shared terminal on the left and an
//! agent conversation panel on the right. Tab switches focus between them.
//!
//! # Public API
//! - [`run`] — starts the app
//!
//! # Dependencies
//! `construct`, `ratatui`, `crossterm`, `tokio`

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal as RatatuiTerminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};
use tokio::sync::{mpsc, oneshot, watch};

use construct::control::{LogEvent, SessionRequest, TuiRequest};
use construct::permissions::Permissions;
use construct::prompt::{DomainModule, InterfaceMode};
use construct::session::Session;
use construct::shell::{ShellIo, spawnShell};

use crate::agent_panel::AgentPanel;
use crate::fork_picker::{ForkAction, ForkPicker};
use crate::lsp_panel::{LspPanel, PanelAction as LspPanelAction};
use crate::mcp_panel::{McpPanel, PanelAction as McpPanelAction};
use crate::rewind_picker::{RewindAction, RewindPicker};
use crate::selection::{self, PanelId, SelectionState};
use crate::session_picker::{PickerAction, SessionPicker};
use crate::terminal::{Terminal as EmbeddedTerminal, TerminalState};

use std::io::{self, Write as _};
use std::time::{Duration, Instant};

/// Resolve main agent permissions from config, falling back to allowReadOnly.
fn mainAgentPermissions(config: &construct::config::Config) -> Permissions {
    config
        .permissions
        .clone()
        .unwrap_or_else(Permissions::allowReadOnly)
}

/// Which panel has input focus.
#[derive(PartialEq)]
enum Focus {
    Terminal,
    Agent,
}

/// Internal deck-update messages. Produced by tasks that await oneshot replies
/// to `TuiRequest`s; drained by `runLoop` alongside `LogEvent`s so slash
/// commands don't block the TUI while the session task is mid-turn.
enum DeckUpdate {
    McpStatus(construct::control::McpStatus),
    LspStatus(construct::control::LspStatus),
    PermissionsStatus(construct::control::PermissionsStatus),
    ContextDisplay(construct::context::ContextState),
    RewindOptions(Vec<construct::transcript::Turn>),
    Forks(Vec<construct::transcript::Fork>),
    /// Generic string result to show in the panel (e.g. cost, session list).
    ShowResult(String),
    /// Result of a mutation (rewind, fork switch, save permissions, clear).
    CommandAck(construct::control::CommandAck),
    /// Resume completed — picker should close on ok, show error on failure.
    ResumeResult(construct::control::CommandAck),
}

/// Axis lock for trackpad scroll — prevents diagonal scrolling.
///
/// Once scrolling starts on one axis, the other axis is ignored until
/// the gesture pauses (no scroll events for `TIMEOUT`).
#[derive(PartialEq, Clone, Copy)]
enum ScrollAxis {
    Vertical,
    Horizontal,
}

struct ScrollAxisLock {
    axis: Option<ScrollAxis>,
    lastEvent: Instant,
}

const SCROLL_LOCK_TIMEOUT_MS: u64 = 150;

impl ScrollAxisLock {
    fn new() -> Self {
        Self {
            axis: None,
            lastEvent: Instant::now(),
        }
    }

    /// Returns true if the given axis is allowed right now.
    fn allow(&mut self, incoming: ScrollAxis) -> bool {
        let now = Instant::now();
        let expired =
            now.duration_since(self.lastEvent).as_millis() > SCROLL_LOCK_TIMEOUT_MS as u128;

        if expired || self.axis.is_none() {
            self.axis = Some(incoming);
            self.lastEvent = now;
            true
        } else if self.axis == Some(incoming) {
            self.lastEvent = now;
            true
        } else {
            // Locked to the other axis — swallow this event.
            false
        }
    }
}

/// Static glyph shown in the title when a topic is set but the agent is idle.
/// Ace of hearts — deliberately distinct from every animation frame so a
/// still title reads as "idle" rather than "animation stuck on a card".
const TITLE_IDLE_GLYPH: &str = "\u{1F0B1}";

/// Set the host terminal's window title via OSC 2.
fn writeTerminalTitle(glyph: &str, topic: Option<&str>) {
    let title = match topic {
        Some(t) if !t.is_empty() => format!("{glyph} {t}"),
        _ => "flatline".to_string(),
    };
    let _ = write!(io::stdout(), "\x1b]2;{title}\x07");
    let _ = io::stdout().flush();
}

/// Restore the terminal title to whatever it was (best-effort reset).
fn resetTerminalTitle() {
    let _ = write!(io::stdout(), "\x1b]2;\x07");
    let _ = io::stdout().flush();
}

/// Animated card-flip spinner for the terminal title. Cycles through a
/// pre-built frame list with per-frame dwell times, advanced by a wall-clock
/// tick on the main render loop.
struct TitleSpinner {
    frames: Vec<(&'static str, Duration)>,
    idx: usize,
    lastAdvance: Instant,
}

impl TitleSpinner {
    fn new() -> Self {
        const CARDS: &[&str] = &[
            "\u{1F0A0}", // card back
            "\u{1F0A1}", // ace of spades
            "\u{1F0A0}", // card back
            "\u{1F0DF}", // black joker
        ];
        // Trailing NBSP (not regular space) pads each 1-col bar to 2 cols.
        // The title writer also injects a separator space between glyph and
        // topic — using NBSP here prevents the terminal from collapsing
        // those two consecutive whitespace chars and dropping the topic a
        // column to the left on every non-card frame.
        const BARS: &[&str] = &["\u{2337}", "|\u{00A0}"];
        // Two NBSPs for the edge: leading one survives leading-whitespace
        // stripping, trailing one survives the adjacent-space collapse.
        const EDGE: &str = "\u{00A0}\u{00A0}";
        let cardHold = Duration::from_millis(600);
        let barHold = Duration::from_millis(80);
        let edgeHold = Duration::from_millis(100);
        let mut frames: Vec<(&'static str, Duration)> = Vec::new();
        for card in CARDS {
            frames.push((*card, cardHold));
            for b in BARS {
                frames.push((*b, barHold));
            }
            frames.push((EDGE, edgeHold));
            for b in BARS.iter().rev() {
                frames.push((*b, barHold));
            }
        }
        Self {
            frames,
            idx: 0,
            lastAdvance: Instant::now(),
        }
    }

    fn current(&self) -> &'static str {
        self.frames[self.idx].0
    }

    /// Advance by one frame if the current dwell has elapsed. Returns true
    /// if the frame actually changed (caller should repaint the title).
    fn tick(&mut self) -> bool {
        let dwell = self.frames[self.idx].1;
        if self.lastAdvance.elapsed() >= dwell {
            self.idx = (self.idx + 1) % self.frames.len();
            self.lastAdvance = Instant::now();
            true
        } else {
            false
        }
    }
}

/// Run the deck TUI.
pub async fn run() -> Result<()> {
    // Load config BEFORE touching the terminal — a config error here would
    // otherwise skip the cleanup block and leave the terminal in raw mode
    // with mouse capture still on.
    let config = construct::config::load()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        crossterm::event::EnableBracketedPaste,
        crossterm::event::EnableMouseCapture,
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | crossterm::event::KeyboardEnhancementFlags::REPORT_EVENT_TYPES,
        ),
        crossterm::cursor::SetCursorStyle::SteadyBar,
    )?;
    writeTerminalTitle(TITLE_IDLE_GLYPH, None);

    // From here until cleanup, any `?` bypasses terminal restore — wrap
    // fallible setup so errors get cleaned up before propagating.
    let setupResult = (|| -> Result<_> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = RatatuiTerminal::new(backend)?;
        let size = terminal.size()?;
        // Terminal gets ~60% width minus borders, full height minus status bar and borders.
        let termCols = (size.width * 3 / 5).saturating_sub(2);
        let termRows = size.height.saturating_sub(3);
        let (shell, shellIo) = spawnShell(termCols, termRows)?;
        let termState = TerminalState::new(termCols, termRows);
        Ok((terminal, shell, shellIo, termState, termCols, termRows))
    })();

    let (mut terminal, shell, mut shellIo, mut termState, _termCols, _termRows) = match setupResult
    {
        Ok(v) => v,
        Err(e) => {
            let _ = disable_raw_mode();
            let _ = execute!(
                io::stdout(),
                crossterm::cursor::SetCursorStyle::DefaultUserShape,
                crossterm::event::PopKeyboardEnhancementFlags,
                crossterm::event::DisableMouseCapture,
                crossterm::event::DisableBracketedPaste,
                LeaveAlternateScreen,
            );
            resetTerminalTitle();
            return Err(e);
        }
    };

    let contextWindow = config.heavy.contextWindow;
    let cachingEnabled = config.heavy.cachingActive();
    tracing::info!(
        model = %config.heavy.model,
        cachingEnabled,
        supportsAnthropicCache = ?config.heavy.supportsAnthropicCache,
        "deck startup cache config"
    );
    let rollingBaseline =
        construct::cost::rollingWindowCost(config.projectRoot.as_deref().and_then(|p| p.to_str()));

    let mut agentPanel = AgentPanel::new();
    let mut focus = Focus::Terminal;
    let mut selState = SelectionState::new();
    let mut scrollLock = ScrollAxisLock::new();
    // Session channels — Log (session → TUI), SessionRequest (session → TUI),
    // TuiRequest (TUI → session), plus user input / cancel / steer.
    let (logTx, mut logRx) = mpsc::channel::<LogEvent>(256);
    let (sessionRequestTx, mut sessionRequestRx) = mpsc::channel::<SessionRequest>(16);
    let (userInputTx, mut userInputRx) = mpsc::channel::<construct::session::UserInput>(16);
    let (requestTx, mut requestRx) = mpsc::channel::<TuiRequest>(16);
    let (cancelTx, cancelRx) = watch::channel(false);
    let (steerTx, steerRx) = mpsc::channel::<construct::session::UserInput>(16);
    let (deckUpdateTx, mut deckUpdateRx) = mpsc::channel::<DeckUpdate>(32);

    // Spawn the agent session task.
    let mut cancelRx = cancelRx;
    let mut steerRx = steerRx;
    tokio::spawn(async move {
        let config = match construct::config::load() {
            Ok(c) => c,
            Err(e) => {
                let _ = logTx
                    .send(LogEvent::Error(format!("Config error: {e}")))
                    .await;
                return;
            }
        };

        // Main agent auto-approves read-only tools but still prompts on writes/mutations.
        let permissions = mainAgentPermissions(&config);

        // Deck is the shared terminal harness — SWE domain by default.
        let mut session = match Session::new(
            &config,
            permissions,
            shell,
            InterfaceMode::SharedTerminal,
            &[DomainModule::Swe],
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = logTx
                    .send(LogEvent::Error(format!("Session error: {e}")))
                    .await;
                return;
            }
        };

        // Initialize MCP servers from .mcp.json files.
        match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
            Ok(servers) if !servers.is_empty() => {
                session.initMcp(servers).await;
            }
            Err(e) => tracing::warn!("failed to load MCP config: {e}"),
            _ => {}
        }

        loop {
            tokio::select! {
                msg = userInputRx.recv() => {
                    match msg {
                        Some(msg) => {
                            // Clear any stale cancel notification from a previous turn.
                            cancelRx.borrow_and_update();
                            if let Err(e) = session.send(&msg, &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx).await {
                                let _ = logTx
                                    .send(LogEvent::Error(format!("Agent error: {e}")))
                                    .await;
                            }
                        }
                        None => break,
                    }
                }
                req = requestRx.recv() => {
                    match req {
                        Some(TuiRequest::ResumeSession { sessionId: id, reply }) => {
                            // Consume old session, keep the shell.
                            let shell = session.intoShell();
                            match Session::resume(
                                &config,
                                mainAgentPermissions(&config),
                                shell,
                                InterfaceMode::SharedTerminal,
                                &[DomainModule::Swe],
                                &id,
                            ).await {
                                Ok(s) => {
                                    session = s;
                                    // Re-init MCP for the resumed session.
                                    match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
                                        Ok(servers) if !servers.is_empty() => {
                                            session.initMcp(servers).await;
                                        }
                                        Err(e) => tracing::warn!("failed to load MCP config: {e}"),
                                        _ => {}
                                    }

                                    // Load display branch — includes the full un-branched chain
                                    // past the rewind point until the user sends a new message.
                                    let turns = session.loadDisplayTurns().unwrap_or_default();
                                    let markers = session.compactionMarkers();
                                    let _ = logTx
                                        .send(LogEvent::SessionRestored { turns, markers })
                                        .await;
                                    // Set window title to the current topic on resume.
                                    let label = session.currentTopicLabel();
                                    if !label.is_empty() {
                                        let _ = logTx
                                            .send(LogEvent::TopicChanged {
                                                label: label.to_string(),
                                            })
                                            .await;
                                    }
                                    let _ = reply.send(construct::control::CommandAck::ok(
                                        format!("Resumed session {id}"),
                                    ));
                                }
                                Err((e, shell)) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to resume {id}: {e}"),
                                    ));
                                    // Shell returned — recreate a fresh session.
                                    match Session::new(
                                        &config,
                                        mainAgentPermissions(&config),
                                        shell,
                                        InterfaceMode::SharedTerminal,
                                        &[DomainModule::Swe],
                                    ) {
                                        Ok(mut s) => {
                                            match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
                                                Ok(servers) if !servers.is_empty() => {
                                                    s.initMcp(servers).await;
                                                }
                                                Err(e) => tracing::warn!("failed to load MCP config: {e}"),
                                                _ => {}
                                            }
                                            session = s;
                                        }
                                        Err(e2) => {
                                            let _ = logTx
                                                .send(LogEvent::Error(
                                                    format!("Session lost after failed resume: {e2}"),
                                                ))
                                                .await;
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Some(TuiRequest::Clear { reply }) => {
                            let shell = session.intoShell();
                            match Session::new(
                                &config,
                                mainAgentPermissions(&config),
                                shell,
                                InterfaceMode::SharedTerminal,
                                &[DomainModule::Swe],
                            ) {
                                Ok(s) => {
                                    session = s;
                                    match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
                                        Ok(servers) if !servers.is_empty() => {
                                            session.initMcp(servers).await;
                                        }
                                        Err(e) => tracing::warn!("failed to load MCP config: {e}"),
                                        _ => {}
                                    }

                                    let _ = logTx.send(LogEvent::Cleared).await;
                                    let _ = reply.send(construct::control::CommandAck::ok("Session cleared."));
                                }
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to create new session: {e}"),
                                    ));
                                    return;
                                }
                            }
                        }
                        Some(TuiRequest::GetLsp { reply }) => {
                            let servers = session.lspStatusData();
                            let _ = reply.send(construct::control::LspStatus { servers });
                        }
                        Some(TuiRequest::GetMcp { reply }) => {
                            let (servers, totalTools, searchMode, configPath) =
                                session.mcpStatusData().await;
                            let _ = reply.send(construct::control::McpStatus {
                                servers,
                                totalTools,
                                searchMode,
                                configPath,
                            });
                        }
                        Some(TuiRequest::GetPermissions { reply }) => {
                            let (defaultMode, rules, source, configPath) =
                                session.permissionsStatusData();
                            let _ = reply.send(construct::control::PermissionsStatus {
                                defaultMode,
                                rules,
                                source,
                                configPath,
                            });
                        }
                        Some(TuiRequest::SavePermissions { defaultMode, rules, reply }) => {
                            if let Some(ref root) = config.projectRoot {
                                match construct::config::savePermissions(
                                    root,
                                    &defaultMode,
                                    &rules,
                                ) {
                                    Ok(()) => {
                                        session.setPermissions(construct::permissions::Permissions {
                                            defaultMode,
                                            rules,
                                            source: construct::permissions::PermissionsSource::Project,
                                        });
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            "Permissions saved.",
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Failed to save permissions: {e}"),
                                        ));
                                    }
                                }
                            } else {
                                let _ = reply.send(construct::control::CommandAck::err(
                                    "No project root; permissions not persisted.",
                                ));
                            }
                        }
                        Some(TuiRequest::GetRewindOptions { reply }) => {
                            let turns = session.loadDisplayTurns().unwrap_or_default();
                            let _ = reply.send(turns);
                        }
                        Some(TuiRequest::Rewind { target, saveFork, reply }) => {
                            let result = session.rewind(&target, saveFork, &logTx).await;
                            let _ = reply.send(construct::control::CommandAck::ok(result));
                        }
                        Some(TuiRequest::GetForks { reply }) => {
                            let forks = session.listForks();
                            let _ = reply.send(forks);
                        }
                        Some(TuiRequest::SwitchFork { forkId, reply }) => {
                            let result = session.switchFork(&forkId, &logTx).await;
                            let _ = reply.send(construct::control::CommandAck::ok(result));
                        }
                        Some(TuiRequest::ShowContext { reply }) => {
                            let state = session.buildContextState();
                            let _ = reply.send(state);
                        }
                        Some(TuiRequest::Undo { reply }) => {
                            let ack = session.undoCheckpoint().await;
                            let _ = reply.send(ack);
                        }
                        Some(TuiRequest::ListSessions { reply }) => {
                            let _ = reply.send(session.listSessionsText());
                        }
                        Some(TuiRequest::ShowCost { reply }) => {
                            let _ = reply.send(session.formatCostBreakdown());
                        }
                        Some(TuiRequest::Shutdown) => {
                            session.shutdownLsp().await;
                            session.shutdownMcp().await;
                            break;
                        }
                        Some(TuiRequest::RetryLastTurn { reply }) => {
                            cancelRx.borrow_and_update();
                            match session.retryLastTurn(
                                &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx,
                            ).await {
                                Ok(()) => {
                                    let _ = reply.send(construct::control::CommandAck::ok(""));
                                }
                                Err(e) => {
                                    let _ = logTx
                                        .send(LogEvent::Error(format!("Retry failed: {e}")))
                                        .await;
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Retry failed: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::ContinueLastTurn { reply }) => {
                            cancelRx.borrow_and_update();
                            match session.continueLastTurn(
                                &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx,
                            ).await {
                                Ok(()) => {
                                    let _ = reply.send(construct::control::CommandAck::ok(""));
                                }
                                Err(e) => {
                                    let _ = logTx
                                        .send(LogEvent::Error(format!("Continue failed: {e}")))
                                        .await;
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Continue failed: {e}"),
                                    ));
                                }
                            }
                        }
                        None => break,
                    }
                }
            }
        }
    });

    let result = runLoop(
        &mut terminal,
        &mut shellIo,
        &mut termState,
        &mut agentPanel,
        &mut focus,
        &mut selState,
        &mut scrollLock,
        &mut logRx,
        &mut sessionRequestRx,
        &mut deckUpdateRx,
        &userInputTx,
        &requestTx,
        &deckUpdateTx,
        &cancelTx,
        &steerTx,
        contextWindow,
        rollingBaseline,
        cachingEnabled,
    )
    .await;

    // Gracefully shut down background services before exiting.
    let _ = requestTx.send(TuiRequest::Shutdown).await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::cursor::SetCursorStyle::DefaultUserShape,
        crossterm::event::PopKeyboardEnhancementFlags,
        crossterm::event::DisableMouseCapture,
        crossterm::event::DisableBracketedPaste,
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;
    resetTerminalTitle();

    result
}

#[allow(clippy::too_many_arguments)]
async fn runLoop(
    terminal: &mut RatatuiTerminal<CrosstermBackend<io::Stdout>>,
    shellIo: &mut ShellIo,
    termState: &mut TerminalState,
    agentPanel: &mut AgentPanel,
    focus: &mut Focus,
    selState: &mut SelectionState,
    scrollLock: &mut ScrollAxisLock,
    logRx: &mut mpsc::Receiver<LogEvent>,
    sessionRequestRx: &mut mpsc::Receiver<SessionRequest>,
    deckUpdateRx: &mut mpsc::Receiver<DeckUpdate>,
    userInputTx: &mpsc::Sender<construct::session::UserInput>,
    requestTx: &mpsc::Sender<TuiRequest>,
    deckUpdateTx: &mpsc::Sender<DeckUpdate>,
    cancelTx: &watch::Sender<bool>,
    steerTx: &mpsc::Sender<construct::session::UserInput>,
    contextWindow: usize,
    rollingBaseline: f64,
    cachingEnabled: bool,
) -> Result<()> {
    let mut tokenCount: usize = 0;
    let mut sessionCost: f64 = 0.0;
    let mut lastCacheHitAt: Option<Instant> = None;
    let mut helpPopupOpen = false;
    let _ = rollingBaseline; // No longer displayed; kept in signature for API stability.
    let mut sessionPicker: Option<SessionPicker> = None;
    let mut rewindPicker: Option<RewindPicker> = None;
    let mut forkPicker: Option<ForkPicker> = None;
    let mut pendingRewindMessage: Option<String> = None;
    let mut pendingRewindAttachments: Option<Vec<construct::transcript::TurnAttachment>> = None;
    let mut mcpPanel: Option<McpPanel> = None;
    let mut lspPanel: Option<LspPanel> = None;
    let mut permissionsPanel: Option<crate::permissions_panel::PermissionsPanel> = None;
    let mut subagentPanel: Option<crate::subagent_panel::SubagentPanel> = None;
    // Stash for the oneshot reply to the currently-open permit prompt (either
    // top-level or subagent). Set when we receive a `SessionRequest::Permit`
    // and show the prompt; consumed when the user responds.
    let mut pendingPermitReply: Option<oneshot::Sender<construct::permissions::PermitResponse>> =
        None;
    let projectDir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let mut needsRedraw = true;
    let mut lastQuitPress: Option<Instant> = None;
    // Title state: the current topic and the animated spinner that fronts it
    // in the OS terminal title whenever the agent is actively working.
    let mut currentTopic: Option<String> = None;
    let mut titleSpinner = TitleSpinner::new();
    let mut titleWasAnimating = false;
    // Cap draws at ~30fps. Prevents strobing when the PTY floods us with
    // updates (rich progress bars, keystroke echo) — ratatui's buffer diff
    // absorbs all the intermediate state into one frame.
    const DRAW_MIN_INTERVAL: Duration = Duration::from_millis(33);
    let mut lastDraw = Instant::now() - DRAW_MIN_INTERVAL;

    loop {
        // Mark the frame dirty when the VT emulator has new content. Ratatui's
        // buffer diff handles the update — no full clear needed.
        if termState.takeDirty() {
            needsRedraw = true;
        }

        // Draw only when state has changed and the throttle allows it.
        if needsRedraw && lastDraw.elapsed() >= DRAW_MIN_INTERVAL {
            needsRedraw = false;
            lastDraw = Instant::now();
            terminal.draw(|frame| {
                let area = frame.area();

                // Top area: terminal + agent panel. Bottom: status bar.
                let vChunks = Layout::default()
                    .constraints([Constraint::Min(1), Constraint::Length(1)])
                    .split(area);

                // Horizontal split: terminal | agent panel.
                let hChunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .split(vChunks[0]);

                // Terminal.
                let termBorder = if *focus == Focus::Terminal {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                let termTitle = if termState.displayOffset() > 0 {
                    format!(" terminal [\u{2191}{}\u{FE0E}] ", termState.displayOffset())
                } else {
                    " terminal ".to_string()
                };
                let termBlock = Block::default()
                    .borders(Borders::ALL)
                    .border_style(termBorder)
                    .title(termTitle);
                let termInner = termBlock.inner(hChunks[0]);

                // Sync terminal grid size with render area — a mismatch causes
                // content to overflow or underflow the panel borders.
                let gridCols = termState.columns();
                if gridCols != termInner.width as usize
                    || termState.screenLines() != termInner.height as usize
                {
                    termState.resize(termInner.width, termInner.height);
                    let _ = shellIo
                        .resizeTx
                        .try_send((termInner.width, termInner.height));
                }

                frame.render_widget(termBlock, hChunks[0]);
                frame.render_stateful_widget(EmbeddedTerminal, termInner, termState);

                // Capture content rects for mouse hit-testing.
                selState.termContentRect = termInner;

                // Agent panel. When the popup is open showing a subagent permit,
                // suppress the duplicate prompt in the main panel — they share
                // permit state, so a visible duplicate would scroll/copy in
                // lockstep with the popup version.
                agentPanel.permitDisplaySuppressed =
                    subagentPanel.is_some() && agentPanel.pendingPermitIsSubagent();
                let agentChatArea =
                    agentPanel.render(hChunks[1], frame.buffer_mut(), *focus == Focus::Agent);
                // Full chat area for hit-testing (includes prefix columns).
                selState.agentPanelRect = agentChatArea;
                // Content rect offset past the 2-column prefix for selection/highlight.
                let agentContentArea = Rect {
                    x: agentChatArea.x + 2,
                    y: agentChatArea.y,
                    width: agentChatArea.width.saturating_sub(2),
                    height: agentChatArea.height,
                };
                selState.agentContentRect = agentContentArea;
                selState.inputContentRect = agentPanel.lastInputRect;

                // Expand selection for double/triple/quad click (needs Buffer).
                let termOffset = termState.displayOffset() as u16;
                let agentOffset = agentPanel.displayOffset();
                if let Some((panel, clickCount)) = selState.pendingExpand.take() {
                    if panel == PanelId::Input {
                        // Input selection expansion not supported.
                    } else {
                        let (sel, rect, offset) = match panel {
                            PanelId::Terminal => {
                                (&mut selState.termSelection, termInner, termOffset)
                            }
                            PanelId::Agent => {
                                (&mut selState.agentSelection, agentContentArea, agentOffset)
                            }
                            PanelId::Input => unreachable!(),
                        };
                        if let Some(sel) = sel {
                            if clickCount == 4 {
                                match panel {
                                    PanelId::Terminal => {
                                        // Select the command output region under the cursor.
                                        let clickGrid = sel.startGridLine();
                                        if let Some((startGrid, endGrid)) =
                                            termState.commandRegionAt(clickGrid)
                                        {
                                            sel.setBounds(0, startGrid, rect.width, endGrid);
                                        } else {
                                            // No OSC 133 region — fall back to block selection.
                                            selection::expandSelection(
                                                sel,
                                                clickCount,
                                                frame.buffer_mut(),
                                                rect,
                                                offset,
                                            );
                                        }
                                    }
                                    PanelId::Agent => {
                                        // Select the logical entry under the cursor.
                                        let clickGrid = sel.startGridLine();
                                        if let Some((startGrid, endGrid)) =
                                            agentPanel.entryBoundsAtGridLine(clickGrid)
                                        {
                                            sel.setBounds(0, startGrid, rect.width, endGrid);
                                        }
                                    }
                                    PanelId::Input => unreachable!(),
                                }
                            } else {
                                selection::expandSelection(
                                    sel,
                                    clickCount,
                                    frame.buffer_mut(),
                                    rect,
                                    offset,
                                );
                            }
                            sel.finalize();
                            selState.pendingCopy = Some(panel);
                            selState.selectingIn = None;
                        }
                    }
                }

                // Apply selection highlights after widgets have rendered.
                if let Some(ref sel) = selState.termSelection {
                    selection::applyHighlight(sel, termInner, frame.buffer_mut(), termOffset);
                }
                if let Some(ref sel) = selState.agentSelection {
                    selection::applyHighlight(
                        sel,
                        agentContentArea,
                        frame.buffer_mut(),
                        agentOffset,
                    );
                }

                // Deferred clipboard copy (Buffer only available during draw).
                if let Some(panel) = selState.pendingCopy.take() {
                    match panel {
                        PanelId::Terminal => {
                            if let Some(sel) = &selState.termSelection {
                                let text = extractTerminalUnwrapped(
                                    sel,
                                    termInner,
                                    frame.buffer_mut(),
                                    termOffset,
                                    termState,
                                );
                                selection::copyToClipboard(&text);
                            }
                        }
                        PanelId::Agent => {
                            if let Some(sel) = &selState.agentSelection {
                                let text = agentPanel.extractUnwrappedText(
                                    sel,
                                    agentContentArea,
                                    frame.buffer_mut(),
                                    agentOffset,
                                );
                                selection::copyToClipboard(&text);
                            }
                        }
                        PanelId::Input => {}
                    }
                }

                // Hardware cursor for agent text input (hidden during permission prompt).
                if *focus == Focus::Agent && !agentPanel.pendingPermit {
                    if let Some((col, row)) = agentPanel.textArea.cursorScreenPos {
                        frame.set_cursor_position(ratatui::layout::Position::new(col, row));
                    }
                } else if *focus == Focus::Terminal {
                    if let Some((col, row)) = termState.cursorViewportPos() {
                        frame.set_cursor_position(ratatui::layout::Position::new(
                            termInner.x + col,
                            termInner.y + row,
                        ));
                    }
                }

                // Status bar. Minimal layout: right-aligned cost + ctx% + cache.
                // The "press Ctrl+Q again to quit" hint still hijacks the bar
                // briefly — it's a load-bearing safety affordance.
                let quitHintActive = lastQuitPress
                    .map(|t| t.elapsed() < Duration::from_secs(1))
                    .unwrap_or(false);

                let (barBg, barFg) = if quitHintActive {
                    (Color::Yellow, Color::Black)
                } else {
                    (Color::DarkGray, Color::White)
                };

                let barWidth = vChunks[1].width as usize;
                let mut spans: Vec<Span<'static>> = Vec::with_capacity(8);

                if quitHintActive {
                    let hint = " \u{25B8} press Ctrl+Q again to quit";
                    spans.push(Span::raw(hint.to_string()));
                    let pad = barWidth.saturating_sub(hint.chars().count());
                    spans.push(Span::raw(" ".repeat(pad)));
                } else {
                    let costStr = if sessionCost > 0.0 {
                        construct::cost::formatCost(sessionCost)
                    } else {
                        String::new()
                    };
                    let ctxStr = formatContextPct(tokenCount, contextWindow);
                    let cacheSpans = cacheHeatSpans(cachingEnabled, lastCacheHitAt, barBg, barFg);

                    // Assemble right-aligned spans: cost  ctx  cache-glyph cache-word.
                    let mut rightSpans: Vec<Span<'static>> = Vec::with_capacity(6);
                    if !costStr.is_empty() {
                        rightSpans.push(Span::raw(costStr));
                    }
                    if !ctxStr.is_empty() {
                        if !rightSpans.is_empty() {
                            rightSpans.push(Span::raw("  "));
                        }
                        rightSpans.push(Span::raw(ctxStr));
                    }
                    if !cacheSpans.is_empty() {
                        if !rightSpans.is_empty() {
                            rightSpans.push(Span::raw("  "));
                        }
                        rightSpans.extend(cacheSpans);
                        rightSpans.push(Span::raw(" cache"));
                    }

                    let rightWidth: usize =
                        rightSpans.iter().map(|s| s.content.chars().count()).sum();
                    // Trailing space + 1 char leading pad keeps the bar breathing.
                    let pad = barWidth.saturating_sub(rightWidth + 1);
                    spans.push(Span::raw(" ".repeat(pad)));
                    spans.extend(rightSpans);
                    spans.push(Span::raw(" "));
                }

                let statusBar =
                    Paragraph::new(Line::from(spans)).style(Style::default().bg(barBg).fg(barFg));
                frame.render_widget(statusBar, vChunks[1]);

                // Session picker overlay.
                if let Some(ref mut picker) = sessionPicker {
                    picker.render(area, frame.buffer_mut());
                }

                // Rewind picker overlay.
                if let Some(ref mut picker) = rewindPicker {
                    picker.render(area, frame.buffer_mut());
                }

                // Fork picker overlay.
                if let Some(ref mut picker) = forkPicker {
                    picker.render(area, frame.buffer_mut());
                }

                // MCP panel overlay.
                if let Some(ref mut panel) = mcpPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // LSP panel overlay.
                if let Some(ref mut panel) = lspPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Permissions panel overlay.
                if let Some(ref mut panel) = permissionsPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Subagent panel overlay.
                if let Some(ref mut panel) = subagentPanel {
                    panel.render(area, frame.buffer_mut(), agentPanel);
                }

                // Help popup overlay.
                if helpPopupOpen {
                    renderHelpPopup(area, frame.buffer_mut());
                }
            })?;
        }

        // Drain PTY output.
        while let Ok(bytes) = shellIo.outputRx.try_recv() {
            termState.process(&bytes);
            needsRedraw = true;
        }

        // Drain session log events (monotone stream, no replies).
        while let Ok(event) = logRx.try_recv() {
            needsRedraw = true;
            match event {
                LogEvent::ContentDelta(text) => agentPanel.appendContent(&text),
                LogEvent::ReasoningDelta(text) => agentPanel.appendReasoning(&text),
                LogEvent::ToolResult { name, output } => {
                    // Task tool results are handled by SubagentComplete — don't double-render.
                    if name != "task" {
                        agentPanel.pushToolResult(&name, &output);
                    }
                }
                LogEvent::ToolStarted { name, summary } => {
                    if name != "task" {
                        agentPanel.toolStarted(&name, &summary);
                    }
                }
                LogEvent::ToolCallPending { index, name } => {
                    agentPanel.toolCallPending(index, &name);
                }
                LogEvent::ToolCallProgress { index, bytes } => {
                    agentPanel.toolCallProgress(index, bytes);
                }
                LogEvent::ToolCallPreview { index, preview } => {
                    agentPanel.toolCallPreview(index, &preview);
                }
                LogEvent::ToolAutoApproved { name, summary } => {
                    agentPanel.toolApproved(&format!("{name}: {summary}"));
                }
                LogEvent::ToolDenied { name } => {
                    agentPanel.toolDenied(&name);
                }
                LogEvent::ToolAutoDenied { name, summary } => {
                    agentPanel.toolAutoDenied(&name, &summary);
                }
                LogEvent::TurnAborted { name } => {
                    agentPanel.pushError(&format!("Turn aborted: {name} not permitted"));
                }
                LogEvent::TurnComplete => {
                    agentPanel.finishTurn();
                }
                LogEvent::TurnCancelled => {
                    agentPanel.finalizeCancelled();
                }
                LogEvent::SteerInjected { texts } => {
                    agentPanel.promoteQueue(&texts);
                }
                LogEvent::TopicChanged { label } => {
                    currentTopic = Some(label);
                    let glyph = if agentPanel.isActive() {
                        titleSpinner.current()
                    } else {
                        TITLE_IDLE_GLYPH
                    };
                    writeTerminalTitle(glyph, currentTopic.as_deref());
                }
                LogEvent::LspHint {
                    serverId,
                    installHint,
                } => {
                    let msg = format!(
                        "\u{2699}\u{FE0E} {} not found \u{2014} `{}`",
                        serverId, installHint,
                    );
                    agentPanel.pushCommandResult(&msg);
                }
                LogEvent::TokenUpdate {
                    contextTokens,
                    sessionCost: sc,
                    cacheReadTokens,
                    ..
                } => {
                    tokenCount = contextTokens;
                    sessionCost = sc;
                    if cacheReadTokens > 0 {
                        lastCacheHitAt = Some(Instant::now());
                    }
                }
                LogEvent::Retrying {
                    attempt,
                    maxAttempts,
                } => {
                    agentPanel.showRetrying(attempt, maxAttempts);
                }
                LogEvent::Error(msg) => {
                    agentPanel.pushError(&msg);
                    // A fatal error ends the turn — stop the throbber so the
                    // user isn't misled into thinking the model is still
                    // working, and surface the retry/continue hint.
                    agentPanel.finishTurn();
                    agentPanel.errorHint = true;
                }
                LogEvent::BudgetWarning {
                    sessionCost: sc,
                    limit,
                } => {
                    let msg = format!(
                        "\u{26A0}\u{FE0E} Session cost ({}) exceeded limit ({})",
                        construct::cost::formatCost(sc),
                        construct::cost::formatCost(limit),
                    );
                    agentPanel.pushCommandResult(&msg);
                }
                LogEvent::CompactionStarted { stage } => {
                    tracing::info!(stage = %stage, "compaction started");
                }
                LogEvent::CompactionComplete {
                    stage,
                    reduction,
                    markerBlock,
                } => {
                    tracing::info!(stage = %stage, reduction = %reduction, "compaction complete");
                    if let Some(blockIdx) = markerBlock {
                        agentPanel.pushCompactionMarker(&stage, blockIdx);
                    }
                }
                LogEvent::Cleared => {
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    currentTopic = None;
                    writeTerminalTitle(TITLE_IDLE_GLYPH, None);
                }
                LogEvent::Rewound { targetTurnId } => {
                    rewindPicker = None;
                    forkPicker = None;
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    if let Some(msg) = pendingRewindMessage.take() {
                        agentPanel.textArea.setText(&msg);
                    }
                    // Restore image attachments from the rewound turn.
                    if let Some(atts) = pendingRewindAttachments.take() {
                        use base64::Engine;
                        for att in atts {
                            let data = base64::engine::general_purpose::STANDARD
                                .decode(&att.data)
                                .unwrap_or_default();
                            agentPanel.addAttachment(construct::session::Attachment {
                                mimeType: att.mimeType.clone(),
                                data,
                                label: format!("restored image"),
                                rgbaDimensions: None,
                            });
                        }
                    }
                    tracing::info!(target = %targetTurnId, "conversation rewound");
                }
                LogEvent::SessionRestored { turns, markers } => {
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    replayTranscript(agentPanel, &turns);
                    for (stage, blockIdx) in &markers {
                        agentPanel.pushCompactionMarker(stage, *blockIdx);
                    }
                }
                LogEvent::SubagentStarted {
                    sessionId,
                    agentType,
                    prompt,
                } => {
                    tracing::info!(agent = %agentType, "subagent started");
                    agentPanel.subagentStarted(&sessionId, &agentType, &prompt);
                }
                LogEvent::SubagentEvent {
                    sessionId: _,
                    event,
                } => {
                    match *event {
                        LogEvent::ToolAutoApproved {
                            ref name,
                            ref summary,
                        } => {
                            agentPanel.subagentToolLine(name, summary);
                        }
                        LogEvent::ToolStarted {
                            ref name,
                            ref summary,
                        } => {
                            agentPanel.subagentToolLine(name, summary);
                        }
                        LogEvent::ToolResult {
                            ref name,
                            ref output,
                        } => {
                            // Brief one-liner for the inline block.
                            let brief = if output.len() > 60 {
                                format!("{}\u{2026}", &output[..output.floor_char_boundary(60)])
                            } else {
                                output.clone()
                            };
                            agentPanel.subagentToolLine(name, &brief);
                            // Full output for the overlay transcript.
                            agentPanel.subagentToolResult(name, output);
                        }
                        LogEvent::ContentDelta(ref text) => {
                            agentPanel.subagentContent(text);
                        }
                        LogEvent::Error(ref msg) => {
                            agentPanel.subagentToolLine("error", msg);
                        }
                        _ => {}
                    }
                }
                LogEvent::SubagentShellOutput { data, .. } => {
                    agentPanel.feedSubagentShell(&data);
                }
                LogEvent::SubagentComplete {
                    agentType,
                    turns,
                    content,
                    ..
                } => {
                    tracing::info!(agent = %agentType, turns = turns, "subagent completed");
                    agentPanel.subagentComplete(&agentType, turns, &content);
                }
                LogEvent::ScratchpadRecovered {
                    matchedTag,
                    snippet,
                    recoveredChars,
                } => {
                    let msg = format!(
                        "\u{26A0}\u{FE0E} scratchpad close recovered (`{}`, {} chars): \"{}\"",
                        matchedTag, recoveredChars, snippet,
                    );
                    agentPanel.pushCommandResult(&msg);
                }
            }
        }

        // Drain session → TUI requests (permits). Each variant carries a
        // oneshot reply that is resolved when the user responds via the
        // permit prompt.
        while let Ok(req) = sessionRequestRx.try_recv() {
            needsRedraw = true;
            match req {
                SessionRequest::Permit {
                    origin,
                    name,
                    summary,
                    args,
                    diff,
                    explanation,
                    impact,
                    reply,
                } => {
                    let isSubagent =
                        matches!(origin, construct::control::PermitOrigin::Subagent { .. });
                    agentPanel.showToolRequest(
                        &name,
                        &summary,
                        &args,
                        diff,
                        explanation,
                        impact,
                        origin,
                    );
                    pendingPermitReply = Some(reply);
                    // Parent permits auto-close the popup so the main-panel
                    // prompt becomes visible. Subagent permits stay routed
                    // through the popup (which renders them inline).
                    if !isSubagent && subagentPanel.is_some() {
                        subagentPanel = None;
                    }
                }
            }
        }

        // Drain deck-internal updates (replies to slash-command requests).
        while let Ok(update) = deckUpdateRx.try_recv() {
            needsRedraw = true;
            match update {
                DeckUpdate::McpStatus(status) => {
                    mcpPanel = Some(McpPanel::new(
                        status.servers,
                        status.totalTools,
                        status.searchMode,
                        status.configPath,
                    ));
                }
                DeckUpdate::LspStatus(status) => {
                    lspPanel = Some(LspPanel::new(status.servers));
                }
                DeckUpdate::PermissionsStatus(status) => {
                    permissionsPanel = Some(crate::permissions_panel::PermissionsPanel::new(
                        status.defaultMode,
                        status.rules,
                        status.source,
                        status.configPath,
                    ));
                }
                DeckUpdate::ContextDisplay(state) => {
                    agentPanel.pushContextDisplay(state);
                }
                DeckUpdate::RewindOptions(turns) => {
                    rewindPicker = Some(RewindPicker::new(&turns));
                }
                DeckUpdate::Forks(forks) => {
                    forkPicker = Some(ForkPicker::new(&forks));
                }
                DeckUpdate::ShowResult(text) => {
                    agentPanel.pushCommandResult(&text);
                }
                DeckUpdate::CommandAck(ack) => {
                    if forkPicker.is_some() && !ack.ok {
                        if let Some(ref mut picker) = forkPicker {
                            picker.switchFailed(ack.message);
                        }
                    } else if !ack.message.is_empty() {
                        agentPanel.pushCommandResult(&ack.message);
                    }
                }
                DeckUpdate::ResumeResult(ack) => {
                    if ack.ok {
                        sessionPicker = None;
                        agentPanel
                            .entries
                            .push(crate::agent_panel::PanelEntry::SessionNotice(ack.message));
                    } else if let Some(ref mut picker) = sessionPicker {
                        picker.resumeFailed(ack.message);
                    } else {
                        agentPanel.pushError(&ack.message);
                    }
                }
            }
        }

        // Tick throbber animation (wall-clock gated).
        if agentPanel.tickThrobber() {
            needsRedraw = true;
        }

        // Tick animated title spinner while a turn is active.
        let nowAnimating = currentTopic.is_some() && agentPanel.isActive();
        if nowAnimating {
            if titleSpinner.tick() {
                writeTerminalTitle(titleSpinner.current(), currentTopic.as_deref());
            }
        } else if titleWasAnimating {
            // Agent just finished — drop back to the static idle glyph.
            writeTerminalTitle(TITLE_IDLE_GLYPH, currentTopic.as_deref());
        }
        titleWasAnimating = nowAnimating;

        // Advance character reveal buffer.
        if agentPanel.tickReveal() {
            needsRedraw = true;
        }

        // Clear the quit hint after the double-tap window expires.
        if let Some(t) = lastQuitPress {
            if t.elapsed() >= Duration::from_secs(1) {
                lastQuitPress = None;
                needsRedraw = true;
            }
        }

        // Handle input.
        let (quit, hadInput, wasResized) = handleInput(
            focus,
            shellIo,
            termState,
            agentPanel,
            selState,
            scrollLock,
            userInputTx,
            requestTx,
            deckUpdateTx,
            cancelTx,
            steerTx,
            &mut sessionPicker,
            &mut rewindPicker,
            &mut forkPicker,
            &mut pendingRewindMessage,
            &mut pendingRewindAttachments,
            &mut mcpPanel,
            &mut lspPanel,
            &mut subagentPanel,
            &mut permissionsPanel,
            &mut pendingPermitReply,
            &projectDir,
            &mut lastQuitPress,
            &mut helpPopupOpen,
        )
        .await?;
        if quit {
            break;
        }
        if hadInput {
            needsRedraw = true;
        }
        if wasResized {
            terminal.clear()?;
        }

        tokio::task::yield_now().await;
    }

    Ok(())
}

/// Drain all pending input events. Returns (quit, hadInput).
#[allow(clippy::too_many_arguments)]
async fn handleInput(
    focus: &mut Focus,
    shellIo: &ShellIo,
    termState: &mut TerminalState,
    agentPanel: &mut AgentPanel,
    selState: &mut SelectionState,
    scrollLock: &mut ScrollAxisLock,
    userInputTx: &mpsc::Sender<construct::session::UserInput>,
    requestTx: &mpsc::Sender<TuiRequest>,
    deckUpdateTx: &mpsc::Sender<DeckUpdate>,
    cancelTx: &watch::Sender<bool>,
    steerTx: &mpsc::Sender<construct::session::UserInput>,
    sessionPicker: &mut Option<SessionPicker>,
    rewindPicker: &mut Option<RewindPicker>,
    forkPicker: &mut Option<ForkPicker>,
    pendingRewindMessage: &mut Option<String>,
    pendingRewindAttachments: &mut Option<Vec<construct::transcript::TurnAttachment>>,
    mcpPanel: &mut Option<McpPanel>,
    lspPanel: &mut Option<LspPanel>,
    subagentPanel: &mut Option<crate::subagent_panel::SubagentPanel>,
    permissionsPanel: &mut Option<crate::permissions_panel::PermissionsPanel>,
    pendingPermitReply: &mut Option<oneshot::Sender<construct::permissions::PermitResponse>>,
    projectDir: &str,
    lastQuitPress: &mut Option<Instant>,
    helpPopupOpen: &mut bool,
) -> Result<(bool, bool, bool)> {
    // Wait up to 16ms for the first event.
    if !event::poll(Duration::from_millis(16))? {
        return Ok((false, false, false));
    }

    // Drain all queued events to avoid input lag (especially trackpad momentum).
    let mut hadInput = false;
    let mut resized = false;
    loop {
        match event::read()? {
            Event::Key(key) => {
                if key.kind == event::KeyEventKind::Release {
                    // Poll for more without blocking.
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                    continue;
                }

                hadInput = true;

                // Clear selections on content/navigation keys, but not system
                // shortcuts (Cmd+key on macOS) so Cmd+C doesn't nuke the highlight.
                if !key.modifiers.contains(KeyModifiers::SUPER) {
                    selState.termSelection = None;
                    selState.agentSelection = None;
                    agentPanel.textArea.clearSelection();
                }

                // Global keybindings.
                // Double-tap Ctrl+Q to quit — prevents accidental exits.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('q') {
                    const DOUBLE_TAP_WINDOW: Duration = Duration::from_secs(1);
                    if let Some(prev) = *lastQuitPress {
                        if prev.elapsed() < DOUBLE_TAP_WINDOW {
                            return Ok((true, true, false));
                        }
                    }
                    *lastQuitPress = Some(Instant::now());
                    break;
                }

                // Ctrl+L: force full terminal redraw to fix rendering artifacts.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
                    resized = true;
                    break;
                }

                // Ctrl+H: toggle the hotkey-tips popup.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('h') {
                    *helpPopupOpen = !*helpPopupOpen;
                    break;
                }

                // Any key dismisses the help popup (and does not propagate).
                if *helpPopupOpen {
                    *helpPopupOpen = false;
                    break;
                }

                // Error-mode recovery bindings. Only active when the last
                // turn fatally errored (hint is showing). Both discard the
                // error state before dispatching the command.
                if agentPanel.errorHint && key.modifiers.contains(KeyModifiers::CONTROL) {
                    match key.code {
                        KeyCode::Char('r') => {
                            agentPanel.errorHint = false;
                            let _ = cancelTx.send(false);
                            spawnAckRequest(requestTx.clone(), deckUpdateTx.clone(), |reply| {
                                TuiRequest::RetryLastTurn { reply }
                            });
                            break;
                        }
                        KeyCode::Char(' ') => {
                            agentPanel.errorHint = false;
                            let _ = cancelTx.send(false);
                            spawnAckRequest(requestTx.clone(), deckUpdateTx.clone(), |reply| {
                                TuiRequest::ContinueLastTurn { reply }
                            });
                            break;
                        }
                        _ => {}
                    }
                }

                // Subagent panel overlay. Navigation keys (Tab, scroll, close)
                // are always consumed. During a subagent permit, action keys
                // (y/n/A/D, Shift+arrows, custom-pattern chars) fall through
                // to the permit dispatcher below.
                if let Some(ref mut panel) = *subagentPanel {
                    let consumed = panel.consumedKey(&key, agentPanel);
                    let shouldClose = panel.handleKey(key, agentPanel);
                    if shouldClose {
                        *subagentPanel = None;
                        break;
                    }
                    if consumed {
                        break;
                    }
                }

                // Cancel running turn with Escape — immediate visual feedback.
                if key.code == KeyCode::Esc && agentPanel.isActive() {
                    let _ = cancelTx.send(true);
                    agentPanel.finalizeCancelled();
                    break;
                }

                if key.code == KeyCode::Tab {
                    // Don't switch focus when an overlay is active or completion menu is open.
                    if sessionPicker.is_some()
                        || rewindPicker.is_some()
                        || forkPicker.is_some()
                        || mcpPanel.is_some()
                        || lspPanel.is_some()
                        || permissionsPanel.is_some()
                    {
                        // Let the overlay handle Tab (falls through to overlay dispatch below).
                    } else if !(*focus == Focus::Agent && agentPanel.completionActive()) {
                        *focus = match focus {
                            Focus::Terminal => Focus::Agent,
                            Focus::Agent => Focus::Terminal,
                        };
                        break;
                    }
                }

                // Cmd+C: copy active selection to clipboard.
                if key.modifiers.contains(KeyModifiers::SUPER) && key.code == KeyCode::Char('c') {
                    if let Some(text) = agentPanel.textArea.selectedText() {
                        selection::copyToClipboard(&text);
                    } else if selState.agentSelection.is_some() {
                        selState.pendingCopy = Some(PanelId::Agent);
                    } else if selState.termSelection.is_some() {
                        selState.pendingCopy = Some(PanelId::Terminal);
                    }
                    break;
                }

                // Permission prompt takes priority regardless of focus.
                if agentPanel.pendingPermit {
                    use construct::permissions::PermitResponse;

                    // Resolve the oneshot reply stashed when SessionRequest::Permit arrived.
                    // Same path for top-level and subagent permits — the origin is folded
                    // into the request variant before it reaches the TUI.
                    macro_rules! sendPermit {
                        ($resp:expr) => {
                            if let Some(tx) = pendingPermitReply.take() {
                                let _ = tx.send($resp);
                            }
                        };
                    }

                    match key.code {
                        KeyCode::Char('y') => {
                            agentPanel.approvePending();
                            sendPermit!(PermitResponse::Allow);
                        }
                        // Shift+A: always allow (persist to project config).
                        KeyCode::Char('A') => {
                            let pattern = agentPanel.selectedPattern();
                            agentPanel.approvePending();
                            sendPermit!(PermitResponse::AlwaysAllow { pattern });
                        }
                        KeyCode::Char('n') => {
                            agentPanel.denyPending();
                            sendPermit!(PermitResponse::Deny);
                        }
                        // Shift+D: always deny (persist to project config).
                        KeyCode::Char('D') => {
                            let pattern = agentPanel.selectedPattern();
                            agentPanel.denyPending();
                            sendPermit!(PermitResponse::AlwaysDeny { pattern });
                        }
                        // Shift+Up/Down: navigate pattern selector (patterns are for persistent decisions).
                        KeyCode::Up if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            agentPanel.prevPattern();
                        }
                        KeyCode::Down if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            agentPanel.nextPattern();
                        }
                        // Custom pattern text input (when custom field is selected).
                        KeyCode::Char(c) if agentPanel.isEditingCustom() => {
                            agentPanel.customPatternInsert(c);
                        }
                        KeyCode::Backspace if agentPanel.isEditingCustom() => {
                            agentPanel.customPatternBackspace();
                        }
                        KeyCode::Char('v') | KeyCode::Char('V') => {
                            // Open the subagent panel — Live mode reads
                            // transcript + shell from agentPanel.activeSubagent.
                            if agentPanel.activeSubagent.is_some() {
                                *subagentPanel = Some(crate::subagent_panel::SubagentPanel::live());
                            }
                        }
                        _ => {}
                    }
                    break;
                }

                // Session picker intercepts all keys when active.
                if let Some(picker) = sessionPicker {
                    match picker.handleKey(key) {
                        PickerAction::Close => {
                            *sessionPicker = None;
                        }
                        PickerAction::Select(id) => {
                            // Picker stays open — closed on ResumeResult reply.
                            let requestTxC = requestTx.clone();
                            let deckTx = deckUpdateTx.clone();
                            tokio::spawn(async move {
                                let (rTx, rRx) = oneshot::channel();
                                let _ = requestTxC
                                    .send(TuiRequest::ResumeSession {
                                        sessionId: id,
                                        reply: rTx,
                                    })
                                    .await;
                                if let Ok(ack) = rRx.await {
                                    let _ = deckTx.send(DeckUpdate::ResumeResult(ack)).await;
                                }
                            });
                        }
                        PickerAction::None => {}
                    }
                    break;
                }

                // Rewind picker intercepts all keys when active.
                if let Some(picker) = rewindPicker {
                    match picker.handleKey(key) {
                        RewindAction::Close => {
                            *rewindPicker = None;
                        }
                        RewindAction::Rewind {
                            target,
                            userMessage,
                            attachments,
                        } => {
                            *pendingRewindMessage = Some(userMessage);
                            *pendingRewindAttachments = attachments;
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::Rewind {
                                    target,
                                    saveFork: false,
                                    reply,
                                },
                            );
                        }
                        RewindAction::ForkAndRewind {
                            target,
                            userMessage,
                            attachments,
                        } => {
                            *pendingRewindMessage = Some(userMessage);
                            *pendingRewindAttachments = attachments;
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::Rewind {
                                    target,
                                    saveFork: true,
                                    reply,
                                },
                            );
                        }
                        RewindAction::None => {}
                    }
                    break;
                }

                // Fork picker intercepts all keys when active.
                if let Some(picker) = forkPicker {
                    match picker.handleKey(key) {
                        ForkAction::Close => {
                            *forkPicker = None;
                        }
                        ForkAction::Switch(id) => {
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SwitchFork { forkId: id, reply },
                            );
                        }
                        ForkAction::None => {}
                    }
                    break;
                }

                // MCP panel intercepts all keys when active.
                if let Some(panel) = mcpPanel {
                    match panel.handleKey(key) {
                        McpPanelAction::Close => {
                            *mcpPanel = None;
                        }
                        McpPanelAction::None => {}
                    }
                    break;
                }

                // LSP panel intercepts all keys when active.
                if let Some(panel) = lspPanel {
                    match panel.handleKey(key) {
                        LspPanelAction::Close => {
                            *lspPanel = None;
                        }
                        LspPanelAction::Install { serverId, command } => {
                            *lspPanel = None;
                            // Run install in the shared terminal.
                            agentPanel.pushCommandResult(&format!(
                                "\u{2699}\u{FE0E} Installing {serverId}: {command}",
                            ));
                            let cmdBytes = format!("{command}\n").into_bytes();
                            let _ = shellIo.inputTx.try_send(cmdBytes);
                        }
                        LspPanelAction::None => {}
                    }
                    break;
                }

                // Permissions panel intercepts all keys when active.
                if let Some(panel) = permissionsPanel {
                    use crate::permissions_panel::PermPanelAction;
                    match panel.handleKey(key) {
                        PermPanelAction::Close => {
                            *permissionsPanel = None;
                        }
                        PermPanelAction::Save { defaultMode, rules } => {
                            *permissionsPanel = None;
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SavePermissions {
                                    defaultMode,
                                    rules,
                                    reply,
                                },
                            );
                        }
                        PermPanelAction::None => {}
                    }
                    break;
                }

                match focus {
                    Focus::Terminal => {
                        if key.modifiers.contains(KeyModifiers::SUPER) {
                            if !event::poll(Duration::ZERO)? {
                                break;
                            }
                            continue;
                        }
                        // Ctrl+C triggers the killchain for captured commands.
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c')
                        {
                            let _ = shellIo.killTx.try_send(());
                        } else if let Some(bytes) = keyToBytes(&key) {
                            if termState.displayOffset() > 0 {
                                termState.scrollToBottom();
                            }
                            let _ = shellIo.inputTx.try_send(bytes);
                        }
                    }
                    Focus::Agent => {
                        let mods = key.modifiers;
                        let ctrl = mods.contains(KeyModifiers::CONTROL);
                        let shift = mods.contains(KeyModifiers::SHIFT);
                        let supr = mods.contains(KeyModifiers::SUPER);
                        let alt = mods.contains(KeyModifiers::ALT);

                        // Completion menu intercepts (before textarea borrow).
                        let mut completionHandled = false;
                        if agentPanel.completionActive() {
                            completionHandled = true;
                            match key.code {
                                KeyCode::Tab => {
                                    if let Some(completed) = agentPanel.completeSelected() {
                                        agentPanel.textArea.setText(&completed);
                                    }
                                }
                                KeyCode::Up => agentPanel.selectPrev(),
                                KeyCode::Down => agentPanel.selectNext(),
                                KeyCode::Esc => agentPanel.dismissCompletion(),
                                KeyCode::Enter if !shift => {
                                    // Accept completion then fall through to execute.
                                    if let Some(completed) = agentPanel.completeSelected() {
                                        agentPanel.textArea.setText(&completed);
                                    }
                                    completionHandled = false;
                                }
                                _ => completionHandled = false,
                            }
                        }

                        let mut navigatingHistory = false;
                        if !completionHandled {
                            // Handle attachment-related keys before borrowing textArea.
                            if ctrl
                                && key.code == KeyCode::Char('d')
                                && agentPanel.attachmentCount() > 0
                            {
                                agentPanel.removeLastAttachment();
                                break;
                            }
                            if ctrl && key.code == KeyCode::Char('v') {
                                if let Ok(mut cb) = arboard::Clipboard::new() {
                                    if let Ok(imgData) = cb.get_image() {
                                        // Store raw RGBA — PNG encoding deferred to submit.
                                        agentPanel.addAttachment(construct::session::Attachment {
                                            mimeType: "image/rgba".into(),
                                            data: imgData.bytes.to_vec(),
                                            label: format!(
                                                "pasted image ({}x{})",
                                                imgData.width, imgData.height,
                                            ),
                                            rgbaDimensions: Some((
                                                imgData.width as u32,
                                                imgData.height as u32,
                                            )),
                                        });
                                    } else if let Ok(text) = cb.get_text() {
                                        let normalized =
                                            text.replace("\r\n", "\n").replace('\r', "\n");
                                        agentPanel.textArea.insertStr(&normalized);
                                    }
                                }
                                break;
                            }

                            // Pop queued message on Up before borrowing textArea.
                            if key.code == KeyCode::Up
                                && agentPanel.textArea.isAtFirstLine()
                                && agentPanel.isActive()
                                && agentPanel.queuedCount() > 0
                            {
                                if let Some(input) = agentPanel.popQueuedMessage() {
                                    agentPanel.textArea.setText(&input.text);
                                    agentPanel.restoreAttachments(input.attachments);
                                }
                                break;
                            }

                            let ta = &mut agentPanel.textArea;
                            match key.code {
                                KeyCode::Enter if shift => ta.insert('\n'),
                                KeyCode::Enter => {
                                    if let Some(msg) = ta.submit() {
                                        if let Some(output) = crate::command::tryHandle(&msg) {
                                            agentPanel.history.push(&msg);
                                            agentPanel.pushCommand(&msg);
                                            match output {
                                                crate::command::CommandOutput::Inline(text) => {
                                                    agentPanel.pushCommandResult(&text);
                                                }
                                                crate::command::CommandOutput::Action(
                                                    crate::command::CommandAction::Resume {
                                                        sessionId: None,
                                                    },
                                                ) => {
                                                    // Open the interactive picker.
                                                    *sessionPicker =
                                                        Some(SessionPicker::new(projectDir));
                                                }
                                                crate::command::CommandOutput::Action(action) => {
                                                    dispatchSlashCommand(
                                                        action,
                                                        requestTx.clone(),
                                                        deckUpdateTx.clone(),
                                                    );
                                                }
                                            }
                                        } else if agentPanel.isActive() {
                                            // Queue for mid-turn injection.
                                            agentPanel.history.push(&msg);
                                            let input = construct::session::UserInput {
                                                text: msg,
                                                attachments: agentPanel.takeAttachments(),
                                            };
                                            agentPanel.queueMessage(input.clone());
                                            let _ = steerTx.try_send(input);
                                        } else {
                                            let _ = cancelTx.send(false);
                                            agentPanel.history.push(&msg);
                                            agentPanel.pushUser(&msg);
                                            // New user input — supersedes any prior error state.
                                            agentPanel.errorHint = false;
                                            let input = construct::session::UserInput {
                                                text: msg,
                                                attachments: agentPanel.takeAttachments(),
                                            };
                                            let _ = userInputTx.send(input).await;
                                        }
                                    }
                                }
                                KeyCode::Char('a') if ctrl => ta.moveHome(),
                                KeyCode::Char('e') if ctrl => ta.moveEnd(),
                                KeyCode::Char('k') if ctrl => ta.killToEnd(),
                                KeyCode::Char('u') if ctrl => ta.killToStart(),
                                KeyCode::Char('y') if ctrl => ta.yank(),
                                KeyCode::Char('t') if ctrl => agentPanel.toggleThinking(),
                                KeyCode::Char(c) if !supr => ta.insert(c),
                                KeyCode::Backspace if supr => ta.killToStart(),
                                KeyCode::Backspace if alt => ta.deleteWordLeft(),
                                KeyCode::Backspace => ta.backspace(),
                                KeyCode::Delete => ta.delete(),
                                KeyCode::Left if supr => ta.moveHome(),
                                KeyCode::Right if supr => ta.moveEnd(),
                                KeyCode::Left if alt => ta.moveWordLeft(),
                                KeyCode::Right if alt => ta.moveWordRight(),
                                KeyCode::Left if ctrl => ta.moveWordLeft(),
                                KeyCode::Right if ctrl => ta.moveWordRight(),
                                KeyCode::Left => ta.moveLeft(),
                                KeyCode::Right => ta.moveRight(),
                                KeyCode::Home => ta.moveHome(),
                                KeyCode::End => ta.moveEnd(),
                                KeyCode::Up => {
                                    if ta.isAtFirstLine() {
                                        navigatingHistory = true;
                                        let currentText = ta.text().to_string();
                                        if let Some(entry) =
                                            agentPanel.history.navigateUp(&currentText)
                                        {
                                            let entry = entry.to_string();
                                            ta.setText(&entry);
                                        }
                                    } else {
                                        ta.moveUp();
                                    }
                                }
                                KeyCode::Down => {
                                    if ta.isAtLastLine() {
                                        navigatingHistory = true;
                                        if let Some(entry) = agentPanel.history.navigateDown() {
                                            let entry = entry.to_string();
                                            ta.setText(&entry);
                                        }
                                    } else {
                                        ta.moveDown();
                                    }
                                }
                                KeyCode::PageUp => agentPanel.scrollUp(10),
                                KeyCode::PageDown => agentPanel.scrollDown(10),
                                _ => {}
                            }
                        }

                        // Reset history navigation on any non-navigation key.
                        if !navigatingHistory {
                            agentPanel.history.resetCursor();
                        }

                        // Update completion after every keystroke.
                        let currentText = agentPanel.textArea.text().to_string();
                        agentPanel.updateCompletion(&currentText);
                    }
                }
            }
            Event::Mouse(mouse) => {
                // Subagent overlay: route ALL mouse events here when open.
                // Click-outside closes the popup; click-inside on the permit
                // overlay falls through to the main mouse handler so the
                // existing permit click logic (copy button etc.) still works.
                if let Some(ref mut panel) = *subagentPanel {
                    use crate::subagent_panel::SubagentMouseAction;
                    match panel.handleMouse(mouse, agentPanel) {
                        SubagentMouseAction::Handled => {
                            hadInput = true;
                        }
                        SubagentMouseAction::ClickOutside => {
                            *subagentPanel = None;
                            hadInput = true;
                        }
                    }
                    break;
                }
                // Other overlay panels consume all mouse events to prevent click-through.
                if sessionPicker.is_some()
                    || rewindPicker.is_some()
                    || forkPicker.is_some()
                    || mcpPanel.is_some()
                    || lspPanel.is_some()
                    || permissionsPanel.is_some()
                {
                    break;
                }
                if handleMouse(
                    mouse,
                    focus,
                    agentPanel,
                    termState,
                    selState,
                    shellIo,
                    scrollLock,
                    subagentPanel,
                ) {
                    hadInput = true;
                }
            }
            Event::Paste(text) => {
                hadInput = true;
                // Popup overlay swallows paste — main panel input is covered.
                if subagentPanel.is_some() {
                    if agentPanel.pendingPermit && agentPanel.isEditingCustom() {
                        for ch in text.chars() {
                            agentPanel.customPatternInsert(ch);
                        }
                    }
                    break;
                }
                match *focus {
                    Focus::Agent if !agentPanel.pendingPermit => {
                        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                        agentPanel.textArea.insertStr(&normalized);
                    }
                    Focus::Terminal => {
                        if termState.displayOffset() > 0 {
                            termState.scrollToBottom();
                        }
                        // Shells treat CR as "execute"; normalize to LF so multi-line
                        // pastes land as a single buffered command when possible.
                        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                        let payload = if termState.bracketedPaste() {
                            let mut buf = Vec::with_capacity(normalized.len() + 12);
                            buf.extend_from_slice(b"\x1b[200~");
                            buf.extend_from_slice(normalized.as_bytes());
                            buf.extend_from_slice(b"\x1b[201~");
                            buf
                        } else {
                            normalized.into_bytes()
                        };
                        let _ = shellIo.inputTx.try_send(payload);
                    }
                    _ => {}
                }
            }
            Event::Resize(cols, rows) => {
                hadInput = true;
                resized = true;
                let termCols = (cols * 3 / 5).saturating_sub(2);
                let termRows = rows.saturating_sub(3);
                let _ = shellIo.resizeTx.try_send((termCols, termRows));
                termState.resize(termCols, termRows);
            }
            _ => {}
        }

        // Keep draining if more events are queued.
        if !event::poll(Duration::ZERO)? {
            break;
        }
    }

    Ok((false, hadInput, resized))
}

/// Handle mouse events — selection, scroll wheel.
/// Returns true if the event modified state (needs redraw).
fn handleMouse(
    mouse: event::MouseEvent,
    focus: &mut Focus,
    agentPanel: &mut AgentPanel,
    termState: &mut TerminalState,
    selState: &mut SelectionState,
    _shellIo: &ShellIo,
    scrollLock: &mut ScrollAxisLock,
    subagentPanel: &mut Option<crate::subagent_panel::SubagentPanel>,
) -> bool {
    // Resolve display offset for the given panel.
    fn panelOffset(panel: PanelId, termState: &TerminalState, agentPanel: &AgentPanel) -> u16 {
        match panel {
            PanelId::Terminal => termState.displayOffset() as u16,
            PanelId::Agent => agentPanel.displayOffset(),
            PanelId::Input => 0,
        }
    }

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            // Check input area first (not part of panel hit-testing).
            if selState
                .inputContentRect
                .contains((mouse.column, mouse.row).into())
            {
                // Permission prompt: check for copy button click on code block top border.
                if agentPanel.pendingPermit {
                    let localRow = mouse.row.saturating_sub(selState.inputContentRect.y);
                    let localCol = mouse.column.saturating_sub(selState.inputContentRect.x);
                    // Top border row — check if click is on "copy" label.
                    if localRow == 0 && localCol + 6 >= selState.inputContentRect.width {
                        if let Some(cmd) = agentPanel.pendingCommand() {
                            crate::selection::copyToClipboard(cmd);
                            agentPanel.flashCopied();
                        }
                    }
                    return true;
                }
                *focus = Focus::Agent;
                let localCol = mouse.column.saturating_sub(selState.inputContentRect.x);
                let localRow = mouse.row.saturating_sub(selState.inputContentRect.y);
                let contentWidth = selState.inputContentRect.width.saturating_sub(2);
                let contentCol = localCol.saturating_sub(2);
                agentPanel
                    .textArea
                    .mouseDown(contentCol, localRow, contentWidth);
                selState.selectingIn = Some(PanelId::Input);
                selState.termSelection = None;
                selState.agentSelection = None;
                return true;
            }

            // Clear input selection when clicking in a panel.
            agentPanel.textArea.clearSelection();

            let panel = selState.hitTest(mouse.column, mouse.row);
            if let Some(panel) = panel {
                *focus = match panel {
                    PanelId::Terminal => Focus::Terminal,
                    PanelId::Agent | PanelId::Input => Focus::Agent,
                };

                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let gridLine =
                    selection::toGridLine(screenRow, panelOffset(panel, termState, agentPanel));

                // Single click on a reasoning header toggles it.
                if panel == PanelId::Agent && agentPanel.toggleReasoningAtGridLine(gridLine) {
                    return true;
                }

                // Click on code block "copy" label copies the block content.
                if panel == PanelId::Agent && agentPanel.tryCopyCodeBlock(gridLine, col) {
                    return true;
                }

                // Click on code block top/bottom border toggles expand/collapse.
                if panel == PanelId::Agent && agentPanel.tryToggleCodeBlock(gridLine) {
                    return true;
                }

                // Click on subagent header [view] opens the overlay panel.
                if panel == PanelId::Agent && agentPanel.isSubagentHeaderLine(gridLine) {
                    if agentPanel.activeSubagent.is_some() {
                        // Live subagent — popup reads from agentPanel each frame.
                        *subagentPanel = Some(crate::subagent_panel::SubagentPanel::live());
                        return true;
                    } else if let Some((agentType, sid)) = agentPanel.lastSubagentSession() {
                        // Resumed session — load child transcript on demand.
                        let agentType = agentType.to_string();
                        let sid = sid.to_string();
                        if let Ok(transcript) = construct::transcript::Transcript::open(&sid) {
                            if let Ok(turns) = transcript.loadAll() {
                                // Build a temporary AgentPanel to replay the
                                // child transcript into PanelEntries.
                                let mut tmp = crate::agent_panel::AgentPanel::new();
                                replayTranscript(&mut tmp, &turns);
                                let entries = tmp.entries;
                                *subagentPanel =
                                    Some(crate::subagent_panel::SubagentPanel::frozen(
                                        &agentType, entries,
                                    ));
                                return true;
                            }
                        }
                    }
                }

                // Click on subagent content toggle (expand/collapse).
                if panel == PanelId::Agent && agentPanel.tryToggleSubagentContent(gridLine) {
                    return true;
                }

                let isAlt = mouse.modifiers.contains(KeyModifiers::ALT);

                if isAlt {
                    *selState.selectionForMut(panel) =
                        Some(selection::Selection::newRectangular(col, gridLine));
                    selState.selectingIn = Some(panel);
                } else {
                    let clickCount = selState.click.record(col, screenRow);
                    *selState.selectionForMut(panel) =
                        Some(selection::Selection::new(col, gridLine));
                    selState.selectingIn = Some(panel);

                    if clickCount >= 2 {
                        selState.pendingExpand = Some((panel, clickCount));
                    }
                }

                // Clear selection in the other panel.
                match panel {
                    PanelId::Terminal => {
                        selState.agentSelection = None;
                    }
                    PanelId::Agent => {
                        selState.termSelection = None;
                    }
                    PanelId::Input => {}
                }
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if selState.selectingIn == Some(PanelId::Input) {
                let localCol = mouse
                    .column
                    .saturating_sub(selState.inputContentRect.x)
                    .min(selState.inputContentRect.width.saturating_sub(1));
                let localRow = mouse
                    .row
                    .saturating_sub(selState.inputContentRect.y)
                    .min(selState.inputContentRect.height.saturating_sub(1));
                let contentWidth = selState.inputContentRect.width.saturating_sub(2);
                let contentCol = localCol.saturating_sub(2);
                agentPanel
                    .textArea
                    .mouseDrag(contentCol, localRow, contentWidth);
                return true;
            }
            if let Some(panel) = selState.selectingIn {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine =
                    selection::toGridLine(screenRow, panelOffset(panel, termState, agentPanel));
                if let Some(sel) = selState.selectionForMut(panel) {
                    sel.update(col, gridLine);
                }
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            if selState.selectingIn == Some(PanelId::Input) {
                selState.selectingIn = None;
                if let Some(text) = agentPanel.textArea.selectedText() {
                    selection::copyToClipboard(&text);
                }
                return true;
            }
            if let Some(panel) = selState.selectingIn.take() {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine =
                    selection::toGridLine(screenRow, panelOffset(panel, termState, agentPanel));

                if let Some(sel) = selState.selectionForMut(panel) {
                    sel.update(col, gridLine);
                    sel.finalize();

                    if sel.isEmpty() {
                        *selState.selectionForMut(panel) = None;
                    } else {
                        selState.pendingCopy = Some(panel);
                    }
                }
            }
        }
        MouseEventKind::ScrollUp => {
            if !scrollLock.allow(ScrollAxis::Vertical) {
                return false;
            }
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => {
                    if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                        // Shift+ScrollUp = scroll code block left.
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                        let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                        if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                            agentPanel.scrollCodeBlockH(blockId, -3);
                        } else {
                            agentPanel.scrollUp(3);
                        }
                    } else {
                        agentPanel.scrollUp(3);
                    }
                }
                Some(PanelId::Terminal) => {
                    termState.scrollUp(3);
                    // Extend selection into scrollback during drag.
                    if selState.selectingIn == Some(PanelId::Terminal) {
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Terminal, mouse.column, mouse.row);
                        let offset = termState.displayOffset() as u16;
                        if let Some(sel) = &mut selState.termSelection {
                            sel.extendRow(selection::toGridLine(screenRow, offset));
                        }
                    }
                }
                Some(PanelId::Input) | None => {}
            }
        }
        MouseEventKind::ScrollDown => {
            if !scrollLock.allow(ScrollAxis::Vertical) {
                return false;
            }
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => {
                    if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                        // Shift+ScrollDown = scroll code block right.
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                        let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                        if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                            agentPanel.scrollCodeBlockH(blockId, 3);
                        } else {
                            agentPanel.scrollDown(3);
                        }
                    } else {
                        agentPanel.scrollDown(3);
                    }
                }
                Some(PanelId::Terminal) => {
                    termState.scrollDown(3);
                    if selState.selectingIn == Some(PanelId::Terminal) {
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Terminal, mouse.column, mouse.row);
                        let offset = termState.displayOffset() as u16;
                        if let Some(sel) = &mut selState.termSelection {
                            sel.extendRow(selection::toGridLine(screenRow, offset));
                        }
                    }
                }
                Some(PanelId::Input) | None => {}
            }
        }
        MouseEventKind::ScrollLeft => {
            if !scrollLock.allow(ScrollAxis::Horizontal) {
                return false;
            }
            // Permission code block scroll (in input area).
            if agentPanel.pendingPermit
                && selState
                    .inputContentRect
                    .contains((mouse.column, mouse.row).into())
            {
                agentPanel.scrollPermitCode(-3);
            } else if let Some(PanelId::Agent) = selState.hitTest(mouse.column, mouse.row) {
                let (_, screenRow) = selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                    agentPanel.scrollCodeBlockH(blockId, -3);
                }
            }
        }
        MouseEventKind::ScrollRight => {
            if !scrollLock.allow(ScrollAxis::Horizontal) {
                return false;
            }
            if agentPanel.pendingPermit
                && selState
                    .inputContentRect
                    .contains((mouse.column, mouse.row).into())
            {
                agentPanel.scrollPermitCode(3);
            } else if let Some(PanelId::Agent) = selState.hitTest(mouse.column, mouse.row) {
                let (_, screenRow) = selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                    agentPanel.scrollCodeBlockH(blockId, 3);
                }
            }
        }
        _ => {
            return false;
        }
    }
    true
}

/// Convert a crossterm key event to raw bytes for the PTY.
fn keyToBytes(key: &event::KeyEvent) -> Option<Vec<u8>> {
    match key.code {
        KeyCode::Char(c) => {
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                let ctrl = (c as u8).wrapping_sub(b'a').wrapping_add(1);
                Some(vec![ctrl])
            } else {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                Some(s.as_bytes().to_vec())
            }
        }
        KeyCode::Enter => Some(vec![b'\r']),
        KeyCode::Backspace => Some(vec![0x7f]),
        KeyCode::Tab => Some(vec![b'\t']),
        KeyCode::Esc => Some(vec![0x1b]),
        KeyCode::Up => Some(b"\x1b[A".to_vec()),
        KeyCode::Down => Some(b"\x1b[B".to_vec()),
        KeyCode::Right => Some(b"\x1b[C".to_vec()),
        KeyCode::Left => Some(b"\x1b[D".to_vec()),
        KeyCode::Home => Some(b"\x1b[H".to_vec()),
        KeyCode::End => Some(b"\x1b[F".to_vec()),
        KeyCode::Delete => Some(b"\x1b[3~".to_vec()),
        _ => None,
    }
}

/// Status-bar cache-heat indicator — just the glyph.
///
/// Returns a single glyph span, color-coded by how long it's been since the
/// last cache hit. Empty vec when caching is disabled. The caller appends
/// the literal word " cache" so the user can't mistake the glyph for a
/// sampling-temperature indicator.
fn cacheHeatSpans(
    enabled: bool,
    lastHit: Option<Instant>,
    barBg: Color,
    _barFg: Color,
) -> Vec<Span<'static>> {
    if !enabled {
        return Vec::new();
    }

    let glyphSpan = if let Some(ts) = lastHit {
        let elapsed = ts.elapsed().as_secs();
        let (glyph, fg, dim) = if elapsed < 60 {
            ("\u{2668}\u{FE0E}", Color::Red, false)
        } else if elapsed < 180 {
            ("\u{2668}\u{FE0E}", Color::LightRed, false)
        } else if elapsed < 300 {
            ("\u{2668}\u{FE0E}", Color::Yellow, true)
        } else {
            ("\u{2744}\u{FE0E}", Color::Cyan, false)
        };
        let mut style = Style::default().bg(barBg).fg(fg);
        if dim {
            style = style.add_modifier(Modifier::DIM);
        }
        Span::styled(glyph.to_string(), style)
    } else {
        // Caching configured but no hit yet this session.
        Span::styled(
            "\u{25CB}".to_string(),
            Style::default().bg(barBg).fg(Color::Cyan),
        )
    };

    vec![glyphSpan]
}

/// Short "ctx 3%" string for the status bar. Empty when no context consumed yet.
fn formatContextPct(tokens: usize, window: usize) -> String {
    if tokens == 0 || window == 0 {
        return String::new();
    }
    let pct = (tokens as f64 / window as f64 * 100.0).round() as usize;
    format!("ctx {pct}%")
}

/// Render the hotkey-tips popup — a centered dialog listing every bound
/// shortcut. Dismisses on any keypress (handled in input loop).
fn renderHelpPopup(area: Rect, buf: &mut ratatui::buffer::Buffer) {
    const ROWS: &[(&str, &str)] = &[
        ("Tab", "Switch focus between terminal and agent"),
        ("Esc", "Cancel running turn / close overlay"),
        ("Ctrl+Q \u{00d7}2", "Quit flatline (double-tap)"),
        ("Ctrl+L", "Force terminal redraw"),
        ("Ctrl+H", "Toggle this help"),
        (
            "\u{2191} / \u{2193}",
            "Scroll / history navigation in agent panel",
        ),
    ];

    // Center a 56x(rows+4) box inside the given area.
    let innerW: u16 = 56;
    let innerH: u16 = ROWS.len() as u16 + 4;
    let w = innerW.min(area.width.saturating_sub(2));
    let h = innerH.min(area.height.saturating_sub(2));
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let popup = Rect {
        x,
        y,
        width: w,
        height: h,
    };

    // Clear underlying cells so the popup reads cleanly.
    Clear.render(popup, buf);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" hotkeys \u{2014} any key to dismiss ")
        .style(Style::default().bg(Color::Black).fg(Color::White));
    let inner = block.inner(popup);
    block.render(popup, buf);

    let lines: Vec<Line<'static>> = ROWS
        .iter()
        .map(|(key, desc)| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{:<10}", key),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(*desc),
            ])
        })
        .collect();
    Paragraph::new(lines).render(inner, buf);
}

/// Extract text from the terminal selection, rejoining soft-wrapped lines.
///
/// Uses alacritty's WRAPLINE flag on the last cell of each row to detect
/// lines that were soft-wrapped by the terminal emulator.
fn extractTerminalUnwrapped(
    sel: &selection::Selection,
    area: ratatui::layout::Rect,
    buf: &ratatui::buffer::Buffer,
    displayOffset: u16,
    termState: &TerminalState,
) -> String {
    if sel.isEmpty() {
        return String::new();
    }

    let ((sc, sr), (ec, er)) = sel.sorted();
    let mut segments: Vec<(String, bool)> = Vec::new();

    for gridLine in sr..=er {
        let screenRow = match selection::toScreenRow(gridLine, displayOffset, area.height) {
            Some(r) => r,
            None => continue,
        };

        let colStart = if gridLine == sr { sc } else { 0 };
        let colEnd = if gridLine == er { ec } else { area.width };

        let mut line = String::new();
        for col in colStart..colEnd {
            if col >= area.width {
                break;
            }
            if let Some(cell) = buf.cell((area.x + col, area.y + screenRow)) {
                line.push_str(cell.symbol());
            }
        }
        let trimmed = line.trim_end().to_string();

        // Check if the PREVIOUS line was soft-wrapped (making this a continuation).
        let isCont = gridLine > sr && termState.isLineWrapped(gridLine - 1);
        segments.push((trimmed, isCont));
    }

    // Remove trailing empty lines.
    while segments.last().is_some_and(|(l, _)| l.is_empty()) {
        segments.pop();
    }

    let mut result = String::new();
    for (i, (line, isCont)) in segments.iter().enumerate() {
        if i > 0 {
            if *isCont {
                // Soft-wrapped continuation — join without newline.
                result.push_str(line);
                continue;
            } else {
                result.push('\n');
            }
        }
        result.push_str(line);
    }

    result
}

/// Convert deck's CommandAction to construct's CommandAction.
/// Replay transcript turns into the agent panel for a resumed session.
fn replayTranscript(panel: &mut AgentPanel, turns: &[construct::transcript::Turn]) {
    use construct::transcript::TurnRole;

    // Track pending task (subagent) calls: toolCallId -> entry index.
    let mut pendingTasks: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for turn in turns {
        match turn.role {
            TurnRole::User => {
                let display = match &turn.attachments {
                    Some(atts) if !atts.is_empty() => {
                        let n = atts.len();
                        let suffix = if n == 1 {
                            "1 image".to_string()
                        } else {
                            format!("{n} images")
                        };
                        format!("{}\n[+{suffix} attached]", turn.content)
                    }
                    _ => turn.content.clone(),
                };
                panel
                    .entries
                    .push(crate::agent_panel::PanelEntry::User(display));
            }
            TurnRole::Assistant => {
                if let Some(ref reasoning) = turn.reasoning {
                    if !reasoning.is_empty() {
                        panel
                            .entries
                            .push(crate::agent_panel::PanelEntry::Reasoning {
                                text: reasoning.clone(),
                                expanded: false,
                            });
                    }
                }
                if !turn.content.is_empty() {
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::Assistant(
                            turn.content.clone(),
                        ));
                }
            }
            TurnRole::ToolCall => {
                let name = turn.tool.as_deref().unwrap_or("tool");

                if name == "task" {
                    // Reconstruct SubagentBlock from the task tool call.
                    let prompt = turn
                        .args
                        .as_ref()
                        .and_then(|a| a["prompt"].as_str())
                        .unwrap_or("")
                        .to_string();
                    let agentType = turn
                        .args
                        .as_ref()
                        .and_then(|a| a["agent"].as_str())
                        .unwrap_or("general")
                        .to_string();

                    let entryIdx = panel.entries.len();
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::SubagentBlock {
                            agentType,
                            prompt,
                            toolLines: Vec::new(),
                            done: true,
                            turns: 0,
                            content: None,
                            contentExpanded: false,
                            sessionId: None, // Filled when ToolResult arrives.
                        });

                    if let Some(ref callId) = turn.toolCallId {
                        pendingTasks.insert(callId.clone(), entryIdx);
                    }
                } else {
                    let summary = turn
                        .args
                        .as_ref()
                        .map(|a| {
                            let s = a.to_string();
                            if s.len() > 80 {
                                format!("{}\u{2026}", &s[..s.floor_char_boundary(79)])
                            } else {
                                s
                            }
                        })
                        .unwrap_or_default();
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::ToolApproved {
                            name: format!("{name}: {summary}"),
                        });
                }
            }
            TurnRole::ToolResult => {
                // Check if this result belongs to a pending task.
                if let Some(ref callId) = turn.toolCallId {
                    if let Some(entryIdx) = pendingTasks.remove(callId) {
                        // Extract child session ID and content from
                        // "[subagent session: {id}]\n\n{content}".
                        let raw = &turn.content;
                        let (childSessionId, content) =
                            if let Some(start) = raw.find("[subagent session: ") {
                                let idStart = start + "[subagent session: ".len();
                                if let Some(end) = raw[idStart..].find(']') {
                                    let sid = raw[idStart..idStart + end].to_string();
                                    let bodyStart = idStart + end + 1;
                                    let body = if raw[bodyStart..].starts_with("\n\n") {
                                        raw[bodyStart + 2..].to_string()
                                    } else {
                                        raw[bodyStart..].to_string()
                                    };
                                    (Some(sid), body)
                                } else {
                                    (None, raw.clone())
                                }
                            } else {
                                (None, raw.clone())
                            };

                        // Load the child transcript to reconstruct tool lines and turn count.
                        let (childToolLines, childTurns) = if let Some(ref csid) = childSessionId {
                            loadChildToolLines(csid)
                        } else {
                            (Vec::new(), 0)
                        };

                        if let Some(crate::agent_panel::PanelEntry::SubagentBlock {
                            content: c,
                            sessionId: sid,
                            toolLines: tl,
                            turns: t,
                            ..
                        }) = panel.entries.get_mut(entryIdx)
                        {
                            *sid = childSessionId;
                            *tl = childToolLines;
                            *t = childTurns;
                            if !content.is_empty() && content != "Task completed (no text output)."
                            {
                                *c = Some(content);
                            }
                        }
                        // Skip pushing a ToolResult — SubagentBlock handles display.
                        continue;
                    }
                }

                let name = turn.tool.as_deref().unwrap_or("tool");
                panel
                    .entries
                    .push(crate::agent_panel::PanelEntry::ToolResult {
                        name: name.to_string(),
                        output: turn.content.clone(),
                    });
            }
            // System turns are ephemeral — never appear in transcript chains.
            TurnRole::System => {}
        }
    }
}

/// Load tool lines and turn count from a child session transcript.
fn loadChildToolLines(sessionId: &str) -> (Vec<(String, String)>, usize) {
    use construct::transcript::TurnRole;

    let transcript = match construct::transcript::Transcript::open(sessionId) {
        Ok(t) => t,
        Err(_) => return (Vec::new(), 0),
    };
    let turns = match transcript.loadAll() {
        Ok(t) => t,
        Err(_) => return (Vec::new(), 0),
    };

    let mut toolLines: Vec<(String, String)> = Vec::new();
    let mut turnCount: usize = 0;

    for turn in &turns {
        match turn.role {
            TurnRole::Assistant => {
                turnCount += 1;
            }
            TurnRole::ToolCall => {
                let name = turn.tool.as_deref().unwrap_or("tool");
                let argsJson = turn
                    .args
                    .as_ref()
                    .map(|a| a.to_string())
                    .unwrap_or_default();
                let summary = match construct::tool::parse(name, &argsJson) {
                    Ok(action) => construct::tool::summarize(&action),
                    Err(_) => format!("{name} (parse error)"),
                };
                toolLines.push((name.to_string(), summary));
            }
            TurnRole::ToolResult => {
                let name = turn.tool.as_deref().unwrap_or("tool");
                let output = &turn.content;
                let brief = if output.len() > 60 {
                    format!("{}\u{2026}", &output[..output.floor_char_boundary(60)])
                } else {
                    output.clone()
                };
                toolLines.push((name.to_string(), brief));
            }
            _ => {}
        }
    }

    (toolLines, turnCount)
}

/// Send a TuiRequest whose reply is a `CommandAck`, and forward the ack
/// onto the deck update channel.
fn spawnAckRequest<F>(
    requestTx: mpsc::Sender<TuiRequest>,
    deckUpdateTx: mpsc::Sender<DeckUpdate>,
    build: F,
) where
    F: FnOnce(oneshot::Sender<construct::control::CommandAck>) -> TuiRequest + Send + 'static,
{
    tokio::spawn(async move {
        let (rTx, rRx) = oneshot::channel();
        let req = build(rTx);
        let _ = requestTx.send(req).await;
        if let Ok(ack) = rRx.await {
            let _ = deckUpdateTx.send(DeckUpdate::CommandAck(ack)).await;
        }
    });
}

/// Dispatch a parsed slash-command action. Each variant constructs a
/// `TuiRequest` with an appropriate oneshot reply channel and spawns a
/// task to forward the reply to the deck update channel.
fn dispatchSlashCommand(
    action: crate::command::CommandAction,
    requestTx: mpsc::Sender<TuiRequest>,
    deckUpdateTx: mpsc::Sender<DeckUpdate>,
) {
    match action {
        crate::command::CommandAction::ShowContext => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ShowContext { reply: rTx }).await;
                if let Ok(state) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ContextDisplay(state)).await;
                }
            });
        }
        crate::command::CommandAction::Undo => {
            spawnAckRequest(requestTx, deckUpdateTx, |reply| TuiRequest::Undo { reply });
        }
        crate::command::CommandAction::Rewind { target: _ } => {
            // /rewind opens the picker. Fetch options first.
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::GetRewindOptions { reply: rTx })
                    .await;
                if let Ok(turns) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::RewindOptions(turns)).await;
                }
            });
        }
        crate::command::CommandAction::Forks { forkId: None } => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetForks { reply: rTx }).await;
                if let Ok(forks) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::Forks(forks)).await;
                }
            });
        }
        crate::command::CommandAction::Forks { forkId: Some(id) } => {
            spawnAckRequest(requestTx, deckUpdateTx, move |reply| {
                TuiRequest::SwitchFork { forkId: id, reply }
            });
        }
        crate::command::CommandAction::Resume {
            sessionId: Some(id),
        } => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::ResumeSession {
                        sessionId: id,
                        reply: rTx,
                    })
                    .await;
                if let Ok(ack) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ResumeResult(ack)).await;
                }
            });
        }
        crate::command::CommandAction::Resume { sessionId: None } => {
            // Opening the interactive picker is handled inline; this branch
            // is unreachable but handled defensively with a list fallback.
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::ListSessions { reply: rTx })
                    .await;
                if let Ok(text) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ShowResult(text)).await;
                }
            });
        }
        crate::command::CommandAction::Clear => {
            spawnAckRequest(requestTx, deckUpdateTx, |reply| TuiRequest::Clear { reply });
        }
        crate::command::CommandAction::Mcp => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetMcp { reply: rTx }).await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::McpStatus(status)).await;
                }
            });
        }
        crate::command::CommandAction::Lsp => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetLsp { reply: rTx }).await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::LspStatus(status)).await;
                }
            });
        }
        crate::command::CommandAction::Permissions => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::GetPermissions { reply: rTx })
                    .await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx
                        .send(DeckUpdate::PermissionsStatus(status))
                        .await;
                }
            });
        }
        crate::command::CommandAction::ShowCost => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ShowCost { reply: rTx }).await;
                if let Ok(text) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ShowResult(text)).await;
                }
            });
        }
    }
}
