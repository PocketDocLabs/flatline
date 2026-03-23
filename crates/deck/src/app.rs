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
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    },
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    Terminal as RatatuiTerminal,
};
use tokio::sync::{mpsc, watch};

use construct::permissions::Permissions;
use construct::prompt::{DomainModule, InterfaceMode};
use construct::session::{CommandAction, Session, SessionEvent};
use construct::shell::{ShellIo, spawnShell};

use crate::agent_panel::AgentPanel;
use crate::lsp_panel::{LspPanel, PanelAction as LspPanelAction};
use crate::mcp_panel::{McpPanel, PanelAction as McpPanelAction};
use crate::fork_picker::{ForkPicker, ForkAction};
use crate::rewind_picker::{RewindPicker, RewindAction};
use crate::selection::{self, PanelId, SelectionState};
use crate::session_picker::{PickerAction, SessionPicker};
use crate::terminal::{Terminal as EmbeddedTerminal, TerminalState};

use std::io::{self, Write as _};
use std::time::{Duration, Instant};

/// Resolve main agent permissions from config, falling back to allowReadOnly.
fn mainAgentPermissions(config: &construct::config::Config) -> Permissions {
    config.permissions.clone().unwrap_or_else(Permissions::allowReadOnly)
}

/// Which panel has input focus.
#[derive(PartialEq)]
enum Focus {
    Terminal,
    Agent,
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
        let expired = now.duration_since(self.lastEvent).as_millis()
            > SCROLL_LOCK_TIMEOUT_MS as u128;

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

/// Set the host terminal's window title via OSC 2.
fn setTerminalTitle(topic: Option<&str>) {
    let title = match topic {
        Some(t) if !t.is_empty() => format!("\u{1f0a1} {t}"),
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

/// Run the deck TUI.
pub async fn run() -> Result<()> {
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
    setTerminalTitle(None);

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = RatatuiTerminal::new(backend)?;

    let size = terminal.size()?;
    // Terminal gets ~60% width minus borders, full height minus status bar and borders.
    let termCols = (size.width * 3 / 5).saturating_sub(2);
    let termRows = size.height.saturating_sub(3);
    let (shell, mut shellIo) = spawnShell(termCols, termRows)?;
    let mut termState = TerminalState::new(termCols, termRows);

    let config = construct::config::load()?;
    let contextWindow = config.main.contextWindow;

    let mut agentPanel = AgentPanel::new();
    let mut focus = Focus::Terminal;
    let mut selState = SelectionState::new();
    let mut scrollLock = ScrollAxisLock::new();
    // Session channels.
    let (eventTx, mut eventRx) = mpsc::channel::<SessionEvent>(256);
    let (permitTx, permitRx) = mpsc::channel::<construct::permissions::PermitResponse>(1);
    let (userInputTx, mut userInputRx) = mpsc::channel::<construct::session::UserInput>(16);
    let (commandTx, mut commandRx) = mpsc::channel::<CommandAction>(16);
    let (cancelTx, cancelRx) = watch::channel(false);

    // Spawn the agent session task.
    let mut cancelRx = cancelRx;
    tokio::spawn(async move {
        let config = match construct::config::load() {
            Ok(c) => c,
            Err(e) => {
                let _ = eventTx
                    .send(SessionEvent::Error(format!("Config error: {e}")))
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
                let _ = eventTx
                    .send(SessionEvent::Error(format!("Session error: {e}")))
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

        let mut permitRx = permitRx;

        loop {
            tokio::select! {
                msg = userInputRx.recv() => {
                    match msg {
                        Some(msg) => {
                            // Clear any stale cancel notification from a previous turn.
                            cancelRx.borrow_and_update();
                            if let Err(e) = session.send(&msg, &eventTx, &mut permitRx, &mut cancelRx).await {
                                let _ = eventTx
                                    .send(SessionEvent::Error(format!("Agent error: {e}")))
                                    .await;
                            }
                        }
                        None => break,
                    }
                }
                cmd = commandRx.recv() => {
                    match cmd {
                        Some(CommandAction::Resume { sessionId: Some(id) }) => {
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
                                    let _ = eventTx
                                        .send(SessionEvent::SessionRestored { turns, markers })
                                        .await;
                                    // Set window title to the current topic on resume.
                                    let label = session.currentTopicLabel();
                                    if !label.is_empty() {
                                        let _ = eventTx
                                            .send(SessionEvent::TopicChanged {
                                                label: label.to_string(),
                                            })
                                            .await;
                                    }
                                    let _ = eventTx
                                        .send(SessionEvent::ResumeComplete {
                                            success: true,
                                            message: format!("Resumed session {id}"),
                                        })
                                        .await;
                                }
                                Err((e, shell)) => {
                                    let _ = eventTx
                                        .send(SessionEvent::ResumeComplete {
                                            success: false,
                                            message: format!("Failed to resume {id}: {e}"),
                                        })
                                        .await;
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
                                            let _ = eventTx
                                                .send(SessionEvent::Error(
                                                    format!("Session lost after failed resume: {e2}"),
                                                ))
                                                .await;
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Some(CommandAction::Clear) => {
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

                                    let _ = eventTx.send(SessionEvent::Cleared).await;
                                }
                                Err(e) => {
                                    let _ = eventTx
                                        .send(SessionEvent::Error(
                                            format!("Failed to create new session: {e}"),
                                        ))
                                        .await;
                                    return;
                                }
                            }
                        }
                        Some(CommandAction::Lsp) => {
                            let servers = session.lspStatusData();
                            let _ = eventTx
                                .send(SessionEvent::LspStatus { servers })
                                .await;
                        }
                        Some(CommandAction::Mcp) => {
                            let (servers, totalTools, searchMode, configPath) =
                                session.mcpStatusData().await;
                            let _ = eventTx
                                .send(SessionEvent::McpStatus {
                                    servers,
                                    totalTools,
                                    searchMode,
                                    configPath,
                                })
                                .await;
                        }
                        Some(CommandAction::Permissions) => {
                            let (defaultMode, rules, source, configPath) =
                                session.permissionsStatusData();
                            let _ = eventTx
                                .send(SessionEvent::PermissionsStatus {
                                    defaultMode,
                                    rules,
                                    source,
                                    configPath,
                                })
                                .await;
                        }
                        Some(CommandAction::SavePermissions { defaultMode, rules }) => {
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
                                        let _ = eventTx
                                            .send(SessionEvent::CommandResult(
                                                "Permissions saved.".into(),
                                            ))
                                            .await;
                                    }
                                    Err(e) => {
                                        let _ = eventTx
                                            .send(SessionEvent::CommandResult(
                                                format!("Failed to save permissions: {e}"),
                                            ))
                                            .await;
                                    }
                                }
                            }
                        }
                        Some(CommandAction::Rewind { target }) if target.is_empty() => {
                            let turns = session.loadDisplayTurns().unwrap_or_default();
                            let _ = eventTx
                                .send(SessionEvent::RewindPickerData { turns })
                                .await;
                        }
                        Some(CommandAction::Rewind { target }) => {
                            let result = session.rewind(&target, false, &eventTx).await;
                            let _ = eventTx
                                .send(SessionEvent::CommandResult(result))
                                .await;
                        }
                        Some(CommandAction::ForkAndRewind { target }) => {
                            let result = session.rewind(&target, true, &eventTx).await;
                            let _ = eventTx
                                .send(SessionEvent::CommandResult(result))
                                .await;
                        }
                        Some(CommandAction::Forks { forkId: None }) => {
                            let forks = session.listForks();
                            let _ = eventTx
                                .send(SessionEvent::ForkPickerData { forks })
                                .await;
                        }
                        Some(CommandAction::Forks { forkId: Some(id) }) => {
                            let result = session.switchFork(&id, &eventTx).await;
                            let _ = eventTx
                                .send(SessionEvent::CommandResult(result))
                                .await;
                        }
                        Some(action) => {
                            let result = session.executeCommand(&action).await;
                            let _ = eventTx
                                .send(SessionEvent::CommandResult(result))
                                .await;
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
        &mut eventRx,
        &permitTx,
        &userInputTx,
        &commandTx,
        &cancelTx,
        contextWindow,
    )
    .await;

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
    eventRx: &mut mpsc::Receiver<SessionEvent>,
    permitTx: &mpsc::Sender<construct::permissions::PermitResponse>,
    userInputTx: &mpsc::Sender<construct::session::UserInput>,
    commandTx: &mpsc::Sender<CommandAction>,
    cancelTx: &watch::Sender<bool>,
    contextWindow: usize,
) -> Result<()> {
    let mut tokenCount: usize = 0;
    let mut sessionPicker: Option<SessionPicker> = None;
    let mut rewindPicker: Option<RewindPicker> = None;
    let mut forkPicker: Option<ForkPicker> = None;
    let mut pendingRewindMessage: Option<String> = None;
    let mut mcpPanel: Option<McpPanel> = None;
    let mut lspPanel: Option<LspPanel> = None;
    let mut permissionsPanel: Option<crate::permissions_panel::PermissionsPanel> = None;
    let mut subagentPanel: Option<crate::subagent_panel::SubagentPanel> = None;
    let mut subagentPermitTx: Option<mpsc::Sender<construct::permissions::PermitResponse>> = None;
    let projectDir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let mut needsRedraw = true;
    let mut lastQuitPress: Option<Instant> = None;

    loop {
        // Draw only when state has changed.
        if needsRedraw {
        needsRedraw = false;
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
            if gridCols != termInner.width as usize || termState.screenLines() != termInner.height as usize {
                termState.resize(termInner.width, termInner.height);
                let _ = shellIo.resizeTx.try_send((termInner.width, termInner.height));
            }

            frame.render_widget(termBlock, hChunks[0]);
            frame.render_stateful_widget(EmbeddedTerminal, termInner, termState);

            // Capture content rects for mouse hit-testing.
            selState.termContentRect = termInner;

            // Agent panel.
            let agentChatArea = agentPanel.render(hChunks[1], frame.buffer_mut(), *focus == Focus::Agent);
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
                    PanelId::Terminal => (&mut selState.termSelection, termInner, termOffset),
                    PanelId::Agent => (&mut selState.agentSelection, agentContentArea, agentOffset),
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
                selection::applyHighlight(sel, agentContentArea, frame.buffer_mut(), agentOffset);
            }

            // Deferred clipboard copy (Buffer only available during draw).
            if let Some(panel) = selState.pendingCopy.take() {
                match panel {
                    PanelId::Terminal => {
                        if let Some(sel) = &selState.termSelection {
                            let text = extractTerminalUnwrapped(sel, termInner, frame.buffer_mut(), termOffset, termState);
                            selection::copyToClipboard(&text);
                        }
                    }
                    PanelId::Agent => {
                        if let Some(sel) = &selState.agentSelection {
                            let text = agentPanel.extractUnwrappedText(sel, agentContentArea, frame.buffer_mut(), agentOffset);
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
            }

            // Status bar.
            let modeStr = match focus {
                Focus::Terminal => "terminal",
                Focus::Agent => "agent",
            };

            // Show "press again" hint briefly after a single Ctrl+Q tap.
            let quitHintActive = lastQuitPress
                .map(|t| t.elapsed() < Duration::from_secs(1))
                .unwrap_or(false);
            let controls = if quitHintActive {
                "▸ press Ctrl+Q again to quit"
            } else if agentPanel.isActive() {
                "Esc: cancel  Ctrl+Q\u{00d7}2: quit"
            } else {
                "Tab: switch  Ctrl+Q\u{00d7}2: quit"
            };

            let (barBg, barFg) = if quitHintActive {
                (Color::Yellow, Color::Black)
            } else {
                (Color::DarkGray, Color::White)
            };

            let leftText = format!(" [{modeStr}]  {controls}");
            let tokenStr = formatTokens(tokenCount, contextWindow);
            let barWidth = vChunks[1].width as usize;
            // Pad between left and right-justified token string.
            let gap = barWidth.saturating_sub(leftText.len() + tokenStr.len() + 1);
            let statusLine = format!("{leftText}{:>width$} ", tokenStr, width = gap + tokenStr.len());
            let statusBar = Paragraph::new(statusLine)
                .style(Style::default().bg(barBg).fg(barFg));
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
                panel.render(area, frame.buffer_mut());
            }
        })?;
        }

        // Drain PTY output.
        while let Ok(bytes) = shellIo.outputRx.try_recv() {
            termState.process(&bytes);
            needsRedraw = true;
        }

        // Drain session events.
        while let Ok(event) = eventRx.try_recv() {
            needsRedraw = true;
            match event {
                SessionEvent::ContentDelta(text) => agentPanel.appendContent(&text),
                SessionEvent::ReasoningDelta(text) => agentPanel.appendReasoning(&text),
                SessionEvent::ToolRequest { name, summary, args, diff, explanation, impact } => {
                    agentPanel.showToolRequest(&name, &summary, &args, diff, explanation, impact);
                }
                SessionEvent::ToolResult { name, output } => {
                    // Task tool results are handled by SubagentComplete — don't double-render.
                    if name != "task" {
                        agentPanel.pushToolResult(&name, &output);
                    }
                }
                SessionEvent::ToolStarted { name, summary } => {
                    if name != "task" {
                        agentPanel.toolStarted(&name, &summary);
                    }
                }
                SessionEvent::ToolAutoApproved { name, summary } => {
                    agentPanel.toolApproved(&format!("{name}: {summary}"));
                }
                SessionEvent::ToolDenied { name } => {
                    agentPanel.toolDenied(&name);
                }
                SessionEvent::ToolAutoDenied { name, summary } => {
                    agentPanel.toolAutoDenied(&name, &summary);
                }
                SessionEvent::TurnAborted { name } => {
                    agentPanel.pushError(&format!("Turn aborted: {name} not permitted"));
                }
                SessionEvent::TurnComplete => {
                    agentPanel.finishTurn();
                }
                SessionEvent::TurnCancelled => {
                    agentPanel.finalizeCancelled();
                }
                SessionEvent::TopicChanged { label } => {
                    setTerminalTitle(Some(&label));
                }
                SessionEvent::LspHint { serverId, installHint } => {
                    let msg = format!(
                        "\u{2699}\u{FE0E} {} not found \u{2014} `{}`",
                        serverId, installHint,
                    );
                    agentPanel.pushCommandResult(&msg);
                }
                SessionEvent::TokenUpdate {
                    contextTokens,
                    ..
                } => {
                    tokenCount = contextTokens;
                }
                SessionEvent::Retrying { attempt, maxAttempts } => {
                    agentPanel.showRetrying(attempt, maxAttempts);
                }
                SessionEvent::Error(msg) => {
                    agentPanel.pushError(&msg);
                }
                SessionEvent::CommandResult(text) => {
                    if forkPicker.is_some() {
                        // Fork switch failed — Rewound would have cleared the picker first.
                        if let Some(ref mut picker) = forkPicker {
                            picker.switchFailed(text);
                        }
                    } else {
                        agentPanel.pushCommandResult(&text);
                    }
                }
                SessionEvent::CompactionStarted { stage } => {
                    tracing::info!(stage = %stage, "compaction started");
                }
                SessionEvent::CompactionComplete { stage, reduction, markerBlock } => {
                    tracing::info!(stage = %stage, reduction = %reduction, "compaction complete");
                    if let Some(blockIdx) = markerBlock {
                        agentPanel.pushCompactionMarker(&stage, blockIdx);
                    }
                }
                SessionEvent::Cleared => {
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    setTerminalTitle(None);
                }
                SessionEvent::Rewound { targetTurnId } => {
                    rewindPicker = None;
                    forkPicker = None;
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    if let Some(msg) = pendingRewindMessage.take() {
                        agentPanel.textArea.setText(&msg);
                    }
                    tracing::info!(target = %targetTurnId, "conversation rewound");
                }
                SessionEvent::SessionRestored { turns, markers } => {
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    replayTranscript(agentPanel, &turns);
                    for (stage, blockIdx) in &markers {
                        agentPanel.pushCompactionMarker(stage, *blockIdx);
                    }
                }
                SessionEvent::McpStatus { servers, totalTools, searchMode, configPath } => {
                    mcpPanel = Some(McpPanel::new(servers, totalTools, searchMode, configPath));
                }
                SessionEvent::LspStatus { servers } => {
                    lspPanel = Some(LspPanel::new(servers));
                }
                SessionEvent::PermissionsStatus { defaultMode, rules, source, configPath } => {
                    permissionsPanel = Some(crate::permissions_panel::PermissionsPanel::new(
                        defaultMode, rules, source, configPath,
                    ));
                }
                SessionEvent::RewindPickerData { turns } => {
                    rewindPicker = Some(RewindPicker::new(&turns));
                }
                SessionEvent::ForkPickerData { forks } => {
                    forkPicker = Some(ForkPicker::new(&forks));
                }
                SessionEvent::ResumeComplete { success, message } => {
                    if success {
                        sessionPicker = None;
                        agentPanel.entries.push(
                            crate::agent_panel::PanelEntry::SessionNotice(message),
                        );
                    } else if let Some(ref mut picker) = sessionPicker {
                        picker.resumeFailed(message);
                    } else {
                        agentPanel.pushError(&message);
                    }
                }
                SessionEvent::SubagentStarted { sessionId, agentType, prompt } => {
                    tracing::info!(agent = %agentType, "subagent started");
                    agentPanel.subagentStarted(&sessionId, &agentType, &prompt);
                }
                SessionEvent::SubagentEvent { sessionId: _, event } => {
                    match *event {
                        SessionEvent::ToolAutoApproved { ref name, ref summary } => {
                            agentPanel.subagentToolLine(name, summary);
                        }
                        SessionEvent::ToolStarted { ref name, ref summary } => {
                            agentPanel.subagentToolLine(name, summary);
                        }
                        SessionEvent::ToolResult { ref name, ref output } => {
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
                        SessionEvent::ContentDelta(ref text) => {
                            agentPanel.subagentContent(text);
                        }
                        SessionEvent::Error(ref msg) => {
                            agentPanel.subagentToolLine("error", msg);
                        }
                        _ => {}
                    }
                }
                SessionEvent::SubagentPermitRequest {
                    sessionId: _, name, summary, args, diff, responseTx,
                } => {
                    // Show permission prompt for the subagent (reuse same UI).
                    agentPanel.showToolRequest(
                        &name, &summary, &args, diff,
                        None, construct::tool::ShellImpact::MinorMod,
                    );
                    subagentPermitTx = Some(responseTx);
                }
                SessionEvent::SubagentShellOutput { data, .. } => {
                    if let Some(ref mut sub) = agentPanel.activeSubagent {
                        // Cap at 512KB.
                        const MAX: usize = 512 * 1024;
                        if sub.shellScrollback.len() + data.len() > MAX {
                            let excess = (sub.shellScrollback.len() + data.len()) - MAX;
                            sub.shellScrollback.drain(..excess.min(sub.shellScrollback.len()));
                        }
                        sub.shellScrollback.extend_from_slice(&data);
                    }
                }
                SessionEvent::SubagentComplete { agentType, turns, content, .. } => {
                    tracing::info!(agent = %agentType, turns = turns, "subagent completed");
                    agentPanel.subagentComplete(&agentType, turns, &content);
                }
            }
        }

        // Tick throbber animation (wall-clock gated).
        if agentPanel.tickThrobber() {
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
            permitTx,
            userInputTx,
            commandTx,
            cancelTx,
            &mut sessionPicker,
            &mut rewindPicker,
            &mut forkPicker,
            &mut pendingRewindMessage,
            &mut mcpPanel,
            &mut lspPanel,
            &mut subagentPanel,
            &mut permissionsPanel,
            &mut subagentPermitTx,
            &projectDir,
            &mut lastQuitPress,
        )
        .await?;
        if quit {
            break;
        }
        if hadInput {
            needsRedraw = true;
        }
        // Force full redraw on resize to clear any cursor-drift artifacts
        // from characters whose display width differs between unicode-width
        // and the host terminal.
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
    permitTx: &mpsc::Sender<construct::permissions::PermitResponse>,
    userInputTx: &mpsc::Sender<construct::session::UserInput>,
    commandTx: &mpsc::Sender<CommandAction>,
    cancelTx: &watch::Sender<bool>,
    sessionPicker: &mut Option<SessionPicker>,
    rewindPicker: &mut Option<RewindPicker>,
    forkPicker: &mut Option<ForkPicker>,
    pendingRewindMessage: &mut Option<String>,
    mcpPanel: &mut Option<McpPanel>,
    lspPanel: &mut Option<LspPanel>,
    subagentPanel: &mut Option<crate::subagent_panel::SubagentPanel>,
    permissionsPanel: &mut Option<crate::permissions_panel::PermissionsPanel>,
    subagentPermitTx: &mut Option<mpsc::Sender<construct::permissions::PermitResponse>>,
    projectDir: &str,
    lastQuitPress: &mut Option<Instant>,
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
                    if !event::poll(Duration::ZERO)? { break; }
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
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('q')
                {
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
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('l')
                {
                    resized = true;
                    break;
                }

                // Subagent panel overlay intercepts ALL keys when active (including Esc, Tab).
                if let Some(ref mut panel) = *subagentPanel {
                    if panel.handleKey(key) {
                        *subagentPanel = None;
                    }
                    break;
                }

                // Cancel running turn with Escape — immediate visual feedback.
                if key.code == KeyCode::Esc && agentPanel.isActive() {
                    let _ = cancelTx.send(true);
                    agentPanel.finalizeCancelled();
                    break;
                }

                if key.code == KeyCode::Tab {
                    // Don't switch focus when an overlay is active or completion menu is open.
                    if sessionPicker.is_some() || rewindPicker.is_some() || forkPicker.is_some() || mcpPanel.is_some() || lspPanel.is_some() || permissionsPanel.is_some() {
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
                if key.modifiers.contains(KeyModifiers::SUPER)
                    && key.code == KeyCode::Char('c')
                {
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

                    // Helper: send response to subagent escalation or parent permit channel.
                    macro_rules! sendPermit {
                        ($resp:expr) => {
                            if let Some(tx) = subagentPermitTx.take() {
                                let _ = tx.send($resp).await;
                            } else {
                                let _ = permitTx.send($resp).await;
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
                            // Open subagent panel if a subagent is active.
                            if let Some(ref sub) = agentPanel.activeSubagent {
                                let mut panel = crate::subagent_panel::SubagentPanel::new(
                                    &sub.agentType, &sub.sessionId,
                                );
                                panel.transcript = sub.transcript.clone();
                                panel.shellScrollback = sub.shellScrollback.clone();
                                *subagentPanel = Some(panel);
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
                            // Picker stays open — closed on ResumeComplete event.
                            let action = CommandAction::Resume {
                                sessionId: Some(id),
                            };
                            let _ = commandTx.send(action).await;
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
                        RewindAction::Rewind { target, userMessage } => {
                            *pendingRewindMessage = Some(userMessage);
                            let action = CommandAction::Rewind { target };
                            let _ = commandTx.send(action).await;
                        }
                        RewindAction::ForkAndRewind { target, userMessage } => {
                            *pendingRewindMessage = Some(userMessage);
                            let action = CommandAction::ForkAndRewind { target };
                            let _ = commandTx.send(action).await;
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
                            let action = CommandAction::Forks { forkId: Some(id) };
                            let _ = commandTx.send(action).await;
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
                            let _ = commandTx
                                .send(CommandAction::SavePermissions { defaultMode, rules })
                                .await;
                        }
                        PermPanelAction::None => {}
                    }
                    break;
                }

                match focus {
                    Focus::Terminal => {
                        if key.modifiers.contains(KeyModifiers::SUPER) {
                            if !event::poll(Duration::ZERO)? { break; }
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

                        if !completionHandled {
                            // Handle attachment-related keys before borrowing textArea.
                            if ctrl && key.code == KeyCode::Char('d') && agentPanel.attachmentCount() > 0 {
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
                                            rgbaDimensions: Some((imgData.width as u32, imgData.height as u32)),
                                        });
                                    } else if let Ok(text) = cb.get_text() {
                                        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                                        agentPanel.textArea.insertStr(&normalized);
                                    }
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
                                                    crate::command::CommandAction::Resume { sessionId: None },
                                                ) => {
                                                    // Open the interactive picker.
                                                    *sessionPicker = Some(SessionPicker::new(projectDir));
                                                }
                                                crate::command::CommandOutput::Action(action) => {
                                                    let constructAction = convertAction(action);
                                                    let _ = commandTx.send(constructAction).await;
                                                }
                                            }
                                        } else {
                                            let _ = cancelTx.send(false);
                                            agentPanel.history.push(&msg);
                                            agentPanel.pushUser(&msg);
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
                                    if ta.isEmpty() || ta.lineCount() == 1 {
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
                                    if ta.isEmpty() || ta.lineCount() == 1 {
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

                        // Update completion after every keystroke.
                        let currentText = agentPanel.textArea.text().to_string();
                        agentPanel.updateCompletion(&currentText);
                    }
                }
            }
            Event::Mouse(mouse) => {
                // Subagent overlay handles scroll events.
                if let Some(ref mut panel) = *subagentPanel {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            panel.scrollUp();
                            hadInput = true;
                        }
                        MouseEventKind::ScrollDown => {
                            panel.scrollDown();
                            hadInput = true;
                        }
                        _ => {}
                    }
                    break;
                }
                // Other overlay panels consume all mouse events to prevent click-through.
                if sessionPicker.is_some() || rewindPicker.is_some() || forkPicker.is_some() || mcpPanel.is_some() || lspPanel.is_some() || permissionsPanel.is_some() {
                    break;
                }
                if handleMouse(mouse, focus, agentPanel, termState, selState, shellIo, scrollLock, subagentPanel) {
                    hadInput = true;
                }
            }
            Event::Paste(text) => {
                hadInput = true;
                if *focus == Focus::Agent && !agentPanel.pendingPermit {
                    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                    agentPanel.textArea.insertStr(&normalized);
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
    shellIo: &ShellIo,
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
            if selState.inputContentRect.contains((mouse.column, mouse.row).into()) {
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
                agentPanel.textArea.mouseDown(contentCol, localRow, contentWidth);
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
                let gridLine = selection::toGridLine(screenRow, panelOffset(panel, termState, agentPanel));

                // Single click on a reasoning header toggles it.
                if panel == PanelId::Agent
                    && agentPanel.toggleReasoningAtGridLine(gridLine)
                {
                    return true;
                }

                // Click on code block "copy" label copies the block content.
                if panel == PanelId::Agent
                    && agentPanel.tryCopyCodeBlock(gridLine, col)
                {
                    return true;
                }

                // Click on code block top/bottom border toggles expand/collapse.
                if panel == PanelId::Agent
                    && agentPanel.tryToggleCodeBlock(gridLine)
                {
                    return true;
                }

                // Click on subagent header [view] opens the overlay panel.
                if panel == PanelId::Agent
                    && agentPanel.isSubagentHeaderLine(gridLine)
                {
                    if let Some(ref sub) = agentPanel.activeSubagent {
                        // Live subagent — use cached transcript.
                        let mut panel = crate::subagent_panel::SubagentPanel::new(
                            &sub.agentType, &sub.sessionId,
                        );
                        panel.transcript = sub.transcript.clone();
                        panel.shellScrollback = sub.shellScrollback.clone();
                        *subagentPanel = Some(panel);
                        return true;
                    } else if let Some((agentType, sid)) = agentPanel.lastSubagentSession() {
                        // Resumed session — load child transcript on demand.
                        let agentType = agentType.to_string();
                        let sid = sid.to_string();
                        if let Ok(transcript) = construct::transcript::Transcript::open(&sid) {
                            if let Ok(turns) = transcript.loadAll() {
                                let mut panel = crate::subagent_panel::SubagentPanel::new(
                                    &agentType, &sid,
                                );
                                replayTranscript(&mut panel.transcriptPanel, &turns);
                                panel.transcript = panel.transcriptPanel.entries.clone();
                                *subagentPanel = Some(panel);
                                return true;
                            }
                        }
                    }
                }

                // Click on subagent content toggle (expand/collapse).
                if panel == PanelId::Agent
                    && agentPanel.tryToggleSubagentContent(gridLine)
                {
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
                let localCol = mouse.column.saturating_sub(selState.inputContentRect.x)
                    .min(selState.inputContentRect.width.saturating_sub(1));
                let localRow = mouse.row.saturating_sub(selState.inputContentRect.y)
                    .min(selState.inputContentRect.height.saturating_sub(1));
                let contentWidth = selState.inputContentRect.width.saturating_sub(2);
                let contentCol = localCol.saturating_sub(2);
                agentPanel.textArea.mouseDrag(contentCol, localRow, contentWidth);
                return true;
            }
            if let Some(panel) = selState.selectingIn {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine = selection::toGridLine(screenRow, panelOffset(panel, termState, agentPanel));
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
                let gridLine = selection::toGridLine(screenRow, panelOffset(panel, termState, agentPanel));

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
            if !scrollLock.allow(ScrollAxis::Vertical) { return false; }
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => {
                    if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                        // Shift+ScrollUp = scroll code block left.
                        let (_, screenRow) = selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
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
            if !scrollLock.allow(ScrollAxis::Vertical) { return false; }
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => {
                    if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                        // Shift+ScrollDown = scroll code block right.
                        let (_, screenRow) = selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
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
            if !scrollLock.allow(ScrollAxis::Horizontal) { return false; }
            // Permission code block scroll (in input area).
            if agentPanel.pendingPermit
                && selState.inputContentRect.contains((mouse.column, mouse.row).into())
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
            if !scrollLock.allow(ScrollAxis::Horizontal) { return false; }
            if agentPanel.pendingPermit
                && selState.inputContentRect.contains((mouse.column, mouse.row).into())
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
        _ => { return false; }
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

/// Format token usage for the status bar: "42.1k / 256k (16%)".
fn formatTokens(tokens: usize, window: usize) -> String {
    if tokens == 0 {
        return String::new();
    }

    let fmtK = |n: usize| -> String {
        if n >= 1_000_000 {
            format!("{:.1}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}k", n as f64 / 1_000.0)
        } else {
            format!("{n}")
        }
    };

    let pct = if window > 0 {
        (tokens as f64 / window as f64 * 100.0) as usize
    } else {
        0
    };

    format!("{} / {} ({}%)", fmtK(tokens), fmtK(window), pct)
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
                        let suffix = if n == 1 { "1 image".to_string() } else { format!("{n} images") };
                        format!("{}\n[+{suffix} attached]", turn.content)
                    }
                    _ => turn.content.clone(),
                };
                panel.entries.push(crate::agent_panel::PanelEntry::User(display));
            }
            TurnRole::Assistant => {
                if let Some(ref reasoning) = turn.reasoning {
                    if !reasoning.is_empty() {
                        panel.entries.push(crate::agent_panel::PanelEntry::Reasoning {
                            text: reasoning.clone(),
                            expanded: false,
                        });
                    }
                }
                if !turn.content.is_empty() {
                    panel.entries.push(crate::agent_panel::PanelEntry::Assistant(turn.content.clone()));
                }
            }
            TurnRole::ToolCall => {
                let name = turn.tool.as_deref().unwrap_or("tool");

                if name == "task" {
                    // Reconstruct SubagentBlock from the task tool call.
                    let prompt = turn.args.as_ref()
                        .and_then(|a| a["prompt"].as_str())
                        .unwrap_or("")
                        .to_string();
                    let agentType = turn.args.as_ref()
                        .and_then(|a| a["agent"].as_str())
                        .unwrap_or("general")
                        .to_string();

                    let entryIdx = panel.entries.len();
                    panel.entries.push(crate::agent_panel::PanelEntry::SubagentBlock {
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
                    panel.entries.push(crate::agent_panel::PanelEntry::ToolApproved {
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
                        let (childSessionId, content) = if let Some(start) = raw.find("[subagent session: ") {
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
                            content: c, sessionId: sid, toolLines: tl, turns: t, ..
                        }) = panel.entries.get_mut(entryIdx) {
                            *sid = childSessionId;
                            *tl = childToolLines;
                            *t = childTurns;
                            if !content.is_empty()
                                && content != "Task completed (no text output)."
                            {
                                *c = Some(content);
                            }
                        }
                        // Skip pushing a ToolResult — SubagentBlock handles display.
                        continue;
                    }
                }

                let name = turn.tool.as_deref().unwrap_or("tool");
                panel.entries.push(crate::agent_panel::PanelEntry::ToolResult {
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
                let argsJson = turn.args.as_ref()
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

fn convertAction(action: crate::command::CommandAction) -> CommandAction {
    match action {
        crate::command::CommandAction::ShowContext => CommandAction::ShowContext,
        crate::command::CommandAction::Undo => CommandAction::Undo,
        crate::command::CommandAction::Rewind { target } => CommandAction::Rewind { target },
        crate::command::CommandAction::Forks { forkId } => CommandAction::Forks { forkId },
        crate::command::CommandAction::Resume { sessionId } => {
            CommandAction::Resume { sessionId }
        }
        crate::command::CommandAction::Clear => CommandAction::Clear,
        crate::command::CommandAction::Mcp => CommandAction::Mcp,
        crate::command::CommandAction::Lsp => CommandAction::Lsp,
        crate::command::CommandAction::Permissions => CommandAction::Permissions,
        crate::command::CommandAction::SavePermissions { defaultMode, rules } => {
            CommandAction::SavePermissions { defaultMode, rules }
        }
    }
}
