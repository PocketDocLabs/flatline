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
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    Terminal as RatatuiTerminal,
};
use tokio::sync::mpsc;

use construct::permissions::Permissions;
use construct::session::{Session, SessionEvent};
use construct::shell::{ShellIo, spawnShell};

use crate::agent_panel::AgentPanel;
use crate::selection::{self, PanelId, SelectionState};
use crate::terminal::{Terminal as EmbeddedTerminal, TerminalState};

use std::io;
use std::time::Duration;

/// Which panel has input focus.
#[derive(PartialEq)]
enum Focus {
    Terminal,
    Agent,
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
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = RatatuiTerminal::new(backend)?;

    let size = terminal.size()?;
    // Terminal gets ~60% width minus borders, full height minus status bar and borders.
    let termCols = (size.width * 3 / 5).saturating_sub(2);
    let termRows = size.height.saturating_sub(3);
    let (shell, mut shellIo) = spawnShell(termCols, termRows)?;
    let mut termState = TerminalState::new(termCols, termRows);

    let mut agentPanel = AgentPanel::new();
    let mut focus = Focus::Terminal;
    let mut selState = SelectionState::new();

    // Session channels.
    let (eventTx, mut eventRx) = mpsc::channel::<SessionEvent>(256);
    let (permitTx, permitRx) = mpsc::channel::<bool>(1);
    let (userInputTx, mut userInputRx) = mpsc::channel::<String>(16);

    // Spawn the agent session task.
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

        // Default: ask for permission on every tool call.
        let permissions = Permissions::askForEverything();

        let mut session = match Session::new(&config, permissions, shell) {
            Ok(s) => s,
            Err(e) => {
                let _ = eventTx
                    .send(SessionEvent::Error(format!("Session error: {e}")))
                    .await;
                return;
            }
        };

        let mut permitRx = permitRx;

        while let Some(msg) = userInputRx.recv().await {
            if let Err(e) = session.send(&msg, &eventTx, &mut permitRx).await {
                let _ = eventTx
                    .send(SessionEvent::Error(format!("Agent error: {e}")))
                    .await;
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
        &mut eventRx,
        &permitTx,
        &userInputTx,
    )
    .await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::event::DisableMouseCapture,
        crossterm::event::DisableBracketedPaste,
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;

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
    eventRx: &mut mpsc::Receiver<SessionEvent>,
    permitTx: &mpsc::Sender<bool>,
    userInputTx: &mpsc::Sender<String>,
) -> Result<()> {
    loop {
        // Draw.
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
            frame.render_widget(termBlock, hChunks[0]);
            frame.render_stateful_widget(EmbeddedTerminal, termInner, termState);

            // Capture content rects for mouse hit-testing.
            selState.termContentRect = termInner;

            // Agent panel.
            let agentChatArea = agentPanel.render(hChunks[1], frame.buffer_mut(), *focus == Focus::Agent);
            selState.agentContentRect = agentChatArea;

            // Expand selection for double/triple/quad click (needs Buffer).
            let termOffset = termState.displayOffset() as u16;
            let agentOffset = agentPanel.displayOffset();
            if let Some((panel, clickCount)) = selState.pendingExpand.take() {
                let (sel, rect, offset) = match panel {
                    PanelId::Terminal => (&mut selState.termSelection, termInner, termOffset),
                    PanelId::Agent => (&mut selState.agentSelection, agentChatArea, agentOffset),
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

            // Apply selection highlights after widgets have rendered.
            if let Some(ref sel) = selState.termSelection {
                selection::applyHighlight(sel, termInner, frame.buffer_mut(), termOffset);
            }
            if let Some(ref sel) = selState.agentSelection {
                selection::applyHighlight(sel, agentChatArea, frame.buffer_mut(), agentOffset);
            }

            // Deferred clipboard copy (Buffer only available during draw).
            if let Some(panel) = selState.pendingCopy.take() {
                let (sel, rect, offset) = match panel {
                    PanelId::Terminal => (&selState.termSelection, termInner, termOffset),
                    PanelId::Agent => (&selState.agentSelection, agentChatArea, agentOffset),
                };
                if let Some(sel) = sel {
                    let text = selection::extractText(sel, rect, frame.buffer_mut(), offset);
                    selection::copyToClipboard(&text);
                }
            }

            // Status bar.
            let modeStr = match focus {
                Focus::Terminal => "terminal",
                Focus::Agent => "agent",
            };
            let statusText = format!(" [{modeStr}]  Tab: switch  Ctrl+Q: quit");
            let statusBar = Paragraph::new(statusText)
                .style(Style::default().bg(Color::DarkGray).fg(Color::White));
            frame.render_widget(statusBar, vChunks[1]);
        })?;

        // Drain PTY output.
        while let Ok(bytes) = shellIo.outputRx.try_recv() {
            termState.process(&bytes);
        }

        // Drain session events.
        while let Ok(event) = eventRx.try_recv() {
            match event {
                SessionEvent::ContentDelta(text) => agentPanel.appendContent(&text),
                SessionEvent::ReasoningDelta(text) => agentPanel.appendReasoning(&text),
                SessionEvent::ToolRequest { name, summary, .. } => {
                    agentPanel.showToolRequest(&name, &summary);
                }
                SessionEvent::ToolResult { name, output } => {
                    agentPanel.pushToolResult(&name, &output);
                }
                SessionEvent::ToolAutoApproved { name, summary } => {
                    agentPanel.toolApproved(&format!("{name}: {summary}"));
                }
                SessionEvent::ToolDenied { name } => {
                    agentPanel.toolDenied(&name);
                }
                SessionEvent::TurnAborted { name } => {
                    agentPanel.pushError(&format!("Turn aborted: {name} not permitted"));
                }
                SessionEvent::TurnComplete => {
                    agentPanel.finalizeStreaming();
                }
                SessionEvent::Error(msg) => {
                    agentPanel.pushError(&msg);
                }
            }
        }

        // Handle input.
        if handleInput(
            focus,
            shellIo,
            termState,
            agentPanel,
            selState,
            permitTx,
            userInputTx,
        )
        .await?
        {
            break;
        }

        tokio::task::yield_now().await;
    }

    Ok(())
}

/// Handle one input event. Returns true if the user wants to quit.
#[allow(clippy::too_many_arguments)]
async fn handleInput(
    focus: &mut Focus,
    shellIo: &ShellIo,
    termState: &mut TerminalState,
    agentPanel: &mut AgentPanel,
    selState: &mut SelectionState,
    permitTx: &mpsc::Sender<bool>,
    userInputTx: &mpsc::Sender<String>,
) -> Result<bool> {
    if !event::poll(Duration::from_millis(16))? {
        return Ok(false);
    }

    match event::read()? {
        Event::Key(key) => {
            // Any keypress clears selections.
            selState.termSelection = None;
            selState.agentSelection = None;

            // Global keybindings.
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('q') {
                return Ok(true);
            }

            if key.code == KeyCode::Tab {
                *focus = match focus {
                    Focus::Terminal => Focus::Agent,
                    Focus::Agent => Focus::Terminal,
                };
                return Ok(false);
            }

            // Permission prompt takes priority regardless of focus.
            if agentPanel.pendingPermit {
                match key.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        agentPanel.approvePending();
                        let _ = permitTx.send(true).await;
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') => {
                        agentPanel.denyPending();
                        let _ = permitTx.send(false).await;
                    }
                    _ => {}
                }
                return Ok(false);
            }

            match focus {
                Focus::Terminal => {
                    if let Some(bytes) = keyToBytes(&key) {
                        // Snap to bottom on user input.
                        if termState.displayOffset() > 0 {
                            termState.scrollToBottom();
                        }
                        let _ = shellIo.inputTx.try_send(bytes);
                    }
                }
                Focus::Agent => {
                    match key.code {
                        KeyCode::Enter => {
                            if !agentPanel.inputBuf.is_empty() {
                                let msg = std::mem::take(&mut agentPanel.inputBuf);
                                agentPanel.pushUser(&msg);
                                let _ = userInputTx.send(msg).await;
                            }
                        }
                        KeyCode::Char(c) => {
                            agentPanel.inputBuf.push(c);
                        }
                        KeyCode::Backspace => {
                            if agentPanel.isLargePaste() {
                                agentPanel.inputBuf.clear();
                            } else {
                                agentPanel.inputBuf.pop();
                            }
                        }
                        KeyCode::Up => {
                            agentPanel.scrollUp(3);
                        }
                        KeyCode::Down => {
                            agentPanel.scrollDown(3);
                        }
                        _ => {}
                    }
                }
            }
        }
        Event::Mouse(mouse) => {
            handleMouse(mouse, focus, agentPanel, termState, selState, shellIo);
        }
        Event::Paste(text) => {
            if *focus == Focus::Agent && !agentPanel.pendingPermit {
                let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                agentPanel.inputBuf.push_str(&normalized);
            }
        }
        Event::Resize(cols, rows) => {
            let termCols = (cols * 3 / 5).saturating_sub(2);
            let termRows = rows.saturating_sub(3);
            let _ = shellIo.resizeTx.try_send((termCols, termRows));
            termState.resize(termCols, termRows);
        }
        _ => {}
    }

    Ok(false)
}

/// Handle mouse events — selection, scroll wheel.
fn handleMouse(
    mouse: event::MouseEvent,
    focus: &mut Focus,
    agentPanel: &mut AgentPanel,
    termState: &mut TerminalState,
    selState: &mut SelectionState,
    shellIo: &ShellIo,
) {
    // Resolve display offset for coordinate conversion.
    let displayOffset = |panel: PanelId| -> u16 {
        match panel {
            PanelId::Terminal => termState.displayOffset() as u16,
            PanelId::Agent => agentPanel.displayOffset(),
        }
    };

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            let panel = selState.hitTest(mouse.column, mouse.row);
            if let Some(panel) = panel {
                *focus = match panel {
                    PanelId::Terminal => Focus::Terminal,
                    PanelId::Agent => Focus::Agent,
                };

                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let gridLine = selection::toGridLine(screenRow, displayOffset(panel));
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
                let other = match panel {
                    PanelId::Terminal => PanelId::Agent,
                    PanelId::Agent => PanelId::Terminal,
                };
                *selState.selectionForMut(other) = None;
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if let Some(panel) = selState.selectingIn {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine = selection::toGridLine(screenRow, displayOffset(panel));
                if let Some(sel) = selState.selectionForMut(panel) {
                    sel.update(col, gridLine);
                }
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            if let Some(panel) = selState.selectingIn.take() {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine = selection::toGridLine(screenRow, displayOffset(panel));

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
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => agentPanel.scrollUp(3),
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
                None => {}
            }
        }
        MouseEventKind::ScrollDown => {
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => agentPanel.scrollDown(3),
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
                None => {}
            }
        }
        _ => {}
    }
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
