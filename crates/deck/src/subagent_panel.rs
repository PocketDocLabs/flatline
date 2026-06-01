#![allow(non_snake_case)]

//! Live overlay panel for inspecting subagent activity.
//!
//! Two tabs: Transcript (full agent panel rendering, via `AgentPanel`) and
//! Shell (VT-emulated PTY output, via `EmbeddedTerminal`).
//!
//! For a live subagent the panel reads transcript + shell state directly from
//! `agentPanel.activeSubagent` on every render — single source of truth.
//! For resumed/completed sessions, the panel owns a frozen snapshot.
//!
//! Mouse and keyboard input are routed through the panel when it is open;
//! code-block copy/expand, drag-selection, and inline subagent permits all
//! work inside the popup.
//!
//! # Public API
//! - [`SubagentPanel`] — overlay state
//! - [`SubagentSource`] — Live (borrowed) or Frozen (owned snapshot)
//! - [`SubagentMouseAction`] — outcome of a routed mouse event
//!
//! # Dependencies
//! `ratatui`, `crossterm`, `agent_panel`, `terminal`, `selection`

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, StatefulWidget, Widget, Wrap};

use crossterm::event::{KeyEvent, MouseButton, MouseEvent, MouseEventKind};

use crate::agent_panel::{AgentPanel, PanelEntry};
use crate::selection::{self, Click, Selection};
use crate::terminal::{Terminal as EmbeddedTerminal, TerminalState};

/// Which tab is active.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubagentTab {
    Transcript,
    Shell,
}

/// Where the popup reads its content from.
#[allow(clippy::large_enum_variant)]
pub enum SubagentSource {
    /// Live: read from `agentPanel.activeSubagent` on every render.
    /// Stays Live as long as the subagent's data is in `activeSubagent`,
    /// regardless of whether it's still running.
    Live,
    /// Frozen snapshot — used for resumed sessions where there's no live
    /// subagent to borrow from. Owns its own transcript + shell state.
    Frozen {
        agentType: String,
        transcript: Vec<PanelEntry>,
        shellTerm: TerminalState,
    },
}

/// Outcome of routing a mouse event to the popup.
pub enum SubagentMouseAction {
    /// Event handled by the popup; main app should not process it.
    Handled,
    /// User clicked outside the popup; main app may close the popup.
    ClickOutside,
}

/// Live overlay for inspecting subagent activity.
pub struct SubagentPanel {
    pub source: SubagentSource,
    pub tab: SubagentTab,
    /// AgentPanel instance used purely as a renderer for the transcript tab.
    /// Holds its own scroll, code-block expansion, and copy-flash state so
    /// interactions inside the popup are independent of the main panel.
    pub transcriptPanel: AgentPanel,
    /// Active text selection inside the transcript tab.
    pub selection: Option<Selection>,
    /// Click counter for double/triple/quad-click detection.
    pub click: Click,
    /// True while a drag-selection is in progress.
    pub selecting: bool,
    /// Copy the current selection on the next render (after Buffer is built).
    pub pendingCopy: bool,
    /// Expand the current selection on the next render: (clickCount).
    pub pendingExpand: Option<usize>,

    /// Popup outer rect from the last render (for click-outside detection).
    pub lastPopupRect: Rect,
    /// Inner content rect of the active tab from the last render
    /// (transcript or shell). Used for mouse hit-testing on content clicks.
    pub lastContentRect: Rect,
    /// Selection coord rect — content rect shifted right by the 2-col
    /// line prefix so selection columns match the convention used by
    /// `AgentPanel::extractUnwrappedText`. Set when the transcript tab is
    /// active.
    pub lastSelectionRect: Rect,
    /// Tab bar row from the last render (for tab clicks).
    pub lastTabBarRect: Rect,
    /// Per-row click hitboxes for the parallel-subagent tab strip. One
    /// rect per entry in `agentPanel.activeSubagents`, in matching order.
    /// Cleared (length 0) whenever fewer than 2 subagents are alive.
    subagentTabRects: Vec<(Rect, String)>,
    /// Per-tab click rects: (transcriptRect, shellRect). Set during render.
    pub tabRects: [Rect; 2],
    /// Permit overlay rect from the last render (when subagent permit pending).
    pub lastPermitRect: Rect,
}

impl SubagentPanel {
    /// Open in Live mode — content is borrowed from `agentPanel.activeSubagent`.
    pub fn live() -> Self {
        Self::base(SubagentSource::Live)
    }

    /// Open in Frozen mode — owns a snapshot for resumed sessions.
    pub fn frozen(agentType: &str, transcript: Vec<PanelEntry>) -> Self {
        Self::base(SubagentSource::Frozen {
            agentType: agentType.into(),
            transcript,
            shellTerm: TerminalState::new(120, 40),
        })
    }

    fn base(source: SubagentSource) -> Self {
        Self {
            source,
            tab: SubagentTab::Transcript,
            transcriptPanel: AgentPanel::new(),
            selection: None,
            click: Click::new(),
            selecting: false,
            pendingCopy: false,
            pendingExpand: None,
            lastPopupRect: Rect::default(),
            lastContentRect: Rect::default(),
            lastSelectionRect: Rect::default(),
            lastTabBarRect: Rect::default(),
            tabRects: [Rect::default(); 2],
            subagentTabRects: Vec::new(),
            lastPermitRect: Rect::default(),
        }
    }

    /// Display name of the subagent (live or frozen).
    fn agentTypeFor<'a>(&'a self, agentPanel: &'a AgentPanel) -> &'a str {
        match (&self.source, agentPanel.currentSubagent()) {
            (SubagentSource::Live, Some(sub)) => &sub.agentType,
            (SubagentSource::Frozen { agentType, .. }, _) => agentType,
            _ => "subagent",
        }
    }

    /// True when the subagent is still running (live + has a currently
    /// selected subagent whose SubagentBlock is still unfinished).
    fn isRunning(&self, agentPanel: &AgentPanel) -> bool {
        if !matches!(self.source, SubagentSource::Live) {
            return false;
        }
        let Some(sub) = agentPanel.currentSubagent() else {
            return false;
        };
        // Match the SubagentBlock by sessionId — looking for "the most
        // recent unfinished block" is wrong with parallel subagents.
        for entry in agentPanel.entries.iter().rev() {
            if let PanelEntry::SubagentBlock {
                sessionId: Some(sid),
                done,
                ..
            } = entry
                && sid == &sub.sessionId
            {
                return !*done;
            }
        }
        true
    }

    /// Elapsed runtime for the currently-selected live subagent.
    fn elapsedSecs(&self, agentPanel: &AgentPanel) -> Option<u64> {
        match &self.source {
            SubagentSource::Live => agentPanel
                .currentSubagent()
                .map(|s| s.startTime.elapsed().as_secs()),
            SubagentSource::Frozen { .. } => None,
        }
    }

    /// Scroll up in the active tab.
    pub fn scrollUp(&mut self, agentPanel: &mut AgentPanel, lines: u16) {
        match self.tab {
            SubagentTab::Transcript => self.transcriptPanel.scrollUp(lines),
            SubagentTab::Shell => self.shellMut(agentPanel).scrollUp(lines as i32),
        }
    }

    /// Scroll down in the active tab.
    pub fn scrollDown(&mut self, agentPanel: &mut AgentPanel, lines: u16) {
        match self.tab {
            SubagentTab::Transcript => self.transcriptPanel.scrollDown(lines),
            SubagentTab::Shell => self.shellMut(agentPanel).scrollDown(lines as i32),
        }
    }

    /// Mutable reference to the shell TerminalState backing the current view.
    /// Promotes Live → Frozen with an empty terminal if the live subagent
    /// disappeared (graceful degradation; should not happen in practice).
    fn shellMut<'a>(&'a mut self, agentPanel: &'a mut AgentPanel) -> &'a mut TerminalState {
        if matches!(self.source, SubagentSource::Live) && agentPanel.currentSubagent().is_none() {
            self.source = SubagentSource::Frozen {
                agentType: "subagent".into(),
                transcript: Vec::new(),
                shellTerm: TerminalState::new(120, 40),
            };
        }
        match &mut self.source {
            SubagentSource::Frozen { shellTerm, .. } => shellTerm,
            SubagentSource::Live => {
                &mut agentPanel
                    .currentSubagentMut()
                    .expect("checked above")
                    .shellTerm
            }
        }
    }

    /// Handle a key event. Returns true if the panel should close.
    ///
    /// Esc / q / v always close (even mid-permit; the permit state lives on
    /// `agentPanel` and the main panel renders it once the popup is gone).
    /// Tab and scroll keys always navigate the popup.
    ///
    /// When a subagent permit is pending, action keys (y/n/A/D, Shift+arrows,
    /// custom-pattern chars, Backspace) are NOT consumed — they fall through
    /// to the app's permit dispatcher. `consumedKey` tells the caller which
    /// keys the popup swallows so the caller can break vs fall through.
    pub fn handleKey(&mut self, key: KeyEvent, agentPanel: &mut AgentPanel) -> bool {
        use crossterm::event::KeyCode;

        // Popup-close keys are always honored.
        if matches!(
            key.code,
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('v')
        ) {
            return true;
        }

        // Navigation keys always work in the popup (even during a permit).
        match key.code {
            KeyCode::Tab => {
                self.tab = match self.tab {
                    SubagentTab::Transcript => SubagentTab::Shell,
                    SubagentTab::Shell => SubagentTab::Transcript,
                };
            }
            // Cycle parallel subagents. `]` / `[` mirrors `Cmd+Shift+]`
            // tab-cycle conventions; no-ops when only one is live.
            KeyCode::Char(']') => agentPanel.cycleSubagent(1),
            KeyCode::Char('[') => agentPanel.cycleSubagent(-1),
            KeyCode::Up | KeyCode::Char('k') => self.scrollUp(agentPanel, 3),
            KeyCode::Down | KeyCode::Char('j') => self.scrollDown(agentPanel, 3),
            KeyCode::PageUp => self.scrollUp(agentPanel, 20),
            KeyCode::PageDown => self.scrollDown(agentPanel, 20),
            _ => {}
        }
        false
    }

    /// Whether the popup consumed this key (caller should NOT fall through
    /// to the app-level handlers).
    pub fn consumedKey(&self, key: &KeyEvent, agentPanel: &AgentPanel) -> bool {
        use crossterm::event::KeyCode;
        let isNav = matches!(
            key.code,
            KeyCode::Esc
                | KeyCode::Char('q')
                | KeyCode::Char('v')
                | KeyCode::Tab
                | KeyCode::Up
                | KeyCode::Down
                | KeyCode::PageUp
                | KeyCode::PageDown
                | KeyCode::Char('k')
                | KeyCode::Char('j')
                | KeyCode::Char('[')
                | KeyCode::Char(']')
        );
        if isNav {
            return true;
        }
        // During a subagent permit, action keys fall through to the dispatcher.
        !(agentPanel.pendingPermit && agentPanel.pendingPermitIsSubagent())
    }

    /// Route a mouse event to the popup. Click-outside returns
    /// `ClickOutside` so the caller can decide whether to close.
    pub fn handleMouse(
        &mut self,
        ev: MouseEvent,
        agentPanel: &mut AgentPanel,
    ) -> SubagentMouseAction {
        use crossterm::event::KeyModifiers;
        let inPopup = self.lastPopupRect.contains((ev.column, ev.row).into());
        let inPermit = self.lastPermitRect.contains((ev.column, ev.row).into());
        let inTranscriptContent = self.tab == SubagentTab::Transcript
            && self.lastContentRect.contains((ev.column, ev.row).into());

        match ev.kind {
            MouseEventKind::ScrollUp => {
                if inPermit && agentPanel.pendingPermit {
                    // Horizontal scroll on permit code block (matches the
                    // main panel's Shift+ScrollUp / ScrollLeft behavior).
                    agentPanel.scrollPermitCode(-3);
                    return SubagentMouseAction::Handled;
                }
                if inTranscriptContent && ev.modifiers.contains(KeyModifiers::SHIFT) {
                    let localRow = ev.row.saturating_sub(self.lastContentRect.y);
                    let gridLine =
                        selection::toGridLine(localRow, self.transcriptPanel.displayOffset());
                    if let Some(blockId) = self.transcriptPanel.codeBlockAtGridLine(gridLine) {
                        self.transcriptPanel.scrollCodeBlockH(blockId, -3);
                        return SubagentMouseAction::Handled;
                    }
                }
                if inPopup {
                    self.scrollUp(agentPanel, 3);
                }
                SubagentMouseAction::Handled
            }
            MouseEventKind::ScrollDown => {
                if inPermit && agentPanel.pendingPermit {
                    agentPanel.scrollPermitCode(3);
                    return SubagentMouseAction::Handled;
                }
                if inTranscriptContent && ev.modifiers.contains(KeyModifiers::SHIFT) {
                    let localRow = ev.row.saturating_sub(self.lastContentRect.y);
                    let gridLine =
                        selection::toGridLine(localRow, self.transcriptPanel.displayOffset());
                    if let Some(blockId) = self.transcriptPanel.codeBlockAtGridLine(gridLine) {
                        self.transcriptPanel.scrollCodeBlockH(blockId, 3);
                        return SubagentMouseAction::Handled;
                    }
                }
                if inPopup {
                    self.scrollDown(agentPanel, 3);
                }
                SubagentMouseAction::Handled
            }
            MouseEventKind::ScrollLeft => {
                if inPermit && agentPanel.pendingPermit {
                    agentPanel.scrollPermitCode(-3);
                    return SubagentMouseAction::Handled;
                }
                if inTranscriptContent {
                    let localRow = ev.row.saturating_sub(self.lastContentRect.y);
                    let gridLine =
                        selection::toGridLine(localRow, self.transcriptPanel.displayOffset());
                    if let Some(blockId) = self.transcriptPanel.codeBlockAtGridLine(gridLine) {
                        self.transcriptPanel.scrollCodeBlockH(blockId, -3);
                    }
                }
                SubagentMouseAction::Handled
            }
            MouseEventKind::ScrollRight => {
                if inPermit && agentPanel.pendingPermit {
                    agentPanel.scrollPermitCode(3);
                    return SubagentMouseAction::Handled;
                }
                if inTranscriptContent {
                    let localRow = ev.row.saturating_sub(self.lastContentRect.y);
                    let gridLine =
                        selection::toGridLine(localRow, self.transcriptPanel.displayOffset());
                    if let Some(blockId) = self.transcriptPanel.codeBlockAtGridLine(gridLine) {
                        self.transcriptPanel.scrollCodeBlockH(blockId, 3);
                    }
                }
                SubagentMouseAction::Handled
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if !inPopup {
                    return SubagentMouseAction::ClickOutside;
                }

                // Permit code-block "copy" label click.
                if inPermit && agentPanel.pendingPermit && agentPanel.pendingCommand().is_some() {
                    let localRow = ev.row.saturating_sub(self.lastPermitRect.y);
                    let localCol = ev.column.saturating_sub(self.lastPermitRect.x);
                    if localRow == 0
                        && localCol + 6 >= self.lastPermitRect.width
                        && let Some(cmd) = agentPanel.pendingCommand()
                    {
                        selection::copyToClipboard(cmd);
                        agentPanel.flashCopied();
                        return SubagentMouseAction::Handled;
                    }
                }

                // Tab bar click — switch tabs.
                if self.lastTabBarRect.contains((ev.column, ev.row).into()) {
                    self.handleTabBarClick(ev.column);
                    return SubagentMouseAction::Handled;
                }

                // Parallel-subagent tab strip click — switch to that subagent.
                for (rect, sessionId) in &self.subagentTabRects {
                    if rect.contains((ev.column, ev.row).into()) {
                        let sid = sessionId.clone();
                        agentPanel.selectSubagentBySessionId(&sid);
                        return SubagentMouseAction::Handled;
                    }
                }

                // Content click in the active tab.
                if inTranscriptContent {
                    self.handleTranscriptClick(ev.column, ev.row);
                    return SubagentMouseAction::Handled;
                }

                SubagentMouseAction::Handled
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if self.selecting && self.tab == SubagentTab::Transcript {
                    let rect = self.lastSelectionRect;
                    if rect.height > 0 && rect.width > 0 {
                        let col = ev
                            .column
                            .saturating_sub(rect.x)
                            .min(rect.width.saturating_sub(1));
                        let row = ev
                            .row
                            .saturating_sub(rect.y)
                            .min(rect.height.saturating_sub(1));
                        let gridLine =
                            selection::toGridLine(row, self.transcriptPanel.displayOffset());
                        if let Some(sel) = self.selection.as_mut() {
                            sel.update(col, gridLine);
                        }
                    }
                    return SubagentMouseAction::Handled;
                }
                SubagentMouseAction::Handled
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.selecting {
                    self.selecting = false;
                    if let Some(sel) = self.selection.as_mut() {
                        sel.finalize();
                        if sel.isEmpty() {
                            self.selection = None;
                        } else {
                            self.pendingCopy = true;
                        }
                    }
                    return SubagentMouseAction::Handled;
                }
                SubagentMouseAction::Handled
            }
            _ => SubagentMouseAction::Handled,
        }
    }

    /// Tab bar click: hit-test against the per-tab rects computed during render.
    fn handleTabBarClick(&mut self, col: u16) {
        let row = self.lastTabBarRect.y;
        for (i, rect) in self.tabRects.iter().enumerate() {
            if rect.contains((col, row).into()) {
                self.tab = if i == 0 {
                    SubagentTab::Transcript
                } else {
                    SubagentTab::Shell
                };
                return;
            }
        }
    }

    /// Process a left-click inside the transcript tab content.
    fn handleTranscriptClick(&mut self, col: u16, row: u16) {
        // Selection / agent_panel hit tests use "after-prefix" coordinates:
        // col 0 = first content character, prefix excluded. Shift by +2 to
        // match the convention used by the main panel.
        let selRect = self.lastSelectionRect;
        let displayOffset = self.transcriptPanel.displayOffset();
        let localCol = col.saturating_sub(selRect.x);
        let localRow = row.saturating_sub(self.lastContentRect.y);
        let gridLine = selection::toGridLine(localRow, displayOffset);

        // Reasoning toggle (col-independent).
        if self.transcriptPanel.toggleReasoningAtGridLine(gridLine) {
            return;
        }
        // Code block copy (after-prefix col).
        if self.transcriptPanel.tryCopyCodeBlock(gridLine, localCol) {
            return;
        }
        // Code block expand/collapse (col-independent).
        if self.transcriptPanel.tryToggleCodeBlock(gridLine) {
            return;
        }
        // Subagent toggle (col-independent; rare in subagent transcripts).
        if self.transcriptPanel.tryToggleSubagentContent(gridLine) {
            return;
        }

        // Otherwise begin a selection.
        let clickCount = self.click.record(localCol, localRow);
        self.selection = Some(Selection::new(localCol, gridLine));
        self.selecting = true;
        if clickCount >= 2 {
            self.pendingExpand = Some(clickCount);
        }
    }

    /// Render the popup. Reads transcript + shell from `agentPanel.activeSubagent`
    /// when in Live mode; from owned snapshot when Frozen.
    pub fn render(&mut self, screenArea: Rect, buf: &mut Buffer, agentPanel: &mut AgentPanel) {
        let popupWidth = (screenArea.width * 4 / 5).max(40);
        let popupHeight = (screenArea.height * 4 / 5).max(10);
        let x = screenArea.x + (screenArea.width.saturating_sub(popupWidth)) / 2;
        let y = screenArea.y + (screenArea.height.saturating_sub(popupHeight)) / 2;
        let popupArea = Rect::new(x, y, popupWidth, popupHeight);
        self.lastPopupRect = popupArea;

        // Clear popup background. ratatui's `set_style` only overrides
        // fields that are Some, so any cell drawn later by a span/grapheme
        // with `fg = None` keeps the cell's prior fg. Underneath us are the
        // terminal/agent panel borders (cyan/dark gray) — without a hard
        // reset and an explicit fg here, those colors bleed through any
        // popup cell drawn with default-fg content (alacritty cells with
        // default fg, blank padding inside spans, etc).
        for row in popupArea.y..popupArea.y + popupArea.height {
            for col in popupArea.x..popupArea.x + popupArea.width {
                if let Some(cell) = buf.cell_mut((col, row)) {
                    cell.reset();
                    cell.set_char(' ');
                    cell.set_fg(Color::Gray);
                    cell.set_bg(Color::Black);
                }
            }
        }

        // Header title with elapsed/turn info.
        let agentType = self.agentTypeFor(agentPanel).to_string();
        let statusSuffix = if let Some(secs) = self.elapsedSecs(agentPanel) {
            if self.isRunning(agentPanel) {
                format!(" \u{25CF} {secs}s ")
            } else {
                format!(" \u{2713}\u{FE0E} {secs}s ")
            }
        } else {
            " \u{2713}\u{FE0E} resumed ".into()
        };
        let scrollSuffix = if self.tab == SubagentTab::Shell {
            let off = match &self.source {
                SubagentSource::Frozen { shellTerm, .. } => shellTerm.displayOffset(),
                SubagentSource::Live => agentPanel
                    .currentSubagent()
                    .map(|s| s.shellTerm.displayOffset())
                    .unwrap_or(0),
            };
            if off > 0 {
                format!("[\u{2191}{}\u{FE0E}] ", off)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let activeTabStyle = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let inactiveTabStyle = Style::default().fg(Color::Gray).bg(Color::Black);

        // Title sections are: " {agentType}{statusSuffix}" then the two tabs.
        // Compute tab rects so clicks land on the right tab regardless of
        // how wide the agent label is.
        let agentSection = format!(" {agentType}{statusSuffix}");
        let agentSectionW = unicode_width::UnicodeWidthStr::width(agentSection.as_str()) as u16;
        const TRANSCRIPT_LABEL: &str = " Transcript ";
        const SHELL_LABEL: &str = " Shell ";
        let transcriptW = unicode_width::UnicodeWidthStr::width(TRANSCRIPT_LABEL) as u16;
        let shellW = unicode_width::UnicodeWidthStr::width(SHELL_LABEL) as u16;
        // Title starts at popupArea.x + 1 (after the left border corner).
        let titleStart = popupArea.x + 1;
        let transcriptX = titleStart + agentSectionW;
        let shellX = transcriptX + transcriptW + 1; // +1 for the divider char.
        self.tabRects = [
            Rect {
                x: transcriptX,
                y: popupArea.y,
                width: transcriptW,
                height: 1,
            },
            Rect {
                x: shellX,
                y: popupArea.y,
                width: shellW,
                height: 1,
            },
        ];

        let title = Line::from(vec![
            Span::styled(
                agentSection,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                TRANSCRIPT_LABEL,
                if self.tab == SubagentTab::Transcript {
                    activeTabStyle
                } else {
                    inactiveTabStyle
                },
            ),
            Span::styled("\u{2502}", Style::default().fg(Color::DarkGray)),
            Span::styled(
                SHELL_LABEL,
                if self.tab == SubagentTab::Shell {
                    activeTabStyle
                } else {
                    inactiveTabStyle
                },
            ),
            Span::raw(" "),
            Span::styled(scrollSuffix, Style::default().fg(Color::DarkGray)),
        ]);

        let footer = self.footerLine(agentPanel);

        // Border goes yellow when a subagent permit is awaiting response, so
        // the popup visually signals "needs attention" even if the user was
        // scrolled into the transcript.
        let borderColor = if agentPanel.pendingPermit && agentPanel.pendingPermitIsSubagent() {
            Color::Yellow
        } else {
            Color::Cyan
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(borderColor))
            .title(title)
            .title_bottom(footer);

        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        // Tab bar lives on the top border (the title line). Click-test there.
        self.lastTabBarRect = Rect {
            x: popupArea.x + 1,
            y: popupArea.y,
            width: popupArea.width.saturating_sub(2),
            height: 1,
        };

        // Reserve bottom rows for an inline subagent permit prompt.
        let permitHeight = if agentPanel.pendingPermit && agentPanel.pendingPermitIsSubagent() {
            agentPanel
                .permitInlineHeight(inner.width)
                .min(inner.height.saturating_sub(3))
                .max(8)
        } else {
            0
        };

        let (contentArea, permitArea) = if permitHeight > 0 && inner.height > permitHeight + 1 {
            let split = inner.height.saturating_sub(permitHeight + 1);
            let content = Rect {
                x: inner.x,
                y: inner.y,
                width: inner.width,
                height: split,
            };
            let divider = Rect {
                x: inner.x,
                y: inner.y + split,
                width: inner.width,
                height: 1,
            };
            let permit = Rect {
                x: inner.x,
                y: inner.y + split + 1,
                width: inner.width,
                height: permitHeight,
            };
            // Draw divider line.
            let sep = "\u{2500}".repeat(divider.width as usize);
            Paragraph::new(sep)
                .style(Style::default().fg(Color::DarkGray))
                .render(divider, buf);
            (content, Some(permit))
        } else {
            (inner, None)
        };

        // When the parent has fanned out multiple parallel subagents, draw
        // a secondary tab strip just below the title row listing each one.
        // Empty otherwise so the body has the full pane.
        let multiTabHeight: u16 = if agentPanel.activeSubagents.len() > 1 {
            1
        } else {
            0
        };
        let bodyArea = if multiTabHeight > 0 && contentArea.height > multiTabHeight {
            let strip = Rect {
                x: contentArea.x,
                y: contentArea.y,
                width: contentArea.width,
                height: multiTabHeight,
            };
            self.renderSubagentTabStrip(strip, buf, agentPanel);
            Rect {
                x: contentArea.x,
                y: contentArea.y + multiTabHeight,
                width: contentArea.width,
                height: contentArea.height - multiTabHeight,
            }
        } else {
            self.subagentTabRects.clear();
            contentArea
        };

        self.lastContentRect = bodyArea;
        // Selection rect is content rect shifted +2 cols (after-prefix), so
        // selection coords match the convention used by extractUnwrappedText.
        self.lastSelectionRect = Rect {
            x: bodyArea.x + 2,
            y: bodyArea.y,
            width: bodyArea.width.saturating_sub(2),
            height: bodyArea.height,
        };

        match self.tab {
            SubagentTab::Transcript => self.renderTranscript(bodyArea, buf, agentPanel),
            SubagentTab::Shell => self.renderShell(bodyArea, buf, agentPanel),
        }

        if let Some(rect) = permitArea {
            self.lastPermitRect = rect;
            agentPanel.renderPermitInline(rect, buf);
        } else {
            self.lastPermitRect = Rect::default();
        }
    }

    fn footerLine<'a>(&self, agentPanel: &AgentPanel) -> Line<'a> {
        let dim = Style::default().fg(Color::DarkGray);
        if agentPanel.pendingPermit && agentPanel.pendingPermitIsSubagent() {
            Line::from(vec![
                Span::styled(
                    " \u{26A0}\u{FE0E} subagent requests permission \u{2014} ",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("[y]allow [n]deny [A]always [D]never ", dim),
            ])
        } else {
            Line::from(vec![Span::styled(
                " [Tab] switch  [\u{2191}/\u{2193}] scroll  [v/Esc] close ",
                dim,
            )])
        }
    }

    /// Render the parallel-subagent tab strip — one labelled chip per
    /// live subagent. Highlights the currently selected one and records
    /// per-chip click rects.
    fn renderSubagentTabStrip(&mut self, area: Rect, buf: &mut Buffer, agentPanel: &AgentPanel) {
        use unicode_width::UnicodeWidthStr as UWS;

        // Background fill so cell colors don't bleed.
        for col in area.x..area.x + area.width {
            if let Some(cell) = buf.cell_mut((col, area.y)) {
                cell.reset();
                cell.set_char(' ');
                cell.set_bg(Color::Rgb(20, 20, 30));
                cell.set_fg(Color::Gray);
            }
        }

        let active = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let inactive = Style::default().fg(Color::Gray).bg(Color::Rgb(20, 20, 30));
        let pendingPermit = Style::default()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD);

        let mut x = area.x;
        let mut rects: Vec<(Rect, String)> = Vec::with_capacity(agentPanel.activeSubagents.len());
        let isSubagentPermit = agentPanel.pendingPermit && agentPanel.pendingPermitIsSubagent();
        let permitSessionId = if isSubagentPermit {
            agentPanel.pendingPermitSubagentSessionId()
        } else {
            None
        };

        for (idx, sub) in agentPanel.activeSubagents.iter().enumerate() {
            let label = format!(" #{} {} ", idx + 1, sub.agentType);
            let labelW = UWS::width(label.as_str()) as u16;
            if x + labelW > area.x + area.width {
                break;
            }

            let isSelected = agentPanel.selectedSubagent == Some(idx);
            let hasPermit = permitSessionId
                .as_deref()
                .map(|sid| sid == sub.sessionId)
                .unwrap_or(false);
            let style = if hasPermit {
                pendingPermit
            } else if isSelected {
                active
            } else {
                inactive
            };

            let chipRect = Rect {
                x,
                y: area.y,
                width: labelW,
                height: 1,
            };
            buf.set_span(x, area.y, &Span::styled(label, style), labelW);
            rects.push((chipRect, sub.sessionId.clone()));
            x += labelW;

            // Thin divider between chips (except after last).
            if idx + 1 < agentPanel.activeSubagents.len() && x < area.x + area.width {
                buf.set_span(
                    x,
                    area.y,
                    &Span::styled(
                        "\u{2502}",
                        Style::default()
                            .fg(Color::DarkGray)
                            .bg(Color::Rgb(20, 20, 30)),
                    ),
                    1,
                );
                x += 1;
            }
        }

        // Cycle hint pinned to the right of the strip when there's room.
        let hint = " [/] cycle ";
        let hintW = UWS::width(hint) as u16;
        if x + hintW + 1 < area.x + area.width {
            let hintX = area.x + area.width - hintW;
            buf.set_span(
                hintX,
                area.y,
                &Span::styled(
                    hint,
                    Style::default()
                        .fg(Color::DarkGray)
                        .bg(Color::Rgb(20, 20, 30)),
                ),
                hintW,
            );
        }

        self.subagentTabRects = rects;
    }

    fn renderTranscript(&mut self, area: Rect, buf: &mut Buffer, agentPanel: &mut AgentPanel) {
        // Pull live transcript from activeSubagent each frame; fall back to
        // frozen snapshot otherwise.
        let entries: Vec<PanelEntry> = match (&self.source, agentPanel.currentSubagent()) {
            (SubagentSource::Live, Some(sub)) => sub.transcript.clone(),
            (SubagentSource::Frozen { transcript, .. }, _) => transcript.clone(),
            (SubagentSource::Live, None) => Vec::new(),
        };

        if entries.is_empty() {
            let placeholder = vec![Line::from(Span::styled(
                "  Waiting for subagent activity\u{2026}",
                Style::default().fg(Color::DarkGray),
            ))];
            Paragraph::new(placeholder).render(area, buf);
            return;
        }

        self.transcriptPanel.entries = entries;
        self.transcriptPanel.renderChatOnly(area, buf);

        // Apply selection highlight using the after-prefix rect so columns
        // line up with the convention used by extractUnwrappedText.
        let selRect = self.lastSelectionRect;
        if let Some(sel) = &self.selection {
            selection::applyHighlight(sel, selRect, buf, self.transcriptPanel.displayOffset());
        }

        // Pending double/triple-click expansion (needs Buffer).
        if let Some(clickCount) = self.pendingExpand.take()
            && let Some(sel) = self.selection.as_mut()
        {
            selection::expandSelection(
                sel,
                clickCount,
                buf,
                selRect,
                self.transcriptPanel.displayOffset(),
            );
            sel.finalize();
            self.pendingCopy = true;
            self.selecting = false;
        }

        // Deferred clipboard copy.
        if self.pendingCopy {
            self.pendingCopy = false;
            if let Some(sel) = &self.selection {
                let text = self.transcriptPanel.extractUnwrappedText(
                    sel,
                    selRect,
                    buf,
                    self.transcriptPanel.displayOffset(),
                );
                selection::copyToClipboard(&text);
            }
        }
    }

    fn renderShell(&mut self, area: Rect, buf: &mut Buffer, agentPanel: &mut AgentPanel) {
        // Resolve the shell TerminalState (live or frozen).
        let term: &mut TerminalState = match &mut self.source {
            SubagentSource::Frozen { shellTerm, .. } => shellTerm,
            SubagentSource::Live => match agentPanel.currentSubagentMut() {
                Some(sub) => &mut sub.shellTerm,
                None => {
                    let lines = vec![Line::from(Span::styled(
                        "  No shell output\u{2026}",
                        Style::default().fg(Color::DarkGray),
                    ))];
                    Paragraph::new(lines)
                        .wrap(Wrap { trim: false })
                        .render(area, buf);
                    return;
                }
            },
        };

        // Resize the display grid to match render area. This reflows the
        // emulated grid only — the underlying construct PTY is a separate
        // concern and is not affected.
        if term.columns() != area.width as usize || term.screenLines() != area.height as usize {
            term.resize(area.width, area.height);
        }
        EmbeddedTerminal.render(area, buf, term);
    }
}
