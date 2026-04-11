#![allow(non_snake_case)]

//! Read-only overlay panel for inspecting subagent activity.
//!
//! Two tabs: Transcript (full agent panel rendering) and Shell (raw PTY output).
//! The transcript tab is a real AgentPanel instance loaded with the subagent's
//! entries — same rendering, same scroll, same code blocks, same everything.
//!
//! # Public API
//! - [`SubagentPanel`] — overlay state
//!
//! # Dependencies
//! `ratatui`, `agent_panel`

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::agent_panel::{AgentPanel, PanelEntry};

/// Which tab is active.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubagentTab {
    Transcript,
    Shell,
}

/// Read-only overlay for inspecting subagent activity.
pub struct SubagentPanel {
    pub agentType: String,
    pub tab: SubagentTab,
    /// Real agent panel for transcript rendering — same code as the main panel.
    pub transcriptPanel: AgentPanel,
    pub transcript: Vec<PanelEntry>,
    pub shellScrollback: Vec<u8>,
    pub shellScrollOffset: u16,
}

impl SubagentPanel {
    pub fn new(agentType: &str) -> Self {
        Self {
            agentType: agentType.into(),
            tab: SubagentTab::Transcript,
            transcriptPanel: AgentPanel::new(),
            transcript: Vec::new(),
            shellScrollback: Vec::new(),
            shellScrollOffset: 0,
        }
    }

    pub fn scrollUp(&mut self) {
        match self.tab {
            SubagentTab::Transcript => self.transcriptPanel.scrollUp(3),
            SubagentTab::Shell => {
                self.shellScrollOffset = self.shellScrollOffset.saturating_add(3);
            }
        }
    }

    pub fn scrollDown(&mut self) {
        match self.tab {
            SubagentTab::Transcript => self.transcriptPanel.scrollDown(3),
            SubagentTab::Shell => {
                self.shellScrollOffset = self.shellScrollOffset.saturating_sub(3);
            }
        }
    }

    /// Handle a key event. Returns true if the panel should close.
    pub fn handleKey(&mut self, key: crossterm::event::KeyEvent) -> bool {
        use crossterm::event::KeyCode;
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return true,
            KeyCode::Tab => {
                self.tab = match self.tab {
                    SubagentTab::Transcript => SubagentTab::Shell,
                    SubagentTab::Shell => SubagentTab::Transcript,
                };
            }
            KeyCode::Up | KeyCode::Char('k') => self.scrollUp(),
            KeyCode::Down | KeyCode::Char('j') => self.scrollDown(),
            KeyCode::PageUp => {
                match self.tab {
                    SubagentTab::Transcript => self.transcriptPanel.scrollUp(20),
                    SubagentTab::Shell => {
                        self.shellScrollOffset = self.shellScrollOffset.saturating_add(20);
                    }
                }
            }
            KeyCode::PageDown => {
                match self.tab {
                    SubagentTab::Transcript => self.transcriptPanel.scrollDown(20),
                    SubagentTab::Shell => {
                        self.shellScrollOffset = self.shellScrollOffset.saturating_sub(20);
                    }
                }
            }
            _ => {}
        }
        false
    }

    pub fn render(&mut self, screenArea: Rect, buf: &mut Buffer) {
        let popupWidth = (screenArea.width * 4 / 5).max(40);
        let popupHeight = (screenArea.height * 4 / 5).max(10);
        let x = screenArea.x + (screenArea.width.saturating_sub(popupWidth)) / 2;
        let y = screenArea.y + (screenArea.height.saturating_sub(popupHeight)) / 2;
        let popupArea = Rect::new(x, y, popupWidth, popupHeight);

        // Clear background.
        let bgStyle = Style::default().bg(Color::Black);
        for row in popupArea.y..popupArea.y + popupArea.height {
            for col in popupArea.x..popupArea.x + popupArea.width {
                if let Some(cell) = buf.cell_mut((col, row)) {
                    cell.set_char(' ');
                    cell.set_style(bgStyle);
                }
            }
        }

        let activeStyle = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
        let inactiveStyle = Style::default().fg(Color::DarkGray);

        let title = Line::from(vec![
            Span::styled(
                format!(" {} ", self.agentType),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " Transcript ",
                if self.tab == SubagentTab::Transcript { activeStyle } else { inactiveStyle },
            ),
            Span::styled("\u{2502}", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " Shell ",
                if self.tab == SubagentTab::Shell { activeStyle } else { inactiveStyle },
            ),
        ]);

        let footer = Line::from(vec![
            Span::styled(
                " [Tab] switch  [\u{2191}/\u{2193}] scroll  [Esc] close ",
                Style::default().fg(Color::DarkGray),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(title)
            .title_bottom(footer);

        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        match self.tab {
            SubagentTab::Transcript => {
                self.transcriptPanel.entries = self.transcript.clone();
                self.transcriptPanel.renderChatOnly(inner, buf);
            }
            SubagentTab::Shell => self.renderShell(inner, buf),
        }
    }

    fn renderShell(&self, area: Rect, buf: &mut Buffer) {
        if self.shellScrollback.is_empty() {
            let lines = vec![Line::from(Span::styled(
                "  No shell output yet\u{2026}",
                Style::default().fg(Color::DarkGray),
            ))];
            Paragraph::new(lines).render(area, buf);
            return;
        }

        let text = String::from_utf8_lossy(&self.shellScrollback);
        let lines: Vec<Line> = text
            .lines()
            .map(|l| Line::from(Span::styled(l.to_string(), Style::default().fg(Color::Green))))
            .collect();

        let totalLines = lines.len() as u16;
        let maxScroll = totalLines.saturating_sub(area.height);
        let scrollY = maxScroll.saturating_sub(self.shellScrollOffset.min(maxScroll));

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scrollY, 0))
            .render(area, buf);
    }
}
