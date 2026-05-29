#![allow(non_snake_case)]

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, StatefulWidget, Widget},
};

use construct::storage::TerminalRunRecord;

const BG: Color = Color::Rgb(32, 36, 44);
const FG: Color = Color::Rgb(225, 228, 235);
const DIM: Color = Color::Rgb(120, 125, 135);
const BORDER: Color = Color::Rgb(130, 190, 185);
const SELECTED: Color = Color::Rgb(48, 55, 68);

pub enum RunsAction {
    None,
    Close,
}

pub struct RunsPanel {
    runs: Vec<TerminalRunRecord>,
    selected: usize,
    detail: bool,
    scroll: usize,
}

impl RunsPanel {
    pub fn new(runs: Vec<TerminalRunRecord>) -> Self {
        Self {
            runs,
            selected: 0,
            detail: false,
            scroll: 0,
        }
    }

    pub fn refresh(&mut self, runs: Vec<TerminalRunRecord>) {
        self.runs = runs;
        if self.selected >= self.runs.len() {
            self.selected = self.runs.len().saturating_sub(1);
        }
    }

    pub fn handleKey(&mut self, key: KeyEvent) -> RunsAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => RunsAction::Close,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.detail {
                    self.scroll = self.scroll.saturating_sub(1);
                } else {
                    self.selected = self.selected.saturating_sub(1);
                }
                RunsAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.detail {
                    self.scroll = self.scroll.saturating_add(1);
                } else if self.selected + 1 < self.runs.len() {
                    self.selected += 1;
                }
                RunsAction::None
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.detail = !self.detail;
                self.scroll = 0;
                RunsAction::None
            }
            KeyCode::Backspace | KeyCode::Left => {
                self.detail = false;
                self.scroll = 0;
                RunsAction::None
            }
            _ => RunsAction::None,
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let popup = centerRect(area, 92, 28);
        Clear.render(popup, buf);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER))
            .style(Style::default().bg(BG))
            .title(Span::styled(
                " /runs ",
                Style::default().fg(BORDER).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popup);
        block.render(popup, buf);
        if self.detail {
            self.renderDetail(inner, buf);
        } else {
            self.renderList(inner, buf);
        }
    }

    fn renderList(&self, area: Rect, buf: &mut Buffer) {
        if self.runs.is_empty() {
            Paragraph::new("  no terminal runs archived yet")
                .style(Style::default().fg(DIM).bg(BG))
                .render(area, buf);
            return;
        }
        let height = area.height as usize;
        let start = self.selected.saturating_sub(height.saturating_sub(1));
        for (row, run) in self.runs.iter().enumerate().skip(start).take(height) {
            let y = area.y + (row - start) as u16;
            let selected = row == self.selected;
            let bg = if selected { SELECTED } else { BG };
            let (glyph, color) = crate::impact::shellImpactGlyphColor(&run.impact);
            let status = match run.status.as_str() {
                "running" => "\u{25F4}",
                "completed" => "\u{2713}",
                "failed" | "timed_out" => "\u{2717}",
                _ => "\u{00B7}",
            };
            let mut purpose = run.purpose.clone();
            let maxPurpose = area.width.saturating_sub(30) as usize;
            if purpose.chars().count() > maxPurpose {
                purpose = format!(
                    "{}\u{2026}",
                    purpose
                        .char_indices()
                        .nth(maxPurpose.saturating_sub(1))
                        .map(|(i, _)| &purpose[..i])
                        .unwrap_or(&purpose)
                );
            }
            let line = Line::from(vec![
                Span::styled(" ", Style::default().bg(bg)),
                Span::styled(glyph, Style::default().fg(color).bg(bg)),
                Span::styled(format!(" {status} "), Style::default().fg(FG).bg(bg)),
                Span::styled(purpose, Style::default().fg(FG).bg(bg)),
                Span::styled(
                    format!("  [{}]", run.terminalName),
                    Style::default().fg(DIM).bg(bg),
                ),
            ]);
            buf.set_line(area.x, y, &line, area.width);
        }
    }

    fn renderDetail(&self, area: Rect, buf: &mut Buffer) {
        let Some(run) = self.runs.get(self.selected) else {
            return;
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(area);
        let (glyph, color) = crate::impact::shellImpactGlyphColor(&run.impact);
        let header = vec![
            Line::from(vec![
                Span::styled(glyph, Style::default().fg(color).bg(BG)),
                Span::styled(format!(" {}", run.purpose), Style::default().fg(FG).bg(BG)),
            ]),
            Line::from(Span::styled(
                format!("{} · {} · {}", run.terminalName, run.status, run.command),
                Style::default().fg(DIM).bg(BG),
            )),
        ];
        Paragraph::new(header)
            .style(Style::default().bg(BG))
            .render(chunks[0], buf);
        let replayArea = chunks[1];
        let mut state = crate::terminal::TerminalState::new(replayArea.width, replayArea.height);
        state.process(&run.replayBlob);
        if self.scroll > 0 {
            state.scrollUp(self.scroll as i32);
        }
        crate::terminal::Terminal.render(replayArea, buf, &mut state);
    }
}

fn centerRect(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width.saturating_sub(4)).max(20);
    let h = height.min(area.height.saturating_sub(2)).max(8);
    Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    }
}
