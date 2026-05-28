#![allow(non_snake_case)]

//! Developer log history panel.
//!
//! Keeps deck-local operational events out of the agent transcript while
//! preserving a bounded, inspectable history for debugging.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Clear, Widget},
};

// -- Palette ------------------------------------------------------------------

const BG: Color = Color::Rgb(15, 15, 25);
const BG_SELECTED: Color = Color::Rgb(40, 40, 80);
const FG_PRIMARY: Color = Color::White;
const FG_DIM: Color = Color::Rgb(120, 120, 140);
const FG_MUTED: Color = Color::Rgb(80, 80, 100);
const FG_ACCENT: Color = Color::Cyan;
const FG_BORDER: Color = Color::Magenta;
const FG_OK: Color = Color::Green;
const FG_WARN: Color = Color::Yellow;
const FG_ERR: Color = Color::Red;

// -- Public API ---------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogLevel {
    Debug,
    Info,
    Success,
    Warning,
    Error,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Success => "ok",
            Self::Warning => "warn",
            Self::Error => "error",
        }
    }

    pub fn color(self) -> Color {
        match self {
            Self::Debug => FG_MUTED,
            Self::Info => FG_ACCENT,
            Self::Success => FG_OK,
            Self::Warning => FG_WARN,
            Self::Error => FG_ERR,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LogRecord {
    pub id: u64,
    pub createdAt: Instant,
    pub level: LogLevel,
    pub source: String,
    pub title: String,
    pub detail: Option<String>,
}

pub struct DeveloperLog {
    entries: VecDeque<LogRecord>,
    capacity: usize,
    nextId: u64,
}

impl DeveloperLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
            nextId: 1,
        }
    }

    pub fn push(
        &mut self,
        level: LogLevel,
        source: impl Into<String>,
        title: impl Into<String>,
        detail: Option<String>,
    ) -> LogRecord {
        let record = LogRecord {
            id: self.nextId,
            createdAt: Instant::now(),
            level,
            source: source.into(),
            title: title.into(),
            detail,
        };
        self.nextId = self.nextId.saturating_add(1);
        self.entries.push_back(record.clone());
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
        record
    }

    pub fn snapshot(&self) -> Vec<LogRecord> {
        self.entries.iter().cloned().collect()
    }
}

pub enum PanelAction {
    None,
    Close,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Filter {
    All,
    Level(LogLevel),
}

enum Mode {
    List,
    Detail { recordId: u64, scrollOffset: usize },
}

pub struct LogPanel {
    records: Vec<LogRecord>,
    selected: usize,
    scrollOffset: usize,
    lastVisibleCount: usize,
    filter: Filter,
    mode: Mode,
}

impl LogPanel {
    pub fn new(records: Vec<LogRecord>) -> Self {
        let mut panel = Self {
            records,
            selected: 0,
            scrollOffset: 0,
            lastVisibleCount: 1,
            filter: Filter::All,
            mode: Mode::List,
        };
        panel.selected = panel.filteredCount().saturating_sub(1);
        panel.adjustScroll();
        panel
    }

    pub fn refresh(&mut self, records: Vec<LogRecord>) {
        let previousSelectedId = self.selectedRecord().map(|r| r.id);
        self.records = records;
        if let Some(id) = previousSelectedId {
            if let Some(pos) = self.filteredRecords().iter().position(|r| r.id == id) {
                self.selected = pos;
            } else {
                self.selected = self.filteredCount().saturating_sub(1);
            }
        } else {
            self.selected = self.filteredCount().saturating_sub(1);
        }
        self.adjustScroll();
    }

    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        match self.mode {
            Mode::List => self.handleListKey(key),
            Mode::Detail { .. } => self.handleDetailKey(key),
        }
    }

    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        match self.mode {
            Mode::List => self.renderList(area, buf),
            Mode::Detail { .. } => self.renderDetail(area, buf),
        }
    }

    fn handleListKey(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Up | KeyCode::Char('k') => {
                self.moveUp();
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.moveDown();
                PanelAction::None
            }
            KeyCode::PageUp => {
                let page = self.lastVisibleCount.max(1);
                self.selected = self.selected.saturating_sub(page);
                self.adjustScroll();
                PanelAction::None
            }
            KeyCode::PageDown => {
                let max = self.filteredCount().saturating_sub(1);
                let page = self.lastVisibleCount.max(1);
                self.selected = (self.selected + page).min(max);
                self.adjustScroll();
                PanelAction::None
            }
            KeyCode::Home => {
                self.selected = 0;
                self.adjustScroll();
                PanelAction::None
            }
            KeyCode::End => {
                self.selected = self.filteredCount().saturating_sub(1);
                self.adjustScroll();
                PanelAction::None
            }
            KeyCode::Enter => {
                if let Some(record) = self.selectedRecord() {
                    self.mode = Mode::Detail {
                        recordId: record.id,
                        scrollOffset: 0,
                    };
                }
                PanelAction::None
            }
            KeyCode::Char('1') => {
                self.setFilter(Filter::All);
                PanelAction::None
            }
            KeyCode::Char('2') => {
                self.setFilter(Filter::Level(LogLevel::Info));
                PanelAction::None
            }
            KeyCode::Char('3') => {
                self.setFilter(Filter::Level(LogLevel::Warning));
                PanelAction::None
            }
            KeyCode::Char('4') => {
                self.setFilter(Filter::Level(LogLevel::Error));
                PanelAction::None
            }
            KeyCode::Char('5') => {
                self.setFilter(Filter::Level(LogLevel::Debug));
                PanelAction::None
            }
            _ => PanelAction::None,
        }
    }

    fn handleDetailKey(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Esc => PanelAction::Close,
            KeyCode::Char('q') | KeyCode::Backspace => {
                self.mode = Mode::List;
                PanelAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if let Mode::Detail { scrollOffset, .. } = &mut self.mode {
                    *scrollOffset = scrollOffset.saturating_sub(1);
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Mode::Detail { scrollOffset, .. } = &mut self.mode {
                    *scrollOffset = scrollOffset.saturating_add(1);
                }
                PanelAction::None
            }
            KeyCode::PageUp => {
                if let Mode::Detail { scrollOffset, .. } = &mut self.mode {
                    *scrollOffset = scrollOffset.saturating_sub(8);
                }
                PanelAction::None
            }
            KeyCode::PageDown => {
                if let Mode::Detail { scrollOffset, .. } = &mut self.mode {
                    *scrollOffset = scrollOffset.saturating_add(8);
                }
                PanelAction::None
            }
            KeyCode::Home => {
                if let Mode::Detail { scrollOffset, .. } = &mut self.mode {
                    *scrollOffset = 0;
                }
                PanelAction::None
            }
            _ => PanelAction::None,
        }
    }

    fn setFilter(&mut self, filter: Filter) {
        self.filter = filter;
        self.selected = self.filteredCount().saturating_sub(1);
        self.adjustScroll();
    }

    fn moveUp(&mut self) {
        self.selected = self.selected.saturating_sub(1);
        self.adjustScroll();
    }

    fn moveDown(&mut self) {
        let max = self.filteredCount().saturating_sub(1);
        self.selected = (self.selected + 1).min(max);
        self.adjustScroll();
    }

    fn adjustScroll(&mut self) {
        let visible = self.lastVisibleCount.max(1);
        if self.selected < self.scrollOffset {
            self.scrollOffset = self.selected;
        } else if self.selected >= self.scrollOffset + visible {
            self.scrollOffset = self.selected.saturating_sub(visible - 1);
        }
    }

    fn selectedRecord(&self) -> Option<&LogRecord> {
        self.filteredRecords().get(self.selected).copied()
    }

    fn filteredCount(&self) -> usize {
        self.records
            .iter()
            .filter(|record| self.matchesFilter(record))
            .count()
    }

    fn filteredRecords(&self) -> Vec<&LogRecord> {
        self.records
            .iter()
            .filter(|record| self.matchesFilter(record))
            .collect()
    }

    fn matchesFilter(&self, record: &LogRecord) -> bool {
        match self.filter {
            Filter::All => true,
            Filter::Level(level) => record.level == level,
        }
    }

    fn renderList(&mut self, area: Rect, buf: &mut Buffer) {
        let popupRect = centered(area, 96, 28);
        Clear.render(popupRect, buf);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FG_BORDER))
            .style(Style::default().bg(BG).fg(FG_PRIMARY))
            .title(Span::styled(
                " /logs ",
                Style::default().fg(FG_ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popupRect);
        block.render(popupRect, buf);
        if inner.height < 5 || inner.width < 20 {
            return;
        }

        self.lastVisibleCount = inner.height.saturating_sub(4) as usize;
        let filtered = self.filteredRecords();
        let filterLabel = match self.filter {
            Filter::All => "all",
            Filter::Level(level) => level.label(),
        };
        let header = format!(
            " {} entries · filter: {} · 1 all 2 info 3 warn 4 error 5 debug ",
            self.records.len(),
            filterLabel,
        );
        line(
            buf,
            inner.x,
            inner.y,
            inner.width,
            &truncate(&header, inner.width as usize),
            style(FG_DIM, BG),
        );

        let colsY = inner.y + 1;
        line(
            buf,
            inner.x,
            colsY,
            inner.width,
            &truncate(
                " time      level  source       message",
                inner.width as usize,
            ),
            style(FG_MUTED, BG),
        );
        drawHLine(buf, inner.x, inner.y + 2, inner.width, FG_MUTED);

        let rowsArea = Rect {
            x: inner.x,
            y: inner.y + 3,
            width: inner.width,
            height: inner.height.saturating_sub(4),
        };

        if filtered.is_empty() {
            line(
                buf,
                rowsArea.x,
                rowsArea.y,
                rowsArea.width,
                "  no log entries",
                style(FG_DIM, BG),
            );
        } else {
            for (visibleRow, record) in filtered
                .iter()
                .enumerate()
                .skip(self.scrollOffset)
                .take(rowsArea.height as usize)
            {
                let y = rowsArea.y + (visibleRow - self.scrollOffset) as u16;
                let selected = visibleRow == self.selected;
                self.renderRow(buf, rowsArea.x, y, rowsArea.width, record, selected);
            }
        }

        let hint = " ↑↓/jk select  Enter details  q/Esc close ";
        line(
            buf,
            inner.x,
            inner.y + inner.height.saturating_sub(1),
            inner.width,
            &truncate(hint, inner.width as usize),
            style(FG_MUTED, BG),
        );
    }

    fn renderRow(
        &self,
        buf: &mut Buffer,
        x: u16,
        y: u16,
        width: u16,
        record: &LogRecord,
        selected: bool,
    ) {
        let bg = if selected { BG_SELECTED } else { BG };
        fillRow(buf, x, y, width, Style::default().bg(bg).fg(FG_PRIMARY));

        let age = formatAge(record.createdAt.elapsed());
        let source = truncate(&record.source, 12);
        let fixed = format!(" {:<8} {:<5} {:<12} ", age, record.level.label(), source,);
        let fixedWidth = fixed.chars().count();
        let maxTitle = (width as usize).saturating_sub(fixedWidth);
        let title = truncate(&record.title, maxTitle);

        let mut col = x;
        set(
            buf,
            col,
            y,
            &fixed,
            Style::default().fg(FG_DIM).bg(bg),
            width,
        );
        col += fixedWidth.min(width as usize) as u16;
        if col < x + width {
            set(
                buf,
                col,
                y,
                &title,
                Style::default().fg(record.level.color()).bg(bg),
                x + width - col,
            );
        }
    }

    fn renderDetail(&mut self, area: Rect, buf: &mut Buffer) {
        let popupRect = centered(area, 100, 30);
        Clear.render(popupRect, buf);

        let Mode::Detail {
            recordId,
            scrollOffset,
        } = self.mode
        else {
            return;
        };
        let Some(record) = self.records.iter().find(|r| r.id == recordId).cloned() else {
            self.mode = Mode::List;
            return;
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FG_BORDER))
            .style(Style::default().bg(BG).fg(FG_PRIMARY))
            .title(Span::styled(
                format!(" /logs › #{} ", record.id),
                Style::default().fg(FG_ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popupRect);
        block.render(popupRect, buf);
        if inner.height < 5 || inner.width < 20 {
            return;
        }

        let meta = format!(
            " {} · {} · {} ago",
            record.level.label(),
            record.source,
            formatAge(record.createdAt.elapsed()),
        );
        line(
            buf,
            inner.x,
            inner.y,
            inner.width,
            &truncate(&meta, inner.width as usize),
            Style::default().fg(record.level.color()).bg(BG),
        );
        line(
            buf,
            inner.x,
            inner.y + 1,
            inner.width,
            &truncate(&record.title, inner.width as usize),
            Style::default()
                .fg(FG_PRIMARY)
                .bg(BG)
                .add_modifier(Modifier::BOLD),
        );
        drawHLine(buf, inner.x, inner.y + 2, inner.width, FG_MUTED);

        let body = record
            .detail
            .as_deref()
            .filter(|d| !d.trim().is_empty())
            .unwrap_or("(no detail)");
        let wrapped = wrapText(body, inner.width as usize);
        let rowsArea = Rect {
            x: inner.x,
            y: inner.y + 3,
            width: inner.width,
            height: inner.height.saturating_sub(4),
        };
        let maxScroll = wrapped.len().saturating_sub(rowsArea.height as usize);
        let clampedScroll = scrollOffset.min(maxScroll);
        if let Mode::Detail { scrollOffset, .. } = &mut self.mode {
            *scrollOffset = clampedScroll;
        }

        for (rowIdx, text) in wrapped
            .iter()
            .skip(clampedScroll)
            .take(rowsArea.height as usize)
            .enumerate()
        {
            line(
                buf,
                rowsArea.x,
                rowsArea.y + rowIdx as u16,
                rowsArea.width,
                text,
                style(FG_PRIMARY, BG),
            );
        }

        let hint = " ↑↓/jk scroll  Backspace/q list  Esc close ";
        line(
            buf,
            inner.x,
            inner.y + inner.height.saturating_sub(1),
            inner.width,
            &truncate(hint, inner.width as usize),
            style(FG_MUTED, BG),
        );
    }
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width.saturating_sub(4)).max(1);
    let h = height.min(area.height.saturating_sub(2)).max(1);
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

fn fillRow(buf: &mut Buffer, x: u16, y: u16, width: u16, style: Style) {
    for col in x..x + width {
        if let Some(cell) = buf.cell_mut((col, y)) {
            cell.set_char(' ');
            cell.set_style(style);
        }
    }
}

fn drawHLine(buf: &mut Buffer, x: u16, y: u16, width: u16, color: Color) {
    for col in x..x + width {
        set(buf, col, y, "─", Style::default().fg(color).bg(BG), 1);
    }
}

fn line(buf: &mut Buffer, x: u16, y: u16, width: u16, text: &str, style: Style) {
    fillRow(buf, x, y, width, style);
    set(buf, x, y, text, style, width);
}

fn set(buf: &mut Buffer, x: u16, y: u16, text: &str, style: Style, width: u16) {
    if width == 0 {
        return;
    }
    buf.set_span(x, y, &Span::styled(text.to_string(), style), width);
}

fn style(fg: Color, bg: Color) -> Style {
    Style::default().fg(fg).bg(bg)
}

fn truncate(s: &str, maxChars: usize) -> String {
    if maxChars == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count <= maxChars {
        return s.to_string();
    }
    if maxChars == 1 {
        return "…".to_string();
    }
    let mut out: String = s.chars().take(maxChars - 1).collect();
    out.push('…');
    out
}

fn wrapText(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for rawLine in text.lines() {
        if rawLine.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in rawLine.split_whitespace() {
            let nextLen = if current.is_empty() {
                word.chars().count()
            } else {
                current.chars().count() + 1 + word.chars().count()
            };
            if nextLen > width && !current.is_empty() {
                out.push(current);
                current = String::new();
            }
            if word.chars().count() > width {
                if !current.is_empty() {
                    out.push(current);
                }
                let mut chunk = String::new();
                for ch in word.chars() {
                    if chunk.chars().count() >= width {
                        out.push(chunk);
                        chunk = String::new();
                    }
                    chunk.push(ch);
                }
                current = chunk;
            } else {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
            }
        }
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn formatAge(d: Duration) -> String {
    let s = d.as_secs();
    if s < 2 {
        "now".to_string()
    } else if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m", s / 60)
    } else {
        format!("{}h", s / 3600)
    }
}
