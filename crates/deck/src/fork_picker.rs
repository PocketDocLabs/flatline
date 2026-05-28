#![allow(non_snake_case)]

//! Fork picker — interactive popup for switching conversation forks.
//!
//! Shows saved forks with their label, relative time, and fork ID.
//! User picks one to switch to, or presses Esc to close.
//!
//! # Public API
//! - [`ForkPicker`] — picker state and rendering
//! - [`ForkAction`] — result of handling a key event
//!
//! # Dependencies
//! `construct::transcript`, `ratatui`

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Widget},
};

use construct::transcript::Fork;

/// Result of handling a key event in the picker.
pub enum ForkAction {
    /// Key consumed, no state change.
    None,
    /// Close the picker.
    Close,
    /// Switch to a fork by ID. Picker stays open in pending state.
    Switch(String),
}

/// Whether the picker is waiting for a switch result.
enum PickerState {
    Browsing,
    Pending,
}

/// Display-ready fork data for one picker row.
struct ForkRow {
    forkId: String,
    label: String,
    relativeTime: String,
}

/// Interactive fork picker.
pub struct ForkPicker {
    rows: Vec<ForkRow>,
    selected: usize,
    scrollOffset: usize,
    state: PickerState,
    error: Option<String>,
    lastVisibleCount: usize,
}

impl ForkPicker {
    /// Build the picker from saved forks.
    pub fn new(forks: &[Fork]) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let rows: Vec<ForkRow> = forks
            .iter()
            .map(|f| ForkRow {
                forkId: f.id.clone(),
                label: f.label.clone(),
                relativeTime: formatRelativeTime(f.createdAt, now),
            })
            .collect();

        Self {
            rows,
            selected: 0,
            scrollOffset: 0,
            state: PickerState::Browsing,
            error: None,
            lastVisibleCount: 5,
        }
    }

    /// Handle a key event.
    pub fn handleKey(&mut self, key: KeyEvent) -> ForkAction {
        // While waiting for switch result, only allow Esc.
        if matches!(self.state, PickerState::Pending) {
            return match key.code {
                KeyCode::Esc => ForkAction::Close,
                _ => ForkAction::None,
            };
        }

        // Clear error on any keypress.
        self.error = None;

        match key.code {
            KeyCode::Esc => ForkAction::Close,
            KeyCode::Enter => {
                if let Some(row) = self.rows.get(self.selected) {
                    self.state = PickerState::Pending;
                    ForkAction::Switch(row.forkId.clone())
                } else {
                    ForkAction::None
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.adjustScroll();
                }
                ForkAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.rows.len() {
                    self.selected += 1;
                    self.adjustScroll();
                }
                ForkAction::None
            }
            _ => ForkAction::None,
        }
    }

    /// Switch failed — show error and return to browsing.
    pub fn switchFailed(&mut self, msg: String) {
        self.error = Some(msg);
        self.state = PickerState::Browsing;
    }

    /// Render the picker as a centered popup overlay.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let popupWidth = (area.width * 7 / 10)
            .max(40)
            .min(area.width.saturating_sub(4));
        let popupHeight = (area.height * 7 / 10)
            .max(10)
            .min(area.height.saturating_sub(2));
        let popupX = area.x + (area.width.saturating_sub(popupWidth)) / 2;
        let popupY = area.y + (area.height.saturating_sub(popupHeight)) / 2;

        let popupArea = Rect {
            x: popupX,
            y: popupY,
            width: popupWidth,
            height: popupHeight,
        };

        // Clear background.
        let bgStyle = Style::default().bg(Color::Rgb(15, 15, 25)).fg(Color::White);
        for row in popupArea.y..popupArea.y + popupArea.height {
            for col in popupArea.x..popupArea.x + popupArea.width {
                if let Some(cell) = buf.cell_mut((col, row)) {
                    cell.set_char(' ');
                    cell.set_style(bgStyle);
                }
            }
        }

        // Border.
        let borderStyle = Style::default().fg(Color::Cyan).bg(Color::Rgb(15, 15, 25));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" Forks ");
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 3 || inner.width < 10 {
            return;
        }

        let contentWidth = inner.width as usize;
        let mut y = inner.y;

        // Footer takes 2 lines.
        let footerHeight: u16 = 2;
        let availableRows = inner.height.saturating_sub(footerHeight);
        let rowHeight: u16 = 2;
        let visibleCount = (availableRows / rowHeight).max(1) as usize;
        self.lastVisibleCount = visibleCount;

        if self.rows.is_empty() {
            let emptyStyle = Style::default()
                .fg(Color::DarkGray)
                .bg(Color::Rgb(15, 15, 25));
            renderLine(
                buf,
                inner.x + 1,
                y,
                contentWidth - 1,
                "No saved forks.",
                emptyStyle,
            );
        } else {
            for visIdx in 0..visibleCount {
                let listIdx = self.scrollOffset + visIdx;
                if listIdx >= self.rows.len() {
                    break;
                }
                let row = &self.rows[listIdx];
                let isSelected = listIdx == self.selected;

                let labelStyle = if isSelected {
                    Style::default().fg(Color::White).bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default().fg(Color::White).bg(Color::Rgb(15, 15, 25))
                };
                let metaStyle = if isSelected {
                    Style::default()
                        .fg(Color::DarkGray)
                        .bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default()
                        .fg(Color::DarkGray)
                        .bg(Color::Rgb(15, 15, 25))
                };

                // Clear row background for selected item.
                if isSelected {
                    for line in 0..rowHeight {
                        let ry = y + line;
                        for col in inner.x..inner.x + inner.width {
                            if let Some(cell) = buf.cell_mut((col, ry)) {
                                cell.set_char(' ');
                                cell.set_style(labelStyle);
                            }
                        }
                    }
                }

                // Line 1: marker + label.
                let marker = if isSelected { "\u{25B8} " } else { "  " };
                let labelText = format!(
                    "{marker}{}",
                    truncate(&row.label, contentWidth.saturating_sub(3))
                );
                renderLine(buf, inner.x, y, contentWidth, &labelText, labelStyle);

                // Line 2: fork ID + relative time.
                let metaText = format!("  {}  {}", row.forkId, row.relativeTime);
                renderLine(
                    buf,
                    inner.x,
                    y + 1,
                    contentWidth,
                    &truncate(&metaText, contentWidth),
                    metaStyle,
                );

                y += rowHeight;
            }
        }

        // Footer.
        let footerY = popupArea.y + popupArea.height - 2;
        let footerStyle = Style::default()
            .fg(Color::DarkGray)
            .bg(Color::Rgb(15, 15, 25));

        if let Some(ref err) = self.error {
            let errorStyle = Style::default().fg(Color::Red).bg(Color::Rgb(15, 15, 25));
            renderLine(
                buf,
                inner.x,
                footerY,
                contentWidth,
                &truncate(err, contentWidth),
                errorStyle,
            );
        } else if matches!(self.state, PickerState::Pending) {
            let pendingStyle = Style::default()
                .fg(Color::Yellow)
                .bg(Color::Rgb(15, 15, 25));
            renderLine(
                buf,
                inner.x,
                footerY,
                contentWidth,
                "Switching\u{2026}",
                pendingStyle,
            );
        }

        let hints = if matches!(self.state, PickerState::Pending) {
            "Esc: cancel".to_string()
        } else {
            "\u{2191}\u{2193}/jk: navigate  Enter: switch  Esc: close".to_string()
        };
        renderLine(
            buf,
            inner.x,
            footerY + 1,
            contentWidth,
            &truncate(&hints, contentWidth),
            footerStyle,
        );
    }

    /// Ensure the selected item is visible in the scroll window.
    fn adjustScroll(&mut self) {
        if self.selected < self.scrollOffset {
            self.scrollOffset = self.selected;
        }
        if self.selected >= self.scrollOffset + self.lastVisibleCount {
            self.scrollOffset = self.selected.saturating_sub(self.lastVisibleCount - 1);
        }
    }
}

/// Render a single line of text into the buffer, truncated to width.
fn renderLine(buf: &mut Buffer, x: u16, y: u16, maxWidth: usize, text: &str, style: Style) {
    let area = Rect {
        x,
        y,
        width: maxWidth as u16,
        height: 1,
    };
    Paragraph::new(text.to_string())
        .style(style)
        .render(area, buf);
}

/// Format a unix timestamp as a relative time string.
fn formatRelativeTime(ts: u64, now: u64) -> String {
    let delta = now.saturating_sub(ts);

    if delta < 60 {
        "just now".to_string()
    } else if delta < 3600 {
        let mins = delta / 60;
        format!("{mins}m ago")
    } else if delta < 86400 {
        let hours = delta / 3600;
        format!("{hours}h ago")
    } else if delta < 172800 {
        "yesterday".to_string()
    } else if delta < 604800 {
        let days = delta / 86400;
        format!("{days}d ago")
    } else {
        let weeks = delta / 604800;
        format!("{weeks}w ago")
    }
}

/// Truncate a string to fit within maxChars, adding ellipsis if needed.
fn truncate(s: &str, maxChars: usize) -> String {
    if s.chars().count() <= maxChars {
        s.to_string()
    } else if maxChars > 3 {
        let end = s
            .char_indices()
            .nth(maxChars - 1)
            .map_or(s.len(), |(i, _)| i);
        format!("{}\u{2026}", &s[..end])
    } else {
        let end = s.char_indices().nth(maxChars).map_or(s.len(), |(i, _)| i);
        s[..end].to_string()
    }
}
