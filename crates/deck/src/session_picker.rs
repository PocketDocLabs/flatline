#![allow(non_snake_case)]

//! Session resume picker — interactive popup for selecting a saved session.
//!
//! Renders a centered overlay with a searchable, scrollable list of
//! saved sessions. Each row shows the session name, relative time,
//! and topic range.
//!
//! # Public API
//! - [`SessionPicker`] — picker state and rendering
//! - [`PickerAction`] — result of handling a key event
//!
//! # Dependencies
//! `construct::transcript`, `ratatui`

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
};

use construct::transcript;

/// Display-ready session data for one picker row.
struct SessionRow {
    sessionId: String,
    /// Session name or first user message (truncated).
    name: String,
    /// Relative time string ("2h ago", "yesterday", etc).
    relativeTime: String,
    /// Topic range ("auth \u{2192} routing") or single topic or empty.
    topicRange: String,
    /// Project directory path.
    projectDir: String,
}

/// Result of handling a key event in the picker.
pub enum PickerAction {
    /// Key consumed, no state change.
    None,
    /// Close the picker without resuming.
    Close,
    /// Resume the selected session. Picker stays open until confirmation.
    Select(String),
}

/// Whether the picker is waiting for a resume result.
enum PickerState {
    Browsing,
    Pending,
}

/// Interactive session resume picker.
pub struct SessionPicker {
    rows: Vec<SessionRow>,
    filteredIndices: Vec<usize>,
    selected: usize,
    scrollOffset: usize,
    search: String,
    allProjects: bool,
    error: Option<String>,
    projectDir: String,
    state: PickerState,
    /// Cached from the last render pass for scroll math.
    lastVisibleCount: usize,
}

impl SessionPicker {
    /// Load sessions and build the picker.
    pub fn new(projectDir: &str) -> Self {
        let sessions = transcript::listSessions(Some(projectDir)).unwrap_or_default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let rows: Vec<SessionRow> = sessions
            .into_iter()
            .map(|meta| {
                let name = meta
                    .name
                    .unwrap_or_else(|| truncate(&meta.sessionId, 40).to_string());
                let topicRange = formatTopicRange(&meta.topicLabels);

                SessionRow {
                    sessionId: meta.sessionId,
                    name,
                    relativeTime: formatRelativeTime(meta.updatedAt, now),
                    topicRange,
                    projectDir: meta.projectDir,
                }
            })
            .collect();

        let filteredIndices: Vec<usize> = (0..rows.len()).collect();

        Self {
            rows,
            filteredIndices,
            selected: 0,
            scrollOffset: 0,
            search: String::new(),
            allProjects: false,
            error: None,
            projectDir: projectDir.to_string(),
            state: PickerState::Browsing,
            lastVisibleCount: 5,
        }
    }

    /// Handle a key event. Returns the resulting action.
    pub fn handleKey(&mut self, key: KeyEvent) -> PickerAction {
        // While waiting for resume result, only allow Esc.
        if matches!(self.state, PickerState::Pending) {
            return match key.code {
                KeyCode::Esc => PickerAction::Close,
                _ => PickerAction::None,
            };
        }

        // Clear error on any keypress.
        self.error = None;

        match key.code {
            KeyCode::Esc => PickerAction::Close,
            KeyCode::Enter => {
                if let Some(&idx) = self.filteredIndices.get(self.selected) {
                    self.state = PickerState::Pending;
                    let id = self.rows[idx].sessionId.clone();
                    PickerAction::Select(id)
                } else {
                    PickerAction::None
                }
            }
            KeyCode::Up => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.adjustScroll();
                }
                PickerAction::None
            }
            KeyCode::Down => {
                if self.selected + 1 < self.filteredIndices.len() {
                    self.selected += 1;
                    self.adjustScroll();
                }
                PickerAction::None
            }
            KeyCode::Char('a') | KeyCode::Char('A')
                if self.search.is_empty() =>
            {
                self.allProjects = !self.allProjects;
                self.reload();
                PickerAction::None
            }
            KeyCode::Char(c) => {
                self.search.push(c);
                self.applyFilter();
                PickerAction::None
            }
            KeyCode::Backspace => {
                self.search.pop();
                self.applyFilter();
                PickerAction::None
            }
            _ => PickerAction::None,
        }
    }

    /// Resume failed — show error and return to browsing.
    pub fn resumeFailed(&mut self, msg: String) {
        self.error = Some(msg);
        self.state = PickerState::Browsing;
    }

    /// Render the picker as a centered popup overlay.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        // Size: ~70% of terminal area, clamped.
        let popupWidth = (area.width * 7 / 10).max(40).min(area.width.saturating_sub(4));
        let popupHeight = (area.height * 7 / 10).max(10).min(area.height.saturating_sub(2));
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
        let modeLabel = if self.allProjects {
            "all projects"
        } else {
            "this project"
        };
        let title = format!(" Resume Session ({modeLabel}) ");
        let borderStyle = Style::default().fg(Color::Cyan).bg(Color::Rgb(15, 15, 25));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(title);
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 3 || inner.width < 10 {
            return;
        }

        let mut y = inner.y;
        let contentWidth = inner.width as usize;

        // Search bar.
        let searchDisplay = if self.search.is_empty() {
            "\u{25B8} Search...".to_string()
        } else {
            format!("\u{25B8} {}", self.search)
        };
        let searchStyle = if self.search.is_empty() {
            Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25))
        } else {
            Style::default().fg(Color::White).bg(Color::Rgb(15, 15, 25))
        };
        renderLine(buf, inner.x, y, contentWidth, &searchDisplay, searchStyle);
        y += 1;

        // Blank separator.
        y += 1;

        // Rows per entry: 2 lines (name + metadata), or 3 in allProjects mode.
        let rowHeight: u16 = if self.allProjects { 3 } else { 2 };
        let availableRows = (inner.y + inner.height).saturating_sub(y + 2); // -2 for footer
        let visibleCount = (availableRows / rowHeight).max(1) as usize;
        self.lastVisibleCount = visibleCount;

        if self.filteredIndices.is_empty() {
            let emptyMsg = if self.rows.is_empty() {
                "No sessions found"
            } else {
                "No matches"
            };
            let emptyStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25));
            renderLine(buf, inner.x + 1, y, contentWidth - 1, emptyMsg, emptyStyle);
        } else {
            for visIdx in 0..visibleCount {
                let listIdx = self.scrollOffset + visIdx;
                if listIdx >= self.filteredIndices.len() {
                    break;
                }
                let rowIdx = self.filteredIndices[listIdx];
                let row = &self.rows[rowIdx];
                let isSelected = listIdx == self.selected;

                let nameStyle = if isSelected {
                    Style::default().fg(Color::White).bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default().fg(Color::White).bg(Color::Rgb(15, 15, 25))
                };
                let metaStyle = if isSelected {
                    Style::default().fg(Color::DarkGray).bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25))
                };

                // Clear row background for selected item.
                if isSelected {
                    for line in 0..rowHeight {
                        let ry = y + line;
                        for col in inner.x..inner.x + inner.width {
                            if let Some(cell) = buf.cell_mut((col, ry)) {
                                cell.set_char(' ');
                                cell.set_style(nameStyle);
                            }
                        }
                    }
                }

                // Line 1: marker + session name.
                let marker = if isSelected { "\u{25B8} " } else { "  " };
                let nameText = format!("{marker}{}", truncate(&row.name, contentWidth - 3));
                renderLine(buf, inner.x, y, contentWidth, &nameText, nameStyle);

                // Line 2: relative time + topic range.
                let metaText = if row.topicRange.is_empty() {
                    format!("  {}", row.relativeTime)
                } else {
                    format!("  {}  {}", row.relativeTime, row.topicRange)
                };
                renderLine(
                    buf,
                    inner.x,
                    y + 1,
                    contentWidth,
                    &truncate(&metaText, contentWidth),
                    metaStyle,
                );

                // Line 3 (allProjects only): project path.
                if self.allProjects {
                    let pathText = format!("  {}", row.projectDir);
                    renderLine(
                        buf,
                        inner.x,
                        y + 2,
                        contentWidth,
                        &truncate(&pathText, contentWidth),
                        metaStyle,
                    );
                }

                y += rowHeight;
            }
        }

        // Footer: keybind hints + optional error.
        let footerY = popupArea.y + popupArea.height - 2;
        let footerStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25));
        let errorStyle = Style::default().fg(Color::Red).bg(Color::Rgb(15, 15, 25));

        if let Some(ref err) = self.error {
            renderLine(buf, inner.x, footerY, contentWidth, &truncate(err, contentWidth), errorStyle);
        } else if matches!(self.state, PickerState::Pending) {
            let pendingStyle = Style::default().fg(Color::Yellow).bg(Color::Rgb(15, 15, 25));
            renderLine(buf, inner.x, footerY, contentWidth, "Resuming\u{2026}", pendingStyle);
        }

        let hints = if matches!(self.state, PickerState::Pending) {
            "Esc: cancel".to_string()
        } else {
            let toggleKey = if self.allProjects { "A: this project" } else { "A: all projects" };
            format!("{toggleKey}  Enter: resume  Esc: close")
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

    /// Rebuild filtered indices from search text.
    fn applyFilter(&mut self) {
        let query = self.search.to_lowercase();
        self.filteredIndices = self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                if query.is_empty() {
                    return true;
                }
                let haystack = format!("{} {}", row.name, row.topicRange).to_lowercase();
                haystack.contains(&query)
            })
            .map(|(i, _)| i)
            .collect();
        self.selected = 0;
        self.scrollOffset = 0;
    }

    /// Reload sessions (after toggling allProjects).
    fn reload(&mut self) {
        let filterDir = if self.allProjects {
            None
        } else {
            Some(self.projectDir.as_str())
        };
        let sessions = transcript::listSessions(filterDir).unwrap_or_default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.rows = sessions
            .into_iter()
            .map(|meta| {
                let name = meta
                    .name
                    .unwrap_or_else(|| truncate(&meta.sessionId, 40).to_string());
                let topicRange = formatTopicRange(&meta.topicLabels);

                SessionRow {
                    sessionId: meta.sessionId,
                    name,
                    relativeTime: formatRelativeTime(meta.updatedAt, now),
                    topicRange,
                    projectDir: meta.projectDir,
                }
            })
            .collect();

        self.applyFilter();
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
    Paragraph::new(text.to_string()).style(style).render(area, buf);
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

/// Format topic labels into a range string.
fn formatTopicRange(labels: &[String]) -> String {
    match labels.len() {
        0 => String::new(),
        1 => labels[0].clone(),
        _ => {
            let first = &labels[0];
            let last = &labels[labels.len() - 1];
            format!("{first} \u{2192} {last}")
        }
    }
}

/// Truncate a string to fit within maxChars, adding ellipsis if needed.
fn truncate(s: &str, maxChars: usize) -> String {
    if s.chars().count() <= maxChars {
        s.to_string()
    } else if maxChars > 3 {
        let end = s.char_indices().nth(maxChars - 1).map_or(s.len(), |(i, _)| i);
        format!("{}\u{2026}", &s[..end])
    } else {
        let end = s.char_indices().nth(maxChars).map_or(s.len(), |(i, _)| i);
        s[..end].to_string()
    }
}

use ratatui::widgets::Widget;
