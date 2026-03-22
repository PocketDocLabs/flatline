#![allow(non_snake_case)]

//! Permissions panel — interactive popup overlay for viewing and managing permission rules.
//!
//! Displays the current effective permission rules with their allow/deny state.
//! Supports deleting and toggling rules when the permissions source is project or local.
//!
//! # Public API
//! - [`PermissionsPanel`] — panel state and rendering
//! - [`PermPanelAction`] — result of handling a key event
//!
//! # Dependencies
//! `ratatui`, `construct`

use std::collections::HashSet;

use construct::permissions::{PermissionsSource, PermitMode, Rule};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Widget},
};

// -- Palette ------------------------------------------------------------------

const BG: Color = Color::Rgb(15, 15, 25);
const BG_SELECTED: Color = Color::Rgb(40, 40, 80);
const FG_PRIMARY: Color = Color::White;
const FG_DIM: Color = Color::Rgb(100, 100, 120);
const FG_MUTED: Color = Color::Rgb(70, 70, 90);
const FG_BORDER: Color = Color::Magenta;

// -- Public API ---------------------------------------------------------------

/// Result of handling a key event.
pub enum PermPanelAction {
    /// Key consumed, no state change.
    None,
    /// Close the panel.
    Close,
    /// Save modified rules to project config.
    Save {
        defaultMode: PermitMode,
        rules: Vec<Rule>,
    },
}

/// Interactive permissions panel.
pub struct PermissionsPanel {
    rules: Vec<Rule>,
    defaultMode: PermitMode,
    source: PermissionsSource,
    configPath: String,
    selected: usize,
    scrollOffset: usize,
    lastVisibleCount: usize,
    /// Indices of rules marked for deletion.
    pendingDeletes: HashSet<usize>,
    /// Indices of rules with toggled allow/deny.
    pendingToggles: HashSet<usize>,
}

impl PermissionsPanel {
    /// Create a new panel from permission status data.
    pub fn new(
        defaultMode: PermitMode,
        rules: Vec<Rule>,
        source: PermissionsSource,
        configPath: String,
    ) -> Self {
        Self {
            rules,
            defaultMode,
            source,
            configPath,
            selected: 0,
            scrollOffset: 0,
            lastVisibleCount: 5,
            pendingDeletes: HashSet::new(),
            pendingToggles: HashSet::new(),
        }
    }

    fn editable(&self) -> bool {
        matches!(self.source, PermissionsSource::Project | PermissionsSource::Local)
    }

    fn dirty(&self) -> bool {
        !self.pendingDeletes.is_empty() || !self.pendingToggles.is_empty()
    }

    /// Handle a key event.
    pub fn handleKey(&mut self, key: KeyEvent) -> PermPanelAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PermPanelAction::Close,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.adjustScroll();
                }
                PermPanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.rules.is_empty() && self.selected + 1 < self.rules.len() {
                    self.selected += 1;
                    self.adjustScroll();
                }
                PermPanelAction::None
            }
            KeyCode::Char('x') if self.editable() => {
                if !self.rules.is_empty() {
                    if self.pendingDeletes.contains(&self.selected) {
                        self.pendingDeletes.remove(&self.selected);
                    } else {
                        self.pendingDeletes.insert(self.selected);
                    }
                }
                PermPanelAction::None
            }
            KeyCode::Char('t') if self.editable() => {
                if !self.rules.is_empty() {
                    if self.pendingToggles.contains(&self.selected) {
                        self.pendingToggles.remove(&self.selected);
                    } else {
                        self.pendingToggles.insert(self.selected);
                    }
                }
                PermPanelAction::None
            }
            KeyCode::Char('s') if self.editable() && self.dirty() => {
                // Build the final rule list with deletes and toggles applied.
                let rules: Vec<Rule> = self
                    .rules
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !self.pendingDeletes.contains(i))
                    .map(|(i, rule)| {
                        let mut r = rule.clone();
                        if self.pendingToggles.contains(&i) {
                            r.allow = !r.allow;
                        }
                        r
                    })
                    .collect();
                PermPanelAction::Save {
                    defaultMode: self.defaultMode.clone(),
                    rules,
                }
            }
            _ => PermPanelAction::None,
        }
    }

    /// Render as a centered popup overlay.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let popupWidth = (area.width * 7 / 10).max(50).min(area.width.saturating_sub(4));
        let popupHeight = (area.height * 7 / 10).max(14).min(area.height.saturating_sub(2));
        let popupX = area.x + (area.width.saturating_sub(popupWidth)) / 2;
        let popupY = area.y + (area.height.saturating_sub(popupHeight)) / 2;

        let popupArea = Rect {
            x: popupX,
            y: popupY,
            width: popupWidth,
            height: popupHeight,
        };

        // Clear background.
        let bgStyle = Style::default().bg(BG).fg(FG_PRIMARY);
        fillRect(buf, popupArea, bgStyle);

        // Border.
        let borderStyle = Style::default().fg(FG_BORDER).bg(BG);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" Permissions ");
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 3 || inner.width < 10 {
            return;
        }

        let w = inner.width as usize;
        let mut y = inner.y;

        self.renderHeader(buf, inner.x, w, &mut y);

        if self.rules.is_empty() {
            line(buf, inner.x + 2, y, w - 2, "No rules configured.", style(FG_DIM, BG));
        } else {
            self.renderRuleList(buf, inner, w, &mut y);
        }

        self.renderFooter(buf, popupArea, inner, w);
    }

    fn renderHeader(&self, buf: &mut Buffer, x: u16, w: usize, y: &mut u16) {
        // Source line.
        let sourceLabel = match self.source {
            PermissionsSource::BuiltIn => "built-in defaults",
            PermissionsSource::User => "user config",
            PermissionsSource::Project => &self.configPath,
            PermissionsSource::Local => &self.configPath,
        };
        let sourceLine = format!(" Source: {sourceLabel}");
        line(buf, x, *y, w, &truncateStr(&sourceLine, w), style(FG_DIM, BG));
        *y += 1;

        // Default mode.
        let modeStr = match self.defaultMode {
            PermitMode::Ask => "ask",
            PermitMode::Deny => "deny",
            PermitMode::Abort => "abort",
        };
        let modeLine = format!(" Default: {modeStr}");
        line(buf, x, *y, w, &modeLine, style(FG_DIM, BG));
        *y += 1;

        // Separator.
        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, x + 1, *y, w - 2, &sep, style(FG_MUTED, BG));
        *y += 1;
    }

    fn renderRuleList(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        // Reserve 3 lines for footer.
        let footerReserve = 3u16;
        let available = (inner.y + inner.height).saturating_sub(*y + footerReserve) as usize;

        let visibleCount = available.min(self.rules.len().saturating_sub(self.scrollOffset));
        self.lastVisibleCount = visibleCount.max(1);

        // Compute column widths from visible rules.
        let toolColWidth = self
            .rules
            .iter()
            .skip(self.scrollOffset)
            .take(visibleCount)
            .map(|r| r.tool.len())
            .max()
            .unwrap_or(8)
            .min(w / 3);

        for visIdx in 0..visibleCount {
            let listIdx = self.scrollOffset + visIdx;
            if listIdx >= self.rules.len() {
                break;
            }
            let rule = &self.rules[listIdx];
            let sel = listIdx == self.selected;
            let deleted = self.pendingDeletes.contains(&listIdx);
            let toggled = self.pendingToggles.contains(&listIdx);

            let bg = if sel { BG_SELECTED } else { BG };
            if sel {
                fillRect(
                    buf,
                    Rect { x: inner.x, y: *y, width: inner.width, height: 1 },
                    style(FG_PRIMARY, bg),
                );
            }

            // Determine effective allow state (with pending toggle).
            let effectiveAllow = if toggled { !rule.allow } else { rule.allow };

            // Icon.
            let (icon, iconColor) = if deleted {
                ("\u{2298} ", Color::DarkGray) // Strikethrough circle.
            } else if effectiveAllow {
                ("\u{2713}\u{FE0E} ", Color::Green)
            } else {
                ("\u{2717}\u{FE0E} ", Color::Red)
            };

            // Marker.
            let marker = if sel { "\u{25B8} " } else { "  " };

            // Tool name (padded).
            let toolPadded = format!("{:width$}", rule.tool, width = toolColWidth);

            // Pattern.
            let pattern = rule
                .pattern
                .as_deref()
                .unwrap_or("*");

            let rowText = format!("{marker}{icon}{toolPadded}  {pattern}");
            let rowStyle = if deleted {
                style(Color::DarkGray, bg)
            } else {
                style(FG_PRIMARY, bg)
            };
            line(buf, inner.x, *y, w, &truncateStr(&rowText, w), rowStyle);

            // Overlay icon color.
            let iconX = inner.x + marker.len() as u16;
            if let Some(cell) = buf.cell_mut((iconX, *y)) {
                cell.set_fg(iconColor);
            }

            *y += 1;
        }

        // Scroll indicator.
        let totalVisible = self.scrollOffset + visibleCount;
        if totalVisible < self.rules.len() {
            let remaining = self.rules.len() - totalVisible;
            let more = format!("  \u{22EE} {remaining} more");
            line(buf, inner.x, *y, w, &more, style(FG_MUTED, BG));
        }
    }

    fn renderFooter(&self, buf: &mut Buffer, popup: Rect, inner: Rect, w: usize) {
        let footerY = popup.y + popup.height - 4;

        // Separator.
        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, inner.x + 1, footerY, w - 2, &sep, style(FG_MUTED, BG));

        // Config path.
        let configLine = format!(" \u{2302} {}", self.configPath);
        line(
            buf,
            inner.x,
            footerY + 1,
            w,
            &truncateStr(&configLine, w),
            style(FG_MUTED, BG),
        );

        // Key hints (context-sensitive).
        let hints = if self.editable() {
            if self.dirty() {
                "\u{2191}\u{2193}: select  [x] delete  [t] toggle  [s] save  Esc: close"
                    .to_string()
            } else {
                "\u{2191}\u{2193}: select  [x] delete  [t] toggle  Esc: close".to_string()
            }
        } else {
            "\u{2191}\u{2193}: navigate  Esc: close".to_string()
        };
        line(
            buf,
            inner.x + 1,
            footerY + 2,
            w - 1,
            &truncateStr(&hints, w - 1),
            style(FG_MUTED, BG),
        );
    }

    fn adjustScroll(&mut self) {
        if self.selected < self.scrollOffset {
            self.scrollOffset = self.selected;
        }
        if self.selected >= self.scrollOffset + self.lastVisibleCount {
            self.scrollOffset = self.selected.saturating_sub(self.lastVisibleCount - 1);
        }
    }
}

// -- Utility functions --------------------------------------------------------

fn line(buf: &mut Buffer, x: u16, y: u16, maxWidth: usize, text: &str, s: Style) {
    let area = Rect { x, y, width: maxWidth as u16, height: 1 };
    Paragraph::new(text.to_string()).style(s).render(area, buf);
}

fn fillRect(buf: &mut Buffer, area: Rect, s: Style) {
    for row in area.y..area.y + area.height {
        for col in area.x..area.x + area.width {
            if let Some(cell) = buf.cell_mut((col, row)) {
                cell.set_char(' ');
                cell.set_style(s);
            }
        }
    }
}

fn style(fg: Color, bg: Color) -> Style {
    Style::default().fg(fg).bg(bg)
}

fn truncateStr(s: &str, maxChars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= maxChars {
        s.to_string()
    } else if maxChars > 1 {
        let mut result: String = chars[..maxChars - 1].iter().collect();
        result.push('\u{2026}');
        result
    } else {
        chars.iter().take(maxChars).collect()
    }
}
