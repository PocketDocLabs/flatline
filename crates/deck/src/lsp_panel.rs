#![allow(non_snake_case)]

//! LSP server status panel — interactive popup overlay.
//!
//! Shows all known LSP servers, their install status, and lets the user
//! install missing ones with Enter. Servers relevant to the current project
//! (matching files found in cwd) are sorted to the top.
//!
//! # Public API
//! - [`LspPanel`] — panel state and rendering
//! - [`PanelAction`] — result of handling a key event
//!
//! # Dependencies
//! `ratatui`, `construct::lsp`

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Widget},
};

use construct::lsp::{FullServerStatus, ServerAvailability};

// -- Palette ------------------------------------------------------------------

const BG: Color = Color::Rgb(15, 15, 25);
const BG_SELECTED: Color = Color::Rgb(40, 40, 80);
const FG_PRIMARY: Color = Color::White;
const FG_DIM: Color = Color::Rgb(100, 100, 120);
const FG_MUTED: Color = Color::Rgb(70, 70, 90);
const FG_ACCENT: Color = Color::Cyan;
const FG_BORDER: Color = Color::Rgb(100, 180, 100);
const FG_INSTALLED: Color = Color::Green;
const FG_NOT_INSTALLED: Color = Color::Rgb(180, 100, 60);

// -- Data structs -------------------------------------------------------------

struct Row {
    id: String,
    extensions: String,
    installHint: String,
    statusLabel: String,
    statusColor: Color,
    installable: bool,
    relevant: bool,
}

// -- Public API ---------------------------------------------------------------

/// Result of handling a key event.
pub enum PanelAction {
    /// Key consumed, no state change.
    None,
    /// Close the panel.
    Close,
    /// Install the selected server — run this command in the shared terminal.
    Install { serverId: String, command: String },
}

/// Interactive LSP server panel.
pub struct LspPanel {
    rows: Vec<Row>,
    selected: usize,
    scrollOffset: usize,
    lastVisibleCount: usize,
    installedCount: usize,
    notInstalledCount: usize,
}

impl LspPanel {
    /// Create a new panel from full server status data.
    pub fn new(servers: Vec<FullServerStatus>) -> Self {
        let cwd = std::env::current_dir().unwrap_or_default();
        let relevantExts = scanProjectExtensions(&cwd);

        let mut installedCount = 0;
        let mut notInstalledCount = 0;

        let mut rows: Vec<Row> = servers
            .into_iter()
            .map(|s| {
                let (statusLabel, statusColor, installable) = match &s.status {
                    ServerAvailability::Active => {
                        installedCount += 1;
                        ("\u{25CF} active".to_string(), FG_INSTALLED, false)
                    }
                    ServerAvailability::Starting => {
                        installedCount += 1;
                        ("\u{25CB} starting".to_string(), Color::Yellow, false)
                    }
                    ServerAvailability::Installed => {
                        installedCount += 1;
                        ("\u{25CF} installed".to_string(), FG_INSTALLED, false)
                    }
                    ServerAvailability::Failed(e) => {
                        installedCount += 1;
                        let msg = if e.len() > 30 {
                            format!("\u{2717}\u{FE0E} {:.30}...", e)
                        } else {
                            format!("\u{2717}\u{FE0E} {e}")
                        };
                        (msg, Color::Red, false)
                    }
                    ServerAvailability::NotInstalled => {
                        notInstalledCount += 1;
                        ("not installed".to_string(), FG_NOT_INSTALLED, true)
                    }
                };

                // Check if any of this server's extensions match files in the project.
                let relevant = s.extensions.iter().any(|ext| relevantExts.contains(ext));

                Row {
                    id: s.id,
                    extensions: s.extensions.join(" "),
                    installHint: s.installHint,
                    statusLabel,
                    statusColor,
                    installable,
                    relevant,
                }
            })
            .collect();

        // Sort: relevant first, then installed before not-installed, then alphabetical.
        rows.sort_by(|a, b| {
            b.relevant
                .cmp(&a.relevant)
                .then(a.installable.cmp(&b.installable))
                .then(a.id.cmp(&b.id))
        });

        Self {
            rows,
            selected: 0,
            scrollOffset: 0,
            lastVisibleCount: 5,
            installedCount,
            notInstalledCount,
        }
    }

    /// Handle a key event.
    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.adjustScroll();
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.rows.len() {
                    self.selected += 1;
                    self.adjustScroll();
                }
                PanelAction::None
            }
            KeyCode::Enter => {
                if self.selected < self.rows.len() {
                    let row = &self.rows[self.selected];
                    if row.installable && !row.installHint.is_empty() {
                        return PanelAction::Install {
                            serverId: row.id.clone(),
                            command: row.installHint.clone(),
                        };
                    }
                }
                PanelAction::None
            }
            _ => PanelAction::None,
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

        fillRect(buf, popupArea, style(FG_PRIMARY, BG));

        let borderStyle = Style::default().fg(FG_BORDER).bg(BG);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" LSP Servers ");
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 3 || inner.width < 10 {
            return;
        }

        let w = inner.width as usize;
        let mut y = inner.y;

        self.renderHeader(buf, inner.x, w, &mut y);
        self.renderRows(buf, inner, w, &mut y);
        self.renderFooter(buf, popupArea, inner, w);
    }

    fn renderHeader(&self, buf: &mut Buffer, x: u16, w: usize, y: &mut u16) {
        let header = format!(
            " {} installed  \u{00B7}  {} available",
            self.installedCount, self.notInstalledCount,
        );
        line(buf, x, *y, w, &truncateStr(&header, w), style(FG_DIM, BG));
        *y += 1;

        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, x + 1, *y, w - 2, &sep, style(FG_MUTED, BG));
        *y += 1;
    }

    fn renderRows(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        let footerReserve = 3u16;
        let available = (inner.y + inner.height).saturating_sub(*y + footerReserve);
        let visibleCount = (available as usize / 2).max(1).min(self.rows.len());
        self.lastVisibleCount = visibleCount;

        for visIdx in 0..visibleCount {
            let listIdx = self.scrollOffset + visIdx;
            if listIdx >= self.rows.len() {
                break;
            }
            let row = &self.rows[listIdx];
            let sel = listIdx == self.selected;
            let bg = if sel { BG_SELECTED } else { BG };

            if sel {
                fillRect(
                    buf,
                    Rect {
                        x: inner.x,
                        y: *y,
                        width: inner.width,
                        height: 2,
                    },
                    style(FG_PRIMARY, bg),
                );
            }

            // Line 1: marker + name + extensions + status (right-aligned).
            let marker = if sel { "\u{25B8} " } else { "  " };
            let relevanceTag = if row.relevant { "\u{2605}\u{FE0E}" } else { " " };
            let namePart = format!("{marker}{relevanceTag} {}", row.id);
            line(buf, inner.x, *y, w, &truncateStr(&namePart, w), style(FG_PRIMARY, bg));

            // Relevance star color.
            if row.relevant {
                let starX = inner.x + marker.len() as u16;
                if let Some(cell) = buf.cell_mut((starX, *y)) {
                    cell.set_fg(Color::Yellow);
                }
            }

            // Right-aligned status.
            let statusLen = row.statusLabel.chars().count();
            let rightX = inner.x + inner.width.saturating_sub(statusLen as u16 + 1);
            line(buf, rightX, *y, statusLen + 1, &row.statusLabel, style(row.statusColor, bg));

            // Line 2: extensions + install hint (if not installed).
            let line2 = if row.installable {
                format!("    {} \u{2014} {}", row.extensions, row.installHint)
            } else {
                format!("    {}", row.extensions)
            };
            let line2Color = if row.installable { FG_ACCENT } else { FG_DIM };
            line(buf, inner.x, *y + 1, w, &truncateStr(&line2, w), style(line2Color, bg));

            *y += 2;
        }

        // Scroll indicator.
        if self.rows.len() > visibleCount {
            let remaining = self.rows.len().saturating_sub(self.scrollOffset + visibleCount);
            if remaining > 0 {
                let more = format!("  \u{22EE} {remaining} more");
                line(buf, inner.x, *y, w, &more, style(FG_MUTED, BG));
            }
        }
    }

    fn renderFooter(&self, buf: &mut Buffer, popup: Rect, inner: Rect, w: usize) {
        let footerY = popup.y + popup.height - 3;

        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, inner.x + 1, footerY, w - 2, &sep, style(FG_MUTED, BG));

        let selectedInstallable = self
            .rows
            .get(self.selected)
            .map(|r| r.installable)
            .unwrap_or(false);

        let hints = if selectedInstallable {
            "\u{2191}\u{2193}: select  Enter: install  Esc: close"
        } else {
            "\u{2191}\u{2193}: select  Esc: close"
        };
        line(
            buf,
            inner.x + 1,
            footerY + 1,
            w - 1,
            &truncateStr(hints, w - 1),
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

// -- Project scanning ---------------------------------------------------------

/// Scan the project directory for file extensions (top-level + src/).
/// Used to determine which servers are relevant.
fn scanProjectExtensions(projectDir: &std::path::Path) -> std::collections::HashSet<String> {
    let mut exts = std::collections::HashSet::new();
    let dirs = [projectDir.to_path_buf(), projectDir.join("src")];

    for dir in &dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension() {
                    exts.insert(format!(".{}", ext.to_string_lossy()));
                }
            }
        }
    }

    // Also check for project markers.
    let markerToExt: &[(&str, &str)] = &[
        ("Cargo.toml", ".rs"),
        ("pyproject.toml", ".py"),
        ("package.json", ".ts"),
        ("go.mod", ".go"),
        ("CMakeLists.txt", ".c"),
        ("compile_commands.json", ".c"),
    ];
    for (marker, ext) in markerToExt {
        if projectDir.join(marker).exists() {
            exts.insert(ext.to_string());
        }
    }

    exts
}

// -- Utility functions --------------------------------------------------------

fn line(buf: &mut Buffer, x: u16, y: u16, maxWidth: usize, text: &str, s: Style) {
    Paragraph::new(text.to_string())
        .style(s)
        .render(
            Rect {
                x,
                y,
                width: maxWidth as u16,
                height: 1,
            },
            buf,
        );
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
