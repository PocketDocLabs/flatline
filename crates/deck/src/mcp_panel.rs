#![allow(non_snake_case)]

//! MCP server status panel — interactive popup overlay.
//!
//! Renders a centered overlay showing connected MCP servers, their
//! connection state, transport info, tool counts, and tool search mode
//! status. Supports scrolling through servers and expanding to view
//! per-server tool lists with descriptions.
//!
//! # Public API
//! - [`McpPanel`] — panel state and rendering
//! - [`PanelAction`] — result of handling a key event
//!
//! # Dependencies
//! `ratatui`

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
const FG_ACCENT: Color = Color::Cyan;
const FG_BORDER: Color = Color::Magenta;


// -- Data structs -------------------------------------------------------------

struct ServerRow {
    name: String,
    state: String,
    error: Option<String>,
    toolCount: usize,
    tools: Vec<ToolRow>,
    transport: String,
}

struct ToolRow {
    qualifiedName: String,
    description: String,
}

// -- Public API ---------------------------------------------------------------

/// Result of handling a key event.
pub enum PanelAction {
    /// Key consumed, no state change.
    None,
    /// Close the panel.
    Close,
}

/// Interactive MCP server status panel.
pub struct McpPanel {
    servers: Vec<ServerRow>,
    selected: usize,
    scrollOffset: usize,
    expanded: bool,
    toolScroll: usize,
    totalTools: usize,
    connectedCount: usize,
    searchMode: bool,
    configPath: String,
    lastVisibleCount: usize,
}

impl McpPanel {
    /// Create a new panel from server status data.
    ///
    /// Args:
    ///     servers: Vec of (name, state, toolCount, tools, transport) tuples.
    ///     totalTools: Total tool count across all servers.
    ///     searchMode: Whether tool search mode is active.
    ///     configPath: Resolved path to the config file.
    pub fn new(
        servers: Vec<(String, String, usize, Vec<(String, String)>, String)>,
        totalTools: usize,
        searchMode: bool,
        configPath: String,
    ) -> Self {
        let mut connectedCount = 0;

        let serverRows: Vec<ServerRow> = servers
            .into_iter()
            .map(|(name, state, toolCount, tools, transport)| {
                // Parse error from Failed("...") Debug format.
                let (displayState, error) = if state.starts_with("Failed(") {
                    let msg = state
                        .strip_prefix("Failed(\"")
                        .and_then(|s| s.strip_suffix("\")"))
                        .unwrap_or("unknown error")
                        .to_string();
                    ("failed".to_string(), Some(msg))
                } else {
                    let lower = match state.as_str() {
                        "Connected" => {
                            connectedCount += 1;
                            "connected"
                        }
                        "Connecting" => "connecting",
                        "Disconnected" => "disconnected",
                        "ShuttingDown" => "shutting down",
                        _ => "unknown",
                    };
                    (lower.to_string(), None)
                };

                ServerRow {
                    name,
                    state: displayState,
                    error,
                    toolCount,
                    tools: tools
                        .into_iter()
                        .map(|(qn, desc)| ToolRow {
                            qualifiedName: qn,
                            description: desc,
                        })
                        .collect(),
                    transport,
                }
            })
            .collect();

        Self {
            servers: serverRows,
            selected: 0,
            scrollOffset: 0,
            expanded: false,
            toolScroll: 0,
            totalTools,
            connectedCount,
            searchMode,
            configPath,
            lastVisibleCount: 5,
        }
    }

    /// Handle a key event.
    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.expanded {
                    // Scroll tool list up.
                    self.toolScroll = self.toolScroll.saturating_sub(1);
                } else if self.selected > 0 {
                    self.selected -= 1;
                    self.adjustScroll();
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.expanded {
                    // Scroll tool list down.
                    let maxTools = self.selectedToolCount();
                    if self.toolScroll + 1 < maxTools {
                        self.toolScroll += 1;
                    }
                } else if self.selected + 1 < self.servers.len() {
                    self.selected += 1;
                    self.adjustScroll();
                }
                PanelAction::None
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.expanded {
                    self.expanded = false;
                    self.toolScroll = 0;
                } else if !self.servers.is_empty()
                    && self.servers[self.selected].toolCount > 0
                {
                    self.expanded = true;
                    self.toolScroll = 0;
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

        // Clear background.
        let bgStyle = Style::default().bg(BG).fg(FG_PRIMARY);
        fillRect(buf, popupArea, bgStyle);

        // Border.
        let borderStyle = Style::default().fg(FG_BORDER).bg(BG);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" MCP Servers ");
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 3 || inner.width < 10 {
            return;
        }

        let w = inner.width as usize;
        let mut y = inner.y;

        if self.servers.is_empty() {
            self.renderEmpty(buf, inner, w, &mut y);
        } else {
            self.renderHeader(buf, inner.x, w, &mut y);
            if self.expanded {
                self.renderToolDetail(buf, inner, w, &mut y);
            } else {
                self.renderServerList(buf, inner, w, &mut y);
            }
        }

        self.renderFooter(buf, popupArea, inner, w);
    }

    // -- Render sections ------------------------------------------------------

    fn renderEmpty(&self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        let dim = style(FG_DIM, BG);
        let accent = style(FG_ACCENT, BG);

        line(buf, inner.x + 1, *y, w - 2, "No MCP servers configured.", dim);
        *y += 2;

        let hint = format!("Add servers to {}:", self.configPath);
        line(buf, inner.x + 1, *y, w - 2, &truncateStr(&hint, w - 2), dim);
        *y += 2;
        line(buf, inner.x + 3, *y, w - 4, "[mcp.github]", accent);
        *y += 1;
        line(buf, inner.x + 3, *y, w - 4, "command = \"npx\"", accent);
        *y += 1;
        line(
            buf,
            inner.x + 3,
            *y,
            w - 4,
            "args = [\"-y\", \"@modelcontextprotocol/server-github\"]",
            accent,
        );
    }

    fn renderHeader(&self, buf: &mut Buffer, x: u16, w: usize, y: &mut u16) {
        let serverWord = if self.servers.len() == 1 {
            "server"
        } else {
            "servers"
        };
        let toolWord = if self.totalTools == 1 {
            "tool"
        } else {
            "tools"
        };

        let mut header = format!(
            " {} {}  \u{00B7}  {} connected  \u{00B7}  {} {}",
            self.servers.len(),
            serverWord,
            self.connectedCount,
            self.totalTools,
            toolWord,
        );

        if self.searchMode {
            header.push_str("  \u{00B7}  \u{26A0}\u{FE0E} search mode");
        }

        line(buf, x, *y, w, &truncateStr(&header, w), style(FG_DIM, BG));
        *y += 1;

        // Separator.
        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, x + 1, *y, w - 2, &sep, style(FG_MUTED, BG));
        *y += 1;
    }

    fn renderServerList(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        // Reserve 3 lines for footer (separator + config + hints).
        let footerReserve = 3u16;
        let available = (inner.y + inner.height).saturating_sub(*y + footerReserve);

        // Each server row = 2 lines (name + transport) + 1 for error if failed.
        let visibleCount = self.computeVisibleCount(available);
        self.lastVisibleCount = visibleCount;

        for visIdx in 0..visibleCount {
            let listIdx = self.scrollOffset + visIdx;
            if listIdx >= self.servers.len() {
                break;
            }
            let server = &self.servers[listIdx];
            let sel = listIdx == self.selected;
            let bg = if sel { BG_SELECTED } else { BG };

            // Calculate row height for background fill.
            let rowH: u16 = if server.error.is_some() { 3 } else { 2 };
            if sel {
                fillRect(
                    buf,
                    Rect {
                        x: inner.x,
                        y: *y,
                        width: inner.width,
                        height: rowH,
                    },
                    style(FG_PRIMARY, bg),
                );
            }

            // Line 1: marker + state icon + name + right-aligned tool count.
            let marker = if sel { "\u{25B8} " } else { "  " };
            let (icon, iconColor) = stateIndicator(&server.state);
            let toolLabel = format!("{} {}", server.toolCount, if server.toolCount == 1 { "tool" } else { "tools" });
            let nameSection = format!("{marker}{icon} {}", server.name);

            // Render name on left, tool count on right.
            line(buf, inner.x, *y, w, &nameSection, style(FG_PRIMARY, bg));

            // Overlay icon color.
            let iconX = inner.x + marker.len() as u16;
            if let Some(cell) = buf.cell_mut((iconX, *y)) {
                cell.set_fg(iconColor);
            }

            // Right-aligned tool count.
            let rightX = inner.x + inner.width.saturating_sub(toolLabel.len() as u16 + 1);
            line(
                buf,
                rightX,
                *y,
                toolLabel.len() + 1,
                &toolLabel,
                style(FG_DIM, bg),
            );

            // Line 2: transport.
            let transportText = format!("    {}", server.transport);
            line(
                buf,
                inner.x,
                *y + 1,
                w,
                &truncateStr(&transportText, w),
                style(FG_DIM, bg),
            );

            // Line 3 (failed only): error message.
            if let Some(ref err) = server.error {
                let errText = format!("    \u{2717}\u{FE0E} {err}");
                line(
                    buf,
                    inner.x,
                    *y + 2,
                    w,
                    &truncateStr(&errText, w),
                    style(Color::Red, bg),
                );
            }

            *y += rowH;
        }

        // Scroll indicator.
        if self.servers.len() > visibleCount {
            let remaining = self.servers.len() - self.scrollOffset - visibleCount;
            if remaining > 0 {
                let more = format!("  \u{22EE} {remaining} more");
                line(buf, inner.x, *y, w, &more, style(FG_MUTED, BG));
            }
        }
    }

    fn renderToolDetail(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        if self.selected >= self.servers.len() {
            return;
        }
        let server = &self.servers[self.selected];

        // Header: server name.
        let header = format!(
            " \u{25C6} {}  \u{2014}  {} tools",
            server.name, server.toolCount
        );
        line(buf, inner.x, *y, w, &truncateStr(&header, w), style(FG_PRIMARY, BG));
        *y += 1;

        // Separator.
        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, inner.x + 1, *y, w - 2, &sep, style(FG_MUTED, BG));
        *y += 1;

        // Reserve footer space.
        let footerReserve = 3u16;
        let maxY = (inner.y + inner.height).saturating_sub(footerReserve);
        let available = maxY.saturating_sub(*y) as usize;

        if server.tools.is_empty() {
            line(buf, inner.x + 2, *y, w - 2, "No tools registered.", style(FG_DIM, BG));
            return;
        }

        // Clamp tool scroll.
        let maxScroll = server.tools.len().saturating_sub(available);
        if self.toolScroll > maxScroll {
            self.toolScroll = maxScroll;
        }

        // Determine column widths: name column fits the longest visible tool name.
        let visibleTools = &server.tools[self.toolScroll..];
        let nameColWidth = visibleTools
            .iter()
            .take(available)
            .map(|t| stripPrefix(&t.qualifiedName).len())
            .max()
            .unwrap_or(10)
            .min(w / 3);

        for (i, tool) in visibleTools.iter().enumerate() {
            if i >= available {
                break;
            }

            let shortName = stripPrefix(&tool.qualifiedName);
            let padded = format!("{:width$}", shortName, width = nameColWidth);
            let desc = if tool.description.is_empty() {
                String::new()
            } else {
                let maxDesc = w.saturating_sub(nameColWidth + 6);
                format!("  {}", truncateStr(&tool.description, maxDesc))
            };
            let toolLine = format!("  {padded}{desc}");

            line(
                buf,
                inner.x,
                *y,
                w,
                &truncateStr(&toolLine, w),
                style(FG_ACCENT, BG),
            );

            // Description in dimmer color — overlay.
            if !tool.description.is_empty() {
                let descStart = inner.x + 2 + nameColWidth as u16 + 2;
                let descEnd = inner.x + inner.width;
                for col in descStart..descEnd {
                    if let Some(cell) = buf.cell_mut((col, *y)) {
                        cell.set_fg(FG_DIM);
                    }
                }
            }

            *y += 1;
        }

        // Scroll hint.
        let totalVisible = visibleTools.len().min(available);
        if self.toolScroll > 0 || totalVisible < server.tools.len() - self.toolScroll {
            let shown = self.toolScroll + totalVisible;
            let hint = format!(
                "  {shown}/{} \u{2014} \u{2191}\u{2193} to scroll",
                server.tools.len()
            );
            if *y < maxY {
                line(buf, inner.x, *y, w, &hint, style(FG_MUTED, BG));
            }
        }
    }

    fn renderFooter(&self, buf: &mut Buffer, popup: Rect, inner: Rect, w: usize) {
        let footerY = popup.y + popup.height - 4;
        let sepY = footerY;
        let configY = footerY + 1;
        let hintsY = footerY + 2;

        // Separator line.
        let sep: String = "\u{2500}".repeat(w.saturating_sub(2));
        line(buf, inner.x + 1, sepY, w - 2, &sep, style(FG_MUTED, BG));

        // Config path.
        let configLine = format!(" \u{2302} {}", self.configPath);
        line(
            buf,
            inner.x,
            configY,
            w,
            &truncateStr(&configLine, w),
            style(FG_MUTED, BG),
        );

        // Keybind hints.
        let hints = if self.servers.is_empty() {
            "Esc: close".to_string()
        } else if self.expanded {
            "\u{2191}\u{2193}: scroll  Enter: back  Esc: close".to_string()
        } else {
            "\u{2191}\u{2193}: select  Enter: tools  Esc: close".to_string()
        };
        line(
            buf,
            inner.x + 1,
            hintsY,
            w - 1,
            &truncateStr(&hints, w - 1),
            style(FG_MUTED, BG),
        );
    }

    // -- Helpers --------------------------------------------------------------

    fn computeVisibleCount(&self, available: u16) -> usize {
        let mut count = 0;
        let mut used = 0u16;
        for i in self.scrollOffset..self.servers.len() {
            let rowH: u16 = if self.servers[i].error.is_some() { 3 } else { 2 };
            if used + rowH > available {
                break;
            }
            used += rowH;
            count += 1;
        }
        count.max(1)
    }

    fn selectedToolCount(&self) -> usize {
        if self.selected < self.servers.len() {
            self.servers[self.selected].tools.len()
        } else {
            0
        }
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

/// Render a single line of text.
fn line(buf: &mut Buffer, x: u16, y: u16, maxWidth: usize, text: &str, style: Style) {
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

/// Fill a rect with a style.
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

/// Build a style from fg/bg.
fn style(fg: Color, bg: Color) -> Style {
    Style::default().fg(fg).bg(bg)
}

/// State indicator icon and color.
fn stateIndicator(state: &str) -> (&'static str, Color) {
    match state {
        "connected" => ("\u{25CF}", Color::Green),
        "connecting" => ("\u{25CB}", Color::Yellow),
        "disconnected" => ("\u{25CC}", Color::DarkGray),
        _ => ("\u{2717}\u{FE0E}", Color::Red),
    }
}

/// Strip the `mcp__serverName__` prefix from a qualified tool name.
fn stripPrefix(qualifiedName: &str) -> &str {
    // Format: mcp__{server}__{tool}
    if let Some(rest) = qualifiedName.strip_prefix("mcp__") {
        if let Some(pos) = rest.find("__") {
            return &rest[pos + 2..];
        }
    }
    qualifiedName
}

/// Truncate a string to fit within maxChars (char-aware).
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
