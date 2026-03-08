//! Agent panel — renders conversation, permission prompts, and input.
//!
//! Displays streamed assistant responses with markdown rendering,
//! tool request approvals, and a text input line.
//!
//! # Public API
//! - [`AgentPanel`] — panel state and rendering
//!
//! # Dependencies
//! `ratatui`

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
};

use crate::markdown;

/// A single entry in the agent panel display.
#[derive(Debug, Clone)]
pub enum PanelEntry {
    User(String),
    Assistant(String),
    Reasoning(String),
    ToolRequest { summary: String },
    ToolApproved { name: String },
    ToolDenied { name: String },
    ToolResult { name: String, output: String },
    Error(String),
}

/// Agent panel state.
pub struct AgentPanel {
    entries: Vec<PanelEntry>,
    streamingContent: String,
    streamingReasoning: String,
    isStreaming: bool,
    pub inputBuf: String,
    pub pendingPermit: bool,
    pendingToolName: String,
    /// Scroll offset from the bottom (in visual lines).
    scrollOffset: u16,
    /// ScrollY value from the last render (for visual-line lookups).
    lastScrollY: u16,
    /// Chat area width from the last render (for wrap estimation).
    lastChatWidth: u16,
}

impl AgentPanel {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            streamingContent: String::new(),
            streamingReasoning: String::new(),
            isStreaming: false,
            inputBuf: String::new(),
            pendingPermit: false,
            pendingToolName: String::new(),
            scrollOffset: 0,
            lastScrollY: 0,
            lastChatWidth: 0,
        }
    }

    pub fn pushUser(&mut self, text: &str) {
        self.entries.push(PanelEntry::User(text.into()));
        self.scrollOffset = 0;
    }

    pub fn appendContent(&mut self, text: &str) {
        self.isStreaming = true;
        self.streamingContent.push_str(text);
        // Pin to bottom on new content.
        self.scrollOffset = 0;
    }

    pub fn appendReasoning(&mut self, text: &str) {
        self.isStreaming = true;
        self.streamingReasoning.push_str(text);
        self.scrollOffset = 0;
    }

    pub fn finalizeStreaming(&mut self) {
        if !self.streamingReasoning.is_empty() {
            self.entries.push(PanelEntry::Reasoning(
                std::mem::take(&mut self.streamingReasoning),
            ));
        }
        if !self.streamingContent.is_empty() {
            self.entries.push(PanelEntry::Assistant(
                std::mem::take(&mut self.streamingContent),
            ));
        }
        self.isStreaming = false;
    }

    pub fn showToolRequest(&mut self, name: &str, summary: &str) {
        self.finalizeStreaming();
        self.entries
            .push(PanelEntry::ToolRequest { summary: summary.into() });
        self.pendingPermit = true;
        self.pendingToolName = name.into();
        self.scrollOffset = 0;
    }

    pub fn approvePending(&mut self) {
        self.pendingPermit = false;
        let name = std::mem::take(&mut self.pendingToolName);
        self.entries.push(PanelEntry::ToolApproved { name });
    }

    pub fn denyPending(&mut self) {
        self.pendingPermit = false;
        let name = std::mem::take(&mut self.pendingToolName);
        self.entries.push(PanelEntry::ToolDenied { name });
    }

    pub fn toolApproved(&mut self, name: &str) {
        self.entries
            .push(PanelEntry::ToolApproved { name: name.into() });
    }

    pub fn toolDenied(&mut self, name: &str) {
        self.entries
            .push(PanelEntry::ToolDenied { name: name.into() });
    }

    pub fn pushToolResult(&mut self, name: &str, output: &str) {
        self.entries.push(PanelEntry::ToolResult {
            name: name.into(),
            output: output.into(),
        });
        self.scrollOffset = 0;
    }

    pub fn pushError(&mut self, msg: &str) {
        self.entries.push(PanelEntry::Error(msg.into()));
    }

    /// Whether the input buffer contains a large paste (>5 lines).
    pub fn isLargePaste(&self) -> bool {
        self.inputBuf.lines().count() > 5
    }

    pub fn scrollUp(&mut self, amount: u16) {
        self.scrollOffset = self.scrollOffset.saturating_add(amount);
    }

    pub fn scrollDown(&mut self, amount: u16) {
        self.scrollOffset = self.scrollOffset.saturating_sub(amount);
    }

    /// Scroll offset from the bottom (analogous to terminal displayOffset).
    /// Both increase when scrolling up, making grid-line math consistent.
    pub fn displayOffset(&self) -> u16 {
        self.scrollOffset
    }

    /// Render the panel. Returns the chat content area Rect.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer, focused: bool) -> Rect {
        let borderStyle = if focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" agent ");

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 3 {
            return Rect::default();
        }

        // Split: chat area + input line.
        let chunks = Layout::default()
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(inner);
        let chatArea = chunks[0];
        let inputArea = chunks[1];

        // Input / permit prompt.
        self.renderInput(inputArea, buf, focused);

        // Build all display lines.
        let lines = self.buildLines();

        // Compute scroll to pin to bottom.
        let totalWrapped = estimateWrappedLines(&lines, chatArea.width);
        let visible = chatArea.height;
        let maxScroll = totalWrapped.saturating_sub(visible);
        // Clamp to prevent scroll accumulation past content top.
        self.scrollOffset = self.scrollOffset.min(maxScroll);
        let scrollY = maxScroll.saturating_sub(self.scrollOffset);

        self.lastScrollY = scrollY;
        self.lastChatWidth = chatArea.width;

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scrollY, 0))
            .render(chatArea, buf);

        chatArea
    }

    fn renderInput(&self, area: Rect, buf: &mut Buffer, focused: bool) {
        if self.pendingPermit {
            let prompt = Line::from(vec![
                Span::styled(
                    " Allow? ",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("[y]es ", Style::default().fg(Color::Green)),
                Span::styled("[n]o", Style::default().fg(Color::Red)),
            ]);
            Paragraph::new(prompt).render(area, buf);
        } else if self.isLargePaste() {
            let lineCount = self.inputBuf.lines().count();
            let line = Line::from(vec![
                Span::styled(
                    format!("[{lineCount} lines pasted]"),
                    Style::default().fg(Color::Magenta),
                ),
                Span::styled(
                    " \u{23CE} send",
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            Paragraph::new(line).render(area, buf);
        } else {
            let cursor = if focused { "\u{2588}" } else { "" };
            let inputLine = format!("\u{25B8} {}{cursor}", self.inputBuf);
            let style = if focused {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Paragraph::new(inputLine).style(style).render(area, buf);
        }
    }

    /// Find the visual line range of the message at the given grid line.
    ///
    /// Grid line = screenRow - scrollY. Groups consecutive Reasoning +
    /// Assistant entries into one message since they come from the same
    /// LLM turn. Returns (startGridLine, endGridLine) inclusive.
    pub fn entryBoundsAtGridLine(&self, gridLine: i32) -> Option<(i32, i32)> {
        let w = self.lastChatWidth.max(1) as usize;
        // maxScroll = scrollOffset + scrollY (total content above viewport bottom).
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as u32;

        // Compute visual line ranges for every entry.
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let mut cursor: u32 = 0;

        for entry in &self.entries {
            let mut entryLines: Vec<Line<'static>> = Vec::new();
            self.renderEntry(entry, &mut entryLines);

            let entryStart = cursor;
            for line in &entryLines {
                let lineWidth = line.width();
                if lineWidth == 0 {
                    cursor += 1;
                } else {
                    cursor += ((lineWidth + w - 1) / w) as u32;
                }
            }
            ranges.push((entryStart, cursor));
            // Separator blank line.
            cursor += 1;
        }

        // Find which entry the click landed on.
        let mut matchIdx: Option<usize> = None;
        for (i, &(_start, end)) in ranges.iter().enumerate() {
            // Entry occupies [start, end) content lines + 1 separator.
            if visualLine < end + 1 {
                matchIdx = Some(i);
                break;
            }
        }

        // Handle streaming content as a single entry.
        if matchIdx.is_none() && self.isStreaming {
            let streamStart = cursor;
            if !self.streamingReasoning.is_empty() {
                for line in self.streamingReasoning.lines() {
                    let span = Span::raw(format!("  {line}"));
                    let lineWidth = span.width();
                    cursor += if lineWidth == 0 {
                        1
                    } else {
                        ((lineWidth + w - 1) / w) as u32
                    };
                }
            }
            if !self.streamingContent.is_empty() {
                let md = markdown::render(&self.streamingContent);
                for line in &md.lines {
                    let lineWidth = line.width();
                    cursor += if lineWidth == 0 {
                        1
                    } else {
                        ((lineWidth + w - 1) / w) as u32
                    };
                }
            }
            if visualLine >= streamStart && visualLine < cursor {
                let startGrid = streamStart as i32 - maxScroll;
                let endGrid = cursor as i32 - 1 - maxScroll;
                if endGrid < startGrid {
                    return None;
                }
                return Some((startGrid, endGrid));
            }
            return None;
        }

        let idx = matchIdx?;

        // Expand to include the full message group.
        let (groupStart, groupEnd) = self.messageGroup(idx, &ranges);

        let startGrid = groupStart as i32 - maxScroll;
        let endGrid = groupEnd as i32 - 1 - maxScroll;
        if endGrid < startGrid {
            return None;
        }
        Some((startGrid, endGrid))
    }

    /// Find the visual line range of the message group containing entry `idx`.
    ///
    /// Consecutive Reasoning + Assistant entries are one LLM message.
    /// Everything else (User, ToolRequest, ToolResult, Error) is standalone.
    fn messageGroup(&self, idx: usize, ranges: &[(u32, u32)]) -> (u32, u32) {
        let mut startIdx = idx;
        let mut endIdx = idx;

        // If on Assistant, include preceding Reasoning.
        if matches!(self.entries[idx], PanelEntry::Assistant(_)) {
            if idx > 0 && matches!(self.entries[idx - 1], PanelEntry::Reasoning(_)) {
                startIdx = idx - 1;
            }
        }

        // If on Reasoning, include following Assistant.
        if matches!(self.entries[idx], PanelEntry::Reasoning(_)) {
            if idx + 1 < self.entries.len()
                && matches!(self.entries[idx + 1], PanelEntry::Assistant(_))
            {
                endIdx = idx + 1;
            }
        }

        (ranges[startIdx].0, ranges[endIdx].1)
    }

    fn buildLines(&self) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();

        for entry in &self.entries {
            self.renderEntry(entry, &mut lines);
            lines.push(Line::from(""));
        }

        // Streaming content.
        if self.isStreaming {
            if !self.streamingReasoning.is_empty() {
                for line in self.streamingReasoning.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    )));
                }
            }
            if !self.streamingContent.is_empty() {
                let md = markdown::render(&self.streamingContent);
                lines.extend(md.lines);
            }
        }

        lines
    }

    fn renderEntry(&self, entry: &PanelEntry, lines: &mut Vec<Line<'static>>) {
        match entry {
            PanelEntry::User(text) => {
                lines.push(Line::from(Span::styled(
                    format!("\u{25B8} {text}"),
                    Style::default().fg(Color::Cyan),
                )));
            }
            PanelEntry::Assistant(text) => {
                let md = markdown::render(text);
                lines.extend(md.lines);
            }
            PanelEntry::Reasoning(text) => {
                for line in text.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    )));
                }
            }
            PanelEntry::ToolRequest { summary } => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "\u{2699}\u{FE0E} ",
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(
                        summary.clone(),
                        Style::default().fg(Color::Yellow),
                    ),
                ]));
            }
            PanelEntry::ToolApproved { name } => {
                lines.push(Line::from(Span::styled(
                    format!("\u{2713}\u{FE0E} {name}"),
                    Style::default().fg(Color::Green),
                )));
            }
            PanelEntry::ToolDenied { name } => {
                lines.push(Line::from(Span::styled(
                    format!("\u{2717}\u{FE0E} {name} (denied)"),
                    Style::default().fg(Color::Red),
                )));
            }
            PanelEntry::ToolResult { name, output } => {
                lines.push(Line::from(Span::styled(
                    format!("\u{25B8} {name} result:"),
                    Style::default().fg(Color::DarkGray),
                )));
                // Render output as code-like block (up to 20 lines).
                let outputLines: Vec<&str> = output.lines().collect();
                let showCount = outputLines.len().min(20);
                for line in &outputLines[..showCount] {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(Color::Green),
                    )));
                }
                if outputLines.len() > 20 {
                    lines.push(Line::from(Span::styled(
                        format!("  ... ({} more lines)", outputLines.len() - 20),
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
            PanelEntry::Error(msg) => {
                lines.push(Line::from(Span::styled(
                    format!("\u{26A0}\u{FE0E} {msg}"),
                    Style::default().fg(Color::Red),
                )));
            }
        }
    }
}

/// Estimate total visual lines after wrapping.
fn estimateWrappedLines(lines: &[Line], width: u16) -> u16 {
    let w = width.max(1) as usize;
    let mut total: u16 = 0;
    for line in lines {
        let lineWidth = line.width();
        if lineWidth == 0 {
            total = total.saturating_add(1);
        } else {
            // Ceiling division.
            let wrapped = ((lineWidth + w - 1) / w) as u16;
            total = total.saturating_add(wrapped);
        }
    }
    total
}
