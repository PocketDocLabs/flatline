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

use std::time::Instant;

use crate::history::History;
use crate::markdown;
use crate::selection::{self, Selection};
use crate::text_area::{TextArea, unicode_display_width};
use crate::throbber::Throbber;

/// A single entry in the agent panel display.
#[derive(Debug, Clone)]
pub enum PanelEntry {
    User(String),
    Assistant(String),
    Reasoning { text: String, expanded: bool },
    ToolRequest { summary: String },
    ToolApproved { name: String },
    ToolDenied { name: String },
    ToolResult { name: String, output: String },
    Error(String),
    Cancelled,
}

/// Agent panel state.
pub struct AgentPanel {
    entries: Vec<PanelEntry>,
    streamingContent: String,
    streamingReasoning: String,
    isStreaming: bool,
    pub textArea: TextArea,
    pub history: History,
    pub pendingPermit: bool,
    pendingToolName: String,
    /// Throbber animation for inline thinking indicator.
    throbber: Throbber,
    /// When reasoning started (for elapsed time display).
    thinkingStartTime: Option<Instant>,
    /// Whether the currently-streaming reasoning block is expanded.
    thinkingExpanded: bool,
    /// Whether reasoning is actively streaming right now.
    reasoningActive: bool,
    /// Frame counter for throbber tick throttling.
    throbberTickCounter: u8,
    /// Scroll offset from the bottom (in visual lines).
    scrollOffset: u16,
    /// ScrollY value from the last render (for visual-line lookups).
    lastScrollY: u16,
    /// Chat area width from the last render (for wrap estimation).
    lastChatWidth: u16,
    /// Previous maxScroll value (for stable scroll during streaming).
    lastMaxScroll: u16,
    /// Which visual lines are wrap continuations (not real line breaks).
    lastContinuationMap: Vec<bool>,
    /// Visual line index of each reasoning header (entry index, line index).
    /// `None` entry index means streaming reasoning.
    lastReasoningHeaders: Vec<(Option<usize>, usize)>,
    /// Plain text of each visual line from the last buildLines (for scrollback copy).
    lastLineTexts: Vec<String>,
    /// Input area rect from the last render (for mouse hit-testing).
    pub lastInputRect: Rect,
}

impl AgentPanel {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            streamingContent: String::new(),
            streamingReasoning: String::new(),
            isStreaming: false,
            textArea: TextArea::new(),
            history: History::new(),
            pendingPermit: false,
            pendingToolName: String::new(),
            throbber: Throbber::new(),
            thinkingStartTime: None,
            thinkingExpanded: false,
            reasoningActive: false,
            throbberTickCounter: 0,
            scrollOffset: 0,
            lastScrollY: 0,
            lastChatWidth: 0,
            lastMaxScroll: 0,
            lastContinuationMap: Vec::new(),
            lastReasoningHeaders: Vec::new(),
            lastLineTexts: Vec::new(),
            lastInputRect: Rect::default(),
        }
    }

    pub fn pushUser(&mut self, text: &str) {
        self.entries.push(PanelEntry::User(text.into()));
        self.scrollOffset = 0;
        // Start thinking indicator immediately for responsiveness.
        self.isStreaming = true;
        self.thinkingStartTime = Some(Instant::now());
    }

    pub fn appendContent(&mut self, text: &str) {
        self.isStreaming = true;
        // Content streaming means reasoning phase is over.
        self.reasoningActive = false;
        self.streamingContent.push_str(text);
    }

    pub fn appendReasoning(&mut self, text: &str) {
        self.isStreaming = true;
        self.reasoningActive = true;
        if self.thinkingStartTime.is_none() {
            self.thinkingStartTime = Some(Instant::now());
        }
        self.streamingReasoning.push_str(text);
    }

    pub fn finalizeStreaming(&mut self) {
        if !self.streamingReasoning.is_empty() {
            self.entries.push(PanelEntry::Reasoning {
                text: std::mem::take(&mut self.streamingReasoning),
                expanded: false,
            });
        }
        if !self.streamingContent.is_empty() {
            self.entries.push(PanelEntry::Assistant(
                std::mem::take(&mut self.streamingContent),
            ));
        }
        self.isStreaming = false;
        self.reasoningActive = false;
        self.thinkingStartTime = None;
        self.thinkingExpanded = false;
    }

    /// Whether a turn is currently in progress.
    pub fn isActive(&self) -> bool {
        self.isStreaming || self.pendingPermit
    }

    /// Finalize streaming state after a cancellation.
    pub fn finalizeCancelled(&mut self) {
        self.finalizeStreaming();
        self.pendingPermit = false;
        self.pendingToolName.clear();
        self.entries.push(PanelEntry::Cancelled);
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

    pub fn scrollUp(&mut self, amount: u16) {
        self.scrollOffset = self.scrollOffset.saturating_add(amount);
    }

    pub fn scrollDown(&mut self, amount: u16) {
        self.scrollOffset = self.scrollOffset.saturating_sub(amount);
    }

    /// Scroll offset from the bottom (analogous to terminal displayOffset).
    pub fn displayOffset(&self) -> u16 {
        self.scrollOffset
    }

    /// Advance the throbber animation. Call from the event loop.
    /// Ticks while waiting for first token or during reasoning.
    pub fn tickThrobber(&mut self) {
        self.throbberTickCounter = self.throbberTickCounter.wrapping_add(1);
        let waiting = self.isStreaming
            && self.streamingContent.is_empty()
            && self.streamingReasoning.is_empty();
        if self.throbberTickCounter % 8 == 0 && (waiting || self.reasoningActive) {
            self.throbber.tick();
        }
    }

    /// Toggle the most recent reasoning block (streaming or finalized).
    pub fn toggleThinking(&mut self) {
        if self.isStreaming && !self.streamingReasoning.is_empty() {
            self.thinkingExpanded = !self.thinkingExpanded;
        } else {
            for entry in self.entries.iter_mut().rev() {
                if let PanelEntry::Reasoning { expanded, .. } = entry {
                    *expanded = !*expanded;
                    break;
                }
            }
        }
    }

    /// Toggle a reasoning block if the given grid line is its header.
    ///
    /// Returns true if a toggle occurred (caller should skip selection).
    pub fn toggleReasoningAtGridLine(&mut self, gridLine: i32) -> bool {
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as usize;
        let w = self.lastChatWidth.max(1);

        for &(entryIdx, lineIdx) in &self.lastReasoningHeaders {
            if lineIdx == visualLine {
                match entryIdx {
                    Some(idx) => {
                        if let PanelEntry::Reasoning { text, expanded } = &mut self.entries[idx] {
                            let delta = countReasoningLines(text, w);
                            *expanded = !*expanded;
                            if *expanded {
                                self.scrollOffset = self.scrollOffset.saturating_add(delta);
                                // Preempt streaming compensation so it doesn't double-adjust.
                                self.lastMaxScroll = self.lastMaxScroll.saturating_add(delta);
                            } else {
                                self.scrollOffset = self.scrollOffset.saturating_sub(delta);
                                self.lastMaxScroll = self.lastMaxScroll.saturating_sub(delta);
                            }
                        }
                    }
                    None => {
                        let delta = countReasoningLines(&self.streamingReasoning, w);
                        self.thinkingExpanded = !self.thinkingExpanded;
                        if self.thinkingExpanded {
                            self.scrollOffset = self.scrollOffset.saturating_add(delta);
                            self.lastMaxScroll = self.lastMaxScroll.saturating_add(delta);
                        } else {
                            self.scrollOffset = self.scrollOffset.saturating_sub(delta);
                            self.lastMaxScroll = self.lastMaxScroll.saturating_sub(delta);
                        }
                    }
                }
                return true;
            }
        }

        false
    }

    /// Extract text from the agent panel selection, rejoining wrapped lines.
    ///
    /// Uses the continuation map to detect lines added by word-wrapping
    /// and joins them back together so the clipboard gets unwrapped text.
    pub fn extractUnwrappedText(
        &self,
        sel: &Selection,
        area: Rect,
        buf: &Buffer,
        displayOffset: u16,
    ) -> String {
        if sel.isEmpty() {
            return String::new();
        }

        let ((sc, sr), (ec, er)) = sel.sorted();
        let maxScroll = (self.lastScrollY + self.scrollOffset) as i32;
        // Display column offset for the 2-char prefix that's excluded from content rect.
        let prefixCols: u16 = 2;

        let mut segments: Vec<(String, bool)> = Vec::new();

        for gridLine in sr..=er {
            let visualIdx = (gridLine as i32 + maxScroll) as usize;
            let colStart = if gridLine == sr { sc } else { 0 };
            let colEnd = if gridLine == er { ec } else { area.width };

            let text = if let Some(screenRow) =
                selection::toScreenRow(gridLine, displayOffset, area.height)
            {
                // Visible — read from Buffer (handles styled rendering accurately).
                let mut line = String::new();
                for col in colStart..colEnd {
                    if col >= area.width {
                        break;
                    }
                    if let Some(cell) = buf.cell((area.x + col, area.y + screenRow)) {
                        line.push_str(cell.symbol());
                    }
                }
                line.trim_end().to_string()
            } else if visualIdx < self.lastLineTexts.len() {
                // Off-screen — extract from cached line text.
                sliceByDisplayColumn(
                    &self.lastLineTexts[visualIdx],
                    prefixCols + colStart,
                    prefixCols + colEnd,
                )
            } else {
                String::new()
            };

            let isCont = visualIdx < self.lastContinuationMap.len()
                && self.lastContinuationMap[visualIdx];

            segments.push((text, isCont));
        }

        // Remove trailing empty lines.
        while segments.last().is_some_and(|(l, _)| l.is_empty()) {
            segments.pop();
        }

        // Join lines, merging wrap continuations.
        let mut result = String::new();
        for (i, (line, isCont)) in segments.iter().enumerate() {
            if i > 0 {
                if *isCont {
                    // Continuation from word-wrapping — join without newline.
                    if !result.ends_with(' ') && !result.is_empty() && !line.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(line);
                    continue;
                } else {
                    result.push('\n');
                }
            }
            result.push_str(line);
        }

        result
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

        // Dynamic input height based on content.
        // Width accounts for: 2 right padding + 2 prompt prefix.
        let inputHeight = if self.pendingPermit {
            1
        } else {
            self.textArea.desiredHeight(inner.width.saturating_sub(4)).min(8).max(1)
        };

        // Split: chat area + separator + input.
        let chunks = Layout::default()
            .constraints([
                Constraint::Min(1),
                Constraint::Length(1),
                Constraint::Length(inputHeight),
            ])
            .split(inner);
        let chatArea = chunks[0];
        let separatorArea = chunks[1];
        let inputArea = chunks[2];

        // Separator line.
        let sep = "\u{2500}".repeat(separatorArea.width as usize);
        Paragraph::new(sep)
            .style(Style::default().fg(Color::DarkGray))
            .render(separatorArea, buf);

        // Input / permit prompt (with right padding).
        let paddedInput = Rect {
            x: inputArea.x,
            y: inputArea.y,
            width: inputArea.width.saturating_sub(2),
            height: inputArea.height,
        };
        self.lastInputRect = paddedInput;
        self.renderInput(paddedInput, buf, focused);

        // Right padding so content doesn't touch border.
        let paddedChat = Rect {
            x: chatArea.x,
            y: chatArea.y,
            width: chatArea.width.saturating_sub(2),
            height: chatArea.height,
        };

        // Build all display lines (pre-wrapped to fit paddedChat width).
        let (lines, contMap, reasoningHeaders) = self.buildLines(paddedChat.width);
        self.lastContinuationMap = contMap;
        self.lastReasoningHeaders = reasoningHeaders;
        // Cache plain text of each visual line for scrollback copy.
        self.lastLineTexts = lines.iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();

        // Use ratatui's own word-wrap line counter for exact scroll math.
        let paragraph = Paragraph::new(lines)
            .wrap(Wrap { trim: false });
        let totalWrapped = paragraph.line_count(paddedChat.width) as u16;
        let visible = paddedChat.height;
        let maxScroll = totalWrapped.saturating_sub(visible);
        // Keep view pinned when scrolled up and new content arrives.
        if self.scrollOffset > 0 && maxScroll > self.lastMaxScroll {
            self.scrollOffset = self.scrollOffset.saturating_add(maxScroll - self.lastMaxScroll);
        }
        self.lastMaxScroll = maxScroll;
        // Clamp to prevent scroll accumulation past content top.
        self.scrollOffset = self.scrollOffset.min(maxScroll);
        let scrollY = maxScroll.saturating_sub(self.scrollOffset);

        self.lastScrollY = scrollY;
        self.lastChatWidth = paddedChat.width;

        paragraph
            .scroll((scrollY, 0))
            .render(paddedChat, buf);

        paddedChat
    }

    fn renderInput(&mut self, area: Rect, buf: &mut Buffer, focused: bool) {
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
        } else {
            self.textArea.render(area, buf, focused);
        }
    }

    /// Find the visual line range of the message at the given grid line.
    ///
    /// Grid line = screenRow - scrollY. Groups consecutive Reasoning +
    /// Assistant entries into one message since they come from the same
    /// LLM turn. Returns (startGridLine, endGridLine) inclusive.
    pub fn entryBoundsAtGridLine(&self, gridLine: i32) -> Option<(i32, i32)> {
        let w = self.lastChatWidth.max(1) as usize;
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as u32;

        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let mut cursor: u32 = 0;

        for entry in &self.entries {
            let mut entryLines: Vec<Line<'static>> = Vec::new();
            let mut entryCont: Vec<bool> = Vec::new();
            self.renderEntry(entry, &mut entryLines, &mut entryCont, w as u16);

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
            cursor += 1;
        }

        let mut matchIdx: Option<usize> = None;
        for (i, &(_start, end)) in ranges.iter().enumerate() {
            if visualLine < end + 1 {
                matchIdx = Some(i);
                break;
            }
        }

        // Handle streaming content as a single entry.
        if matchIdx.is_none() && self.isStreaming {
            let streamStart = cursor;
            let waiting = self.streamingContent.is_empty()
                && self.streamingReasoning.is_empty();

            // Throbber: shown while waiting or during reasoning.
            if waiting || !self.streamingReasoning.is_empty() {
                cursor += 2; // blob rows
                if self.thinkingExpanded {
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
                cursor += 1; // separator
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
    fn messageGroup(&self, idx: usize, ranges: &[(u32, u32)]) -> (u32, u32) {
        let mut startIdx = idx;
        let mut endIdx = idx;

        if matches!(self.entries[idx], PanelEntry::Assistant(_)) {
            if idx > 0 && matches!(self.entries[idx - 1], PanelEntry::Reasoning { .. }) {
                startIdx = idx - 1;
            }
        }

        if matches!(self.entries[idx], PanelEntry::Reasoning { .. }) {
            if idx + 1 < self.entries.len()
                && matches!(self.entries[idx + 1], PanelEntry::Assistant(_))
            {
                endIdx = idx + 1;
            }
        }

        (ranges[startIdx].0, ranges[endIdx].1)
    }

    fn buildLines(&self, width: u16) -> (Vec<Line<'static>>, Vec<bool>, Vec<(Option<usize>, usize)>) {
        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut cont: Vec<bool> = Vec::new();
        let mut reasoningHeaders: Vec<(Option<usize>, usize)> = Vec::new();

        for (i, entry) in self.entries.iter().enumerate() {
            if matches!(entry, PanelEntry::Reasoning { .. }) {
                reasoningHeaders.push((Some(i), lines.len()));
            }
            self.renderEntry(entry, &mut lines, &mut cont, width);
            lines.push(Line::from(""));
            cont.push(false);
        }

        // Streaming content.
        if self.isStreaming {
            let waiting = self.streamingContent.is_empty()
                && self.streamingReasoning.is_empty();
            let showThrobber = waiting || self.reasoningActive;
            let hasReasoning = !self.streamingReasoning.is_empty();

            if showThrobber {
                // Record header position for click-to-toggle.
                if hasReasoning {
                    reasoningHeaders.push((None, lines.len()));
                }
                // Animated throbber with elapsed time.
                let blobLines = self.throbber.renderLines();
                let elapsed = self.thinkingStartTime
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                let suffix = if hasReasoning {
                    let icon = if self.thinkingExpanded { "\u{25BE}" } else { "\u{25B8}" };
                    format!(" thinking ({elapsed}s)  {icon}")
                } else {
                    format!(" thinking ({elapsed}s)")
                };

                lines.push(Line::from(vec![
                    blobLines[0].spans[0].clone(),
                    Span::styled(suffix, Style::default().fg(Color::DarkGray)),
                ]));
                cont.push(false);
                lines.push(blobLines[1].clone());
                cont.push(false);
            } else if hasReasoning {
                // Record header position for click-to-toggle.
                reasoningHeaders.push((None, lines.len()));
                // Reasoning finished but text exists — show static collapse header.
                let icon = if self.thinkingExpanded { "\u{25BE}" } else { "\u{25B8}" };
                lines.push(Line::from(Span::styled(
                    format!("{icon} reasoning"),
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
            }

            if self.thinkingExpanded && hasReasoning {
                let style = Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC);
                let textWidth = (width as usize).saturating_sub(2);
                for logicalLine in self.streamingReasoning.lines() {
                    let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
                    let wrapped = wrapSpannedLine(spanLine, textWidth);
                    for (idx, wLine) in wrapped.into_iter().enumerate() {
                        cont.push(idx > 0);
                        let mut spans = vec![Span::styled("  ".to_string(), style)];
                        spans.extend(wLine.spans);
                        lines.push(Line::from(spans));
                    }
                }
            }

            if showThrobber || hasReasoning {
                lines.push(Line::from(""));
                cont.push(false);
            }

            if !self.streamingContent.is_empty() {
                let md = markdown::render(&self.streamingContent);
                prefixFirstLine(&mut lines, &mut cont, md.lines, "\u{25C6} ", Style::default().fg(Color::White), width);
            }
        }

        (lines, cont, reasoningHeaders)
    }

    fn renderEntry(&self, entry: &PanelEntry, lines: &mut Vec<Line<'static>>, cont: &mut Vec<bool>, width: u16) {
        match entry {
            PanelEntry::User(text) => {
                let style = Style::default().fg(Color::Cyan);
                let prefixWidth: usize = 2; // "› " = 2 display columns.
                let textWidth = (width as usize).saturating_sub(prefixWidth);
                let mut isFirst = true;

                for logicalLine in text.lines() {
                    let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
                    let wrapped = wrapSpannedLine(spanLine, textWidth);
                    for (idx, wLine) in wrapped.into_iter().enumerate() {
                        cont.push(idx > 0);
                        let prefix = if isFirst {
                            "\u{203A} "
                        } else {
                            "  "
                        };
                        isFirst = false;
                        let mut spans = vec![Span::styled(prefix.to_string(), style)];
                        spans.extend(wLine.spans);
                        lines.push(Line::from(spans));
                    }
                }
                if text.ends_with('\n') {
                    lines.push(Line::from(Span::styled("  ", style)));
                    cont.push(false);
                }
            }
            PanelEntry::Assistant(text) => {
                let md = markdown::render(text);
                prefixFirstLine(lines, cont, md.lines, "\u{25C6} ", Style::default().fg(Color::White), width);
            }
            PanelEntry::Reasoning { text, expanded } => {
                let icon = if *expanded { "\u{25BE}" } else { "\u{25B8}" };
                lines.push(Line::from(Span::styled(
                    format!("{icon} reasoning"),
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
                if *expanded {
                    let style = Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC);
                    let textWidth = (width as usize).saturating_sub(2);
                    for logicalLine in text.lines() {
                        let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
                        let wrapped = wrapSpannedLine(spanLine, textWidth);
                        for (idx, wLine) in wrapped.into_iter().enumerate() {
                            cont.push(idx > 0);
                            let mut spans = vec![Span::styled("  ".to_string(), style)];
                            spans.extend(wLine.spans);
                            lines.push(Line::from(spans));
                        }
                    }
                }
            }
            PanelEntry::ToolRequest { summary } => {
                let style = Style::default().fg(Color::Yellow);
                let content = vec![Line::from(Span::styled(summary.clone(), style))];
                prefixFirstLine(lines, cont, content, "\u{2699}\u{FE0E} ", style, width);
            }
            PanelEntry::ToolApproved { name } => {
                let style = Style::default().fg(Color::Green);
                let content = vec![Line::from(Span::styled(name.clone(), style))];
                prefixFirstLine(lines, cont, content, "\u{2713}\u{FE0E} ", style, width);
            }
            PanelEntry::ToolDenied { name } => {
                let style = Style::default().fg(Color::Red);
                let content = vec![Line::from(Span::styled(
                    format!("{name} (denied)"),
                    style,
                ))];
                prefixFirstLine(lines, cont, content, "\u{2717}\u{FE0E} ", style, width);
            }
            PanelEntry::ToolResult { name, output } => {
                lines.push(Line::from(Span::styled(
                    format!("\u{25C7} {name} result:"),
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
                let outputLines: Vec<&str> = output.lines().collect();
                let showCount = outputLines.len().min(20);
                let style = Style::default().fg(Color::Green);
                let textWidth = (width as usize).saturating_sub(2);
                for line in &outputLines[..showCount] {
                    let spanLine = Line::from(Span::styled(line.to_string(), style));
                    let wrapped = wrapSpannedLine(spanLine, textWidth);
                    for (idx, wLine) in wrapped.into_iter().enumerate() {
                        cont.push(idx > 0);
                        let mut spans = vec![Span::styled("  ".to_string(), style)];
                        spans.extend(wLine.spans);
                        lines.push(Line::from(spans));
                    }
                }
                if outputLines.len() > 20 {
                    lines.push(Line::from(Span::styled(
                        format!("  ... ({} more lines)", outputLines.len() - 20),
                        Style::default().fg(Color::DarkGray),
                    )));
                    cont.push(false);
                }
            }
            PanelEntry::Error(msg) => {
                let style = Style::default().fg(Color::Red);
                let content = vec![Line::from(Span::styled(msg.clone(), style))];
                prefixFirstLine(lines, cont, content, "\u{26A0}\u{FE0E} ", style, width);
            }
            PanelEntry::Cancelled => {
                lines.push(Line::from(Span::styled(
                    "\u{2500} cancelled",
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
            }
        }
    }
}

/// Extract a substring from a plain text string by display column range.
///
/// Accounts for multi-width characters. Returns the trimmed slice between
/// `colStart` and `colEnd` (exclusive) in display columns.
fn sliceByDisplayColumn(text: &str, colStart: u16, colEnd: u16) -> String {
    let mut result = String::new();
    let mut col: u16 = 0;

    for ch in text.chars() {
        let w = unicode_display_width(ch) as u16;
        if col + w > colEnd {
            break;
        }
        if col >= colStart {
            result.push(ch);
        }
        col += w;
    }

    result.trim_end().to_string()
}

/// Count the visual lines a reasoning text would produce when expanded.
fn countReasoningLines(text: &str, width: u16) -> u16 {
    let textWidth = (width as usize).saturating_sub(2);
    let style = Style::default();
    let mut count: u16 = 0;
    for logicalLine in text.lines() {
        let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
        count += wrapSpannedLine(spanLine, textWidth).len() as u16;
    }
    count
}

/// Prepend a styled symbol to the first line; indent continuations to match.
///
/// Pre-wraps each content line at `(width - prefixWidth)` so that wrapped
/// continuations align with the text start, not column 0.
fn prefixFirstLine(
    out: &mut Vec<Line<'static>>,
    cont: &mut Vec<bool>,
    contentLines: Vec<Line<'static>>,
    symbol: &str,
    symbolStyle: Style,
    width: u16,
) {
    let prefixWidth: usize = symbol.chars().map(unicode_display_width).sum();
    let indent = " ".repeat(prefixWidth);
    let textWidth = (width as usize).saturating_sub(prefixWidth);
    let mut isFirst = true;

    for line in contentLines {
        let wrapped = wrapSpannedLine(line, textWidth);
        for (idx, wLine) in wrapped.into_iter().enumerate() {
            cont.push(idx > 0);
            let prefix = if isFirst {
                Span::styled(symbol.to_string(), symbolStyle)
            } else {
                Span::raw(indent.clone())
            };
            isFirst = false;
            let mut spans = vec![prefix];
            spans.extend(wLine.spans);
            out.push(Line::from(spans));
        }
    }
}

/// Wrap a multi-span Line into multiple lines fitting within maxWidth display columns.
///
/// Prefers breaking at space boundaries. Falls back to character-level
/// splitting when a word exceeds the available width.
fn wrapSpannedLine(line: Line<'static>, maxWidth: usize) -> Vec<Line<'static>> {
    if maxWidth == 0 {
        return vec![line];
    }

    // Flatten spans into (char, Style) pairs.
    let chars: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|span| span.content.chars().map(move |ch| (ch, span.style)))
        .collect();

    let totalWidth: usize = chars.iter().map(|(ch, _)| unicode_display_width(*ch)).sum();
    if totalWidth <= maxWidth {
        return vec![line];
    }

    let mut result: Vec<Line<'static>> = Vec::new();
    let mut lineStart = 0;
    let mut currentWidth: usize = 0;
    let mut lastSpace: Option<usize> = None;

    for i in 0..chars.len() {
        if chars[i].0 == ' ' {
            lastSpace = Some(i);
        }

        let charW = unicode_display_width(chars[i].0);

        if currentWidth + charW > maxWidth && i > lineStart {
            let breakAt = if let Some(sp) = lastSpace {
                if sp > lineStart { sp + 1 } else { i }
            } else {
                i
            };

            result.push(styledCharsToLine(&chars[lineStart..breakAt]));
            lineStart = breakAt;
            // Recount width from the new start through the current character.
            currentWidth = chars[lineStart..=i]
                .iter()
                .map(|(ch, _)| unicode_display_width(*ch))
                .sum();
            lastSpace = None;
        } else {
            currentWidth += charW;
        }
    }

    if lineStart < chars.len() {
        result.push(styledCharsToLine(&chars[lineStart..]));
    }

    result
}

/// Reconstruct a Line from styled character pairs, merging adjacent same-style runs.
fn styledCharsToLine(chars: &[(char, Style)]) -> Line<'static> {
    if chars.is_empty() {
        return Line::from("");
    }

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut currentStr = String::new();
    let mut currentStyle = chars[0].1;

    for &(ch, style) in chars {
        if style == currentStyle {
            currentStr.push(ch);
        } else {
            if !currentStr.is_empty() {
                spans.push(Span::styled(
                    std::mem::take(&mut currentStr),
                    currentStyle,
                ));
            }
            currentStr.push(ch);
            currentStyle = style;
        }
    }

    if !currentStr.is_empty() {
        spans.push(Span::styled(currentStr, currentStyle));
    }

    Line::from(spans)
}

