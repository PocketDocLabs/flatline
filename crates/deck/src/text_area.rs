//! Multi-line text input widget with cursor movement and editing.
//!
//! Grapheme-aware cursor, word/line movement, kill buffer, internal
//! scroll, dynamic height, visual wrapping, and inline paste collapse.
//!
//! # Public API
//! - [`TextArea`] — input state and ratatui rendering
//!
//! # Dependencies
//! `unicode-segmentation`, `ratatui`

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

/// Multi-line text input with cursor, editing, and wrapping.
pub struct TextArea {
    text: String,
    /// Byte offset of the cursor within `text`.
    cursorPos: usize,
    /// Kill buffer (unused for now, reserved for future keybind support).
    #[allow(dead_code)]
    killBuf: String,
    /// Internal scroll offset (in visual lines from the top).
    scroll: u16,
    /// Collapsed paste region [start, end) byte range in `text`.
    /// Set when pasting >5 lines. Rendered as an inline placeholder.
    pasteRegion: Option<(usize, usize)>,
    /// Screen position of the cursor after last render (col, row).
    /// Used by the harness to set the terminal's hardware cursor.
    pub cursorScreenPos: Option<(u16, u16)>,
    /// Selection anchor (byte offset) — set on mouse down.
    selAnchor: Option<usize>,
    /// Selection endpoint (byte offset) — updated on drag.
    selEnd: Option<usize>,
    /// Ghost text shown when the input is empty.
    pub placeholder: &'static str,
}

impl TextArea {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursorPos: 0,
            killBuf: String::new(),
            scroll: 0,
            pasteRegion: None,
            cursorScreenPos: None,
            selAnchor: None,
            selEnd: None,
            placeholder: "Type a message...",
        }
    }

    // --- Queries ---

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Desired widget height for the current content at the given width.
    ///
    /// `width` should be the content width (area minus prefix).
    pub fn desiredHeight(&self, width: u16) -> u16 {
        if self.text.is_empty() {
            return 1;
        }
        let (display, _, _) = self.displayTextAndCursor();
        let w = width.max(1) as usize;
        let count: usize = display
            .split('\n')
            .map(|line| {
                if line.is_empty() {
                    1
                } else {
                    wrapSegmentsFor(line, w).len()
                }
            })
            .sum();
        (count as u16).max(1)
    }

    /// Whether the cursor is on the first logical line.
    pub fn isAtFirstLine(&self) -> bool {
        !self.text[..self.cursorPos].contains('\n')
    }

    /// Whether the cursor is on the last logical line.
    pub fn isAtLastLine(&self) -> bool {
        !self.text[self.cursorPos..].contains('\n')
    }

    // --- Editing ---

    /// Insert a character at the cursor.
    pub fn insert(&mut self, c: char) {
        let len = c.len_utf8();
        self.text.insert(self.cursorPos, c);
        self.adjustPasteInsert(self.cursorPos, len);
        self.cursorPos += len;
    }

    /// Insert a string at the cursor (for paste).
    pub fn insertStr(&mut self, s: &str) {
        let pos = self.cursorPos;
        self.text.insert_str(pos, s);

        if s.len() > 80 {
            // Collapse this paste into an inline placeholder.
            self.adjustPasteInsert(pos, s.len());
            self.pasteRegion = Some((pos, pos + s.len()));
            self.cursorPos = pos + s.len();
        } else {
            self.adjustPasteInsert(pos, s.len());
            self.cursorPos += s.len();
        }
    }

    /// Delete the grapheme before the cursor.
    pub fn backspace(&mut self) {
        if self.cursorPos == 0 {
            return;
        }
        // Backspace at the end of a paste region: delete the whole paste.
        if let Some((start, end)) = self.pasteRegion {
            if self.cursorPos == end {
                self.text.drain(start..end);
                self.cursorPos = start;
                self.pasteRegion = None;
                return;
            }
        }
        let prev = self.prevGraphemeBoundary(self.cursorPos);
        self.adjustPasteDelete(prev, self.cursorPos);
        self.text.drain(prev..self.cursorPos);
        self.cursorPos = prev;
    }

    /// Delete the grapheme after the cursor.
    pub fn delete(&mut self) {
        if self.cursorPos >= self.text.len() {
            return;
        }
        // Delete at the start of a paste region: delete the whole paste.
        if let Some((start, end)) = self.pasteRegion {
            if self.cursorPos == start {
                self.text.drain(start..end);
                self.pasteRegion = None;
                return;
            }
        }
        let next = self.nextGraphemeBoundary(self.cursorPos);
        self.adjustPasteDelete(self.cursorPos, next);
        self.text.drain(self.cursorPos..next);
    }

    /// Delete the word before the cursor (Option+Backspace).
    pub fn deleteWordLeft(&mut self) {
        if self.cursorPos == 0 {
            return;
        }
        let before = &self.text[..self.cursorPos];
        let trimmed = before.trim_end();
        let wordStart = if trimmed.is_empty() {
            0
        } else {
            let lastSpace = trimmed.rfind(|c: char| c.is_whitespace());
            match lastSpace {
                Some(pos) => {
                    let after = &trimmed[pos..];
                    pos + after
                        .find(|c: char| !c.is_whitespace())
                        .unwrap_or(after.len())
                }
                None => 0,
            }
        };
        self.adjustPasteDelete(wordStart, self.cursorPos);
        self.text.drain(wordStart..self.cursorPos);
        self.cursorPos = wordStart;
    }

    /// Kill from cursor to end of current logical line.
    pub fn killToEnd(&mut self) {
        let lineEnd = self.text[self.cursorPos..]
            .find('\n')
            .map(|p| self.cursorPos + p)
            .unwrap_or(self.text.len());
        // If cursor is at newline, kill the newline itself.
        let end = if lineEnd == self.cursorPos && self.cursorPos < self.text.len() {
            self.cursorPos + 1
        } else {
            lineEnd
        };
        self.killBuf = self.text[self.cursorPos..end].to_string();
        self.adjustPasteDelete(self.cursorPos, end);
        self.text.drain(self.cursorPos..end);
    }

    /// Kill from cursor to start of current logical line.
    pub fn killToStart(&mut self) {
        let lineStart = self.text[..self.cursorPos]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        self.killBuf = self.text[lineStart..self.cursorPos].to_string();
        self.adjustPasteDelete(lineStart, self.cursorPos);
        self.text.drain(lineStart..self.cursorPos);
        self.cursorPos = lineStart;
    }

    /// Yank (paste) the kill buffer at the cursor.
    pub fn yank(&mut self) {
        if !self.killBuf.is_empty() {
            let buf = self.killBuf.clone();
            self.insertStr(&buf);
        }
    }

    /// Submit: take the text and reset. Returns None if empty.
    pub fn submit(&mut self) -> Option<String> {
        if self.text.is_empty() {
            return None;
        }
        let text = std::mem::take(&mut self.text);
        self.cursorPos = 0;
        self.scroll = 0;
        self.pasteRegion = None;
        self.selAnchor = None;
        self.selEnd = None;
        Some(text)
    }

    /// Replace the text (for history recall). Cursor moves to end.
    pub fn setText(&mut self, s: &str) {
        self.text = s.to_string();
        self.cursorPos = self.text.len();
        self.scroll = 0;
        self.pasteRegion = None;
        self.selAnchor = None;
        self.selEnd = None;
    }

    // --- Movement ---

    /// Move cursor one grapheme left.
    pub fn moveLeft(&mut self) {
        if self.cursorPos > 0 {
            self.cursorPos = self.prevGraphemeBoundary(self.cursorPos);
            self.skipPasteLeft();
        }
    }

    /// Move cursor one grapheme right.
    pub fn moveRight(&mut self) {
        if self.cursorPos < self.text.len() {
            self.cursorPos = self.nextGraphemeBoundary(self.cursorPos);
            self.skipPasteRight();
        }
    }

    /// Move cursor one word left.
    pub fn moveWordLeft(&mut self) {
        if self.cursorPos == 0 {
            return;
        }
        let before = &self.text[..self.cursorPos];
        let trimmed = before.trim_end();
        if trimmed.is_empty() {
            self.cursorPos = 0;
        } else {
            let lastSpace = trimmed.rfind(|c: char| c.is_whitespace());
            self.cursorPos = match lastSpace {
                Some(pos) => {
                    let after = &trimmed[pos..];
                    pos + after
                        .find(|c: char| !c.is_whitespace())
                        .unwrap_or(after.len())
                }
                None => 0,
            };
        }
        self.skipPasteLeft();
    }

    /// Move cursor one word right.
    pub fn moveWordRight(&mut self) {
        if self.cursorPos >= self.text.len() {
            return;
        }
        let after = &self.text[self.cursorPos..];
        let wordEnd = after
            .find(|c: char| c.is_whitespace())
            .unwrap_or(after.len());
        let rest = &after[wordEnd..];
        let nextWord = rest
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(rest.len());
        self.cursorPos += wordEnd + nextWord;
        self.skipPasteRight();
    }

    /// Move cursor to start of current logical line.
    pub fn moveHome(&mut self) {
        let lineStart = self.text[..self.cursorPos]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        self.cursorPos = lineStart;
        self.skipPasteLeft();
    }

    /// Move cursor to end of current logical line.
    pub fn moveEnd(&mut self) {
        let lineEnd = self.text[self.cursorPos..]
            .find('\n')
            .map(|p| self.cursorPos + p)
            .unwrap_or(self.text.len());
        self.cursorPos = lineEnd;
        self.skipPasteRight();
    }

    /// Move cursor up one logical line (preserving column offset).
    pub fn moveUp(&mut self) {
        let (lineStart, col) = self.currentLineAndCol();
        if lineStart == 0 {
            return;
        }
        let prevLineEnd = lineStart - 1;
        let prevLineStart = self.text[..prevLineEnd]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let prevLineLen = prevLineEnd - prevLineStart;
        self.cursorPos = prevLineStart + col.min(prevLineLen);
        self.skipPasteLeft();
    }

    /// Move cursor down one logical line (preserving column offset).
    pub fn moveDown(&mut self) {
        let (lineStart, col) = self.currentLineAndCol();
        let lineEnd = self.text[lineStart..].find('\n').map(|p| lineStart + p);
        let Some(lineEnd) = lineEnd else { return };
        let nextLineStart = lineEnd + 1;
        let nextLineEnd = self.text[nextLineStart..]
            .find('\n')
            .map(|p| nextLineStart + p)
            .unwrap_or(self.text.len());
        let nextLineLen = nextLineEnd - nextLineStart;
        self.cursorPos = nextLineStart + col.min(nextLineLen);
        self.skipPasteRight();
    }

    // --- Mouse selection ---

    /// Handle mouse down in the input area. Positions cursor and starts selection.
    ///
    /// `contentCol` is the column within the content area (after the 2-char prefix).
    /// `localRow` is relative to the input area top.
    pub fn mouseDown(&mut self, contentCol: u16, localRow: u16, contentWidth: u16) {
        let visualLine = localRow as usize + self.scroll as usize;
        let offset =
            self.visualToByteOffset(contentCol as usize, visualLine, contentWidth as usize);
        self.cursorPos = offset;
        self.selAnchor = Some(offset);
        self.selEnd = Some(offset);
    }

    /// Handle mouse drag. Extends selection from anchor to current position.
    pub fn mouseDrag(&mut self, contentCol: u16, localRow: u16, contentWidth: u16) {
        let visualLine = localRow as usize + self.scroll as usize;
        let offset =
            self.visualToByteOffset(contentCol as usize, visualLine, contentWidth as usize);
        self.selEnd = Some(offset);
        self.cursorPos = offset;
    }

    /// Clear the current selection.
    pub fn clearSelection(&mut self) {
        self.selAnchor = None;
        self.selEnd = None;
    }

    /// Return the selected text, if any.
    pub fn selectedText(&self) -> Option<String> {
        let anchor = self.selAnchor?;
        let end = self.selEnd?;
        let (start, finish) = if anchor <= end {
            (anchor, end)
        } else {
            (end, anchor)
        };
        if start == finish {
            return None;
        }
        Some(self.text[start..finish].to_string())
    }

    // --- Rendering ---

    /// Render the text area into the given rect.
    ///
    /// Uses the terminal's hardware cursor (set via `cursorScreenPos`).
    /// Ghost text placeholder is overlaid — cursor sits on top of it.
    /// Large pastes are collapsed into an inline `[N lines pasted]` indicator.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer, focused: bool) {
        self.cursorScreenPos = None;

        if area.height == 0 || area.width < 3 {
            return;
        }

        if self.text.is_empty() {
            // Ghost text — cursor overlays the placeholder.
            let line = Line::from(vec![
                Span::styled("\u{203A} ", Style::default().fg(Color::DarkGray)),
                Span::styled(self.placeholder, Style::default().fg(Color::DarkGray)),
            ]);
            Paragraph::new(line).render(area, buf);
            if focused {
                self.cursorScreenPos = Some((area.x + 2, area.y));
            }
            return;
        }

        // Build display text — paste region replaced with placeholder.
        let (displayText, displayCursor, phRange) = self.displayTextAndCursor();

        // Swap in display text for rendering.
        let origText = std::mem::replace(&mut self.text, displayText);
        let origCursor = std::mem::replace(&mut self.cursorPos, displayCursor);

        // 2 for prompt prefix (`› ` or `  `).
        let contentWidth = area.width.saturating_sub(2) as usize;
        let lines = self.buildDisplayLines(contentWidth, focused, phRange);

        let (cursorVLine, cursorVCol) = self.cursorVisualPosition(contentWidth);
        if cursorVLine < self.scroll as usize {
            self.scroll = cursorVLine as u16;
        } else if cursorVLine >= (self.scroll as usize + area.height as usize) {
            self.scroll = (cursorVLine + 1).saturating_sub(area.height as usize) as u16;
        }

        // Compute screen position for hardware cursor.
        if focused {
            let screenRow = area.y + (cursorVLine as u16).saturating_sub(self.scroll);
            let screenCol = area.x + 2 + cursorVCol as u16;
            self.cursorScreenPos = Some((screenCol, screenRow));
        }

        // Restore real text before rendering (Paragraph borrows nothing from self).
        self.text = origText;
        self.cursorPos = origCursor;

        Paragraph::new(lines)
            .scroll((self.scroll, 0))
            .render(area, buf);

        // Apply selection highlight over rendered cells.
        if self.pasteRegion.is_none() {
            if let (Some(anchor), Some(end)) = (self.selAnchor, self.selEnd) {
                let start = anchor.min(end);
                let finish = anchor.max(end);
                if start != finish {
                    let (sv, sc) = self.byteToVisual(start, contentWidth);
                    let (ev, ec) = self.byteToVisual(finish, contentWidth);
                    let selStyle = Style::default()
                        .fg(Color::White)
                        .bg(Color::Rgb(60, 60, 120));

                    for vLine in sv..=ev {
                        let row = vLine as i32 - self.scroll as i32;
                        if row < 0 || row >= area.height as i32 {
                            continue;
                        }
                        let cs = if vLine == sv { sc } else { 0 };
                        let ce = if vLine == ev { ec } else { contentWidth };
                        for col in cs..ce {
                            let bufCol = area.x + 2 + col as u16;
                            let bufRow = area.y + row as u16;
                            if bufCol < area.x + area.width {
                                if let Some(cell) = buf.cell_mut((bufCol, bufRow)) {
                                    cell.set_style(selStyle);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Internal: paste region helpers ---

    /// If cursor landed inside the paste region, jump to the start.
    fn skipPasteLeft(&mut self) {
        if let Some((start, end)) = self.pasteRegion {
            if self.cursorPos > start && self.cursorPos < end {
                self.cursorPos = start;
            }
        }
    }

    /// If cursor landed inside the paste region, jump to the end.
    fn skipPasteRight(&mut self) {
        if let Some((start, end)) = self.pasteRegion {
            if self.cursorPos > start && self.cursorPos < end {
                self.cursorPos = end;
            }
        }
    }

    /// Shift paste region after an insertion of `len` bytes at `pos`.
    fn adjustPasteInsert(&mut self, pos: usize, len: usize) {
        if let Some((ref mut start, ref mut end)) = self.pasteRegion {
            if pos <= *start {
                *start += len;
                *end += len;
            }
            // Insertions inside or after the paste don't shift it.
        }
    }

    /// Adjust paste region after deleting bytes [from..to).
    fn adjustPasteDelete(&mut self, from: usize, to: usize) {
        if let Some((start, end)) = self.pasteRegion {
            if to <= start {
                // Deletion entirely before paste.
                let delta = to - from;
                self.pasteRegion = Some((start - delta, end - delta));
            } else if from >= end {
                // Deletion entirely after paste — no change.
            } else {
                // Deletion overlaps paste — expand it inline.
                self.pasteRegion = None;
            }
        }
    }

    /// Build display text with paste placeholder, mapping cursor position.
    ///
    /// Returns (displayText, displayCursor, placeholderByteRange).
    fn displayTextAndCursor(&self) -> (String, usize, Option<(usize, usize)>) {
        match self.pasteRegion {
            Some((start, end)) => {
                let pasteContent = &self.text[start..end];
                let lineCount =
                    pasteContent.lines().count() + if pasteContent.ends_with('\n') { 1 } else { 0 };
                let charCount = pasteContent.len();
                let placeholder = if lineCount > 1 {
                    format!("[{lineCount} lines, {charCount} chars pasted]")
                } else {
                    format!("[{charCount} chars pasted]")
                };
                let phLen = placeholder.len();

                let mut display = String::with_capacity(self.text.len() - (end - start) + phLen);
                display.push_str(&self.text[..start]);
                display.push_str(&placeholder);
                display.push_str(&self.text[end..]);

                let displayCursor = if self.cursorPos <= start {
                    self.cursorPos
                } else if self.cursorPos >= end {
                    start + phLen + (self.cursorPos - end)
                } else {
                    // Shouldn't happen with skip logic, clamp to start.
                    start
                };

                (display, displayCursor, Some((start, start + phLen)))
            }
            None => (self.text.clone(), self.cursorPos, None),
        }
    }

    // --- Internal: grapheme helpers ---

    /// Find the previous grapheme cluster boundary before `pos`.
    fn prevGraphemeBoundary(&self, pos: usize) -> usize {
        let mut cursor = unicode_segmentation::GraphemeCursor::new(pos, self.text.len(), true);
        cursor
            .prev_boundary(&self.text, 0)
            .ok()
            .flatten()
            .unwrap_or(0)
    }

    /// Find the next grapheme cluster boundary after `pos`.
    fn nextGraphemeBoundary(&self, pos: usize) -> usize {
        let mut cursor = unicode_segmentation::GraphemeCursor::new(pos, self.text.len(), true);
        cursor
            .next_boundary(&self.text, 0)
            .ok()
            .flatten()
            .unwrap_or(self.text.len())
    }

    /// (lineStart byte offset, column byte offset from lineStart).
    fn currentLineAndCol(&self) -> (usize, usize) {
        let lineStart = self.text[..self.cursorPos]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        (lineStart, self.cursorPos - lineStart)
    }

    /// Find the visual (line, column) of the cursor (0-indexed).
    fn cursorVisualPosition(&self, contentWidth: usize) -> (usize, usize) {
        self.byteToVisual(self.cursorPos, contentWidth)
    }

    /// Map a byte offset to its visual (line, column) position.
    fn byteToVisual(&self, byteOffset: usize, contentWidth: usize) -> (usize, usize) {
        let w = contentWidth.max(1);
        let mut visualLine = 0;

        for (lineIdx, logicalLine) in self.text.split('\n').enumerate() {
            let lineByteStart = logicalLineByteStartIn(&self.text, lineIdx);

            if logicalLine.is_empty() {
                if byteOffset == lineByteStart {
                    return (visualLine, 0);
                }
                visualLine += 1;
                continue;
            }

            let segments = wrapSegmentsFor(logicalLine, w);
            for seg in &segments {
                let absStart = lineByteStart + seg.byteStart;
                let absEnd = lineByteStart + seg.byteEnd;
                if byteOffset >= absStart
                    && (byteOffset < absEnd || (seg.isLast && byteOffset <= absEnd))
                {
                    let localOffset = byteOffset - lineByteStart;
                    let col: usize = logicalLine[seg.byteStart..localOffset]
                        .chars()
                        .map(unicode_display_width)
                        .sum();
                    return (visualLine, col);
                }
                visualLine += 1;
            }
        }

        (visualLine.saturating_sub(1), 0)
    }

    /// Map a visual (contentCol, visualLine) back to a byte offset.
    fn visualToByteOffset(
        &self,
        contentCol: usize,
        visualLine: usize,
        contentWidth: usize,
    ) -> usize {
        let w = contentWidth.max(1);
        let mut vLine = 0;

        for (lineIdx, logicalLine) in self.text.split('\n').enumerate() {
            let lineByteStart = logicalLineByteStartIn(&self.text, lineIdx);

            if logicalLine.is_empty() {
                if vLine == visualLine {
                    return lineByteStart;
                }
                vLine += 1;
                continue;
            }

            let segments = wrapSegmentsFor(logicalLine, w);
            for seg in &segments {
                if vLine == visualLine {
                    let mut col = 0;
                    for (i, ch) in logicalLine[seg.byteStart..seg.byteEnd].char_indices() {
                        if col >= contentCol {
                            return lineByteStart + seg.byteStart + i;
                        }
                        col += unicode_display_width(ch);
                    }
                    return lineByteStart + seg.byteEnd;
                }
                vLine += 1;
            }
        }

        self.text.len()
    }

    /// Build styled display lines with wrapping and paste placeholder.
    ///
    /// Cursor is handled via hardware cursor (not rendered inline).
    fn buildDisplayLines(
        &self,
        contentWidth: usize,
        _focused: bool,
        phRange: Option<(usize, usize)>,
    ) -> Vec<Line<'static>> {
        let style = Style::default().fg(Color::White);
        let promptStyle = Style::default().fg(Color::DarkGray);
        let pasteStyle = Style::default().fg(Color::Magenta);
        let w = contentWidth.max(1);

        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut isFirstVisualLine = true;

        for (lineIdx, logicalLine) in self.text.split('\n').enumerate() {
            let lineByteStart = logicalLineByteStartIn(&self.text, lineIdx);

            if logicalLine.is_empty() {
                let prefix = if isFirstVisualLine {
                    Span::styled("\u{203A} ", promptStyle)
                } else {
                    Span::styled("  ", promptStyle)
                };
                isFirstVisualLine = false;
                lines.push(Line::from(vec![prefix]));
                continue;
            }

            let segments = wrapSegmentsFor(logicalLine, w);

            for seg in &segments {
                let prefix = if isFirstVisualLine {
                    Span::styled("\u{203A} ", promptStyle)
                } else {
                    Span::styled("  ", promptStyle)
                };
                isFirstVisualLine = false;

                let segAbsStart = lineByteStart + seg.byteStart;
                let segText = &logicalLine[seg.byteStart..seg.byteEnd];

                let spans = buildSegmentSpans(
                    segText,
                    segAbsStart,
                    None,
                    phRange,
                    style,
                    style, // unused — no cursor rendering
                    pasteStyle,
                );

                let mut lineSpans = vec![prefix];
                lineSpans.extend(spans);
                lines.push(Line::from(lineSpans));
            }
        }

        lines
    }
}

/// Build spans for a single wrapped segment, handling cursor and paste placeholder styling.
fn buildSegmentSpans(
    segText: &str,
    segAbsStart: usize,
    cursorAbsPos: Option<usize>,
    phRange: Option<(usize, usize)>,
    style: Style,
    cursorStyle: Style,
    pasteStyle: Style,
) -> Vec<Span<'static>> {
    // Collect split points in local byte offsets.
    let mut splits: Vec<(usize, SplitKind)> = Vec::new();

    if let Some((phStart, phEnd)) = phRange {
        if phStart > segAbsStart && phStart < segAbsStart + segText.len() {
            splits.push((phStart - segAbsStart, SplitKind::PasteStart));
        }
        if phEnd > segAbsStart && phEnd < segAbsStart + segText.len() {
            splits.push((phEnd - segAbsStart, SplitKind::PasteEnd));
        }
    }

    // If no placeholder overlap and no cursor, fast path.
    if splits.is_empty() && cursorAbsPos.is_none() {
        let s = segStyle(segAbsStart, segText.len(), phRange, style, pasteStyle);
        return vec![Span::styled(segText.to_string(), s)];
    }

    // If no placeholder overlap but cursor present, use simple cursor split.
    if splits.is_empty() {
        if let Some(cursorAbs) = cursorAbsPos {
            let local = cursorAbs - segAbsStart;
            let s = segStyle(segAbsStart, segText.len(), phRange, style, pasteStyle);
            return cursorSplitSpans(segText, local, s, cursorStyle);
        }
    }

    // General case: split at placeholder boundaries, then embed cursor in
    // the appropriate chunk.
    splits.sort_by_key(|(pos, _)| *pos);

    let mut spans = Vec::new();
    let mut pos = 0;

    for (splitPos, _kind) in &splits {
        if *splitPos > pos {
            let chunk = &segText[pos..*splitPos];
            let chunkAbs = segAbsStart + pos;
            let s = segStyle(chunkAbs, chunk.len(), phRange, style, pasteStyle);
            if let Some(cursorAbs) = cursorAbsPos {
                if cursorAbs >= chunkAbs && cursorAbs <= chunkAbs + chunk.len() {
                    let local = cursorAbs - chunkAbs;
                    spans.extend(cursorSplitSpans(chunk, local, s, cursorStyle));
                } else {
                    spans.push(Span::styled(chunk.to_string(), s));
                }
            } else {
                spans.push(Span::styled(chunk.to_string(), s));
            }
        }
        pos = *splitPos;
    }

    // Remaining text after last split.
    if pos < segText.len() {
        let chunk = &segText[pos..];
        let chunkAbs = segAbsStart + pos;
        let s = segStyle(chunkAbs, chunk.len(), phRange, style, pasteStyle);
        if let Some(cursorAbs) = cursorAbsPos {
            if cursorAbs >= chunkAbs && cursorAbs <= chunkAbs + chunk.len() {
                let local = cursorAbs - chunkAbs;
                spans.extend(cursorSplitSpans(chunk, local, s, cursorStyle));
            } else {
                spans.push(Span::styled(chunk.to_string(), s));
            }
        } else {
            spans.push(Span::styled(chunk.to_string(), s));
        }
    }

    spans
}

/// Determine the style for a chunk based on whether it falls inside the placeholder range.
fn segStyle(
    absStart: usize,
    len: usize,
    phRange: Option<(usize, usize)>,
    normalStyle: Style,
    pasteStyle: Style,
) -> Style {
    if let Some((phStart, phEnd)) = phRange {
        if absStart >= phStart && absStart + len <= phEnd {
            return pasteStyle;
        }
    }
    normalStyle
}

/// Split a text chunk at the cursor position into [before, cursorChar, after] spans.
fn cursorSplitSpans(
    text: &str,
    localCursor: usize,
    style: Style,
    cursorStyle: Style,
) -> Vec<Span<'static>> {
    let before = &text[..localCursor];
    let (cursorChar, afterStart) = if localCursor < text.len() {
        // Find next grapheme boundary.
        let mut gc = unicode_segmentation::GraphemeCursor::new(localCursor, text.len(), true);
        let next = gc
            .next_boundary(text, 0)
            .ok()
            .flatten()
            .unwrap_or(text.len());
        (&text[localCursor..next], next)
    } else {
        (" ", text.len())
    };
    let after = &text[afterStart..];

    let mut spans = Vec::new();
    if !before.is_empty() {
        spans.push(Span::styled(before.to_string(), style));
    }
    spans.push(Span::styled(cursorChar.to_string(), cursorStyle));
    if !after.is_empty() {
        spans.push(Span::styled(after.to_string(), style));
    }
    spans
}

#[derive(Clone, Copy)]
enum SplitKind {
    PasteStart,
    PasteEnd,
}

/// Byte offset of the start of the Nth logical line (0-indexed) in the given text.
fn logicalLineByteStartIn(text: &str, lineIdx: usize) -> usize {
    if lineIdx == 0 {
        return 0;
    }
    let mut count = 0;
    for (i, ch) in text.char_indices() {
        if ch == '\n' {
            count += 1;
            if count == lineIdx {
                return i + 1;
            }
        }
    }
    text.len()
}

/// A wrapped segment of a logical line.
struct WrapSegment {
    /// Byte offset within the logical line.
    byteStart: usize,
    byteEnd: usize,
    /// Whether this is the last segment of the logical line.
    isLast: bool,
}

/// Break a logical line into wrapped segments by display width.
///
/// Prefers breaking at whitespace boundaries. Falls back to
/// character-level breaking when a single word exceeds the width.
fn wrapSegmentsFor(line: &str, width: usize) -> Vec<WrapSegment> {
    let mut segments = Vec::new();
    let mut segStart = 0;
    let mut colWidth = 0;
    let mut lastBreak: Option<usize> = None;

    for (i, ch) in line.char_indices() {
        if ch.is_whitespace() && ch != '\n' {
            lastBreak = Some(i + ch.len_utf8());
        }

        let charW = unicode_display_width(ch);
        if colWidth + charW > width && segStart < i {
            let breakAt = if let Some(bp) = lastBreak {
                if bp > segStart { bp } else { i }
            } else {
                i
            };

            segments.push(WrapSegment {
                byteStart: segStart,
                byteEnd: breakAt,
                isLast: false,
            });
            segStart = breakAt;
            colWidth = line[breakAt..i + ch.len_utf8()]
                .chars()
                .map(unicode_display_width)
                .sum::<usize>();
            lastBreak = None;
        } else {
            colWidth += charW;
        }
    }
    segments.push(WrapSegment {
        byteStart: segStart,
        byteEnd: line.len(),
        isLast: true,
    });
    segments
}

/// Character display width (CJK = 2, most others = 1).
pub(crate) fn unicode_display_width(c: char) -> usize {
    // NOTE: Simplified — covers common cases. Full solution would use unicode-width crate.
    if c == '\t' {
        4
    } else if c.is_control() || (0xFE00..=0xFE0F).contains(&(c as u32)) {
        // Control chars and variation selectors are zero-width.
        0
    } else if (0x1100..=0x115F).contains(&(c as u32))
        || (0x2E80..=0xA4CF).contains(&(c as u32))
        || (0xAC00..=0xD7AF).contains(&(c as u32))
        || (0xF900..=0xFAFF).contains(&(c as u32))
        || (0xFE10..=0xFE6F).contains(&(c as u32))
        || (0xFF01..=0xFF60).contains(&(c as u32))
        || (0xFFE0..=0xFFE6).contains(&(c as u32))
        || (0x20000..=0x2FFFD).contains(&(c as u32))
        || (0x30000..=0x3FFFD).contains(&(c as u32))
    {
        2
    } else {
        1
    }
}
