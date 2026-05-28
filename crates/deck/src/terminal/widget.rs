//! Ratatui widget that renders terminal emulator state.
//!
//! Uses `alacritty_terminal` to maintain VT state and renders
//! it as a ratatui widget.
//!
//! # Public API
//! - [`Terminal`] — stateful ratatui widget
//! - [`TerminalState`] — VT emulator state
//! - [`CommandRegion`] — tracked command output boundaries
//!
//! # Dependencies
//! `alacritty_terminal`, `ratatui`

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::StatefulWidget,
};

use alacritty_terminal::{
    Term,
    event::VoidListener,
    grid::{Dimensions, Scroll},
    index::{Column, Line},
    term::{Config, TermMode, cell::Flags},
    vte::ansi::{self, Processor},
};

/// Size type that implements `Dimensions` for alacritty_terminal.
struct TermSize {
    columns: usize,
    screenLines: usize,
}

impl Dimensions for TermSize {
    fn total_lines(&self) -> usize {
        self.screenLines
    }

    fn screen_lines(&self) -> usize {
        self.screenLines
    }

    fn columns(&self) -> usize {
        self.columns
    }
}

/// Command output region tracked by OSC 133 shell integration.
///
/// Absolute line indices are stable across scrollback changes.
/// Convert to grid lines via `absoluteLine - historySize`.
#[derive(Debug, Clone)]
pub struct CommandRegion {
    /// Absolute line where command output starts (from OSC 133;C).
    pub outputStart: usize,
    /// Absolute line where command ends (from OSC 133;D), exclusive.
    pub outputEnd: usize,
}

/// Terminal emulator state backed by `alacritty_terminal`.
pub struct TerminalState {
    term: Term<VoidListener>,
    processor: Processor,
    /// Completed command output regions (absolute line indices).
    commandRegions: Vec<CommandRegion>,
    /// Absolute line of the most recent output start (OSC 133;C), not yet closed.
    pendingOutputStart: Option<usize>,
    /// Whether the terminal content changed since the last render.
    /// Consumed by the app loop to trigger a full screen clear.
    dirty: bool,
}

impl TerminalState {
    /// Create a new terminal state.
    ///
    /// Args:
    ///     cols: Column count.
    ///     rows: Row count.
    pub fn new(cols: u16, rows: u16) -> Self {
        let size = TermSize {
            columns: cols as usize,
            screenLines: rows as usize,
        };
        let config = Config::default();
        let term = Term::new(config, &size, VoidListener);

        Self {
            term,
            processor: Processor::new(),
            commandRegions: Vec::new(),
            pendingOutputStart: None,
            dirty: true,
        }
    }

    /// Feed raw PTY output bytes into the terminal emulator.
    ///
    /// Scans for OSC 133 shell integration markers and records
    /// command output boundaries before passing bytes to alacritty.
    pub fn process(&mut self, data: &[u8]) {
        self.dirty = true;
        let mut offset = 0;
        while offset < data.len() {
            if let Some(m) = findOsc133(&data[offset..]) {
                // Process bytes before the marker.
                if m.offset > 0 {
                    self.processor
                        .advance(&mut self.term, &data[offset..offset + m.offset]);
                }
                // Record the marker at the current cursor position.
                self.handleOsc133(m.kind);
                offset += m.offset + m.len;
            } else {
                // No more markers in this chunk.
                self.processor.advance(&mut self.term, &data[offset..]);
                break;
            }
        }
    }

    /// Resize the terminal state.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        let size = TermSize {
            columns: cols as usize,
            screenLines: rows as usize,
        };
        self.term.resize(size);
        self.dirty = true;
    }

    /// Scroll the terminal display up (into scrollback).
    pub fn scrollUp(&mut self, lines: i32) {
        self.term.scroll_display(Scroll::Delta(lines));
    }

    /// Scroll the terminal display down (toward live output).
    pub fn scrollDown(&mut self, lines: i32) {
        self.term.scroll_display(Scroll::Delta(-lines));
    }

    /// Reset scroll to the bottom (live output).
    pub fn scrollToBottom(&mut self) {
        self.term.scroll_display(Scroll::Bottom);
    }

    /// Returns true and clears the dirty flag if the terminal content
    /// changed since the last call. Used by the app loop to decide
    /// when a full screen clear is needed to prevent cursor-drift
    /// artifacts in ratatui's differential renderer.
    pub fn takeDirty(&mut self) -> bool {
        let was = self.dirty;
        self.dirty = false;
        was
    }

    /// Current grid column count.
    pub fn columns(&self) -> usize {
        self.term.grid().columns()
    }

    /// Current grid screen line count.
    pub fn screenLines(&self) -> usize {
        self.term.grid().screen_lines()
    }

    /// Current display offset (0 = at bottom, positive = scrolled up).
    pub fn displayOffset(&self) -> usize {
        self.term.grid().display_offset()
    }

    /// Whether the inner app has enabled bracketed paste mode.
    pub fn bracketedPaste(&self) -> bool {
        self.term.mode().contains(TermMode::BRACKETED_PASTE)
    }

    /// Cursor position within the viewport as (col, row), or None when hidden
    /// or scrolled out of view.
    pub fn cursorViewportPos(&self) -> Option<(u16, u16)> {
        if !self.term.mode().contains(TermMode::SHOW_CURSOR) {
            return None;
        }
        let grid = self.term.grid();
        let displayOffset = grid.display_offset() as i32;
        let row = grid.cursor.point.line.0 + displayOffset;
        if row < 0 || row as usize >= grid.screen_lines() {
            return None;
        }
        let col = grid.cursor.point.column.0;
        Some((col as u16, row as u16))
    }

    /// Number of lines currently in the scrollback buffer.
    fn historySize(&self) -> usize {
        self.term
            .grid()
            .total_lines()
            .saturating_sub(self.term.grid().screen_lines())
    }

    /// Absolute line index of the cursor's current position.
    fn cursorAbsoluteLine(&self) -> usize {
        let cursorLine = self.term.grid().cursor.point.line.0 as usize;
        self.historySize() + cursorLine
    }

    /// Handle an OSC 133 marker at the current cursor position.
    fn handleOsc133(&mut self, kind: u8) {
        let absLine = self.cursorAbsoluteLine();
        match kind {
            b'C' => {
                // Command output starts here.
                self.pendingOutputStart = Some(absLine);
            }
            b'D' => {
                // Command finished. Close the pending region.
                if let Some(start) = self.pendingOutputStart.take() {
                    if absLine > start {
                        self.commandRegions.push(CommandRegion {
                            outputStart: start,
                            outputEnd: absLine,
                        });
                    }
                }
            }
            // A (prompt start) and B (command input start) — not used for selection yet.
            _ => {}
        }
    }

    /// Whether the given grid line is soft-wrapped (continues on the next line).
    ///
    /// Checks the WRAPLINE flag on the last cell of the row.
    pub fn isLineWrapped(&self, gridLine: i32) -> bool {
        let grid = self.term.grid();
        let numCols = grid.columns();
        if numCols == 0 {
            return false;
        }
        let line = &grid[Line(gridLine)];
        line[Column(numCols - 1)].flags.contains(Flags::WRAPLINE)
    }

    /// Find the command output region containing the given grid line.
    ///
    /// Returns the region's grid line bounds (start inclusive, end exclusive).
    /// For in-progress commands (no D marker yet), uses the current cursor as the end.
    pub fn commandRegionAt(&self, gridLine: i32) -> Option<(i32, i32)> {
        let historySize = self.historySize();
        let absLine = (historySize as i32 + gridLine) as usize;

        // Check in-progress command first.
        if let Some(start) = self.pendingOutputStart {
            let cursorAbs = self.cursorAbsoluteLine();
            if absLine >= start && absLine <= cursorAbs {
                let startGrid = start as i32 - historySize as i32;
                let endGrid = cursorAbs as i32 - historySize as i32;
                return Some((startGrid, endGrid));
            }
        }

        // Search completed regions (most recent first).
        for region in self.commandRegions.iter().rev() {
            if absLine >= region.outputStart && absLine < region.outputEnd {
                let startGrid = region.outputStart as i32 - historySize as i32;
                let endGrid = (region.outputEnd as i32 - 1) - historySize as i32;
                return Some((startGrid, endGrid));
            }
        }

        None
    }
}

/// OSC 133 marker found in a byte slice.
struct Osc133Match {
    /// Byte offset of the ESC character in the slice.
    offset: usize,
    /// Total byte length of the sequence (including terminator).
    len: usize,
    /// Marker kind: b'A', b'B', b'C', or b'D'.
    kind: u8,
}

/// Scan a byte slice for the first OSC 133 sequence.
///
/// Looks for `ESC ] 133 ; <kind> [params] BEL` or `ESC ] 133 ; <kind> [params] ST`.
fn findOsc133(data: &[u8]) -> Option<Osc133Match> {
    let needle = b"\x1b]133;";
    if data.len() < needle.len() + 2 {
        return None;
    }

    let mut pos = 0;
    while pos + needle.len() < data.len() {
        if data[pos..].starts_with(needle) {
            let kindIdx = pos + needle.len();
            if kindIdx >= data.len() {
                break;
            }
            let kind = data[kindIdx];

            // Find terminator: BEL (0x07) or ST (ESC \).
            let mut end = kindIdx + 1;
            while end < data.len() {
                if data[end] == 0x07 {
                    return Some(Osc133Match {
                        offset: pos,
                        len: end - pos + 1,
                        kind,
                    });
                }
                if data[end] == 0x1b && end + 1 < data.len() && data[end + 1] == b'\\' {
                    return Some(Osc133Match {
                        offset: pos,
                        len: end - pos + 2,
                        kind,
                    });
                }
                end += 1;
            }
            // Unterminated sequence — skip past the needle to avoid infinite loop.
            pos = kindIdx;
        }
        pos += 1;
    }

    None
}

/// Map an alacritty color to a ratatui color.
fn mapColor(color: &ansi::Color) -> Option<Color> {
    match color {
        ansi::Color::Named(named) => {
            use ansi::NamedColor::*;
            Some(match named {
                Black => Color::Black,
                Red => Color::Red,
                Green => Color::Green,
                Yellow => Color::Yellow,
                Blue => Color::Blue,
                Magenta => Color::Magenta,
                Cyan => Color::Cyan,
                White => Color::White,
                BrightBlack => Color::DarkGray,
                BrightRed => Color::LightRed,
                BrightGreen => Color::LightGreen,
                BrightYellow => Color::LightYellow,
                BrightBlue => Color::LightBlue,
                BrightMagenta => Color::LightMagenta,
                BrightCyan => Color::LightCyan,
                BrightWhite => Color::White,
                // Foreground/background/cursor/dim variants use terminal defaults.
                _ => return None,
            })
        }
        ansi::Color::Spec(rgb) => Some(Color::Rgb(rgb.r, rgb.g, rgb.b)),
        ansi::Color::Indexed(idx) => Some(Color::Indexed(*idx)),
    }
}

/// Map alacritty cell flags to ratatui modifiers.
fn mapFlags(flags: Flags) -> Modifier {
    let mut mods = Modifier::empty();
    if flags.contains(Flags::BOLD) {
        mods |= Modifier::BOLD;
    }
    if flags.contains(Flags::DIM) {
        mods |= Modifier::DIM;
    }
    if flags.contains(Flags::ITALIC) {
        mods |= Modifier::ITALIC;
    }
    if flags.intersects(Flags::ALL_UNDERLINES) {
        mods |= Modifier::UNDERLINED;
    }
    if flags.contains(Flags::INVERSE) {
        mods |= Modifier::REVERSED;
    }
    if flags.contains(Flags::HIDDEN) {
        mods |= Modifier::HIDDEN;
    }
    if flags.contains(Flags::STRIKEOUT) {
        mods |= Modifier::CROSSED_OUT;
    }
    mods
}

/// Ratatui widget that renders the terminal.
pub struct Terminal;

impl StatefulWidget for Terminal {
    type State = TerminalState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let grid = state.term.grid();
        let numLines = grid.screen_lines();
        let numCols = grid.columns();
        let displayOffset = grid.display_offset();

        for row in 0..area.height.min(numLines as u16) {
            // Negative line indices access scrollback history.
            let line = &grid[Line(row as i32 - displayOffset as i32)];
            for col in 0..area.width.min(numCols as u16) {
                let cell = &line[Column(col as usize)];

                // Mark wide char spacer cells as continuations so ratatui
                // knows to skip them during buffer diff rendering.
                if cell.flags.contains(Flags::WIDE_CHAR_SPACER) {
                    if let Some(bufCell) = buf.cell_mut((area.x + col, area.y + row)) {
                        bufCell.reset();
                        bufCell.set_symbol("");
                    }
                    continue;
                }

                // Sanitize control bytes. alacritty_terminal stores some
                // control characters (notably `\t`) in the cell where
                // the cursor sat when they arrived. Passing them through
                // to ratatui makes the host terminal re-interpret them
                // as cursor commands, shifting every following cell on
                // the row and bleeding past the panel border.
                let ch = match cell.c {
                    c if (c as u32) < 0x20 || c == '\u{7f}' => ' ',
                    c => c,
                };

                let mut style = Style::default();
                if let Some(fg) = mapColor(&cell.fg) {
                    style = style.fg(fg);
                }
                if let Some(bg) = mapColor(&cell.bg) {
                    style = style.bg(bg);
                }
                style = style.add_modifier(mapFlags(cell.flags));

                if let Some(bufCell) = buf.cell_mut((area.x + col, area.y + row)) {
                    bufCell.set_char(ch);
                    bufCell.set_style(style);
                }
            }
        }
    }
}
