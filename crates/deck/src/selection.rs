//! Pane-confined mouse selection.
//!
//! Tracks text selection state per panel, renders highlights by
//! overriding cell styles in the ratatui Buffer, and extracts
//! selected text for clipboard copy.
//!
//! # Public API
//! - [`PanelId`] — which panel owns a selection
//! - [`Selection`] — start/end coordinates + active flag
//! - [`Click`] — double/triple click detection
//! - [`SelectionState`] — all selection state for the app
//! - [`applyHighlight`] — render selection overlay
//! - [`extractText`] — read selected characters from Buffer
//!
//! # Dependencies
//! `ratatui`

use std::time::Instant;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

const SELECTION_BG: Color = Color::Rgb(60, 60, 120);
const SELECTION_FG: Color = Color::White;

/// Which panel owns the in-progress selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelId {
    Terminal,
    Agent,
    Input,
}

/// Text selection within a panel's content area.
///
/// Row coordinates are grid lines: `screenRow - displayOffset`.
/// Negative values address scrollback history. This makes the
/// selection stable across scroll changes without rebasing.
/// Column coordinates are panel-local (u16).
/// When `rectangular` is true, only the box region is selected (Alt+drag).
#[derive(Debug, Clone)]
pub struct Selection {
    start: (u16, i32),
    end: (u16, i32),
    active: bool,
    rectangular: bool,
}

impl Selection {
    /// Begin a new selection at the given (col, gridLine) coordinates.
    pub fn new(col: u16, gridLine: i32) -> Self {
        Self {
            start: (col, gridLine),
            end: (col, gridLine),
            active: true,
            rectangular: false,
        }
    }

    /// Begin a rectangular (box) selection.
    pub fn newRectangular(col: u16, gridLine: i32) -> Self {
        Self {
            start: (col, gridLine),
            end: (col, gridLine),
            active: true,
            rectangular: true,
        }
    }

    /// Extend the selection endpoint (called during drag).
    pub fn update(&mut self, col: u16, gridLine: i32) {
        self.end = (col, gridLine);
    }

    /// Extend only the row of the endpoint, keeping the column unchanged.
    /// Used during scroll-drag to avoid snapping the column to an edge.
    pub fn extendRow(&mut self, gridLine: i32) {
        self.end.1 = gridLine;
    }

    /// Override selection bounds programmatically (for entry/viewport expansion).
    pub fn setBounds(&mut self, startCol: u16, startGrid: i32, endCol: u16, endGrid: i32) {
        self.start = (startCol, startGrid);
        self.end = (endCol, endGrid);
    }

    /// Grid line of the selection start point.
    pub fn startGridLine(&self) -> i32 {
        self.start.1
    }

    /// Mark the selection as complete.
    pub fn finalize(&mut self) {
        self.active = false;
    }

    /// Whether start and end are the same point (nothing selected).
    pub fn isEmpty(&self) -> bool {
        self.start == self.end
    }

    /// Return (start, end) in canonical order (top-left first).
    pub fn sorted(&self) -> ((u16, i32), (u16, i32)) {
        let (sc, sr) = self.start;
        let (ec, er) = self.end;
        if sr < er || (sr == er && sc as i32 <= ec as i32) {
            ((sc, sr), (ec, er))
        } else {
            ((ec, er), (sc, sr))
        }
    }

    /// Whether a cell at (gridLine, col) falls within the selection.
    pub fn contains(&self, gridLine: i32, col: u16) -> bool {
        let ((sc, sr), (ec, er)) = self.sorted();

        if self.rectangular {
            let minCol = sc.min(ec);
            let maxCol = sc.max(ec);
            gridLine >= sr && gridLine <= er && col >= minCol && col < maxCol
        } else if sr == er {
            gridLine == sr && col >= sc && col < ec
        } else if gridLine == sr {
            col >= sc
        } else if gridLine == er {
            col < ec
        } else {
            gridLine > sr && gridLine < er
        }
    }
}

/// Double/triple/quad click detection.
///
/// Records click position and time. If a click arrives at the same
/// position within 400ms, the count increments. Wraps at 5 back to 1.
pub struct Click {
    positionAndTime: Option<((u16, u16), Instant)>,
    pub count: usize,
}

const CLICK_THRESHOLD_MS: u128 = 400;

impl Click {
    pub fn new() -> Self {
        Self {
            positionAndTime: None,
            count: 0,
        }
    }

    /// Record a click. Returns the new click count (1-4, wraps at 5).
    pub fn record(&mut self, col: u16, row: u16) -> usize {
        let now = Instant::now();
        let pos = (col, row);

        if let Some((lastPos, lastTime)) = self.positionAndTime {
            if lastPos == pos && now.duration_since(lastTime).as_millis() < CLICK_THRESHOLD_MS {
                self.count += 1;
                if self.count > 4 {
                    self.count = 1;
                }
            } else {
                self.count = 1;
            }
        } else {
            self.count = 1;
        }

        self.positionAndTime = Some((pos, now));
        self.count
    }
}

/// All selection state for the app event loop.
pub struct SelectionState {
    /// Which panel owns the in-progress selection (routing invariance).
    pub selectingIn: Option<PanelId>,
    /// Active or finalized selection in the terminal panel.
    pub termSelection: Option<Selection>,
    /// Active or finalized selection in the agent panel.
    pub agentSelection: Option<Selection>,
    /// Click detector for double/triple click.
    pub click: Click,
    /// Terminal panel content rect (updated each frame).
    pub termContentRect: Rect,
    /// Agent panel full rect for hit-testing (includes prefix columns).
    pub agentPanelRect: Rect,
    /// Agent panel content rect (offset past prefix, for selection/highlight).
    pub agentContentRect: Rect,
    /// Input area content rect (updated each frame).
    pub inputContentRect: Rect,
    /// Panel whose selection needs clipboard copy on the next draw.
    pub pendingCopy: Option<PanelId>,
    /// Expand selection to word/line/block boundaries on next draw (panel, click count).
    pub pendingExpand: Option<(PanelId, usize)>,
}

impl SelectionState {
    pub fn new() -> Self {
        Self {
            selectingIn: None,
            termSelection: None,
            agentSelection: None,
            click: Click::new(),
            termContentRect: Rect::default(),
            agentPanelRect: Rect::default(),
            agentContentRect: Rect::default(),
            inputContentRect: Rect::default(),
            pendingCopy: None,
            pendingExpand: None,
        }
    }

    /// Get a mutable reference to the selection for a given panel.
    pub fn selectionForMut(&mut self, panel: PanelId) -> &mut Option<Selection> {
        match panel {
            PanelId::Terminal => &mut self.termSelection,
            PanelId::Agent => &mut self.agentSelection,
            PanelId::Input => unreachable!("Input selection handled by TextArea"),
        }
    }

    /// Hit-test a screen position against panel content rects.
    pub fn hitTest(&self, col: u16, row: u16) -> Option<PanelId> {
        if self.termContentRect.contains((col, row).into()) {
            Some(PanelId::Terminal)
        } else if self.agentPanelRect.contains((col, row).into()) {
            Some(PanelId::Agent)
        } else {
            None
        }
    }

    /// Convert screen coordinates to panel-local coordinates.
    pub fn toLocal(&self, panel: PanelId, col: u16, row: u16) -> (u16, u16) {
        let rect = match panel {
            PanelId::Terminal => self.termContentRect,
            PanelId::Agent => self.agentContentRect,
            PanelId::Input => self.inputContentRect,
        };
        (col.saturating_sub(rect.x), row.saturating_sub(rect.y))
    }

    /// Clamp panel-local coordinates to content bounds.
    pub fn clampLocal(&self, panel: PanelId, col: u16, row: u16) -> (u16, u16) {
        let rect = match panel {
            PanelId::Terminal => self.termContentRect,
            PanelId::Agent => self.agentContentRect,
            PanelId::Input => self.inputContentRect,
        };
        (
            col.min(rect.width.saturating_sub(1)),
            row.min(rect.height.saturating_sub(1)),
        )
    }
}

// --- Rendering and extraction ---

/// Convert a panel-local screen row to a grid line.
pub fn toGridLine(screenRow: u16, displayOffset: u32) -> i32 {
    screenRow as i32 - displayOffset as i32
}

/// Convert a grid line back to a screen row. Returns None if off-screen.
pub fn toScreenRow(gridLine: i32, displayOffset: u32, height: u16) -> Option<u16> {
    let sr = gridLine + displayOffset as i32;
    if sr >= 0 && sr < height as i32 {
        Some(sr as u16)
    } else {
        None
    }
}

/// Apply selection highlighting to cells in the Buffer.
///
/// `displayOffset` is the panel's current scroll position.
pub fn applyHighlight(selection: &Selection, area: Rect, buf: &mut Buffer, displayOffset: u32) {
    if selection.isEmpty() {
        return;
    }

    let selStyle = Style::default().fg(SELECTION_FG).bg(SELECTION_BG);

    for screenRow in 0..area.height {
        let gridLine = toGridLine(screenRow, displayOffset);
        for col in 0..area.width {
            if selection.contains(gridLine, col)
                && let Some(cell) = buf.cell_mut((area.x + col, area.y + screenRow))
            {
                cell.set_style(selStyle);
            }
        }
    }
}

// --- Expansion helpers ---

/// Find word boundaries around a position in the Buffer.
///
/// A word is a contiguous run of alphanumeric or underscore characters.
/// Returns (startCol, endCol) where endCol is exclusive.
pub fn findWordBounds(buf: &Buffer, area: Rect, screenRow: u16, col: u16) -> (u16, u16) {
    let isWordChar = |c: char| c.is_alphanumeric() || c == '_';

    let clickChar = buf
        .cell((area.x + col, area.y + screenRow))
        .map(|c| c.symbol().chars().next().unwrap_or(' '))
        .unwrap_or(' ');

    let matchWord = isWordChar(clickChar);

    let mut startCol = col;
    while startCol > 0 {
        let ch = buf
            .cell((area.x + startCol - 1, area.y + screenRow))
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .unwrap_or(' ');
        if matchWord != isWordChar(ch) {
            break;
        }
        startCol -= 1;
    }

    let mut endCol = col + 1;
    while endCol < area.width {
        let ch = buf
            .cell((area.x + endCol, area.y + screenRow))
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .unwrap_or(' ');
        if matchWord != isWordChar(ch) {
            break;
        }
        endCol += 1;
    }

    (startCol, endCol)
}

/// Find line boundaries for a row (col 0 to full width if non-empty).
pub fn findLineBounds(buf: &Buffer, area: Rect, screenRow: u16) -> (u16, u16) {
    let mut endCol: u16 = 0;
    for col in 0..area.width {
        let ch = buf
            .cell((area.x + col, area.y + screenRow))
            .map(|c| c.symbol().chars().next().unwrap_or(' '))
            .unwrap_or(' ');
        if ch != ' ' {
            endCol = col + 1;
        }
    }
    if endCol > 0 { (0, area.width) } else { (0, 0) }
}

/// Whether a screen row in the Buffer is entirely whitespace.
fn isEmptyRow(buf: &Buffer, area: Rect, screenRow: u16) -> bool {
    for col in 0..area.width {
        if let Some(cell) = buf.cell((area.x + col, area.y + screenRow)) {
            let ch = cell.symbol().chars().next().unwrap_or(' ');
            if ch != ' ' {
                return false;
            }
        }
    }
    true
}

/// Find the contiguous block of non-empty rows around a screen row.
///
/// Walks up and down from the click position until hitting an empty row
/// or the viewport edge. Returns (startScreenRow, endScreenRow) inclusive.
pub fn findBlockBounds(buf: &Buffer, area: Rect, screenRow: u16) -> (u16, u16) {
    let mut startRow = screenRow;
    while startRow > 0 && !isEmptyRow(buf, area, startRow - 1) {
        startRow -= 1;
    }

    let mut endRow = screenRow;
    while endRow + 1 < area.height && !isEmptyRow(buf, area, endRow + 1) {
        endRow += 1;
    }

    (startRow, endRow)
}

/// Expand a selection to word, line, or block boundaries.
///
/// clickCount 2 = word, 3 = line, 4 = contiguous block of non-empty lines.
pub fn expandSelection(
    selection: &mut Selection,
    clickCount: usize,
    buf: &Buffer,
    area: Rect,
    displayOffset: u32,
) {
    let (col, gridLine) = selection.start;
    let screenRow = match toScreenRow(gridLine, displayOffset, area.height) {
        Some(r) => r,
        None => return,
    };

    match clickCount {
        2 => {
            let (sc, ec) = findWordBounds(buf, area, screenRow, col);
            selection.start = (sc, gridLine);
            selection.end = (ec, gridLine);
        }
        3 => {
            let (sc, ec) = findLineBounds(buf, area, screenRow);
            selection.start = (sc, gridLine);
            selection.end = (ec, gridLine);
        }
        4 => {
            let (startRow, endRow) = findBlockBounds(buf, area, screenRow);
            selection.start = (0, toGridLine(startRow, displayOffset));
            selection.end = (area.width, toGridLine(endRow, displayOffset));
        }
        _ => {}
    }
}

/// Copy text to the system clipboard.
pub fn copyToClipboard(text: &str) {
    if text.is_empty() {
        return;
    }
    if let Ok(mut cb) = arboard::Clipboard::new() {
        let _ = cb.set_text(text.to_string());
    }
}
