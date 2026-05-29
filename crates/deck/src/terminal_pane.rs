#![allow(non_snake_case)]

//! Multi-terminal pane: tab strip + active terminal renderer.
//!
//! Owns a registry of `(name → TerminalEntry)` keyed by shell name. Each
//! entry holds a [`construct::shell::ShellIo`] (for keystroke
//! forwarding, output drainage, resize, kill) plus a [`TerminalState`]
//! (the VT-emulated visible grid).
//!
//! Phase 1 pairs with `crate::layout::Layout` to occupy the left side
//! of the default split. The tab strip occupies the first row of the
//! pane area; the rest renders the active terminal via `EmbeddedTerminal`.
//!
//! # Public API
//! - [`TerminalPane`]
//! - [`TabBarStyle`] (bordered vs flat)
//! - [`TabClick`] — outcome of click hit-testing on the tab strip
//!
//! # Dependencies
//! `ratatui`, [`crate::terminal`], `construct::shell::ShellIo`

use std::collections::HashMap;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

use construct::shell::ShellIo;

use crate::terminal::{Terminal as EmbeddedTerminal, TerminalState};

/// Fill character for an active tab marker.
const ACTIVE_GLYPH: &str = "\u{25C9}"; // ◉
const IDLE_GLYPH: &str = "\u{25CB}"; // ○

/// Per-terminal state held inside the pane.
pub struct TerminalEntry {
    pub io: ShellIo,
    pub state: TerminalState,
    /// True when output arrived since the user last looked at this tab.
    pub unread: bool,
}

impl TerminalEntry {
    pub fn new(io: ShellIo, cols: u16, rows: u16) -> Self {
        Self {
            io,
            state: TerminalState::new(cols, rows),
            unread: false,
        }
    }
}

/// Outcome of routing a mouse click to the pane's tab strip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabClick {
    /// Click hit a specific tab — switch focus to it.
    Switch(String),
    /// Click hit the `[+]` add button — spawn a new terminal.
    NewTab,
    /// Click hit the run-history button.
    History,
    /// Click hit a tab's `[×]` close affordance (none currently).
    #[allow(dead_code)]
    Close(String),
    /// Click landed on the strip but no tab matched.
    Empty,
}

/// Multi-terminal container. Owns ShellIo and TerminalState for each
/// shell, keyed by name. Tracks which tab is active for keystroke routing.
pub struct TerminalPane {
    entries: HashMap<String, TerminalEntry>,
    /// Insertion-ordered names, controls tab strip order.
    order: Vec<String>,
    /// Currently focused (visible) tab.
    active: String,
    /// Last-rendered tab strip click rects, parallel to `order` plus the
    /// trailing `[+]` button. Set by `renderTabBar`.
    lastTabRects: Vec<(String, Rect)>,
    lastPlusRect: Rect,
    lastHistoryRect: Rect,
    lastTabBarRect: Rect,
    lastTabScrollRect: Rect,
    historyHovered: bool,
    plusHovered: bool,
    tabScrollStart: usize,
    lastMaxTabScrollStart: usize,
    tabScrollWheel: i8,
    ensureActiveTabVisible: bool,
}

impl TerminalPane {
    pub fn newWithMain(io: ShellIo, cols: u16, rows: u16) -> Self {
        let mut entries = HashMap::new();
        entries.insert("main".into(), TerminalEntry::new(io, cols, rows));
        Self {
            entries,
            order: vec!["main".into()],
            active: "main".into(),
            lastTabRects: Vec::new(),
            lastPlusRect: Rect::default(),
            lastHistoryRect: Rect::default(),
            lastTabBarRect: Rect::default(),
            lastTabScrollRect: Rect::default(),
            historyHovered: false,
            plusHovered: false,
            tabScrollStart: 0,
            lastMaxTabScrollStart: 0,
            tabScrollWheel: 0,
            ensureActiveTabVisible: true,
        }
    }

    /// Insert a new terminal entry. If a terminal with this name already
    /// exists, the existing entry is replaced (caller is responsible for
    /// avoiding clobbers).
    pub fn add(&mut self, name: impl Into<String>, io: ShellIo, cols: u16, rows: u16) {
        let name = name.into();
        if !self.entries.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.entries
            .insert(name.clone(), TerminalEntry::new(io, cols, rows));
    }

    /// Remove a terminal. If the active tab is removed, focus falls back
    /// to the first tab in order. Returns true if a terminal was removed.
    pub fn remove(&mut self, name: &str) -> bool {
        if self.entries.remove(name).is_none() {
            return false;
        }
        self.order.retain(|n| n != name);
        if self.active == name {
            self.active = self.order.first().cloned().unwrap_or_default();
            self.ensureActiveTabVisible = true;
        }
        true
    }

    /// Set the active (visible) tab. No-op if `name` doesn't exist.
    pub fn setActive(&mut self, name: &str) -> bool {
        if !self.entries.contains_key(name) {
            return false;
        }
        self.active = name.into();
        self.ensureActiveTabVisible = true;
        if let Some(entry) = self.entries.get_mut(name) {
            entry.unread = false;
        }
        true
    }

    /// Cycle to the next tab in order. Wraps around.
    #[allow(dead_code)]
    pub fn cycleNext(&mut self) {
        if self.order.is_empty() {
            return;
        }
        let cur = self
            .order
            .iter()
            .position(|n| n == &self.active)
            .unwrap_or(0);
        let next = (cur + 1) % self.order.len();
        let target = self.order[next].clone();
        self.setActive(&target);
    }

    /// Cycle to the previous tab.
    #[allow(dead_code)]
    pub fn cyclePrev(&mut self) {
        if self.order.is_empty() {
            return;
        }
        let cur = self
            .order
            .iter()
            .position(|n| n == &self.active)
            .unwrap_or(0);
        let prev = (cur + self.order.len() - 1) % self.order.len();
        let target = self.order[prev].clone();
        self.setActive(&target);
    }

    /// Jump to tab N (1-indexed for Ctrl+1..9; 0 = first).
    pub fn jumpTo(&mut self, idx: usize) -> bool {
        if let Some(name) = self.order.get(idx).cloned() {
            self.setActive(&name)
        } else {
            false
        }
    }

    pub fn active(&self) -> &str {
        &self.active
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// Mutable access to the active TerminalState. Panics if there is
    /// no active terminal (registry guarantees `main` always exists, so
    /// this should be unreachable in practice).
    pub fn activeStateMut(&mut self) -> &mut TerminalState {
        let active = self.active.clone();
        &mut self
            .entries
            .get_mut(&active)
            .expect("active terminal exists")
            .state
    }

    /// Read-only access to the active state (for cursor positioning, etc.).
    pub fn activeStateRef(&self) -> &TerminalState {
        &self
            .entries
            .get(&self.active)
            .expect("active terminal exists")
            .state
    }

    /// Mutable access to the active ShellIo.
    pub fn activeIo(&mut self) -> &mut ShellIo {
        let active = self.active.clone();
        &mut self
            .entries
            .get_mut(&active)
            .expect("active terminal exists")
            .io
    }

    /// Forward keystrokes to the active terminal's PTY.
    pub fn sendInput(&mut self, bytes: Vec<u8>) {
        let _ = self.activeIo().inputTx.try_send(bytes);
    }

    /// Send Ctrl+C-style kill to the active terminal.
    pub fn sendKill(&mut self) {
        let _ = self.activeIo().killTx.try_send(());
    }

    /// Resize the active terminal's PTY (and its VT). Called when the
    /// pane area changes.
    #[allow(dead_code)]
    pub fn resizeActive(&mut self, cols: u16, rows: u16) {
        let active = self.active.clone();
        if let Some(entry) = self.entries.get_mut(&active) {
            entry.state.resize(cols, rows);
            let _ = entry.io.resizeTx.try_send((cols, rows));
        }
    }

    /// Drain all terminals' output channels. For non-active terminals,
    /// processed bytes still feed the VT (so when the user switches over
    /// they see current state) but `unread` is set to flag the tab.
    /// Returns true when any byte was drained.
    pub fn drainOutputs(&mut self) -> bool {
        let mut any = false;
        let active = self.active.clone();
        for (name, entry) in self.entries.iter_mut() {
            while let Ok(bytes) = entry.io.outputRx.try_recv() {
                entry.state.process(&bytes);
                any = true;
                if name != &active {
                    entry.unread = true;
                }
            }
        }
        any
    }

    /// Forward a resize to every PTY. Each shell renders its own VT at
    /// the same dimensions (the entire pane area).
    pub fn resizeAll(&mut self, cols: u16, rows: u16) {
        for entry in self.entries.values_mut() {
            if entry.state.columns() != cols as usize || entry.state.screenLines() != rows as usize
            {
                entry.state.resize(cols, rows);
                let _ = entry.io.resizeTx.try_send((cols, rows));
            }
        }
    }

    /// Render the tab strip at `area` (typically the first row of the
    /// terminal pane). Returns the row height used (always 1 in phase 1).
    /// Records click-hit rects into `self.lastTabRects`, `self.lastPlusRect`,
    /// and `self.lastHistoryRect`.
    pub fn renderTabBar(&mut self, area: Rect, buf: &mut Buffer, focused: bool) -> u16 {
        if area.height == 0 || area.width == 0 {
            return 0;
        }
        self.lastTabRects.clear();
        self.lastPlusRect = Rect::default();
        self.lastHistoryRect = Rect::default();
        self.lastTabBarRect = area;
        self.lastTabScrollRect = Rect::default();

        let baseStyle = if focused {
            Style::default().fg(Color::White)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let activeStyle = baseStyle.add_modifier(Modifier::BOLD).fg(Color::Cyan);

        let yRow = area.y;
        let plusLabel = " + ";
        let plusW = plusLabel.chars().count() as u16;
        let histLabel = " \u{25F7} ";
        let histW = histLabel.chars().count() as u16;
        let historyFits = area.width > histW;
        let histX = if historyFits {
            Some(area.x + area.width.saturating_sub(histW))
        } else {
            None
        };
        let actionStart = histX.unwrap_or(area.x + area.width);
        let tabStripRight = actionStart.saturating_sub(u16::from(actionStart > area.x));
        let plusFits = tabStripRight.saturating_sub(area.x) >= plusW;
        let tabCapacity = tabStripRight
            .saturating_sub(area.x)
            .saturating_sub(if plusFits { plusW } else { 0 });

        let tabWidths: Vec<u16> = self
            .order
            .iter()
            .map(|name| Self::tabUnitWidth(name))
            .collect();
        let totalTabWidth: u16 = tabWidths.iter().copied().sum();
        let overflow = totalTabWidth > tabCapacity && !self.order.is_empty();
        self.syncTabScrollStart(&tabWidths, tabCapacity);

        let mut x = area.x;
        for name in self.order.iter().skip(self.tabScrollStart) {
            let isActive = name == &self.active;
            let entry = match self.entries.get(name) {
                Some(e) => e,
                None => continue,
            };
            let glyph = if isActive {
                ACTIVE_GLYPH
            } else if entry.unread {
                "\u{2299}" // ⊙ unread events
            } else {
                IDLE_GLYPH
            };
            let label = format!(" {glyph} {name} ");
            let w = label.chars().count() as u16;

            if x + w > area.x + tabCapacity {
                break;
            }
            let style = if isActive { activeStyle } else { baseStyle };
            let span = Span::styled(label.clone(), style);
            buf.set_span(x, yRow, &span, w);
            self.lastTabRects.push((
                name.clone(),
                Rect {
                    x,
                    y: yRow,
                    width: w,
                    height: 1,
                },
            ));

            x += w;
            if x < area.x + tabCapacity {
                let sepSpan = Span::styled("\u{2502}", baseStyle);
                buf.set_span(x, yRow, &sepSpan, 1);
                x += 1;
            }
        }

        if overflow {
            self.renderTabOverflowScrollbar(area, buf, focused, &tabWidths, tabCapacity);
        }

        // Keep the add button immediately after the visible tab run. The tab
        // viewport reserves room for it, so overflow cannot make it disappear.
        if plusFits {
            let plusStyle = if self.plusHovered {
                Style::default()
                    .fg(Color::Cyan)
                    .bg(Color::Rgb(48, 55, 68))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            };
            let plusSpan = Span::styled(plusLabel, plusStyle);
            buf.set_span(x, yRow, &plusSpan, plusW);
            self.lastPlusRect = Rect {
                x,
                y: yRow,
                width: plusW,
                height: 1,
            };
        }

        // Run history is a terminal-surface affordance, anchored at the far
        // right of the tab strip rather than mixed into the tab list.
        if let Some(histX) = histX {
            let histStyle = if self.historyHovered {
                Style::default()
                    .fg(Color::Cyan)
                    .bg(Color::Rgb(48, 55, 68))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            };
            let histSpan = Span::styled(histLabel, histStyle);
            buf.set_span(histX, yRow, &histSpan, histW);
            self.lastHistoryRect = Rect {
                x: histX,
                y: yRow,
                width: histW,
                height: 1,
            };
        }

        1
    }

    fn tabUnitWidth(name: &str) -> u16 {
        // " ○ name " plus a separator after the tab while more strip space is
        // available. The glyphs are single-cell in the terminal font used here.
        4u16.saturating_add(name.chars().count() as u16)
            .saturating_add(1)
    }

    fn activeTabIndex(&self) -> usize {
        self.order
            .iter()
            .position(|name| name == &self.active)
            .unwrap_or(0)
    }

    fn syncTabScrollStart(&mut self, tabWidths: &[u16], capacity: u16) {
        if self.order.is_empty() {
            self.tabScrollStart = 0;
            self.ensureActiveTabVisible = false;
            return;
        }
        let maxStart = Self::maxUsefulStart(tabWidths, capacity);
        self.lastMaxTabScrollStart = maxStart;
        self.tabScrollStart = self.tabScrollStart.min(maxStart);
        if self.ensureActiveTabVisible {
            let activeIdx = self.activeTabIndex();
            if activeIdx < self.tabScrollStart
                || !Self::rangeFits(tabWidths, self.tabScrollStart, activeIdx, capacity)
            {
                self.tabScrollStart = Self::bestStartForActive(tabWidths, activeIdx, capacity);
            }
            self.ensureActiveTabVisible = false;
        }
        self.tabScrollStart = self.tabScrollStart.min(maxStart);
    }

    fn rangeFits(tabWidths: &[u16], start: usize, end: usize, capacity: u16) -> bool {
        if start > end || end >= tabWidths.len() {
            return false;
        }
        let used: u16 = tabWidths[start..=end].iter().copied().sum();
        used <= capacity
    }

    fn bestStartForActive(tabWidths: &[u16], activeIdx: usize, capacity: u16) -> usize {
        if tabWidths.is_empty() || activeIdx >= tabWidths.len() {
            return 0;
        }
        let mut used = 0u16;
        let mut start = activeIdx;
        for idx in (0..=activeIdx).rev() {
            let next = tabWidths[idx];
            if used.saturating_add(next) > capacity && used > 0 {
                break;
            }
            used = used.saturating_add(next);
            start = idx;
            if used >= capacity {
                break;
            }
        }
        start
    }

    fn maxUsefulStart(tabWidths: &[u16], capacity: u16) -> usize {
        if tabWidths.is_empty() {
            return 0;
        }
        Self::bestStartForActive(tabWidths, tabWidths.len() - 1, capacity)
    }

    fn scrollTabsBackwardImmediate(&mut self) -> bool {
        if self.tabScrollStart == 0 {
            return false;
        }
        self.tabScrollStart -= 1;
        self.ensureActiveTabVisible = false;
        true
    }

    fn scrollTabsForwardImmediate(&mut self) -> bool {
        if self.order.is_empty() {
            return false;
        }
        let next = self.tabScrollStart.saturating_add(1);
        if next > self.lastMaxTabScrollStart {
            return false;
        }
        self.tabScrollStart = next;
        self.ensureActiveTabVisible = false;
        true
    }

    /// Trackpad wheels can send tiny repeated scroll events while the pointer
    /// crosses the tab strip. Require two same-direction ticks before moving
    /// the tab viewport so it feels deliberate instead of twitchy.
    pub fn wheelTabsBackward(&mut self) -> bool {
        self.wheelTabs(-1)
    }

    pub fn wheelTabsForward(&mut self) -> bool {
        self.wheelTabs(1)
    }

    fn wheelTabs(&mut self, direction: i8) -> bool {
        const THRESHOLD: i8 = 2;
        if self.tabScrollWheel.signum() != direction.signum() {
            self.tabScrollWheel = 0;
        }
        self.tabScrollWheel = (self.tabScrollWheel + direction).clamp(-THRESHOLD, THRESHOLD);
        if self.tabScrollWheel.abs() < THRESHOLD {
            return false;
        }
        self.tabScrollWheel = 0;
        if direction < 0 {
            self.scrollTabsBackwardImmediate()
        } else {
            self.scrollTabsForwardImmediate()
        }
    }

    fn renderTabOverflowScrollbar(
        &mut self,
        area: Rect,
        buf: &mut Buffer,
        focused: bool,
        tabWidths: &[u16],
        capacity: u16,
    ) {
        if area.y == 0 || area.width < 18 || capacity == 0 {
            return;
        }
        let total: u16 = tabWidths.iter().copied().sum();
        if total <= capacity {
            return;
        }

        // `area` is the first inner row; the panel's top border is one row up.
        // Reserve the left title gap (" terminal ") and then draw a slim
        // scrollbar directly into the existing outline stroke.
        let trackX = area.x.saturating_add(14);
        // Leave one cell of untouched top-border stroke before the right
        // corner so the thumb never visually runs into the panel joint.
        let trackRight = area.x.saturating_add(area.width.saturating_sub(1));
        if trackX >= trackRight {
            return;
        }
        let trackW = trackRight - trackX;
        if trackW < 8 {
            return;
        }
        self.lastTabScrollRect = Rect {
            x: trackX,
            y: area.y - 1,
            width: trackW,
            height: 1,
        };

        let leftArrowX = trackX;
        let rightArrowX = trackRight - 1;
        // Exclusive right bound. The thumb lives between the arrow cells:
        // [left arrow][thumb travel...][right arrow]. No visual gutter.
        let bodyX = leftArrowX + 1;
        let bodyRight = rightArrowX;
        let bodyW = bodyRight.saturating_sub(bodyX);
        if bodyW < 4 {
            return;
        }

        let viewport = capacity.min(total);
        let thumbW = ((bodyW as u32 * viewport as u32) / total as u32)
            .max(3)
            .min(bodyW as u32) as u16;
        let maxStart = total.saturating_sub(viewport);
        // `tabScrollStart` advances in whole-tab increments, so the summed
        // hidden width can overshoot the continuous scrollbar's max start near
        // the right edge. Clamp before mapping to thumb offset; otherwise the
        // thumb can travel past its lane and collide with the right arrow.
        let startUnits: u16 = tabWidths
            .iter()
            .take(self.tabScrollStart)
            .copied()
            .sum::<u16>()
            .min(maxStart);
        let thumbTravel = bodyW.saturating_sub(thumbW);
        let thumbOffset = if maxStart == 0 {
            0
        } else {
            ((thumbTravel as u32 * startUnits as u32) / maxStart as u32) as u16
        };
        let y = area.y - 1;
        let canScrollLeft = self.tabScrollStart > 0;
        let canScrollRight = self.tabScrollStart < self.lastMaxTabScrollStart;
        let arrowStyle = |enabled: bool| {
            if focused && enabled {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            }
        };
        if let Some(cell) = buf.cell_mut((leftArrowX, y)) {
            cell.set_char(if canScrollLeft {
                '\u{25C0}'
            } else {
                '\u{25C1}'
            });
            cell.set_style(arrowStyle(canScrollLeft));
        }
        if let Some(cell) = buf.cell_mut((rightArrowX, y)) {
            cell.set_char(if canScrollRight {
                '\u{25B6}'
            } else {
                '\u{25B7}'
            });
            cell.set_style(arrowStyle(canScrollRight));
        }
        let thumbStyle = if focused {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD)
        };
        let thumbX = bodyX + thumbOffset;
        for col in thumbX..thumbX + thumbW {
            if let Some(cell) = buf.cell_mut((col, y)) {
                cell.set_char('\u{2501}');
                cell.set_style(thumbStyle);
            }
        }
    }

    /// Update hover state for the right-side terminal action affordances.
    /// Returns true if the state changed and the caller should redraw.
    pub fn setHistoryHover(&mut self, col: u16, row: u16) -> bool {
        let nextHistory = self.lastHistoryRect.contains((col, row).into());
        let nextPlus = self.lastPlusRect.contains((col, row).into());
        if self.historyHovered == nextHistory && self.plusHovered == nextPlus {
            return false;
        }
        self.historyHovered = nextHistory;
        self.plusHovered = nextPlus;
        true
    }

    /// Render the active terminal into `area`. Resizes the active grid
    /// to match if dimensions changed.
    pub fn renderActive(&mut self, area: Rect, buf: &mut Buffer) {
        let active = self.active.clone();
        let entry = match self.entries.get_mut(&active) {
            Some(e) => e,
            None => return,
        };
        if entry.state.columns() != area.width as usize
            || entry.state.screenLines() != area.height as usize
        {
            entry.state.resize(area.width, area.height);
            let _ = entry.io.resizeTx.try_send((area.width, area.height));
        }
        ratatui::widgets::StatefulWidget::render(EmbeddedTerminal, area, buf, &mut entry.state);
    }

    /// Hit-test a click on the tab strip.
    pub fn handleClick(&self, col: u16, row: u16) -> TabClick {
        if self.lastPlusRect.contains((col, row).into()) {
            return TabClick::NewTab;
        }
        if self.lastHistoryRect.contains((col, row).into()) {
            return TabClick::History;
        }
        for (name, rect) in &self.lastTabRects {
            if rect.contains((col, row).into()) {
                return TabClick::Switch(name.clone());
            }
        }
        TabClick::Empty
    }

    /// True when a click landed inside the tab strip rect (any tab or +).
    pub fn clickInTabBar(&self, col: u16, row: u16) -> bool {
        if self.lastPlusRect.contains((col, row).into()) {
            return true;
        }
        if self.lastHistoryRect.contains((col, row).into()) {
            return true;
        }
        self.lastTabRects
            .iter()
            .any(|(_, r)| r.contains((col, row).into()))
    }

    /// True when the coordinate lands anywhere on the rendered tab-strip row.
    pub fn tabBarContains(&self, col: u16, row: u16) -> bool {
        self.lastTabBarRect.contains((col, row).into())
            || self.lastTabScrollRect.contains((col, row).into())
    }
}
