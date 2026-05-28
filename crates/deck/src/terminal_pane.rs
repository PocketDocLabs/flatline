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
        }
        true
    }

    /// Set the active (visible) tab. No-op if `name` doesn't exist.
    pub fn setActive(&mut self, name: &str) -> bool {
        if !self.entries.contains_key(name) {
            return false;
        }
        self.active = name.into();
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
    /// Records click-hit rects into `self.lastTabRects` and `self.lastPlusRect`.
    pub fn renderTabBar(&mut self, area: Rect, buf: &mut Buffer, focused: bool) -> u16 {
        if area.height == 0 || area.width == 0 {
            return 0;
        }
        self.lastTabRects.clear();
        self.lastPlusRect = Rect::default();

        let baseStyle = if focused {
            Style::default().fg(Color::White)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let activeStyle = baseStyle.add_modifier(Modifier::BOLD).fg(Color::Cyan);

        let mut x = area.x;
        let yRow = area.y;

        for name in &self.order {
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

            if x + w + 1 > area.x + area.width {
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

            // Tab separator (skip after the last tab — replaced by [+]).
            x += w;
            let sepSpan = Span::styled("\u{2502}", baseStyle);
            buf.set_span(x, yRow, &sepSpan, 1);
            x += 1;
        }

        // [+] add button.
        let plusLabel = " + ";
        let plusW = plusLabel.chars().count() as u16;
        if x + plusW <= area.x + area.width {
            let plusSpan = Span::styled(
                plusLabel,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            );
            buf.set_span(x, yRow, &plusSpan, plusW);
            self.lastPlusRect = Rect {
                x,
                y: yRow,
                width: plusW,
                height: 1,
            };
        }

        1
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
        self.lastTabRects
            .iter()
            .any(|(_, r)| r.contains((col, row).into()))
    }
}
