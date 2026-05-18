#![allow(non_snake_case)]

//! Window-manager-style layout tree.
//!
//! A `Layout` is a recursive tree of `Split` / `Tabs` / `Window` nodes.
//! The tree is rendered into rectangular regions on top of the existing
//! ratatui frame buffer. Each leaf [`Window`] addresses a logical pane
//! identified by [`WindowId`] (a terminal by name, the agent panel,
//! etc.).
//!
//! Phase 1 ships with a fixed root tree:
//!     Split-h(0.6, Tabs(Window(Terminal("main"))), Window(AgentPanel))
//!
//! Only that canonical shape is renderable today — `isCanonicalPhase1`
//! gatekeeps persisted layouts so a hand-written file with an
//! unsupported shape gets rejected before it can produce invisible
//! panes.
//!
//! # Public API
//! - [`Layout`], [`Orient`], [`WindowId`]
//! - [`SplitArea`] — output of `compute_areas` for a frame
//! - [`Layout::computeAreas`] — flatten the tree into per-window rects
//! - [`Layout::defaultPhase1`] — the canonical phase-1 root
//! - [`Layout::isCanonicalPhase1`] — true iff layout matches the
//!   single shape `app.rs` can render
//!
//! # Dependencies
//! `ratatui`

pub mod discovery;
pub mod control_panel;

use ratatui::layout::{Constraint, Direction, Layout as RatLayout, Rect};
use serde::{Deserialize, Serialize};

/// Direction of a `Split` node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orient {
    Horizontal,
    Vertical,
}

/// Identifier for a leaf window in the tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowId {
    /// A specific terminal by registry name.
    Terminal(String),
    /// The agent conversation panel.
    AgentPanel,
}

/// Recursive layout node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Layout {
    /// Two children laid out along an axis. `ratio` is the fraction of
    /// the container's primary dimension consumed by the first child
    /// (0.0..1.0). Constraints are computed from `ratio` to integer
    /// percentages; a 0.6 split renders as `Percentage(60), Percentage(40)`.
    Split {
        orient: Orient,
        ratio: f32,
        a: Box<Layout>,
        b: Box<Layout>,
    },
    /// N children sharing the same area, with one active at a time.
    /// The deck renders a tab strip and only paints the active child.
    Tabs {
        active: usize,
        children: Vec<Layout>,
    },
    /// A leaf window — addresses a logical pane.
    Window(WindowId),
}

/// Output of `computeAreas` — a flat list of (window, rect) pairs.
/// For `Tabs` only the active child is emitted, since hidden tabs
/// don't need rectangles.
#[derive(Debug, Clone)]
pub struct SplitArea {
    pub window: WindowId,
    pub rect: Rect,
    /// True when this leaf is part of a `Tabs` group; the deck draws
    /// the tab strip on top of the rect's first row.
    pub inTabs: bool,
    /// All siblings in the same `Tabs` group, with their active flag.
    /// Empty when `inTabs` is false. Order matches tree order.
    pub tabSiblings: Vec<(WindowId, bool)>,
}

impl Layout {
    /// The canonical phase-1 layout: terminal tabs left (60%), agent right (40%).
    pub fn defaultPhase1() -> Self {
        Layout::Split {
            orient: Orient::Horizontal,
            ratio: 0.6,
            a: Box::new(Layout::Tabs {
                active: 0,
                children: vec![Layout::Window(WindowId::Terminal("main".into()))],
            }),
            b: Box::new(Layout::Window(WindowId::AgentPanel)),
        }
    }

    /// True iff this tree is the only shape `app.rs` knows how to
    /// render: a horizontal Split with a Tabs container of Terminal
    /// windows on the left and a single AgentPanel window on the right.
    /// Used by discovery to reject hand-written layout files that
    /// would render as invisible panes.
    pub fn isCanonicalPhase1(&self) -> bool {
        let Layout::Split { orient: Orient::Horizontal, ratio, a, b } = self else {
            return false;
        };
        if !(0.05..=0.95).contains(ratio) {
            return false;
        }
        let Layout::Tabs { active, children } = a.as_ref() else {
            return false;
        };
        if children.is_empty() || *active >= children.len() {
            return false;
        }
        if !children
            .iter()
            .all(|c| matches!(c, Layout::Window(WindowId::Terminal(_))))
        {
            return false;
        }
        matches!(b.as_ref(), Layout::Window(WindowId::AgentPanel))
    }

    /// Flatten the tree into per-window rectangles for the given outer area.
    pub fn computeAreas(&self, outer: Rect) -> Vec<SplitArea> {
        let mut out = Vec::new();
        self.computeInto(outer, &mut out);
        out
    }

    fn computeInto(&self, area: Rect, out: &mut Vec<SplitArea>) {
        match self {
            Layout::Split { orient, ratio, a, b } => {
                let pct = (*ratio * 100.0).clamp(5.0, 95.0) as u16;
                let direction = match orient {
                    Orient::Horizontal => Direction::Horizontal,
                    Orient::Vertical => Direction::Vertical,
                };
                let chunks = RatLayout::default()
                    .direction(direction)
                    .constraints([
                        Constraint::Percentage(pct),
                        Constraint::Percentage(100 - pct),
                    ])
                    .split(area);
                a.computeInto(chunks[0], out);
                b.computeInto(chunks[1], out);
            }
            Layout::Tabs { active, children } => {
                if children.is_empty() {
                    return;
                }
                let activeIdx = (*active).min(children.len() - 1);
                let siblings: Vec<(WindowId, bool)> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        // Tabs are flat at this layer — the active child's
                        // window id provides the label. Nested Splits
                        // inside a Tabs container collapse to the first
                        // window in their subtree (rare but legal).
                        Some((firstWindowOf(child)?, i == activeIdx))
                    })
                    .collect();
                let mut tabAreas = Vec::new();
                children[activeIdx].computeInto(area, &mut tabAreas);
                for mut tab in tabAreas {
                    tab.inTabs = true;
                    tab.tabSiblings = siblings.clone();
                    out.push(tab);
                }
            }
            Layout::Window(id) => {
                out.push(SplitArea {
                    window: id.clone(),
                    rect: area,
                    inTabs: false,
                    tabSiblings: Vec::new(),
                });
            }
        }
    }

    /// Find the first `Window` leaf in tree order. Used for tab labels
    /// and focus-fallback when a window is closed.
    pub fn firstWindow(&self) -> Option<&WindowId> {
        match self {
            Layout::Window(id) => Some(id),
            Layout::Split { a, .. } => a.firstWindow(),
            Layout::Tabs { active, children } => {
                let idx = (*active).min(children.len().saturating_sub(1));
                children.get(idx).and_then(|c| c.firstWindow())
            }
        }
    }
}

/// Helper for `Tabs::compute` — extract a representative `WindowId` for
/// a tab's label even if its child is a sub-layout.
fn firstWindowOf(layout: &Layout) -> Option<WindowId> {
    layout.firstWindow().cloned()
}

// ---- Mutators (skeleton for phase 6) ----

impl Layout {
    /// Add or remove a terminal window from the canonical Tabs container.
    /// Phase 1 uses these to reflect ShellRegistry mutations on the
    /// agent's behalf.
    pub fn addTerminal(&mut self, name: &str) {
        if let Some(tabs) = self.findTerminalTabsContainer() {
            // Skip duplicate adds (defensive — registry rejects duplicates).
            let alreadyHas = tabs
                .iter()
                .any(|c| matches!(c, Layout::Window(WindowId::Terminal(n)) if n == name));
            if !alreadyHas {
                tabs.push(Layout::Window(WindowId::Terminal(name.into())));
            }
        }
    }

    /// Remove a named terminal from the canonical Tabs container.
    /// Returns true if the terminal was present and removed.
    pub fn removeTerminal(&mut self, name: &str) -> bool {
        if let Some(tabs) = self.findTerminalTabsContainer() {
            if let Some(pos) = tabs.iter().position(|c| {
                matches!(c, Layout::Window(WindowId::Terminal(n)) if n == name)
            }) {
                tabs.remove(pos);
                return true;
            }
        }
        false
    }

    /// Set the active tab in the canonical Tabs container by terminal name.
    /// Returns true on success.
    pub fn setActiveTerminal(&mut self, name: &str) -> bool {
        if let Some((children, activeRef)) = self.findTerminalTabsParts() {
            if let Some(pos) = children.iter().position(|c| {
                matches!(c, Layout::Window(WindowId::Terminal(n)) if n == name)
            }) {
                *activeRef = pos;
                return true;
            }
        }
        false
    }

    /// Active terminal name in the canonical Tabs container, if any.
    pub fn activeTerminalName(&self) -> Option<&str> {
        let (children, active) = self.findTerminalTabsPartsRef()?;
        match children.get(active)? {
            Layout::Window(WindowId::Terminal(name)) => Some(name.as_str()),
            _ => None,
        }
    }

    /// Walk the tree and return a mutable reference to the first
    /// `Tabs.children` vector that holds at least one Terminal window.
    /// Phase 1: there's exactly one such container.
    fn findTerminalTabsContainer(&mut self) -> Option<&mut Vec<Layout>> {
        match self {
            Layout::Tabs { children, .. } => {
                if children.iter().any(|c| matches!(c, Layout::Window(WindowId::Terminal(_)))) {
                    Some(children)
                } else {
                    None
                }
            }
            Layout::Split { a, b, .. } => {
                if let Some(found) = a.findTerminalTabsContainer() {
                    return Some(found);
                }
                b.findTerminalTabsContainer()
            }
            Layout::Window(_) => None,
        }
    }

    fn findTerminalTabsParts(&mut self) -> Option<(&mut Vec<Layout>, &mut usize)> {
        match self {
            Layout::Tabs { children, active } => {
                if children.iter().any(|c| matches!(c, Layout::Window(WindowId::Terminal(_)))) {
                    Some((children, active))
                } else {
                    None
                }
            }
            Layout::Split { a, b, .. } => {
                if let Some(found) = a.findTerminalTabsParts() {
                    return Some(found);
                }
                b.findTerminalTabsParts()
            }
            Layout::Window(_) => None,
        }
    }

    fn findTerminalTabsPartsRef(&self) -> Option<(&Vec<Layout>, usize)> {
        match self {
            Layout::Tabs { children, active } => {
                if children.iter().any(|c| matches!(c, Layout::Window(WindowId::Terminal(_)))) {
                    Some((children, *active))
                } else {
                    None
                }
            }
            Layout::Split { a, b, .. } => {
                a.findTerminalTabsPartsRef().or_else(|| b.findTerminalTabsPartsRef())
            }
            Layout::Window(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: u16, h: u16) -> Rect {
        Rect { x: 0, y: 0, width: w, height: h }
    }

    #[test]
    fn defaultPhase1Splits60_40() {
        let layout = Layout::defaultPhase1();
        let areas = layout.computeAreas(rect(100, 30));
        assert_eq!(areas.len(), 2);
        let term = &areas[0];
        let agent = &areas[1];
        assert!(matches!(&term.window, WindowId::Terminal(n) if n == "main"));
        assert_eq!(agent.window, WindowId::AgentPanel);
        assert_eq!(term.rect.width, 60);
        assert_eq!(agent.rect.width, 40);
        assert!(term.inTabs);
        assert!(!agent.inTabs);
    }

    #[test]
    fn addRemoveTerminal() {
        let mut layout = Layout::defaultPhase1();
        layout.addTerminal("build");
        layout.addTerminal("logs");
        let areas = layout.computeAreas(rect(100, 30));
        // Tabs only renders the active child; siblings carry the names.
        let active = &areas[0];
        assert_eq!(active.tabSiblings.len(), 3);
        assert!(matches!(&active.tabSiblings[0].0, WindowId::Terminal(n) if n == "main"));
        assert!(matches!(&active.tabSiblings[1].0, WindowId::Terminal(n) if n == "build"));
        assert!(matches!(&active.tabSiblings[2].0, WindowId::Terminal(n) if n == "logs"));
        assert!(layout.removeTerminal("build"));
        assert!(!layout.removeTerminal("nope"));
    }

    #[test]
    fn setActiveTerminal() {
        let mut layout = Layout::defaultPhase1();
        layout.addTerminal("build");
        assert!(layout.setActiveTerminal("build"));
        assert_eq!(layout.activeTerminalName(), Some("build"));
        assert!(!layout.setActiveTerminal("nope"));
    }

    #[test]
    fn defaultPhase1MatchesLegacyHardcodedSplit() {
        // Slice 6b "zero behavior change" guard: the layout tree must
        // produce byte-identical rects to the old hardcoded ratatui
        // call `Layout::default().direction(Horizontal)
        // .constraints([Percentage(60), Percentage(40)]).split(area)`
        // across a range of realistic terminal sizes.
        use ratatui::layout::{Constraint, Direction, Layout as RatLayout};

        for (w, h) in [(80, 24), (100, 30), (132, 50), (200, 75), (40, 20), (300, 100)] {
            let area = Rect { x: 0, y: 0, width: w, height: h };
            let legacy = RatLayout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                .split(area);

            let layout = Layout::defaultPhase1();
            let areas = layout.computeAreas(area);
            let termRect = areas
                .iter()
                .find(|a| matches!(a.window, WindowId::Terminal(_)))
                .map(|a| a.rect)
                .unwrap();
            let agentRect = areas
                .iter()
                .find(|a| a.window == WindowId::AgentPanel)
                .map(|a| a.rect)
                .unwrap();

            assert_eq!(
                termRect, legacy[0],
                "terminal rect mismatch at {w}x{h}: legacy={:?} layout={:?}",
                legacy[0], termRect,
            );
            assert_eq!(
                agentRect, legacy[1],
                "agent rect mismatch at {w}x{h}: legacy={:?} layout={:?}",
                legacy[1], agentRect,
            );
        }
    }

    #[test]
    fn duplicateAddIgnored() {
        let mut layout = Layout::defaultPhase1();
        layout.addTerminal("main");  // already present
        let areas = layout.computeAreas(rect(100, 30));
        let active = &areas[0];
        // Still just one main entry.
        assert_eq!(active.tabSiblings.len(), 1);
    }
}
