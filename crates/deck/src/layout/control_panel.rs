#![allow(non_snake_case)]

//! Ctrl+O layout control panel.
//!
//! Slice 6d ships intentionally narrow: select among built-in presets,
//! preview the resulting shape as an ASCII diagram, apply live, save
//! to disk on demand. Arbitrary split/move/group/tab editing is
//! deferred to a later slice.
//!
//! # Public API
//! - [`ControlPanel`] — panel state + render/key handling
//! - [`PanelAction`] — caller-visible side effects (apply, save, etc.)
//! - [`builtinPresets`] — the shipped presets
//!
//! # Dependencies
//! `ratatui`, `crossterm`, `super::Layout`

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};

use super::{Layout, Orient, WindowId};

/// Built-in preset name + corresponding layout tree.
pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    pub layout: Layout,
}

/// If `layout` matches one of the built-in presets by top-level
/// shape, return its name. Used by app.rs to label what the user is
/// currently looking at. Terminal contents are ignored — they're
/// reconciled per-frame from termPane and not part of the preset
/// identity. Matching tolerance on ratio is 0.01 (1%).
pub fn matchPresetName(layout: &Layout) -> Option<String> {
    for preset in builtinPresets() {
        if layoutsMatchShape(&preset.layout, layout) {
            return Some(preset.name.into());
        }
    }
    None
}

fn layoutsMatchShape(a: &Layout, b: &Layout) -> bool {
    // Preset identity is: same orient, same ratio (within 1%), same
    // child kinds on both sides. Terminal Tabs content doesn't have to
    // match — runtime spawns/closes change the Tabs children freely.
    match (a, b) {
        (
            Layout::Split { orient: oa, ratio: ra, a: aa, b: ab },
            Layout::Split { orient: ob, ratio: rb, a: ba, b: bb },
        ) => {
            oa == ob
                && (ra - rb).abs() < 0.01
                && sameChildKind(aa, ba)
                && sameChildKind(ab, bb)
        }
        _ => false,
    }
}

fn sameChildKind(a: &Layout, b: &Layout) -> bool {
    match (a, b) {
        (Layout::Tabs { .. }, Layout::Tabs { .. }) => true,
        (Layout::Window(WindowId::AgentPanel), Layout::Window(WindowId::AgentPanel)) => true,
        (Layout::Window(WindowId::Terminal(_)), Layout::Window(WindowId::Terminal(_))) => true,
        _ => false,
    }
}

/// Return the built-in preset list. All three presets use a Tabs
/// container for terminals so the per-frame terminal-list
/// reconciliation in app.rs picks up `Ctrl+T` / `terminalSpawn` adds.
/// Future slices can add presets with custom geometry once the
/// recursive layout mutator UI lands.
pub fn builtinPresets() -> Vec<Preset> {
    let tabsOfMain = || Layout::Tabs {
        active: 0,
        children: vec![Layout::Window(WindowId::Terminal("main".into()))],
    };
    let agent = || Layout::Window(WindowId::AgentPanel);

    vec![
        Preset {
            name: "split",
            description: "60/40 horizontal — terminals left, agent right (default)",
            layout: Layout::Split {
                orient: Orient::Horizontal,
                ratio: 0.6,
                a: Box::new(tabsOfMain()),
                b: Box::new(agent()),
            },
        },
        Preset {
            name: "wide",
            description: "50/50 horizontal — more room for the agent panel",
            layout: Layout::Split {
                orient: Orient::Horizontal,
                ratio: 0.5,
                a: Box::new(tabsOfMain()),
                b: Box::new(agent()),
            },
        },
        Preset {
            name: "focus",
            description: "80/20 horizontal — narrow agent, focus on the terminal",
            layout: Layout::Split {
                orient: Orient::Horizontal,
                ratio: 0.8,
                a: Box::new(tabsOfMain()),
                b: Box::new(agent()),
            },
        },
    ]
}

/// Result of handling a key in the panel.
pub enum PanelAction {
    /// Key consumed, no caller action needed.
    None,
    /// Close the panel.
    Close,
    /// Apply the given preset's layout to the session immediately.
    /// Carries the preset name so the caller can show it in `/layout`.
    ApplyPreset { name: String, layout: Layout },
    /// Save the panel's working layout to disk.
    Save,
    /// Reload layout from disk; if no disk file exists, reset to the
    /// `split` preset. The panel re-syncs its `working` state from the
    /// caller's reply via [`ControlPanel::resetTo`].
    Reset,
}

/// Confirm-overlay state for Esc-with-unsaved-changes.
enum DirtyConfirm {
    Hidden,
    Asking,
}

/// Layout control panel.
pub struct ControlPanel {
    presets: Vec<Preset>,
    selected: usize,
    /// Preset name currently applied to the session (set by ApplyPreset
    /// at the caller after each successful application).
    appliedName: Option<String>,
    /// True if `working` has diverged from on-disk state and has not
    /// yet been saved.
    dirty: bool,
    /// Current working layout — used for the live diagram. Mirrors the
    /// session layout; updated whenever the caller informs us via
    /// `applyApplied`.
    working: Layout,
    confirm: DirtyConfirm,
}

impl ControlPanel {
    /// Create a panel initialized against the session's current layout.
    /// `appliedPreset` is the preset name applied at the latest save
    /// (None when the session is using on-disk or unmodified default).
    pub fn new(current: Layout, appliedPreset: Option<String>) -> Self {
        let presets = builtinPresets();
        // Default selection: the currently-applied preset if known,
        // otherwise `split`.
        let selected = appliedPreset
            .as_deref()
            .and_then(|n| presets.iter().position(|p| p.name == n))
            .unwrap_or(0);
        Self {
            presets,
            selected,
            appliedName: appliedPreset,
            dirty: false,
            working: current,
            confirm: DirtyConfirm::Hidden,
        }
    }

    /// Caller's confirmation that an `ApplyPreset` action was accepted
    /// and the session is now running that preset. Updates the panel's
    /// view of the world.
    pub fn confirmApplied(&mut self, name: String, layout: Layout) {
        self.appliedName = Some(name);
        self.working = layout;
        self.dirty = true;
    }

    /// Caller's confirmation that a `Save` action succeeded. Clears
    /// the dirty flag so the next Esc won't prompt.
    pub fn confirmSaved(&mut self) {
        self.dirty = false;
    }

    /// Caller's reply to a `Reset` action: pass the layout loaded from
    /// disk (or the default if no disk file exists) plus the preset
    /// name it corresponds to (None if no match).
    pub fn resetTo(&mut self, layout: Layout, appliedPreset: Option<String>) {
        self.working = layout;
        self.appliedName = appliedPreset.clone();
        if let Some(name) = appliedPreset {
            if let Some(idx) = self.presets.iter().position(|p| p.name == name) {
                self.selected = idx;
            }
        }
        self.dirty = false;
    }

    /// Handle a key event.
    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        // Confirm overlay intercepts everything: y discards + closes,
        // n/Esc cancels back to the panel.
        if matches!(self.confirm, DirtyConfirm::Asking) {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.confirm = DirtyConfirm::Hidden;
                    return PanelAction::Close;
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.confirm = DirtyConfirm::Hidden;
                    return PanelAction::None;
                }
                _ => return PanelAction::None,
            }
        }

        match key.code {
            KeyCode::Esc => {
                if self.dirty {
                    self.confirm = DirtyConfirm::Asking;
                    PanelAction::None
                } else {
                    PanelAction::Close
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.presets.len() {
                    self.selected += 1;
                }
                PanelAction::None
            }
            KeyCode::Enter => {
                let p = &self.presets[self.selected];
                PanelAction::ApplyPreset {
                    name: p.name.to_string(),
                    layout: p.layout.clone(),
                }
            }
            KeyCode::Char('s') | KeyCode::Char('S') => PanelAction::Save,
            KeyCode::Char('r') | KeyCode::Char('R') => PanelAction::Reset,
            _ => PanelAction::None,
        }
    }

    /// Render the modal popup.
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        // Pop the modal at 60% width, ~14 rows tall, centered.
        let popupWidth = area.width.saturating_mul(60).saturating_div(100).max(60);
        let popupHeight = 14u16.min(area.height);
        let xs = area.x + (area.width.saturating_sub(popupWidth)) / 2;
        let ys = area.y + (area.height.saturating_sub(popupHeight)) / 2;
        let popup = Rect {
            x: xs,
            y: ys,
            width: popupWidth.min(area.width),
            height: popupHeight,
        };

        Clear.render(popup, buf);

        let title = if self.dirty {
            " layout (Ctrl+O) \u{00B7} unsaved "
        } else {
            " layout (Ctrl+O) "
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(title);
        let inner = block.inner(popup);
        block.render(popup, buf);

        // Two columns: left = diagram, right = preset list. ~45/55.
        let leftW = inner.width.saturating_mul(45).saturating_div(100);
        let rightW = inner.width.saturating_sub(leftW + 1);
        let left = Rect { x: inner.x, y: inner.y, width: leftW, height: inner.height.saturating_sub(2) };
        let right = Rect {
            x: inner.x + leftW + 1,
            y: inner.y,
            width: rightW,
            height: inner.height.saturating_sub(2),
        };
        let footer = Rect {
            x: inner.x,
            y: inner.y + inner.height.saturating_sub(1),
            width: inner.width,
            height: 1,
        };

        // -- Left: live diagram of the working layout.
        let diagram = renderDiagram(&self.working, left.width, left.height);
        Paragraph::new(diagram)
            .style(Style::default().fg(Color::Rgb(180, 180, 200)))
            .render(left, buf);

        // -- Right: preset list with selection cursor.
        let mut lines: Vec<Line<'static>> = Vec::with_capacity(self.presets.len() * 2 + 2);
        lines.push(Line::from(Span::styled(
            "Presets",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for (i, p) in self.presets.iter().enumerate() {
            let active = self.appliedName.as_deref() == Some(p.name);
            let cursor = if i == self.selected { "\u{25B8} " } else { "  " };
            let suffix = if active { "  (applied)" } else { "" };
            let style = if i == self.selected {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Rgb(180, 180, 200))
            };
            lines.push(Line::from(vec![
                Span::raw(cursor.to_string()),
                Span::styled(p.name.to_string(), style),
                Span::styled(suffix.to_string(), Style::default().fg(Color::Green)),
            ]));
            lines.push(Line::from(Span::styled(
                format!("    {}", p.description),
                Style::default().fg(Color::Rgb(120, 120, 140)),
            )));
        }
        Paragraph::new(lines).render(right, buf);

        // -- Footer: hotkey legend.
        let legend = " \u{2191}\u{2193} select \u{00B7} Enter apply \u{00B7} [s] save \u{00B7} [r] reset to disk \u{00B7} Esc close ";
        Paragraph::new(legend)
            .style(Style::default().fg(Color::Rgb(140, 140, 160)))
            .render(footer, buf);

        // -- Confirm overlay, if showing.
        if matches!(self.confirm, DirtyConfirm::Asking) {
            renderConfirmOverlay(popup, buf);
        }
    }
}

/// Render a small ASCII diagram representing `layout` into a paragraph
/// body of `width` x `height`. Supports the shapes the slice-6d
/// presets produce (single Split with Tabs|Window or Window|Window).
fn renderDiagram(layout: &Layout, width: u16, height: u16) -> Vec<Line<'static>> {
    // Simplify: paint a rough box-art representation of the top-level
    // split's ratio. Everything else collapses to "<custom>".
    let mut out: Vec<Line<'static>> = Vec::new();
    out.push(Line::from(Span::styled(
        "Current",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    out.push(Line::from(""));

    match layout {
        Layout::Split { orient: Orient::Horizontal, ratio, a, b } => {
            let total = width.saturating_sub(4).max(20);
            let leftW = ((total as f32) * ratio).round() as u16;
            let rightW = total.saturating_sub(leftW);
            let aLabel = labelOf(a);
            let bLabel = labelOf(b);
            let rowsAvail = height.saturating_sub(out.len() as u16 + 2).max(3);
            let boxH = rowsAvail.min(5);
            // Top border
            let top = format!(
                "\u{250C}{}\u{252C}{}\u{2510}",
                "\u{2500}".repeat(leftW.saturating_sub(1) as usize),
                "\u{2500}".repeat(rightW.saturating_sub(1) as usize),
            );
            out.push(Line::from(top));
            for r in 0..boxH {
                let leftLabel = if r == boxH / 2 { aLabel.as_str() } else { "" };
                let rightLabel = if r == boxH / 2 { bLabel.as_str() } else { "" };
                let leftCell = padCenter(leftLabel, leftW.saturating_sub(1));
                let rightCell = padCenter(rightLabel, rightW.saturating_sub(1));
                out.push(Line::from(format!(
                    "\u{2502}{leftCell}\u{2502}{rightCell}\u{2502}"
                )));
            }
            let bottom = format!(
                "\u{2514}{}\u{2534}{}\u{2518}",
                "\u{2500}".repeat(leftW.saturating_sub(1) as usize),
                "\u{2500}".repeat(rightW.saturating_sub(1) as usize),
            );
            out.push(Line::from(bottom));
            out.push(Line::from(Span::styled(
                format!("ratio: {:.0}/{:.0}", ratio * 100.0, (1.0 - ratio) * 100.0),
                Style::default().fg(Color::Rgb(140, 140, 160)),
            )));
        }
        _ => {
            out.push(Line::from(Span::styled(
                "<custom layout>",
                Style::default().fg(Color::Rgb(140, 140, 160)),
            )));
        }
    }

    out
}

fn labelOf(layout: &Layout) -> String {
    match layout {
        Layout::Tabs { children, .. } => {
            let count = children.len();
            if count <= 1 { "terminal".into() } else { format!("terminals ({count})") }
        }
        Layout::Window(WindowId::Terminal(name)) => format!("term: {name}"),
        Layout::Window(WindowId::AgentPanel) => "agent".into(),
        Layout::Split { .. } => "<nested split>".into(),
    }
}

fn padCenter(s: &str, w: u16) -> String {
    if w == 0 { return String::new(); }
    let s = if (s.chars().count() as u16) > w {
        s.chars().take(w as usize).collect()
    } else {
        s.to_string()
    };
    let len = s.chars().count() as u16;
    let pad = w.saturating_sub(len);
    let left = pad / 2;
    let right = pad - left;
    format!("{}{}{}", " ".repeat(left as usize), s, " ".repeat(right as usize))
}

fn renderConfirmOverlay(popupRect: Rect, buf: &mut Buffer) {
    let msg = " Unsaved changes \u{00B7} discard & close? [y/n] ";
    let w = (msg.chars().count() as u16 + 4).min(popupRect.width);
    let h = 3u16.min(popupRect.height);
    let x = popupRect.x + (popupRect.width.saturating_sub(w)) / 2;
    let y = popupRect.y + (popupRect.height.saturating_sub(h)) / 2;
    let rect = Rect { x, y, width: w, height: h };
    Clear.render(rect, buf);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" confirm ");
    let inner = block.inner(rect);
    block.render(rect, buf);
    Paragraph::new(msg).render(inner, buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, crossterm::event::KeyModifiers::NONE)
    }

    #[test]
    fn navigateAndApplyPreset() {
        let mut panel = ControlPanel::new(Layout::defaultPhase1(), None);
        // Default selection = "split" (index 0)
        assert!(matches!(panel.handleKey(key(KeyCode::Down)), PanelAction::None));
        // Now selected = 1 = "wide". Enter applies.
        match panel.handleKey(key(KeyCode::Enter)) {
            PanelAction::ApplyPreset { name, layout } => {
                assert_eq!(name, "wide");
                // Render at 100 wide — wide preset is 50/50.
                let area = ratatui::layout::Rect { x: 0, y: 0, width: 100, height: 30 };
                let areas = layout.computeAreas(area);
                let term = areas.iter().find(|a| matches!(a.window, WindowId::Terminal(_))).unwrap();
                assert_eq!(term.rect.width, 50);
            }
            other => panic!("expected ApplyPreset, got {}", actionName(&other)),
        }
    }

    #[test]
    fn dirtyEscPromptsAndYDiscards() {
        let mut panel = ControlPanel::new(Layout::defaultPhase1(), None);
        panel.confirmApplied("wide".into(), builtinPresets()[1].layout.clone());
        // Esc should not close yet — prompt first.
        assert!(matches!(panel.handleKey(key(KeyCode::Esc)), PanelAction::None));
        // y discards and closes.
        assert!(matches!(panel.handleKey(key(KeyCode::Char('y'))), PanelAction::Close));
    }

    #[test]
    fn dirtyEscNCancelsConfirm() {
        let mut panel = ControlPanel::new(Layout::defaultPhase1(), None);
        panel.confirmApplied("focus".into(), builtinPresets()[2].layout.clone());
        assert!(matches!(panel.handleKey(key(KeyCode::Esc)), PanelAction::None));
        assert!(matches!(panel.handleKey(key(KeyCode::Char('n'))), PanelAction::None));
        // Confirm cleared; subsequent Esc still prompts because still dirty.
        assert!(matches!(panel.handleKey(key(KeyCode::Esc)), PanelAction::None));
    }

    #[test]
    fn cleanEscClosesImmediately() {
        let mut panel = ControlPanel::new(Layout::defaultPhase1(), None);
        // No dirty mutations — Esc should close right away.
        assert!(matches!(panel.handleKey(key(KeyCode::Esc)), PanelAction::Close));
    }

    #[test]
    fn savedClearsDirty() {
        let mut panel = ControlPanel::new(Layout::defaultPhase1(), None);
        panel.confirmApplied("wide".into(), builtinPresets()[1].layout.clone());
        match panel.handleKey(key(KeyCode::Char('s'))) {
            PanelAction::Save => {}
            other => panic!("expected Save, got {}", actionName(&other)),
        }
        panel.confirmSaved();
        // After save, Esc closes without prompting.
        assert!(matches!(panel.handleKey(key(KeyCode::Esc)), PanelAction::Close));
    }

    #[test]
    fn matchPresetNameTolerantOfRuntimeTabsContent() {
        // Spawning extra terminals into the Tabs container should not
        // change the preset name — terminals are runtime state, the
        // preset identity is the geometry.
        let mut live = builtinPresets()[0].layout.clone();
        live.addTerminal("build");
        live.addTerminal("logs");
        assert_eq!(matchPresetName(&live), Some("split".into()));
    }

    #[test]
    fn matchPresetNameRejectsDifferentChildKinds() {
        // Same orient + same ratio but agent on the left instead of
        // Tabs — should not match.
        let bogus = Layout::Split {
            orient: Orient::Horizontal,
            ratio: 0.6,
            a: Box::new(Layout::Window(WindowId::AgentPanel)),
            b: Box::new(Layout::Tabs {
                active: 0,
                children: vec![Layout::Window(WindowId::Terminal("main".into()))],
            }),
        };
        assert_eq!(matchPresetName(&bogus), None);
    }

    #[test]
    fn presetsAllUseTabsContainerForTerminals() {
        // Slice 6d invariant: the per-frame terminal reconciliation in
        // app.rs assumes terminals live in a Tabs container. Every
        // shipped preset must respect that, otherwise spawning a new
        // terminal would silently fail to appear.
        for preset in builtinPresets() {
            match &preset.layout {
                Layout::Split { a, .. } => match a.as_ref() {
                    Layout::Tabs { children, .. } => {
                        assert!(
                            children.iter().any(|c| matches!(
                                c, Layout::Window(WindowId::Terminal(_))
                            )),
                            "preset {} has empty Tabs container",
                            preset.name,
                        );
                    }
                    other => panic!(
                        "preset {} left side must be Tabs, got {:?}",
                        preset.name, other,
                    ),
                },
                other => panic!(
                    "preset {} root must be Split, got {:?}",
                    preset.name, other,
                ),
            }
        }
    }

    fn actionName(a: &PanelAction) -> &'static str {
        match a {
            PanelAction::None => "None",
            PanelAction::Close => "Close",
            PanelAction::ApplyPreset { .. } => "ApplyPreset",
            PanelAction::Save => "Save",
            PanelAction::Reset => "Reset",
        }
    }
}
