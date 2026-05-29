#![allow(non_snake_case)]

//! Background-job status panel — interactive popup overlay.
//!
//! Lists unfinished tasks spawned by `shell(runInBackground: true)` by default,
//! with an in-panel toggle for completed/stopped rows.
//! Each row shows id, command preview, age, state, line count. `k` kills
//! the selected task; `Enter` opens an inspect popup that streams the
//! task's stdout/stderr tail with scrolling and autoscroll.
//!
//! # Public API
//! - [`JobsPanel`]
//! - [`PanelAction`]
//! - [`JobsPanel::applyInspectorSnapshot`] — feed fresh output into the inspector
//!
//! # Dependencies
//! `ratatui`, `construct::jobs::{JobInfo, JobState, JobOutputSnapshot}`

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Widget},
};

use construct::jobs::{JobId, JobInfo, JobKind, JobOutputSnapshot, JobState};

// -- Palette ------------------------------------------------------------------
const BG: Color = Color::Rgb(15, 15, 25);
const BG_SELECTED: Color = Color::Rgb(40, 40, 80);
const FG_PRIMARY: Color = Color::White;
const FG_DIM: Color = Color::Rgb(120, 120, 140);
const FG_MUTED: Color = Color::Rgb(80, 80, 100);
const FG_ACCENT: Color = Color::Cyan;
const FG_BORDER: Color = Color::Magenta;
const FG_OK: Color = Color::Green;
const FG_ERR: Color = Color::Red;
const FG_RUNNING: Color = Color::Yellow;

// -- Public API ---------------------------------------------------------------

/// Outcome of a key event.
pub enum PanelAction {
    /// Nothing to propagate, panel stays open.
    None,
    /// Close the panel.
    Close,
    /// Kill the task at the selected row.
    Kill(JobId),
    /// Open the inspect popup for a task. Caller fetches the snapshot via
    /// [`construct::control::TuiRequest::GetTaskOutput`] and feeds it back
    /// with [`JobsPanel::applyInspectorSnapshot`].
    Inspect(JobId),
    /// The inspector's window changed (paged backward/forward) — the
    /// caller should refetch `GetTaskOutput` with the panel's current
    /// `inspectorSinceLine()` so the view updates even for a completed
    /// or idle task.
    Refetch(JobId),
}

/// Inspector subview state — cached output + scroll position.
struct InspectorState {
    id: JobId,
    command: String,
    /// All cached lines (capped by source ring buffer).
    lines: Vec<String>,
    /// First index of `lines` in the task's monotonic line counter.
    firstLine: u64,
    /// Total lines emitted by the task.
    totalLines: u64,
    /// Earliest line still kept in the source ring buffer. Distinguishes
    /// "lines you didn't ask for" (firstLine > earliestBuffered) from
    /// "lines actually evicted from the buffer" (earliestBuffered > 0).
    earliestBuffered: u64,
    state: JobState,
    /// Scroll offset measured in lines from the top of `lines`.
    scrollOffset: usize,
    /// When true, render keeps the bottom of the buffer pinned as new
    /// output arrives. Cleared when the user pages backward.
    autoscroll: bool,
    /// Last rendered viewport height, used by PageUp/PageDown.
    lastViewportRows: usize,
    /// Pinned start line for explicit pagination. When `Some`, the
    /// inspector fetches `taskOutput(sinceLine: N)` instead of the
    /// default tail.
    requestedSinceLine: Option<u64>,
}

/// Top-level panel mode.
enum Mode {
    List,
    Inspect(InspectorState),
}

/// Interactive `/jobs` panel — jobs section + schedules section.
pub struct JobsPanel {
    jobs: Vec<JobInfo>,
    wakes: Vec<construct::wakes::WakeSourceInfo>,
    /// Selected visible row. Hidden finished jobs are never selectable.
    selected: usize,
    scrollOffset: usize,
    lastVisibleCount: usize,
    showFinished: bool,
    mode: Mode,
}

impl JobsPanel {
    pub fn new(jobs: Vec<JobInfo>) -> Self {
        Self {
            jobs,
            wakes: Vec::new(),
            selected: 0,
            scrollOffset: 0,
            lastVisibleCount: 0,
            showFinished: false,
            mode: Mode::List,
        }
    }

    /// Replace the wake list (drives the schedules section).
    pub fn refreshWakes(&mut self, wakes: Vec<construct::wakes::WakeSourceInfo>) {
        self.wakes = wakes;
    }

    /// Update the underlying list while preserving selection by job id.
    /// If the inspector is open on a job that has disappeared from the
    /// new list (e.g. after `/clear` or a session resume rebuilt the
    /// JobPlane), the inspector falls back to the list view rather than
    /// continuing to show a frozen snapshot.
    pub fn refresh(&mut self, jobs: Vec<JobInfo>) {
        let prevId = self.selectedJob().map(|t| t.id);
        self.jobs = jobs;
        if let Some(id) = prevId
            && let Some(pos) = self
                .visibleJobIndexes()
                .iter()
                .position(|idx| self.jobs[*idx].id == id)
        {
            self.selected = pos;
        }
        self.clampSelection();
        if let Mode::Inspect(state) = &self.mode {
            let inspectedId = state.id;
            if !self.jobs.iter().any(|t| t.id == inspectedId) {
                self.mode = Mode::List;
            }
        }
    }

    /// Returns the id of the task currently being inspected, if any.
    /// The app uses this to drive periodic output refresh.
    pub fn inspectorTaskId(&self) -> Option<JobId> {
        match &self.mode {
            Mode::Inspect(state) => Some(state.id),
            Mode::List => None,
        }
    }

    /// Replace the inspector's cached output with a fresh snapshot for
    /// the given task id, only if the snapshot was fetched against the
    /// inspector's current pagination pin (`requestedSinceLine`).
    /// Returns `true` if applied, `false` if rejected (closed inspector,
    /// wrong task, or stale fetch window). Preserves user-driven scroll
    /// position unless autoscroll is active, in which case it pins to
    /// the tail.
    pub fn applyInspectorSnapshot(
        &mut self,
        id: JobId,
        fetchedSinceLine: Option<u64>,
        snap: JobOutputSnapshot,
    ) -> bool {
        let Mode::Inspect(state) = &mut self.mode else {
            return false;
        };
        if state.id != id {
            return false;
        }
        // Race guard: if `fetchedSinceLine` differs from the current
        // `requestedSinceLine`, the user paged after this fetch was
        // issued and the snapshot is for a now-stale window. Drop it so
        // the most-recent paginate decision wins.
        if fetchedSinceLine != state.requestedSinceLine {
            return false;
        }
        state.lines = snap.lines;
        state.firstLine = snap.firstLine;
        state.totalLines = snap.totalLines;
        state.earliestBuffered = snap.earliestBuffered;
        state.state = snap.state;
        state.command = snap.command;
        if state.autoscroll {
            state.scrollOffset = state
                .lines
                .len()
                .saturating_sub(state.lastViewportRows.max(1));
        }
        true
    }

    /// Open the inspector for a specific task id with an initial output
    /// snapshot. If the inspector is already open for the same id, this
    /// just refreshes the cache.
    pub fn applyInspectorOpen(&mut self, id: JobId, snap: JobOutputSnapshot) {
        let viewport = match &self.mode {
            Mode::Inspect(s) => s.lastViewportRows.max(1),
            Mode::List => 1,
        };
        let scrollOffset = snap.lines.len().saturating_sub(viewport);
        self.mode = Mode::Inspect(InspectorState {
            id,
            command: snap.command,
            lines: snap.lines,
            firstLine: snap.firstLine,
            totalLines: snap.totalLines,
            earliestBuffered: snap.earliestBuffered,
            state: snap.state,
            scrollOffset,
            autoscroll: true,
            lastViewportRows: viewport,
            requestedSinceLine: None,
        });
    }

    /// The `sinceLine` the next refresh fetch should pass. `None` means
    /// "tail" — the default for live following; `Some(N)` is set when
    /// the user paged backwards and pinned a starting line.
    pub fn inspectorSinceLine(&self) -> Option<u64> {
        match &self.mode {
            Mode::Inspect(s) => s.requestedSinceLine,
            Mode::List => None,
        }
    }

    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        match &mut self.mode {
            Mode::List => self.handleKeyList(key),
            Mode::Inspect(_) => self.handleKeyInspector(key),
        }
    }

    fn handleKeyList(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Up => {
                self.moveUp();
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.moveDown();
                PanelAction::None
            }
            KeyCode::PageUp => {
                for _ in 0..10 {
                    self.moveUp();
                }
                PanelAction::None
            }
            KeyCode::PageDown => {
                for _ in 0..10 {
                    self.moveDown();
                }
                PanelAction::None
            }
            KeyCode::Char('f') if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.toggleFinished();
                PanelAction::None
            }
            KeyCode::Enter => self
                .selectedJob()
                .map(|t| PanelAction::Inspect(t.id))
                .unwrap_or(PanelAction::None),
            KeyCode::Char('k') if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(t) = self.selectedJob()
                    && matches!(t.state, JobState::Running)
                {
                    return PanelAction::Kill(t.id);
                }
                PanelAction::None
            }
            _ => PanelAction::None,
        }
    }

    fn handleKeyInspector(&mut self, key: KeyEvent) -> PanelAction {
        let Mode::Inspect(state) = &mut self.mode else {
            return PanelAction::None;
        };
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Backspace => {
                self.mode = Mode::List;
                PanelAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                state.autoscroll = false;
                state.scrollOffset = state.scrollOffset.saturating_sub(1);
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let max = state
                    .lines
                    .len()
                    .saturating_sub(state.lastViewportRows.max(1));
                if state.scrollOffset >= max {
                    state.autoscroll = true;
                } else {
                    state.scrollOffset += 1;
                    if state.scrollOffset >= max {
                        state.autoscroll = true;
                    }
                }
                PanelAction::None
            }
            KeyCode::PageUp => {
                let id = state.id;
                let page = state.lastViewportRows.max(1);
                state.autoscroll = false;
                if state.scrollOffset > 0 {
                    state.scrollOffset = state.scrollOffset.saturating_sub(page);
                    PanelAction::None
                } else if state.firstLine > state.earliestBuffered {
                    // At the top of the current slice but earlier lines
                    // are still in the ring — pin a backward sinceLine
                    // AND request a fetch. Without the fetch the view
                    // would not update for completed/idle tasks.
                    let pageU64 = page as u64;
                    let backStart = state
                        .firstLine
                        .saturating_sub(pageU64)
                        .max(state.earliestBuffered);
                    state.requestedSinceLine = Some(backStart);
                    PanelAction::Refetch(id)
                } else {
                    PanelAction::None
                }
            }
            KeyCode::PageDown => {
                let id = state.id;
                let page = state.lastViewportRows.max(1);
                let max = state.lines.len().saturating_sub(page);
                state.scrollOffset = (state.scrollOffset + page).min(max);
                if state.scrollOffset >= max {
                    state.autoscroll = true;
                    // Resume live tail. If we'd been paging back, clear
                    // the pin and refetch to land on the latest output.
                    let wasPaginating = state.requestedSinceLine.is_some();
                    state.requestedSinceLine = None;
                    if wasPaginating {
                        return PanelAction::Refetch(id);
                    }
                }
                PanelAction::None
            }
            KeyCode::Home => {
                let id = state.id;
                state.autoscroll = false;
                state.scrollOffset = 0;
                // Jump to the very start of what's still buffered. The
                // ring's earliest is the floor; lines below it are gone.
                let earliest = state.earliestBuffered;
                if state.firstLine > earliest {
                    state.requestedSinceLine = Some(earliest);
                    PanelAction::Refetch(id)
                } else {
                    PanelAction::None
                }
            }
            KeyCode::End => {
                let id = state.id;
                state.autoscroll = true;
                state.scrollOffset = state
                    .lines
                    .len()
                    .saturating_sub(state.lastViewportRows.max(1));
                let wasPaginating = state.requestedSinceLine.is_some();
                state.requestedSinceLine = None;
                if wasPaginating {
                    return PanelAction::Refetch(id);
                }
                PanelAction::None
            }
            // Killing the inspected task works in inspect mode too.
            KeyCode::Char('K') => {
                if matches!(state.state, JobState::Running) {
                    PanelAction::Kill(state.id)
                } else {
                    PanelAction::None
                }
            }
            _ => PanelAction::None,
        }
    }

    fn moveUp(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            if self.selected < self.scrollOffset {
                self.scrollOffset = self.selected;
            }
        }
    }

    fn moveDown(&mut self) {
        let visibleLen = self.visibleJobIndexes().len();
        if self.selected + 1 < visibleLen {
            self.selected += 1;
            // `lastVisibleCount` is 0 before the first render and could be
            // 0 again on tiny viewports — saturate so `count - 1` doesn't
            // underflow the unsigned subtraction and panic in debug builds.
            let visible = self.lastVisibleCount.max(1);
            if self.selected >= self.scrollOffset + visible {
                self.scrollOffset = self.selected.saturating_sub(visible - 1);
            }
        }
    }

    fn toggleFinished(&mut self) {
        let prevId = self.selectedJob().map(|t| t.id);
        self.showFinished = !self.showFinished;
        if let Some(id) = prevId
            && let Some(pos) = self
                .visibleJobIndexes()
                .iter()
                .position(|idx| self.jobs[*idx].id == id)
        {
            self.selected = pos;
        }
        self.clampSelection();
    }

    fn selectedJob(&self) -> Option<&JobInfo> {
        let indexes = self.visibleJobIndexes();
        indexes
            .get(self.selected)
            .and_then(|idx| self.jobs.get(*idx))
    }

    fn visibleJobIndexes(&self) -> Vec<usize> {
        self.jobs
            .iter()
            .enumerate()
            .filter_map(|(idx, job)| {
                if self.showFinished || !job.state.isTerminal() {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    fn clampSelection(&mut self) {
        let visibleLen = self.visibleJobIndexes().len();
        if visibleLen == 0 {
            self.selected = 0;
            self.scrollOffset = 0;
            return;
        }
        if self.selected >= visibleLen {
            self.selected = visibleLen - 1;
        }
        if self.scrollOffset >= visibleLen {
            self.scrollOffset = visibleLen.saturating_sub(1);
        }
    }

    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        match &self.mode {
            Mode::List => self.renderList(area, buf),
            Mode::Inspect(_) => self.renderInspector(area, buf),
        }
    }

    fn renderList(&mut self, area: Rect, buf: &mut Buffer) {
        let popupRect = centerRect(area, 80, 24);
        ratatui::widgets::Clear.render(popupRect, buf);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FG_BORDER))
            .style(Style::default().bg(BG))
            .title(Span::styled(
                " /jobs ",
                Style::default().fg(FG_ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popupRect);
        block.render(popupRect, buf);

        if inner.height < 4 {
            return;
        }

        let unfinishedCount = self.jobs.iter().filter(|t| !t.state.isTerminal()).count();
        let finishedCount = self.jobs.iter().filter(|t| t.state.isTerminal()).count();
        let header = format!(
            " {} unfinished \u{00B7} {} finished {} \u{00B7} {} schedule{} ",
            unfinishedCount,
            finishedCount,
            if self.showFinished { "shown" } else { "hidden" },
            self.wakes.len(),
            if self.wakes.len() == 1 { "" } else { "s" },
        );
        buf.set_span(
            inner.x,
            inner.y,
            &Span::styled(header, Style::default().fg(FG_DIM)),
            inner.width,
        );

        let sepY = inner.y + 1;
        for x in inner.x..inner.x + inner.width {
            buf.set_span(
                x,
                sepY,
                &Span::styled("\u{2500}", Style::default().fg(FG_MUTED)),
                1,
            );
        }

        // Split the body vertically: jobs on top, schedules below. Give
        // schedules a fixed footprint (1 header + N rows) and jobs the
        // remainder. If wakes is empty, schedules collapses to a single
        // dim line so the section is discoverable.
        let bodyHeight = inner.height.saturating_sub(3);
        let scheduleSectionRows = if self.wakes.is_empty() {
            2u16
        } else {
            (self.wakes.len() as u16 + 1).min(bodyHeight.saturating_sub(3))
        };
        let jobsHeight = bodyHeight.saturating_sub(scheduleSectionRows);
        let rowsArea = Rect {
            x: inner.x,
            y: inner.y + 2,
            width: inner.width,
            height: jobsHeight,
        };
        self.lastVisibleCount = rowsArea.height as usize;
        let visibleIndexes = self.visibleJobIndexes();

        if visibleIndexes.is_empty() {
            let message = if self.jobs.is_empty() {
                "  no jobs \u{2014} call shell with runInBackground: true to start one"
            } else if self.showFinished {
                "  no jobs"
            } else {
                "  no unfinished jobs \u{2014} press f to show finished"
            };
            let msg = Span::styled(message, Style::default().fg(FG_DIM));
            buf.set_span(rowsArea.x, rowsArea.y, &msg, rowsArea.width);
        } else {
            for (visibleIdx, jobIdx) in visibleIndexes
                .iter()
                .copied()
                .enumerate()
                .skip(self.scrollOffset)
                .take(rowsArea.height as usize)
            {
                let task = &self.jobs[jobIdx];
                let row = rowsArea.y + (visibleIdx - self.scrollOffset) as u16;
                let isSelected = visibleIdx == self.selected;
                Self::renderRow(buf, row, rowsArea.x, rowsArea.width, task, isSelected);
            }
        }

        // Schedules section header.
        let schedY = rowsArea.y + jobsHeight;
        let schedHeader = " schedules ";
        buf.set_span(
            inner.x,
            schedY,
            &Span::styled(
                schedHeader,
                Style::default().fg(FG_ACCENT).add_modifier(Modifier::BOLD),
            ),
            inner.width,
        );
        if self.wakes.is_empty() {
            buf.set_span(
                inner.x + 2,
                schedY + 1,
                &Span::styled(
                    "no schedules \u{2014} use scheduleWakeup/cronCreate/fileWatch",
                    Style::default().fg(FG_DIM),
                ),
                inner.width.saturating_sub(2),
            );
        } else {
            for (i, wake) in self
                .wakes
                .iter()
                .enumerate()
                .take(scheduleSectionRows.saturating_sub(1) as usize)
            {
                let row = schedY + 1 + i as u16;
                Self::renderWakeRow(buf, row, inner.x, inner.width, wake);
            }
        }

        let hintY = inner.y + inner.height.saturating_sub(1);
        let finishedHint = if self.showFinished {
            "f hide finished"
        } else {
            "f show finished"
        };
        let hint = format!(
            " \u{2191}\u{2193} select  \u{21B5} inspect  k kill  {finishedHint}  q/Esc close "
        );
        buf.set_span(
            inner.x,
            hintY,
            &Span::styled(hint, Style::default().fg(FG_MUTED)),
            inner.width,
        );
    }

    fn renderWakeRow(
        buf: &mut Buffer,
        y: u16,
        x: u16,
        width: u16,
        wake: &construct::wakes::WakeSourceInfo,
    ) {
        use construct::control::WakeKind;
        let glyph = match wake.kind {
            WakeKind::Delay => "\u{23F2}\u{FE0E}",
            WakeKind::Cron => "\u{23F1}\u{FE0E}",
            WakeKind::FileWatch => "\u{2399}",
            WakeKind::MonitorMatch => "\u{2299}",
            WakeKind::TaskComplete => "\u{25F4}",
        };
        let promptPreview = wake
            .prompt
            .as_deref()
            .filter(|p| !p.is_empty())
            .map(|p| {
                if p.len() > 30 {
                    format!(
                        " \u{2014} {}\u{2026}",
                        &p[..p.char_indices().nth(30).map(|(i, _)| i).unwrap_or(p.len())]
                    )
                } else {
                    format!(" \u{2014} {p}")
                }
            })
            .unwrap_or_default();
        let line = format!(
            "  {glyph} #{} [{}] {}{} \u{00B7} {} fires",
            wake.id,
            wake.kind.asStr(),
            wake.summary,
            promptPreview,
            wake.firesSoFar,
        );
        buf.set_span(
            x,
            y,
            &Span::styled(line, Style::default().fg(FG_PRIMARY)),
            width,
        );
    }

    fn renderRow(buf: &mut Buffer, y: u16, x: u16, width: u16, task: &JobInfo, isSelected: bool) {
        let bg = if isSelected { BG_SELECTED } else { BG };
        let rowStyle = Style::default().fg(FG_PRIMARY).bg(bg);

        // State glyph (running/done/failed). Different "running" glyph for
        // agent threads so /tasks stays focused on subagent-style work.
        let (glyph, glyphStyle) = match (&task.kind, &task.state) {
            (JobKind::Subagent { .. }, JobState::Running) => {
                ("\u{2982}", Style::default().fg(FG_ACCENT).bg(bg))
            }
            (_, JobState::Running) => ("\u{25F4}", Style::default().fg(FG_RUNNING).bg(bg)),
            (_, JobState::Completed { exitCode: 0 }) => {
                ("\u{2713}", Style::default().fg(FG_OK).bg(bg))
            }
            (_, JobState::Completed { .. }) => ("\u{2717}", Style::default().fg(FG_ERR).bg(bg)),
            (_, JobState::Killed) => ("\u{2717}", Style::default().fg(FG_DIM).bg(bg)),
            (_, JobState::Errored(_)) => ("\u{2717}", Style::default().fg(FG_ERR).bg(bg)),
        };

        let age = task
            .completedAt
            .unwrap_or_else(std::time::Instant::now)
            .saturating_duration_since(task.spawnedAt);
        let ageStr = formatAge(age);

        let stateLabel = match &task.state {
            JobState::Running => "running".into(),
            JobState::Completed { exitCode } => format!("exit {exitCode}"),
            JobState::Killed => "killed".into(),
            JobState::Errored(_) => "errored".into(),
        };

        let cmdMax = width.saturating_sub(40) as usize;
        let cmdPreview = if task.command.len() > cmdMax && cmdMax > 4 {
            format!(
                "{}\u{2026}",
                &task.command[..task.command.floor_char_boundary(cmdMax - 1)]
            )
        } else {
            task.command.clone()
        };

        let spans: Vec<Span> = vec![
            Span::styled(format!(" {glyph} "), glyphStyle),
            Span::styled(
                format!("#{:<4}", task.id),
                Style::default().fg(FG_DIM).bg(bg),
            ),
            Span::styled(" ", rowStyle),
            Span::styled(cmdPreview, rowStyle),
        ];
        let leftWidth: u16 = spans.iter().map(|s| s.content.chars().count() as u16).sum();

        let rightText = format!("{stateLabel}  {ageStr} ");
        let rightWidth = rightText.chars().count() as u16;

        for col in x..x + width {
            buf.set_span(col, y, &Span::styled(" ", rowStyle), 1);
        }

        let mut cursor = x;
        for span in &spans {
            let w = span.content.chars().count() as u16;
            buf.set_span(cursor, y, span, w);
            cursor += w;
        }

        if rightWidth + leftWidth + 2 < width {
            let rightStart = x + width - rightWidth;
            buf.set_span(
                rightStart,
                y,
                &Span::styled(rightText, Style::default().fg(FG_DIM).bg(bg)),
                rightWidth,
            );
        }
    }

    fn renderInspector(&mut self, area: Rect, buf: &mut Buffer) {
        // Larger popup for output tail.
        let popupRect = centerRect(area, 100, 30);
        ratatui::widgets::Clear.render(popupRect, buf);

        let Mode::Inspect(state) = &mut self.mode else {
            return;
        };

        let title = format!(" /tasks \u{203A} #{} ", state.id);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FG_BORDER))
            .style(Style::default().bg(BG))
            .title(Span::styled(
                title,
                Style::default().fg(FG_ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popupRect);
        block.render(popupRect, buf);

        if inner.height < 4 {
            return;
        }

        // Header: command preview.
        let cmdMax = inner.width.saturating_sub(2) as usize;
        let cmdShown = if state.command.len() > cmdMax && cmdMax > 4 {
            format!(
                "$ {}\u{2026}",
                &state.command[..state.command.floor_char_boundary(cmdMax - 3)]
            )
        } else {
            format!("$ {}", state.command)
        };
        buf.set_span(
            inner.x,
            inner.y,
            &Span::styled(cmdShown, Style::default().fg(FG_DIM)),
            inner.width,
        );

        // Status line.
        let (stateLabel, stateColor): (String, Color) = match &state.state {
            JobState::Running => ("running".into(), FG_RUNNING),
            JobState::Completed { exitCode: 0 } => ("exit 0".into(), FG_OK),
            JobState::Completed { exitCode } => (format!("exit {exitCode}"), FG_ERR),
            JobState::Killed => ("killed".into(), FG_DIM),
            JobState::Errored(_) => ("errored".into(), FG_ERR),
        };
        // Two cases for "you're not seeing earlier lines":
        //  - earliestBuffered > 0: those lines were genuinely evicted
        //    from the ring buffer; gone for good.
        //  - firstLine > earliestBuffered: lines are still in the ring,
        //    just not in this slice; PgUp pages back to them.
        let dropped = state.earliestBuffered;
        let recoverable = state.firstLine.saturating_sub(state.earliestBuffered);
        let scope = if state.totalLines == 0 {
            String::new()
        } else if state.lines.is_empty() {
            format!("{} total", state.totalLines)
        } else {
            let lastShown = state.firstLine + state.lines.len() as u64 - 1;
            format!(
                "{} total, showing {}-{}",
                state.totalLines, state.firstLine, lastShown,
            )
        };
        let mut hints = Vec::new();
        if dropped > 0 {
            hints.push(format!("{dropped} dropped"));
        }
        if recoverable > 0 {
            hints.push(format!("{recoverable} earlier (PgUp)"));
        }
        let suffix = if hints.is_empty() {
            String::new()
        } else {
            format!(" \u{00B7} {}", hints.join(", "))
        };
        let footnote = format!(" {stateLabel} \u{00B7} {scope}{suffix} ");
        let statusLineY = inner.y + 1;
        buf.set_span(
            inner.x,
            statusLineY,
            &Span::styled(footnote, Style::default().fg(stateColor)),
            inner.width,
        );

        // Separator.
        let sepY = inner.y + 2;
        for x in inner.x..inner.x + inner.width {
            buf.set_span(
                x,
                sepY,
                &Span::styled("\u{2500}", Style::default().fg(FG_MUTED)),
                1,
            );
        }

        // Output rows.
        let rowsArea = Rect {
            x: inner.x,
            y: inner.y + 3,
            width: inner.width,
            height: inner.height.saturating_sub(4),
        };
        state.lastViewportRows = rowsArea.height as usize;

        // Re-clamp scroll on autoscroll (lines may have grown since last apply).
        if state.autoscroll {
            state.scrollOffset = state
                .lines
                .len()
                .saturating_sub(state.lastViewportRows.max(1));
        }

        if state.lines.is_empty() {
            let msg = Span::styled("  (no output yet)", Style::default().fg(FG_DIM));
            buf.set_span(rowsArea.x, rowsArea.y, &msg, rowsArea.width);
        } else {
            let lineMax = rowsArea.width as usize;
            for (rowIdx, line) in state
                .lines
                .iter()
                .skip(state.scrollOffset)
                .take(rowsArea.height as usize)
                .enumerate()
            {
                let y = rowsArea.y + rowIdx as u16;
                let shown = truncateLine(line, lineMax);
                buf.set_span(
                    rowsArea.x,
                    y,
                    &Span::styled(shown, Style::default().fg(FG_PRIMARY).bg(BG)),
                    rowsArea.width,
                );
            }
        }

        // Hint line.
        let hintY = inner.y + inner.height.saturating_sub(1);
        let mode = if state.autoscroll { "tail" } else { "scrolled" };
        let killHint = if matches!(state.state, JobState::Running) {
            "  K kill"
        } else {
            ""
        };
        let hint = format!(
            " {mode} \u{00B7} \u{2191}\u{2193}/jk PgUp/PgDn Home/End scroll{killHint}  q/Esc back ",
        );
        buf.set_span(
            inner.x,
            hintY,
            &Span::styled(hint, Style::default().fg(FG_MUTED)),
            inner.width,
        );
    }
}

fn truncateLine(line: &str, max: usize) -> String {
    if line.chars().count() <= max || max == 0 {
        line.to_string()
    } else if max <= 1 {
        "\u{2026}".to_string()
    } else {
        let mut out: String = line.chars().take(max - 1).collect();
        out.push('\u{2026}');
        out
    }
}

fn formatAge(d: std::time::Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

fn centerRect(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width.saturating_sub(4));
    let h = height.min(area.height.saturating_sub(2));
    let x = area.x + (area.width - w) / 2;
    let y = area.y + (area.height - h) / 2;
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn job(id: JobId, state: JobState) -> JobInfo {
        let now = Instant::now();
        JobInfo {
            id,
            kind: JobKind::Bash,
            command: format!("echo job-{id}"),
            completedAt: state.isTerminal().then_some(now + Duration::from_millis(1)),
            state,
            spawnedAt: now,
            totalLines: 0,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn defaultsToUnfinishedJobsOnly() {
        let panel = JobsPanel::new(vec![
            job(1, JobState::Completed { exitCode: 0 }),
            job(2, JobState::Running),
            job(3, JobState::Killed),
        ]);

        let visible: Vec<JobId> = panel
            .visibleJobIndexes()
            .iter()
            .map(|idx| panel.jobs[*idx].id)
            .collect();
        assert_eq!(visible, vec![2]);
    }

    #[test]
    fn toggleFinishedShowsAllRowsAndEnterUsesVisibleSelection() {
        let mut panel = JobsPanel::new(vec![
            job(1, JobState::Completed { exitCode: 0 }),
            job(2, JobState::Running),
            job(3, JobState::Killed),
        ]);

        match panel.handleKey(key(KeyCode::Enter)) {
            PanelAction::Inspect(id) => assert_eq!(id, 2),
            _ => panic!("expected visible running job to inspect"),
        }

        panel.handleKey(key(KeyCode::Char('f')));
        let visible: Vec<JobId> = panel
            .visibleJobIndexes()
            .iter()
            .map(|idx| panel.jobs[*idx].id)
            .collect();
        assert_eq!(visible, vec![1, 2, 3]);

        // Selection preserves job #2 across the toggle, so Enter still
        // targets the same row after completed jobs become visible.
        match panel.handleKey(key(KeyCode::Enter)) {
            PanelAction::Inspect(id) => assert_eq!(id, 2),
            _ => panic!("expected preserved visible job to inspect"),
        }
    }

    #[test]
    fn refreshClampsWhenSelectedJobBecomesHidden() {
        let mut panel = JobsPanel::new(vec![job(1, JobState::Running), job(2, JobState::Running)]);
        panel.handleKey(key(KeyCode::Down));
        assert_eq!(panel.selectedJob().map(|j| j.id), Some(2));

        panel.refresh(vec![
            job(1, JobState::Running),
            job(2, JobState::Completed { exitCode: 0 }),
        ]);
        assert_eq!(panel.selectedJob().map(|j| j.id), Some(1));
    }
}
