//! Application loop — ratatui event loop with terminal and agent panel.
//!
//! Boots a fullscreen TUI with a shared terminal on the left and an
//! agent conversation panel on the right. Tab switches focus between them.
//!
//! # Public API
//! - [`run`] — starts the app
//!
//! # Dependencies
//! `construct`, `ratatui`, `crossterm`, `tokio`

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal as RatatuiTerminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};
use tokio::sync::{mpsc, oneshot, watch};

use construct::control::{LogEvent, SessionRequest, TuiRequest};
use construct::permissions::Permissions;
use construct::prompt::{DomainModule, InterfaceMode};
use construct::session::Session;
use construct::shell::ShellIo;
use construct::shells::{ShellRegistry, SpawnedBy};

use crate::agent_panel::AgentPanel;
use crate::fork_picker::{ForkAction, ForkPicker};
use crate::log_panel::{DeveloperLog, LogLevel, LogPanel};
use crate::lsp_panel::{LspPanel, PanelAction as LspPanelAction};
use crate::mcp_panel::{McpPanel, PanelAction as McpPanelAction};
use crate::model_panel::{ModelPanel, PanelAction as ModelPanelAction};
use crate::rewind_picker::{RewindAction, RewindPicker};
use crate::selection::{self, PanelId, SelectionState};
use crate::session_picker::{PickerAction, SessionPicker};
use crate::terminal::TerminalState;
use crate::terminal_pane::TerminalPane;
use crate::toast::ToastCenter;

use std::io::{self, Write as _};
use std::time::{Duration, Instant};

/// Resolve main agent permissions from config, falling back to allowReadOnly.
fn mainAgentPermissions(config: &construct::config::Config) -> Permissions {
    config
        .permissions
        .clone()
        .unwrap_or_else(Permissions::allowReadOnly)
}

fn buildModelStatus(config: &construct::config::Config) -> construct::control::ModelStatus {
    let codexStatus = construct::auth::openAiCodexStatus();
    let codexConfigured = codexStatus.configured;
    let saveScope = construct::config::defaultModelSaveScope(config);
    let scopes = construct::config::modelConfigScopes(config)
        .into_iter()
        .filter_map(|scope| {
            let path = construct::config::configPathForScope(
                scope,
                config.projectRoot.as_deref(),
                &config.launchDir,
            )?;
            Some(construct::control::ModelConfigScopeStatus {
                scope,
                label: scope.label().to_string(),
                path: path.display().to_string(),
            })
        })
        .collect::<Vec<_>>();
    let profiles = config
        .profiles
        .iter()
        .map(|(name, model)| {
            let configured = match model.provider.as_str() {
                "openai-codex" => codexConfigured,
                _ => !model.key.is_empty(),
            };
            construct::control::ModelProfileStatus {
                name: name.clone(),
                provider: model.provider.clone(),
                model: model.model.clone(),
                contextWindow: model.contextWindow,
                maxContextWindow: model.maxContextWindow,
                promptThinking: model.promptThinking,
                reasoningEffort: model.reasoning.as_ref().and_then(|r| r.effort.clone()),
                reasoningEfforts: construct::model_catalog::knownModelReasoningEfforts(
                    &model.provider,
                    &model.model,
                ),
                reasoningSummary: model.reasoning.as_ref().and_then(|r| r.summary.clone()),
                configured,
            }
        })
        .collect();
    let configPath = construct::config::configPathForScope(
        saveScope,
        config.projectRoot.as_deref(),
        &config.launchDir,
    )
    .unwrap_or_else(|| construct::config::configDir().join("config.toml"))
    .display()
    .to_string();

    construct::control::ModelStatus {
        heavyProfile: config.heavyProfile.clone(),
        lightProfile: config.lightProfile.clone(),
        utilityProfile: config.utilityProfile.clone(),
        profiles,
        saveScope,
        scopes,
        configPath,
        openAiCodex: codexStatus,
    }
}

/// Which panel has input focus.
#[derive(PartialEq)]
enum Focus {
    Terminal,
    Agent,
}

/// Status-bar items that can be highlighted + activated. Cache is left
/// out — no overlay to open. Cost/Context fire the same requests as the
/// `/cost` and `/context` slash commands.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum StatusChipKind {
    Jobs,
    Cost,
    Context,
}

/// Internal deck-update messages. Produced by tasks that await oneshot replies
/// to `TuiRequest`s; drained by `runLoop` alongside `LogEvent`s so slash
/// commands don't block the TUI while the session task is mid-turn.
enum DeckUpdate {
    McpStatus(construct::control::McpStatus),
    LspStatus(construct::control::LspStatus),
    PermissionsStatus(construct::control::PermissionsStatus),
    ModelStatus(construct::control::ModelStatus),
    ModelCatalog {
        provider: String,
        result: std::result::Result<Vec<construct::model_catalog::ModelCatalogEntry>, String>,
    },
    ContextDisplay(construct::context::ContextState),
    RewindOptions(Vec<construct::transcript::Turn>),
    Forks(Vec<construct::transcript::Fork>),
    /// Generic string result to show in the panel (e.g. cost, session list).
    ShowResult(String),
    /// Result of a mutation (rewind, fork switch, save permissions, clear).
    CommandAck(construct::control::CommandAck),
    /// Resume completed — picker should close on ok, show error on failure.
    ResumeResult(construct::control::CommandAck),
    /// Background-job snapshot for the `/jobs` panel.
    TasksList(Vec<construct::jobs::JobInfo>),
    /// Wake-source snapshot for the `/jobs` panel schedules section.
    WakesList(Vec<construct::wakes::WakeSourceInfo>),
    /// Initial output snapshot for the inspect popup — opens the popup.
    TaskOutputOpen {
        id: construct::jobs::JobId,
        snap: construct::jobs::JobOutputSnapshot,
    },
    /// Periodic refresh while the inspect popup is open. `snap` is None
    /// when the task vanished (e.g. /clear rebuilt the JobPlane); the
    /// app uses that signal to clear the in-flight refresh flag without
    /// applying a snapshot. `sinceLine` is the request parameter the
    /// fetch was issued with — the inspector compares it against its
    /// current `requestedSinceLine` and ignores snapshots whose
    /// pagination window no longer matches (avoids the race where a
    /// pre-paginate tail fetch arrives after the user has paged back).
    TaskOutputRefresh {
        id: construct::jobs::JobId,
        sinceLine: Option<u64>,
        snap: Option<construct::jobs::JobOutputSnapshot>,
    },
}

/// Send a `ListJobs` request to the session and forward the reply
/// through `deckUpdateTx` as a `TasksList` deck update. Only fires when
/// the panel is currently open — no point repopulating if there's no
/// rendering target. Used by `TaskSpawned/Complete/Stopped` handlers to
/// keep the panel fresh without per-event diffing.
fn refreshTasksPanel(
    tasksPanel: &Option<crate::jobs_panel::JobsPanel>,
    requestTx: &mpsc::Sender<TuiRequest>,
    deckUpdateTx: &mpsc::Sender<DeckUpdate>,
) {
    let Some(panel) = tasksPanel else { return };

    let listRequestTx = requestTx.clone();
    let listDeckTx = deckUpdateTx.clone();
    tokio::spawn(async move {
        let (rTx, rRx) = oneshot::channel();
        let _ = listRequestTx
            .send(TuiRequest::ListJobs { reply: rTx })
            .await;
        if let Ok(list) = rRx.await {
            let _ = listDeckTx.send(DeckUpdate::TasksList(list)).await;
        }
    });
    // Same shape for the schedules section (delay/cron/file-watch).
    let wakesRequestTx = requestTx.clone();
    let wakesDeckTx = deckUpdateTx.clone();
    tokio::spawn(async move {
        let (rTx, rRx) = oneshot::channel();
        let _ = wakesRequestTx
            .send(TuiRequest::ListWakes { reply: rTx })
            .await;
        if let Ok(list) = rRx.await {
            let _ = wakesDeckTx.send(DeckUpdate::WakesList(list)).await;
        }
    });

    // Also refresh inspector output if the popup is open. If the user
    // paged backward, their pinned `sinceLine` is honored; otherwise the
    // default tail fetch flows.
    if let Some(id) = panel.inspectorTaskId() {
        let sinceLine = panel.inspectorSinceLine();
        spawnInspectorFetch(requestTx.clone(), deckUpdateTx.clone(), id, sinceLine);
    }
}

fn spawnInspectorFetch(
    requestTx: mpsc::Sender<TuiRequest>,
    deckUpdateTx: mpsc::Sender<DeckUpdate>,
    id: construct::jobs::JobId,
    sinceLine: Option<u64>,
) {
    tokio::spawn(async move {
        let (rTx, rRx) = oneshot::channel();
        let _ = requestTx
            .send(TuiRequest::GetTaskOutput {
                id,
                sinceLine,
                reply: rTx,
            })
            .await;
        // ALWAYS emit a refresh — even when the task is gone (snap=None)
        // — so the runLoop's `inspectorInFlight` flag can clear and the
        // coalescing logic doesn't get stuck suppressing future fetches.
        // Echo `sinceLine` so the inspector can ignore snapshots whose
        // pagination window has been superseded by a newer page key.
        let snap = rRx.await.ok().flatten();
        let _ = deckUpdateTx
            .send(DeckUpdate::TaskOutputRefresh {
                id,
                sinceLine,
                snap,
            })
            .await;
    });
}

#[allow(clippy::too_many_arguments)]
fn pushOperationalLog(
    developerLog: &mut DeveloperLog,
    toastCenter: &mut ToastCenter,
    logPanel: &mut Option<LogPanel>,
    level: LogLevel,
    source: impl Into<String>,
    title: impl Into<String>,
    detail: Option<String>,
    showToast: bool,
) {
    let record = developerLog.push(level, source, title, detail);
    if showToast {
        toastCenter.push(&record);
    }
    if let Some(panel) = logPanel.as_mut() {
        panel.refresh(developerLog.snapshot());
    }
}

/// Axis lock for trackpad scroll — prevents diagonal scrolling.
///
/// Once scrolling starts on one axis, the other axis is ignored until
/// the gesture pauses (no scroll events for `TIMEOUT`).
#[derive(PartialEq, Clone, Copy)]
enum ScrollAxis {
    Vertical,
    Horizontal,
}

struct ScrollAxisLock {
    axis: Option<ScrollAxis>,
    lastEvent: Instant,
}

const SCROLL_LOCK_TIMEOUT_MS: u64 = 150;

impl ScrollAxisLock {
    fn new() -> Self {
        Self {
            axis: None,
            lastEvent: Instant::now(),
        }
    }

    /// Returns true if the given axis is allowed right now.
    fn allow(&mut self, incoming: ScrollAxis) -> bool {
        let now = Instant::now();
        let expired =
            now.duration_since(self.lastEvent).as_millis() > SCROLL_LOCK_TIMEOUT_MS as u128;

        if expired || self.axis.is_none() {
            self.axis = Some(incoming);
            self.lastEvent = now;
            true
        } else if self.axis == Some(incoming) {
            self.lastEvent = now;
            true
        } else {
            // Locked to the other axis — swallow this event.
            false
        }
    }
}

/// Static glyph shown in the title when a topic is set but the agent is idle.
/// Ace of hearts — deliberately distinct from every animation frame so a
/// still title reads as "idle" rather than "animation stuck on a card".
const TITLE_IDLE_GLYPH: &str = "\u{1F0B1}"; // Ace of Hearts

/// Glyph shown when agent completed work while window was not focused.
/// U+FE0E forces text presentation so the title shows a glyph, not an emoji.
const TITLE_UNSEEN_GLYPH: &str = "\u{2709}\u{FE0E}"; // Envelope

/// Glyph shown when a permission prompt is waiting while window was not focused.
const TITLE_PERMIT_GLYPH: &str = "\u{26A0}\u{FE0E}"; // Warning

/// Set the host terminal's window title via OSC 2.
fn writeTerminalTitle(glyph: &str, topic: Option<&str>) {
    let title = match topic {
        Some(t) if !t.is_empty() => format!("{glyph} {t}"),
        _ => "flatline".to_string(),
    };
    let _ = write!(io::stdout(), "\x1b]2;{title}\x07");
    let _ = io::stdout().flush();
}

/// Restore the terminal title to whatever it was (best-effort reset).
fn resetTerminalTitle() {
    let _ = write!(io::stdout(), "\x1b]2;\x07");
    let _ = io::stdout().flush();
}

/// Animated card-flip spinner for the terminal title. Cycles through a
/// pre-built frame list with per-frame dwell times, advanced by a wall-clock
/// tick on the main render loop.
struct TitleSpinner {
    frames: Vec<(&'static str, Duration)>,
    idx: usize,
    lastAdvance: Instant,
}

impl TitleSpinner {
    fn new() -> Self {
        const CARDS: &[&str] = &[
            "\u{1F0A0}", // card back
            "\u{1F0A1}", // ace of spades
            "\u{1F0A0}", // card back
            "\u{1F0DF}", // black joker
        ];
        // Trailing NBSP (not regular space) pads each 1-col bar to 2 cols.
        // The title writer also injects a separator space between glyph and
        // topic — using NBSP here prevents the terminal from collapsing
        // those two consecutive whitespace chars and dropping the topic a
        // column to the left on every non-card frame.
        const BARS: &[&str] = &["\u{2337}", "|\u{00A0}"];
        // Two NBSPs for the edge: leading one survives leading-whitespace
        // stripping, trailing one survives the adjacent-space collapse.
        const EDGE: &str = "\u{00A0}\u{00A0}";
        let cardHold = Duration::from_millis(600);
        let barHold = Duration::from_millis(80);
        let edgeHold = Duration::from_millis(100);
        let mut frames: Vec<(&'static str, Duration)> = Vec::new();
        for card in CARDS {
            frames.push((*card, cardHold));
            for b in BARS {
                frames.push((*b, barHold));
            }
            frames.push((EDGE, edgeHold));
            for b in BARS.iter().rev() {
                frames.push((*b, barHold));
            }
        }
        Self {
            frames,
            idx: 0,
            lastAdvance: Instant::now(),
        }
    }

    fn current(&self) -> &'static str {
        self.frames[self.idx].0
    }

    /// Advance by one frame if the current dwell has elapsed. Returns true
    /// if the frame actually changed (caller should repaint the title).
    fn tick(&mut self) -> bool {
        let dwell = self.frames[self.idx].1;
        if self.lastAdvance.elapsed() >= dwell {
            self.idx = (self.idx + 1) % self.frames.len();
            self.lastAdvance = Instant::now();
            true
        } else {
            false
        }
    }
}

/// Run the deck TUI.
pub async fn run() -> Result<()> {
    // Load config BEFORE touching the terminal — a config error here would
    // otherwise skip the cleanup block and leave the terminal in raw mode
    // with mouse capture still on.
    let config = construct::config::load()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        crossterm::event::EnableBracketedPaste,
        crossterm::event::EnableMouseCapture,
        crossterm::event::PushKeyboardEnhancementFlags(
            crossterm::event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | crossterm::event::KeyboardEnhancementFlags::REPORT_EVENT_TYPES,
        ),
        crossterm::cursor::SetCursorStyle::SteadyBar,
        crossterm::event::EnableFocusChange,
    )?;
    writeTerminalTitle(TITLE_IDLE_GLYPH, None);

    // From here until cleanup, any `?` bypasses terminal restore — wrap
    // fallible setup so errors get cleaned up before propagating.
    let setupResult = (|| -> Result<_> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = RatatuiTerminal::new(backend)?;
        let size = terminal.size()?;
        // Terminal gets ~60% width minus borders, full height minus status bar and borders.
        let termCols = (size.width * 3 / 5).saturating_sub(2);
        let termRows = size.height.saturating_sub(3);
        // Channel for the registry to deliver newly-spawned ShellIos to deck.
        // Tuple carries (name, io, spawnedBy) so the deck can auto-switch
        // to user-initiated tabs without coordinating with LogEvent ordering.
        let (shellIoTx, shellIoRx) = mpsc::channel::<(String, ShellIo, SpawnedBy)>(8);
        let (shells, mainIo) = ShellRegistry::newWithMain(termCols, termRows, shellIoTx)?;
        let shells = std::sync::Arc::new(tokio::sync::Mutex::new(shells));
        let termPane = TerminalPane::newWithMain(mainIo, termCols, termRows);
        Ok((terminal, shells, shellIoRx, termPane, termCols, termRows))
    })();

    let (mut terminal, shells, mut shellIoRx, mut termPane, _termCols, _termRows) =
        match setupResult {
            Ok(v) => v,
            Err(e) => {
                let _ = disable_raw_mode();
                let _ = execute!(
                    io::stdout(),
                    crossterm::cursor::SetCursorStyle::DefaultUserShape,
                    crossterm::event::PopKeyboardEnhancementFlags,
                    crossterm::event::DisableMouseCapture,
                    crossterm::event::DisableFocusChange,
                    crossterm::event::DisableBracketedPaste,
                    LeaveAlternateScreen,
                );
                resetTerminalTitle();
                return Err(e);
            }
        };

    let contextWindow = config.heavy.contextWindow;
    let cachingEnabled = config.heavy.cachingActive();
    tracing::info!(
        model = %config.heavy.model,
        cachingEnabled,
        supportsAnthropicCache = ?config.heavy.supportsAnthropicCache,
        "deck startup cache config"
    );
    let rollingBaseline =
        construct::cost::rollingWindowCost(config.projectRoot.as_deref().and_then(|p| p.to_str()));

    let mut agentPanel = AgentPanel::new();
    let mut focus = Focus::Terminal;
    let mut selState = SelectionState::new();
    let mut scrollLock = ScrollAxisLock::new();
    // Session channels — Log (session → TUI), SessionRequest (session → TUI),
    // TuiRequest (TUI → session), plus user input / cancel / steer.
    let (logTx, mut logRx) = mpsc::channel::<LogEvent>(256);
    let (sessionRequestTx, mut sessionRequestRx) = mpsc::channel::<SessionRequest>(16);
    let (userInputTx, mut userInputRx) = mpsc::channel::<construct::session::UserInput>(16);
    let (requestTx, mut requestRx) = mpsc::channel::<TuiRequest>(16);
    let (cancelTx, cancelRx) = watch::channel(false);
    let (steerTx, steerRx) = mpsc::channel::<construct::session::UserInput>(16);
    let (userBgTx, userBgRx) = mpsc::channel::<()>(4);
    let (deckUpdateTx, mut deckUpdateRx) = mpsc::channel::<DeckUpdate>(32);

    // Spawn the agent session task.
    let mut cancelRx = cancelRx;
    let mut steerRx = steerRx;
    let mut userBgRx = userBgRx;
    let sessionCancelTx = cancelTx.clone();
    tokio::spawn(async move {
        let mut config = match construct::config::load() {
            Ok(c) => c,
            Err(e) => {
                let _ = logTx
                    .send(LogEvent::Error(format!("Config error: {e}")))
                    .await;
                return;
            }
        };

        // Main agent auto-approves read-only tools but still prompts on writes/mutations.
        let permissions = mainAgentPermissions(&config);

        // Deck is the shared terminal harness — SWE domain by default.
        let mut session = match Session::new(
            &config,
            permissions,
            shells,
            InterfaceMode::SharedTerminal,
            &[DomainModule::Swe],
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = logTx
                    .send(LogEvent::Error(format!("Session error: {e}")))
                    .await;
                return;
            }
        };

        // Initialize MCP servers from .mcp.json files.
        match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
            Ok(servers) if !servers.is_empty() => {
                session.initMcp(servers).await;
            }
            Err(e) => tracing::warn!("failed to load MCP config: {e}"),
            _ => {}
        }

        // Take the wake-batch receiver. The session's batcher coalesces
        // wake fires into `WakeBatch` values; we select on this alongside
        // userInputRx so a batch only injects when the session is idle
        // — wakes that arrive mid-turn queue in the receiver and fire
        // immediately after the current turn completes. Slot-replaced
        // on /clear and /resume from the new session.
        let mut wakeBatchRx = session.takeWakeBatchRx();

        // Split the inbound TuiRequest stream so jobs-panel queries can run
        // concurrently with `session.send`. Without this, a JobPlane
        // request stays in the channel until the agent's turn ends,
        // which defeats the whole point of the control panel (inspect/kill
        // the task that's running RIGHT NOW).
        //
        // The slot pattern (`RwLock<Arc<Mutex<JobPlane>>>`) lets us
        // hot-swap the inner plane on /clear and /resume: the session
        // actor writes a new Arc when it rebuilds `session`, the
        // handler reads the current Arc on every request, and the old
        // plane drops as soon as nothing references it (which kills its
        // running tasks via `JobPlane::Drop`).
        let jobPlaneSlot: std::sync::Arc<
            std::sync::RwLock<std::sync::Arc<std::sync::Mutex<construct::jobs::JobPlane>>>,
        > = std::sync::Arc::new(std::sync::RwLock::new(session.jobPlaneHandle()));
        // Same hot-swap pattern for the wake registry — it lives on the
        // session and gets a fresh instance on /clear and /resume.
        let wakeRegistrySlot: std::sync::Arc<
            std::sync::RwLock<std::sync::Arc<tokio::sync::Mutex<construct::wakes::WakeRegistry>>>,
        > = std::sync::Arc::new(std::sync::RwLock::new(session.wakeRegistryHandle()));
        // Shell registry handle stays valid across /clear and /resume
        // because `intoShells()` returns the same Arc that gets passed
        // to the new Session — no slot swap needed.
        let shellsArc = session.shellsHandle();
        let (taskPlaneReqTx, mut taskPlaneReqRx) = mpsc::channel::<TuiRequest>(16);
        let (terminalReqTx, mut terminalReqRx) = mpsc::channel::<TuiRequest>(16);
        let (otherReqTx, mut otherReqRx) = mpsc::channel::<TuiRequest>(16);
        tokio::spawn(async move {
            while let Some(req) = requestRx.recv().await {
                match &req {
                    TuiRequest::ListJobs { .. }
                    | TuiRequest::ListWakes { .. }
                    | TuiRequest::KillTask { .. }
                    | TuiRequest::GetTaskOutput { .. } => {
                        let _ = taskPlaneReqTx.send(req).await;
                    }
                    TuiRequest::SpawnTerminal { .. }
                    | TuiRequest::KillTerminal { .. }
                    | TuiRequest::ListTerminals { .. } => {
                        let _ = terminalReqTx.send(req).await;
                    }
                    _ => {
                        let _ = otherReqTx.send(req).await;
                    }
                }
            }
        });
        let shellsHandler = shellsArc.clone();
        let handlerLogTx = logTx.clone();
        tokio::spawn(async move {
            use construct::shells::SpawnedBy;
            while let Some(req) = terminalReqRx.recv().await {
                match req {
                    TuiRequest::SpawnTerminal { name, reply } => {
                        let result = {
                            let mut guard = shellsHandler.lock().await;
                            guard.spawn(name, SpawnedBy::User).await
                        };
                        match result {
                            Ok(resolved) => {
                                let _ = handlerLogTx
                                    .send(LogEvent::TerminalSpawned {
                                        name: resolved.clone(),
                                        spawnedBy: SpawnedBy::User.into(),
                                    })
                                    .await;
                                let _ = reply.send(Ok(resolved));
                            }
                            Err(e) => {
                                let _ = reply.send(Err(e.to_string()));
                            }
                        }
                    }
                    TuiRequest::KillTerminal { name, reply } => {
                        let result = {
                            let mut guard = shellsHandler.lock().await;
                            guard.kill(&name)
                        };
                        let ack = match result {
                            Ok(()) => {
                                let _ = handlerLogTx
                                    .send(LogEvent::TerminalClosed { name: name.clone() })
                                    .await;
                                construct::control::CommandAck::ok(format!(
                                    "Killed terminal '{name}'."
                                ))
                            }
                            Err(e) => construct::control::CommandAck::err(e.to_string()),
                        };
                        let _ = reply.send(ack);
                    }
                    TuiRequest::ListTerminals { reply } => {
                        let list = shellsHandler.lock().await.list();
                        let _ = reply.send(list);
                    }
                    _ => {}
                }
            }
        });
        let handlerSlot = jobPlaneSlot.clone();
        let wakeSlotForHandler = wakeRegistrySlot.clone();
        tokio::spawn(async move {
            while let Some(req) = taskPlaneReqRx.recv().await {
                // Read the slot fresh per request so /clear and /resume
                // hot-swaps take effect immediately.
                let plane = handlerSlot.read().unwrap().clone();
                match req {
                    TuiRequest::ListJobs { reply } => {
                        let list = plane.lock().unwrap().list();
                        let _ = reply.send(list);
                    }
                    TuiRequest::ListWakes { reply } => {
                        let reg = wakeSlotForHandler.read().unwrap().clone();
                        let list = reg.lock().await.list();
                        let _ = reply.send(list);
                    }
                    TuiRequest::KillTask { id, reply } => {
                        let result = plane.lock().unwrap().stop(id);
                        let ack = match result {
                            Ok(()) => {
                                construct::control::CommandAck::ok(format!("Killing task #{id}."))
                            }
                            Err(e) => construct::control::CommandAck::err(e.to_string()),
                        };
                        let _ = reply.send(ack);
                    }
                    TuiRequest::GetTaskOutput {
                        id,
                        sinceLine,
                        reply,
                    } => {
                        let snap = plane.lock().unwrap().output(id, sinceLine, 500).ok();
                        let _ = reply.send(snap);
                    }
                    _ => {}
                }
            }
        });

        loop {
            // Helper future: yields the next wake batch if the receiver
            // is present, otherwise blocks forever so the select arm
            // becomes a no-op. Wakes that arrive while a turn is
            // running stay in the receiver until this select wins
            // again — no racing spawns, no out-of-order delivery.
            let nextWakeBatch = async {
                match &mut wakeBatchRx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            };

            tokio::select! {
                batch = nextWakeBatch => {
                    if let Some(batch) = batch {
                        // A wake starts a fresh model turn, just like a typed
                        // user message. If Esc left the shared cancel watch at
                        // `true`, clear it before `sendInner` checks the flag;
                        // otherwise the wake chip appears but the turn cancels
                        // before streaming anything.
                        let _ = sessionCancelTx.send(false);
                        cancelRx.borrow_and_update();
                        if let Err(e) = session.injectWakeBatch(
                            batch,
                            &logTx,
                            &sessionRequestTx,
                            &mut cancelRx,
                            &mut steerRx,
                            &mut userBgRx,
                        ).await {
                            let _ = logTx
                                .send(LogEvent::Error(format!("Wake injection error: {e}")))
                                .await;
                        }
                    }
                }
                msg = userInputRx.recv() => {
                    match msg {
                        Some(msg) => {
                            // Clear any stale cancel notification from a previous turn.
                            cancelRx.borrow_and_update();
                            if let Err(e) = session.send(&msg, &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx, &mut userBgRx).await {
                                let _ = logTx
                                    .send(LogEvent::Error(format!("Agent error: {e}")))
                                    .await;
                            }
                        }
                        None => break,
                    }
                }
                req = otherReqRx.recv() => {
                    match req {
                        Some(TuiRequest::ResumeSession { sessionId: id, reply }) => {
                            // Disarm the OLD wake registry's schedulers BEFORE awaiting
                            // resume — otherwise a pending delay/cron/file-watch can
                            // fire during the await window and enqueue a synthetic
                            // wake that hits the new session.
                            {
                                let oldWakes = wakeRegistrySlot.read().unwrap().clone();
                                oldWakes.lock().await.disarmAll();
                            }
                            // Consume old session, keep the shell.
                            let shells = session.intoShells();
                            match Session::resume(
                                &config,
                                mainAgentPermissions(&config),
                                shells,
                                InterfaceMode::SharedTerminal,
                                &[DomainModule::Swe],
                                &id,
                            ).await {
                                Ok(s) => {
                                    session = s;
                                    // Point the task-plane handler at the new session's
                                    // plane so /tasks reflects the resumed state. The
                                    // old Arc drops with the previous Session, which
                                    // kills any tasks it was hosting via JobPlane::Drop.
                                    *jobPlaneSlot.write().unwrap() =
                                        session.jobPlaneHandle();
                                    *wakeRegistrySlot.write().unwrap() =
                                        session.wakeRegistryHandle();
                                    // Replace the wake-batch receiver — the old one
                                    // belonged to the old session's batcher, which has
                                    // been cancelled by disarmAll above.
                                    wakeBatchRx = session.takeWakeBatchRx();
                                    // Re-init MCP for the resumed session.
                                    match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
                                        Ok(servers) if !servers.is_empty() => {
                                            session.initMcp(servers).await;
                                        }
                                        Err(e) => tracing::warn!("failed to load MCP config: {e}"),
                                        _ => {}
                                    }

                                    // Load display branch — includes the full un-branched chain
                                    // past the rewind point until the user sends a new message.
                                    let turns = session.loadDisplayTurns().unwrap_or_default();
                                    let markers = session.compactionMarkers();
                                    let _ = logTx
                                        .send(LogEvent::SessionRestored { turns, markers })
                                        .await;
                                    // Set window title to the current topic on resume.
                                    let label = session.currentTopicLabel();
                                    if !label.is_empty() {
                                        let _ = logTx
                                            .send(LogEvent::TopicChanged {
                                                label: label.to_string(),
                                            })
                                            .await;
                                    }
                                    let _ = reply.send(construct::control::CommandAck::ok(
                                        format!("Resumed session {id}"),
                                    ));
                                }
                                Err((e, shell)) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to resume {id}: {e}"),
                                    ));
                                    // Shell returned — recreate a fresh session.
                                    match Session::new(
                                        &config,
                                        mainAgentPermissions(&config),
                                        shell,
                                        InterfaceMode::SharedTerminal,
                                        &[DomainModule::Swe],
                                    ) {
                                        Ok(mut s) => {
                                            match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
                                                Ok(servers) if !servers.is_empty() => {
                                                    s.initMcp(servers).await;
                                                }
                                                Err(e) => tracing::warn!("failed to load MCP config: {e}"),
                                                _ => {}
                                            }
                                            // (Old wakes already disarmed above before
                                            // the resume attempt — no race here.)
                                            session = s;
                                            *jobPlaneSlot.write().unwrap() =
                                                session.jobPlaneHandle();
                                            *wakeRegistrySlot.write().unwrap() =
                                                session.wakeRegistryHandle();
                                            wakeBatchRx = session.takeWakeBatchRx();
                                        }
                                        Err(e2) => {
                                            let _ = logTx
                                                .send(LogEvent::Error(
                                                    format!("Session lost after failed resume: {e2}"),
                                                ))
                                                .await;
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        Some(TuiRequest::Clear { reply }) => {
                            // Disarm the OLD wake registry BEFORE constructing the
                            // replacement session so a pending delay/cron/file-watch
                            // can't fire during the construction window.
                            {
                                let oldWakes = wakeRegistrySlot.read().unwrap().clone();
                                oldWakes.lock().await.disarmAll();
                            }
                            let shells = session.intoShells();
                            match Session::new(
                                &config,
                                mainAgentPermissions(&config),
                                shells,
                                InterfaceMode::SharedTerminal,
                                &[DomainModule::Swe],
                            ) {
                                Ok(s) => {
                                    session = s;
                                    // Hot-swap the task-plane handler's view of the
                                    // current plane. Dropping the old Arc cancels any
                                    // tasks the cleared session was hosting.
                                    *jobPlaneSlot.write().unwrap() =
                                        session.jobPlaneHandle();
                                    *wakeRegistrySlot.write().unwrap() =
                                        session.wakeRegistryHandle();
                                    wakeBatchRx = session.takeWakeBatchRx();
                                    match construct::mcp::config::loadMcpServers(config.projectRoot.as_deref()) {
                                        Ok(servers) if !servers.is_empty() => {
                                            session.initMcp(servers).await;
                                        }
                                        Err(e) => tracing::warn!("failed to load MCP config: {e}"),
                                        _ => {}
                                    }

                                    let _ = logTx.send(LogEvent::Cleared).await;
                                    let _ = reply.send(construct::control::CommandAck::ok("Session cleared."));
                                }
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to create new session: {e}"),
                                    ));
                                    return;
                                }
                            }
                        }
                        Some(TuiRequest::GetLsp { reply }) => {
                            let servers = session.lspStatusData();
                            let _ = reply.send(construct::control::LspStatus { servers });
                        }
                        Some(TuiRequest::GetMcp { reply }) => {
                            let (servers, totalTools, searchMode, configPath) =
                                session.mcpStatusData().await;
                            let _ = reply.send(construct::control::McpStatus {
                                servers,
                                totalTools,
                                searchMode,
                                configPath,
                            });
                        }
                        Some(TuiRequest::GetPermissions { reply }) => {
                            let (defaultMode, rules, source, configPath) =
                                session.permissionsStatusData();
                            let _ = reply.send(construct::control::PermissionsStatus {
                                defaultMode,
                                rules,
                                source,
                                configPath,
                            });
                        }
                        Some(TuiRequest::GetModels { reply }) => {
                            let _ = reply.send(buildModelStatus(&config));
                        }
                        Some(TuiRequest::SaveModelSelection { scope, tier, profile, reply }) => {
                            if !config.profiles.contains_key(&profile) {
                                let _ = reply.send(construct::control::CommandAck::err(
                                    format!("Unknown model profile: {profile}"),
                                ));
                                continue;
                            }
                            match construct::config::saveModelSelectionInScope(
                                &config,
                                scope,
                                tier,
                                &profile,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Saved model profile to {}. It will be used after /clear or restart.",
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Saved, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to save model profile: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::DiscoverModels { provider, reply }) => {
                            let result = construct::model_catalog::discoverModels(&config, &provider)
                                .await
                                .map_err(|e| e.to_string());
                            let _ = reply.send(result);
                        }
                        Some(TuiRequest::SaveDiscoveredModel { scope, profile, model, reply }) => {
                            if !config.profiles.contains_key(&profile) {
                                let _ = reply.send(construct::control::CommandAck::err(
                                    format!("Unknown model profile: {profile}"),
                                ));
                                continue;
                            }
                            match construct::config::saveDiscoveredModelInScope(
                                &config,
                                scope,
                                &profile,
                                &model,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Saved model {} into profile {profile} at {}. It will be used after /clear or restart.",
                                                model.id,
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Saved, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to save discovered model: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::CreateModelProfile {
                            scope,
                            profile,
                            sourceProfile,
                            reply,
                        }) => {
                            match construct::config::createModelProfileInScope(
                                &config,
                                scope,
                                &profile,
                                &sourceProfile,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Created model profile {profile} at {}.",
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Created, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to create model profile: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::RenameModelProfile {
                            scope,
                            oldProfile,
                            newProfile,
                            reply,
                        }) => {
                            match construct::config::renameModelProfileInScope(
                                &config,
                                scope,
                                &oldProfile,
                                &newProfile,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Renamed model profile {oldProfile} to {newProfile} at {}.",
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Renamed, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to rename model profile: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::DeleteModelProfile { scope, profile, reply }) => {
                            match construct::config::deleteModelProfileInScope(
                                &config,
                                scope,
                                &profile,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Deleted model profile {profile} from {}.",
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Deleted, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to delete model profile: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::SaveModelProfileContext {
                            scope,
                            profile,
                            contextWindow,
                            reply,
                        }) => {
                            match construct::config::saveModelProfileContextInScope(
                                &config,
                                scope,
                                &profile,
                                contextWindow,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Saved context window for profile {profile} to {}.",
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Saved, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to save context window: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::SaveModelProfileThinking {
                            scope,
                            profile,
                            promptThinking,
                            reasoningEffort,
                            reasoningSummary,
                            reply,
                        }) => {
                            match construct::config::saveModelProfileThinkingInScope(
                                &config,
                                scope,
                                &profile,
                                promptThinking,
                                reasoningEffort,
                                reasoningSummary,
                            ) {
                                Ok(path) => match construct::config::load() {
                                    Ok(next) => {
                                        config = next;
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            format!(
                                                "Saved thinking settings for profile {profile} to {}.",
                                                path.display()
                                            ),
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Saved, but failed to reload config: {e}"),
                                        ));
                                    }
                                },
                                Err(e) => {
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Failed to save thinking settings: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::SavePermissions { defaultMode, rules, reply }) => {
                            if let Some(ref root) = config.projectRoot {
                                match construct::config::savePermissions(
                                    root,
                                    &defaultMode,
                                    &rules,
                                ) {
                                    Ok(()) => {
                                        session.setPermissions(construct::permissions::Permissions {
                                            defaultMode,
                                            rules,
                                            source: construct::permissions::PermissionsSource::Project,
                                        });
                                        let _ = reply.send(construct::control::CommandAck::ok(
                                            "Permissions saved.",
                                        ));
                                    }
                                    Err(e) => {
                                        let _ = reply.send(construct::control::CommandAck::err(
                                            format!("Failed to save permissions: {e}"),
                                        ));
                                    }
                                }
                            } else {
                                let _ = reply.send(construct::control::CommandAck::err(
                                    "No project root; permissions not persisted.",
                                ));
                            }
                        }
                        Some(TuiRequest::GetRewindOptions { reply }) => {
                            let turns = session.loadDisplayTurns().unwrap_or_default();
                            let _ = reply.send(turns);
                        }
                        Some(TuiRequest::Rewind { target, saveFork, reply }) => {
                            let result = session.rewind(&target, saveFork, &logTx).await;
                            let _ = reply.send(construct::control::CommandAck::ok(result));
                        }
                        Some(TuiRequest::GetForks { reply }) => {
                            let forks = session.listForks();
                            let _ = reply.send(forks);
                        }
                        Some(TuiRequest::SwitchFork { forkId, reply }) => {
                            let result = session.switchFork(&forkId, &logTx).await;
                            let _ = reply.send(construct::control::CommandAck::ok(result));
                        }
                        Some(TuiRequest::ShowContext { reply }) => {
                            let state = session.buildContextState();
                            let _ = reply.send(state);
                        }
                        Some(TuiRequest::Undo { reply }) => {
                            let ack = session.undoCheckpoint().await;
                            let _ = reply.send(ack);
                        }
                        Some(TuiRequest::ListSessions { reply }) => {
                            let _ = reply.send(session.listSessionsText());
                        }
                        Some(TuiRequest::ShowCost { reply }) => {
                            let _ = reply.send(session.formatCostBreakdown());
                        }
                        Some(TuiRequest::Shutdown) => {
                            session.shutdownLsp().await;
                            session.shutdownMcp().await;
                            break;
                        }
                        Some(TuiRequest::RetryLastTurn { reply }) => {
                            cancelRx.borrow_and_update();
                            match session.retryLastTurn(
                                &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx, &mut userBgRx,
                            ).await {
                                Ok(()) => {
                                    let _ = reply.send(construct::control::CommandAck::ok(""));
                                }
                                Err(e) => {
                                    let _ = logTx
                                        .send(LogEvent::Error(format!("Retry failed: {e}")))
                                        .await;
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Retry failed: {e}"),
                                    ));
                                }
                            }
                        }
                        Some(TuiRequest::ContinueLastTurn { reply }) => {
                            cancelRx.borrow_and_update();
                            match session.continueLastTurn(
                                &logTx, &sessionRequestTx, &mut cancelRx, &mut steerRx, &mut userBgRx,
                            ).await {
                                Ok(()) => {
                                    let _ = reply.send(construct::control::CommandAck::ok(""));
                                }
                                Err(e) => {
                                    let _ = logTx
                                        .send(LogEvent::Error(format!("Continue failed: {e}")))
                                        .await;
                                    let _ = reply.send(construct::control::CommandAck::err(
                                        format!("Continue failed: {e}"),
                                    ));
                                }
                            }
                        }
                        // Terminal-management requests are routed to the
                        // dedicated handler above so they don't block on
                        // `session.send`. Listed here exhaustively so a
                        // future router change doesn't silently drop
                        // them.
                        Some(TuiRequest::SpawnTerminal { .. })
                        | Some(TuiRequest::KillTerminal { .. })
                        | Some(TuiRequest::ListTerminals { .. }) => {
                            tracing::warn!(
                                "terminal-mgmt request slipped past the router; ignoring",
                            );
                        }
                        // ListJobs / KillTask / GetTaskOutput are routed
                        // to the dedicated task-plane handler spawned
                        // above. They are filtered out at the router, so
                        // they should never reach this loop. Match them
                        // exhaustively to keep the compiler happy in
                        // case the router is later changed.
                        Some(TuiRequest::ListJobs { .. })
                        | Some(TuiRequest::ListWakes { .. })
                        | Some(TuiRequest::KillTask { .. })
                        | Some(TuiRequest::GetTaskOutput { .. }) => {
                            tracing::warn!(
                                "task-plane request slipped past the router; ignoring",
                            );
                        }
                        None => break,
                    }
                }
            }
        }
    });

    let result = runLoop(
        &mut terminal,
        &mut termPane,
        &mut shellIoRx,
        &mut agentPanel,
        &mut focus,
        &mut selState,
        &mut scrollLock,
        &mut logRx,
        &mut sessionRequestRx,
        &mut deckUpdateRx,
        &userInputTx,
        &requestTx,
        &deckUpdateTx,
        &cancelTx,
        &steerTx,
        &userBgTx,
        contextWindow,
        rollingBaseline,
        cachingEnabled,
    )
    .await;

    // Gracefully shut down background services before exiting.
    let _ = requestTx.send(TuiRequest::Shutdown).await;

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        crossterm::cursor::SetCursorStyle::DefaultUserShape,
        crossterm::event::PopKeyboardEnhancementFlags,
        crossterm::event::DisableMouseCapture,
        crossterm::event::DisableFocusChange,
        crossterm::event::DisableBracketedPaste,
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;
    resetTerminalTitle();

    result
}

#[allow(clippy::too_many_arguments)]
async fn runLoop(
    terminal: &mut RatatuiTerminal<CrosstermBackend<io::Stdout>>,
    termPane: &mut TerminalPane,
    shellIoRx: &mut mpsc::Receiver<(String, ShellIo, SpawnedBy)>,
    agentPanel: &mut AgentPanel,
    focus: &mut Focus,
    selState: &mut SelectionState,
    scrollLock: &mut ScrollAxisLock,
    logRx: &mut mpsc::Receiver<LogEvent>,
    sessionRequestRx: &mut mpsc::Receiver<SessionRequest>,
    deckUpdateRx: &mut mpsc::Receiver<DeckUpdate>,
    userInputTx: &mpsc::Sender<construct::session::UserInput>,
    requestTx: &mpsc::Sender<TuiRequest>,
    deckUpdateTx: &mpsc::Sender<DeckUpdate>,
    cancelTx: &watch::Sender<bool>,
    steerTx: &mpsc::Sender<construct::session::UserInput>,
    userBgTx: &mpsc::Sender<()>,
    contextWindow: usize,
    rollingBaseline: f64,
    cachingEnabled: bool,
) -> Result<()> {
    let mut tokenCount: usize = 0;
    let mut sessionCost: f64 = 0.0;
    let mut lastCacheHitAt: Option<Instant> = None;
    let mut helpPopupOpen = false;
    let _ = rollingBaseline; // No longer displayed; kept in signature for API stability.
    let mut sessionPicker: Option<SessionPicker> = None;
    let mut rewindPicker: Option<RewindPicker> = None;
    let mut forkPicker: Option<ForkPicker> = None;
    let mut pendingRewindMessage: Option<String> = None;
    let mut pendingRewindAttachments: Option<Vec<construct::transcript::TurnAttachment>> = None;
    let mut mcpPanel: Option<McpPanel> = None;
    let mut lspPanel: Option<LspPanel> = None;
    let mut modelPanel: Option<ModelPanel> = None;
    let mut permissionsPanel: Option<crate::permissions_panel::PermissionsPanel> = None;
    let mut subagentPanel: Option<crate::subagent_panel::SubagentPanel> = None;
    let mut tasksPanel: Option<crate::jobs_panel::JobsPanel> = None;
    let mut logPanel: Option<LogPanel> = None;
    let mut layoutPanel: Option<crate::layout::control_panel::ControlPanel> = None;
    let mut developerLog = DeveloperLog::new(1000);
    let mut toastCenter = ToastCenter::new();
    // Status-bar chips: each frame the draw closure refills this with the
    // chips it rendered and their on-screen x range, so mouse + keyboard
    // handlers between frames can hit-test / navigate them.
    let mut statusChipsLayout: Vec<(StatusChipKind, u16, u16, u16)> = Vec::new();
    let mut statusFocus: Option<usize> = None;
    // Deck-side mirror of running-task count, kept up-to-date from
    // TaskSpawned / TaskComplete / TaskStopped log events. Drives the
    // status-strip running-task indicator. Per-task detail is fetched
    // on demand via TuiRequest::ListJobs when the panel opens.
    let mut taskRunningCount: usize = 0;
    // Completed/stopped task ids the user hasn't yet acknowledged by
    // opening /tasks. Keeps the status chip visible after a short-lived
    // task finishes so a user away from the screen still sees there was
    // activity. Cleared on panel open.
    let mut unreadCompletedTaskIds: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    // Active monitors mirror: incremented on MonitorRegistered, decremented
    // on MonitorStopped or MonitorAutoStopped.
    let mut monitorActiveCount: usize = 0;
    // Armed wake-source count: WakeRegistered minus WakeDisarmed. Note:
    // monitor and bg-task registrations each emit a WakeRegistered too,
    // so this counts ALL active wake sources, not just user-armed
    // delays/crons/fileWatches.
    let mut wakeSourceCount: usize = 0;
    // Inspector refresh coalescing — without this, a task emitting many
    // lines per second would spawn one full GetTaskOutput fetch per line.
    // We keep at most one fetch in flight at a time; if `TaskOutput`
    // arrives while one is mid-flight we just mark the inspector dirty
    // and refire on completion.
    let mut inspectorInFlight: bool = false;
    let mut inspectorDirty: bool = false;
    // Stash for the oneshot reply to the currently-open permit prompt (either
    // top-level or subagent). Set when we receive a `SessionRequest::Permit`
    // and show the prompt; consumed when the user responds.
    let mut pendingPermitReply: Option<oneshot::Sender<construct::permissions::PermitResponse>> =
        None;
    let projectDir = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    // Slice 6c: discover the persisted layout. The discovered tree is
    // the session-scope template (ratio/orient/structure); per-frame
    // we sync the terminal list into it from termPane. On parse error
    // we fall back to defaults and push a one-line notice so the user
    // knows the file was ignored without a scary modal.
    let (mut sessionLayout, mut sessionLayoutPath) = {
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        match crate::layout::discovery::discoverLayout(&cwd) {
            Some(disc) => match disc.result {
                Ok(layout) => (layout, Some(disc.path)),
                Err(msg) => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "layout",
                        "layout file ignored",
                        Some(format!("{}: {msg}", disc.path.display())),
                        true,
                    );
                    (crate::layout::Layout::defaultPhase1(), Some(disc.path))
                }
            },
            None => (crate::layout::Layout::defaultPhase1(), None),
        }
    };
    // Name of the currently-applied layout preset (None when the
    // discovered layout doesn't match a built-in preset). The Ctrl+O
    // panel and /layout slash command both read this.
    let mut activePresetName: Option<String> =
        crate::layout::control_panel::matchPresetName(&sessionLayout);
    let mut needsRedraw = true;
    let mut lastQuitPress: Option<Instant> = None;
    // Title state: the current topic and the animated spinner that fronts it
    // in the OS terminal title whenever the agent is actively working.
    let mut currentTopic: Option<String> = None;
    let mut titleSpinner = TitleSpinner::new();
    let mut titleWasAnimating = false;
    // Window focus tracking — enabled via EnableFocusChange.
    let mut windowFocused = true;
    // True when agent completed work while window was not focused.
    let mut unseenWorkPending = false;
    // True when a permission prompt arrived while window was not focused.
    let mut unseenPermitPending = false;
    // Cap draws at ~30fps. Prevents strobing when the PTY floods us with
    // updates (rich progress bars, keystroke echo) — ratatui's buffer diff
    // absorbs all the intermediate state into one frame.
    const DRAW_MIN_INTERVAL: Duration = Duration::from_millis(33);
    let mut lastDraw = Instant::now() - DRAW_MIN_INTERVAL;

    loop {
        // Mark the frame dirty when the active VT emulator has new content.
        if termPane.activeStateMut().takeDirty() {
            needsRedraw = true;
        }

        // Draw only when state has changed and the throttle allows it.
        if needsRedraw && lastDraw.elapsed() >= DRAW_MIN_INTERVAL {
            needsRedraw = false;
            lastDraw = Instant::now();
            terminal.draw(|frame| {
                let area = frame.area();

                // Top area: terminal + agent panel. Bottom: status bar.
                let vChunks = Layout::default()
                    .constraints([Constraint::Min(1), Constraint::Length(1)])
                    .split(area);

                // Phase 6b/c: render geometry comes from the layout
                // tree. The session-scope `sessionLayout` is loaded
                // from disk at startup (or default); each frame we
                // clone it and reconcile terminal names from termPane
                // before computing rects. Terminal list is reconciled
                // here — not persisted — because terminals are
                // session-runtime, not config.
                let mut layoutTree = sessionLayout.clone();
                for name in termPane.names() {
                    layoutTree.addTerminal(name); // idempotent
                }
                layoutTree.setActiveTerminal(termPane.active());
                let layoutAreas = layoutTree.computeAreas(vChunks[0]);
                let termRect = layoutAreas
                    .iter()
                    .find(|a| matches!(a.window, crate::layout::WindowId::Terminal(_)))
                    .map(|a| a.rect)
                    .unwrap_or(vChunks[0]);
                let agentRect = layoutAreas
                    .iter()
                    .find(|a| a.window == crate::layout::WindowId::AgentPanel)
                    .map(|a| a.rect)
                    .unwrap_or(vChunks[0]);
                let hChunks = [termRect, agentRect];

                // Terminal.
                let termBorder = if *focus == Focus::Terminal {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                let displayOffset = termPane.activeStateRef().displayOffset();
                let termTitle = if displayOffset > 0 {
                    format!(" terminal [\u{2191}{}\u{FE0E}] ", displayOffset)
                } else {
                    " terminal ".to_string()
                };
                let termBlock = Block::default()
                    .borders(Borders::ALL)
                    .border_style(termBorder)
                    .title(termTitle);
                let outerTermArea = hChunks[0];
                let termBlockInner = termBlock.inner(outerTermArea);

                // Tab strip occupies the first row of the inner area.
                let (tabBarRect, termInner) = if termBlockInner.height > 1 {
                    (
                        Rect {
                            x: termBlockInner.x,
                            y: termBlockInner.y,
                            width: termBlockInner.width,
                            height: 1,
                        },
                        Rect {
                            x: termBlockInner.x,
                            y: termBlockInner.y + 1,
                            width: termBlockInner.width,
                            height: termBlockInner.height - 1,
                        },
                    )
                } else {
                    (Rect::default(), termBlockInner)
                };

                frame.render_widget(termBlock, outerTermArea);
                if tabBarRect.height > 0 {
                    termPane.renderTabBar(
                        tabBarRect,
                        frame.buffer_mut(),
                        *focus == Focus::Terminal,
                    );
                }
                termPane.renderActive(termInner, frame.buffer_mut());

                // Capture content rects for mouse hit-testing.
                selState.termContentRect = termInner;

                // Agent panel. When the popup is open showing a subagent permit,
                // suppress the duplicate prompt in the main panel — they share
                // permit state, so a visible duplicate would scroll/copy in
                // lockstep with the popup version.
                agentPanel.permitDisplaySuppressed =
                    subagentPanel.is_some() && agentPanel.pendingPermitIsSubagent();
                let agentChatArea =
                    agentPanel.render(hChunks[1], frame.buffer_mut(), *focus == Focus::Agent);
                // Full chat area for hit-testing (includes prefix columns).
                selState.agentPanelRect = agentChatArea;
                // Content rect offset past the 2-column prefix for selection/highlight.
                let agentContentArea = Rect {
                    x: agentChatArea.x + 2,
                    y: agentChatArea.y,
                    width: agentChatArea.width.saturating_sub(2),
                    height: agentChatArea.height,
                };
                selState.agentContentRect = agentContentArea;
                selState.inputContentRect = agentPanel.lastInputRect;

                // Expand selection for double/triple/quad click (needs Buffer).
                let termOffset = termPane.activeStateRef().displayOffset() as u16;
                let agentOffset = agentPanel.displayOffset();
                if let Some((panel, clickCount)) = selState.pendingExpand.take() {
                    if panel == PanelId::Input {
                        // Input selection expansion not supported.
                    } else {
                        let (sel, rect, offset) = match panel {
                            PanelId::Terminal => {
                                (&mut selState.termSelection, termInner, termOffset)
                            }
                            PanelId::Agent => {
                                (&mut selState.agentSelection, agentContentArea, agentOffset)
                            }
                            PanelId::Input => unreachable!(),
                        };
                        if let Some(sel) = sel {
                            if clickCount == 4 {
                                match panel {
                                    PanelId::Terminal => {
                                        // Select the command output region under the cursor.
                                        let clickGrid = sel.startGridLine();
                                        if let Some((startGrid, endGrid)) =
                                            termPane.activeStateRef().commandRegionAt(clickGrid)
                                        {
                                            sel.setBounds(0, startGrid, rect.width, endGrid);
                                        } else {
                                            // No OSC 133 region — fall back to block selection.
                                            selection::expandSelection(
                                                sel,
                                                clickCount,
                                                frame.buffer_mut(),
                                                rect,
                                                offset,
                                            );
                                        }
                                    }
                                    PanelId::Agent => {
                                        // Select the logical entry under the cursor.
                                        let clickGrid = sel.startGridLine();
                                        if let Some((startGrid, endGrid)) =
                                            agentPanel.entryBoundsAtGridLine(clickGrid)
                                        {
                                            sel.setBounds(0, startGrid, rect.width, endGrid);
                                        }
                                    }
                                    PanelId::Input => unreachable!(),
                                }
                            } else {
                                selection::expandSelection(
                                    sel,
                                    clickCount,
                                    frame.buffer_mut(),
                                    rect,
                                    offset,
                                );
                            }
                            sel.finalize();
                            selState.pendingCopy = Some(panel);
                            selState.selectingIn = None;
                        }
                    }
                }

                // Apply selection highlights after widgets have rendered.
                if let Some(ref sel) = selState.termSelection {
                    selection::applyHighlight(sel, termInner, frame.buffer_mut(), termOffset);
                }
                if let Some(ref sel) = selState.agentSelection {
                    selection::applyHighlight(
                        sel,
                        agentContentArea,
                        frame.buffer_mut(),
                        agentOffset,
                    );
                }

                // Deferred clipboard copy (Buffer only available during draw).
                if let Some(panel) = selState.pendingCopy.take() {
                    match panel {
                        PanelId::Terminal => {
                            if let Some(sel) = &selState.termSelection {
                                let text = extractTerminalUnwrapped(
                                    sel,
                                    termInner,
                                    frame.buffer_mut(),
                                    termOffset,
                                    termPane.activeStateRef(),
                                );
                                selection::copyToClipboard(&text);
                            }
                        }
                        PanelId::Agent => {
                            if let Some(sel) = &selState.agentSelection {
                                let text = agentPanel.extractUnwrappedText(
                                    sel,
                                    agentContentArea,
                                    frame.buffer_mut(),
                                    agentOffset,
                                );
                                selection::copyToClipboard(&text);
                            }
                        }
                        PanelId::Input => {}
                    }
                }

                // Hardware cursor for agent text input (hidden during permission prompt).
                if *focus == Focus::Agent && !agentPanel.pendingPermit {
                    if let Some((col, row)) = agentPanel.textArea.cursorScreenPos {
                        frame.set_cursor_position(ratatui::layout::Position::new(col, row));
                    }
                } else if *focus == Focus::Terminal
                    && let Some((col, row)) = termPane.activeStateRef().cursorViewportPos()
                {
                    frame.set_cursor_position(ratatui::layout::Position::new(
                        termInner.x + col,
                        termInner.y + row,
                    ));
                }

                // Status bar. Minimal layout: right-aligned cost + ctx% + cache.
                // The "press Ctrl+Q again to quit" hint still hijacks the bar
                // briefly — it's a load-bearing safety affordance.
                let quitHintActive = lastQuitPress
                    .map(|t| t.elapsed() < Duration::from_secs(1))
                    .unwrap_or(false);

                let (barBg, barFg) = if quitHintActive {
                    (Color::Yellow, Color::Black)
                } else {
                    (Color::DarkGray, Color::White)
                };

                let barWidth = vChunks[1].width as usize;
                let mut spans: Vec<Span<'static>> = Vec::with_capacity(8);

                // Refill chip-layout each frame; mouse + keyboard nav read
                // it between frames.
                statusChipsLayout.clear();

                if quitHintActive {
                    let hint = " \u{25B8} press Ctrl+Q again to quit";
                    spans.push(Span::raw(hint.to_string()));
                    let pad = barWidth.saturating_sub(hint.chars().count());
                    spans.push(Span::raw(" ".repeat(pad)));
                } else {
                    let costStr = if sessionCost > 0.0 {
                        construct::cost::formatCost(sessionCost)
                    } else {
                        String::new()
                    };
                    let ctxStr = formatContextPct(tokenCount, contextWindow);
                    let cacheSpans = cacheHeatSpans(cachingEnabled, lastCacheHitAt, barBg, barFg);

                    // Build the items in render order (= nav order, left
                    // to right). Each entry carries its kind so we can
                    // hit-test it, plus its span text so we can compute
                    // its on-screen x range. Cache isn't in this list —
                    // it has no overlay to open.
                    let mut items: Vec<(StatusChipKind, String)> = Vec::new();
                    // Status chip shows whenever ANY task/monitor/wake
                    // surface state exists. Running task count is the
                    // primary signal when present; otherwise the chip
                    // still appears so unread completions, registered
                    // monitors, and armed wake sources stay discoverable.
                    let hasBackground = !unreadCompletedTaskIds.is_empty()
                        || monitorActiveCount > 0
                        || wakeSourceCount > 0;
                    if taskRunningCount > 0 {
                        items.push((
                            StatusChipKind::Jobs,
                            format!(
                                "\u{25F4} {taskRunningCount} task{}",
                                if taskRunningCount == 1 { "" } else { "s" }
                            ),
                        ));
                    } else if hasBackground {
                        items.push((StatusChipKind::Jobs, "\u{25F4} tasks".into()));
                    }
                    if !costStr.is_empty() {
                        items.push((StatusChipKind::Cost, costStr.clone()));
                    }
                    if !ctxStr.is_empty() {
                        items.push((StatusChipKind::Context, ctxStr.clone()));
                    }

                    // Clamp focus if the navigable set shrunk this frame.
                    if let Some(idx) = statusFocus {
                        if items.is_empty() {
                            statusFocus = None;
                        } else if idx >= items.len() {
                            statusFocus = Some(items.len() - 1);
                        }
                    }

                    // Assemble right-aligned spans exactly as before
                    // (no glyph affordances, no brackets) — just paint
                    // the focused item in inverse so the user can see
                    // which one Enter will activate. Track each item's
                    // offset so we can resolve its x range after pad
                    // is computed.
                    let mut rightSpans: Vec<Span<'static>> = Vec::with_capacity(8);
                    let mut itemRanges: Vec<(StatusChipKind, usize, usize)> = Vec::new();
                    let mut cursorOff: usize = 0;
                    for (i, (kind, text)) in items.iter().enumerate() {
                        if i > 0 {
                            rightSpans.push(Span::raw("  "));
                            cursorOff += 2;
                        }
                        let len = text.chars().count();
                        if statusFocus == Some(i) {
                            rightSpans.push(Span::styled(
                                text.clone(),
                                Style::default()
                                    .bg(barFg)
                                    .fg(barBg)
                                    .add_modifier(Modifier::BOLD),
                            ));
                        } else {
                            rightSpans.push(Span::raw(text.clone()));
                        }
                        itemRanges.push((*kind, cursorOff, len));
                        cursorOff += len;
                    }

                    // Cache: plain info, not navigable.
                    if !cacheSpans.is_empty() {
                        if !rightSpans.is_empty() {
                            rightSpans.push(Span::raw("  "));
                            cursorOff += 2;
                        }
                        let cacheLen: usize =
                            cacheSpans.iter().map(|s| s.content.chars().count()).sum();
                        rightSpans.extend(cacheSpans);
                        rightSpans.push(Span::raw(" cache"));
                        cursorOff += cacheLen + " cache".chars().count();
                    }

                    // Trailing space + 1 char leading pad keeps the bar breathing.
                    let pad = barWidth.saturating_sub(cursorOff + 1);
                    spans.push(Span::raw(" ".repeat(pad)));
                    let leftEdge = vChunks[1].x + pad as u16;
                    for (kind, off, len) in itemRanges {
                        let xs = leftEdge + off as u16;
                        let xe = xs + len as u16;
                        statusChipsLayout.push((kind, xs, xe, vChunks[1].y));
                    }
                    spans.extend(rightSpans);
                    spans.push(Span::raw(" "));
                }

                let statusBar =
                    Paragraph::new(Line::from(spans)).style(Style::default().bg(barBg).fg(barFg));
                frame.render_widget(statusBar, vChunks[1]);

                // Ephemeral notifications float over the base panes only:
                // no layout shift, and modal overlays still get priority.
                toastCenter.render(area, frame.buffer_mut());

                // Session picker overlay.
                if let Some(ref mut picker) = sessionPicker {
                    picker.render(area, frame.buffer_mut());
                }

                // Rewind picker overlay.
                if let Some(ref mut picker) = rewindPicker {
                    picker.render(area, frame.buffer_mut());
                }

                // Fork picker overlay.
                if let Some(ref mut picker) = forkPicker {
                    picker.render(area, frame.buffer_mut());
                }

                // MCP panel overlay.
                if let Some(ref mut panel) = mcpPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Tasks panel overlay.
                if let Some(ref mut panel) = tasksPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Developer log overlay.
                if let Some(ref mut panel) = logPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Layout control panel overlay (Ctrl+O).
                if let Some(ref panel) = layoutPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // LSP panel overlay.
                if let Some(ref mut panel) = lspPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Model profile panel overlay.
                if let Some(ref mut panel) = modelPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Permissions panel overlay.
                if let Some(ref mut panel) = permissionsPanel {
                    panel.render(area, frame.buffer_mut());
                }

                // Subagent panel overlay.
                if let Some(ref mut panel) = subagentPanel {
                    panel.render(area, frame.buffer_mut(), agentPanel);
                }

                // Help popup overlay.
                if helpPopupOpen {
                    renderHelpPopup(area, frame.buffer_mut());
                }
            })?;
        }

        // Drain PTY output for every terminal — non-active tabs feed
        // their state too so switching tabs shows up-to-date content.
        if termPane.drainOutputs() {
            needsRedraw = true;
        }

        // Drain registry-spawned ShellIos. New entries land here when
        // the agent's `terminalSpawn` tool or the user's Ctrl+T fires.
        // User-initiated spawns auto-focus the new tab; agent-initiated
        // ones don't disturb the user's current focus.
        while let Ok((name, io, spawnedBy)) = shellIoRx.try_recv() {
            // Match dimensions of the active terminal so initial render
            // doesn't double-resize on first frame.
            let cols = termPane.activeStateRef().columns() as u16;
            let rows = termPane.activeStateRef().screenLines() as u16;
            termPane.add(name.clone(), io, cols, rows);
            if matches!(spawnedBy, SpawnedBy::User) {
                termPane.setActive(&name);
                *focus = Focus::Terminal;
            }
            needsRedraw = true;
        }

        // Drain session log events (monotone stream, no replies).
        while let Ok(event) = logRx.try_recv() {
            needsRedraw = true;
            match event {
                LogEvent::ContentDelta(text) => agentPanel.appendContent(&text),
                LogEvent::ReasoningDelta(text) => agentPanel.appendReasoning(&text),
                LogEvent::ToolResult { name, output } => {
                    // Task tool results are handled by SubagentComplete — don't double-render.
                    if name != "task" && !agentPanel.finishWakeToolResult(&name, &output) {
                        agentPanel.pushToolResult(&name, &output);
                    }
                }
                LogEvent::ToolStarted { name, summary } => {
                    if name != "task" {
                        agentPanel.toolStarted(&name, &summary);
                    }
                }
                LogEvent::ToolCallPending { index, name } => {
                    agentPanel.toolCallPending(index, &name);
                }
                LogEvent::ToolCallProgress { index, bytes } => {
                    agentPanel.toolCallProgress(index, bytes);
                }
                LogEvent::ToolCallPreview { index, preview } => {
                    agentPanel.toolCallPreview(index, &preview);
                }
                LogEvent::ToolAutoApproved { name, summary } => {
                    if !crate::agent_panel::isWakeToolName(&name) {
                        agentPanel.toolApproved(&format!("{name}: {summary}"));
                    }
                }
                LogEvent::ToolDenied { name } => {
                    agentPanel.toolDenied(&name);
                }
                LogEvent::ToolAutoDenied { name, summary } => {
                    agentPanel.toolAutoDenied(&name, &summary);
                }
                LogEvent::TurnAborted { name } => {
                    agentPanel.pushError(&format!("Turn aborted: {name} not permitted"));
                }
                LogEvent::TurnComplete => {
                    agentPanel.finishTurn();
                    // If completed while window not focused, show envelope indicator.
                    if !windowFocused {
                        unseenWorkPending = true;
                        writeTerminalTitle(TITLE_UNSEEN_GLYPH, currentTopic.as_deref());
                    }
                }
                LogEvent::TurnCancelled => {
                    agentPanel.finalizeCancelled();
                }
                LogEvent::SteerInjected { texts } => {
                    agentPanel.promoteQueue(&texts);
                }
                LogEvent::TopicChanged { label } => {
                    currentTopic = Some(label);
                    // Preserve unseen indicators across topic changes so the user
                    // still sees the envelope/warning when they refocus.
                    let glyph = if unseenPermitPending {
                        TITLE_PERMIT_GLYPH
                    } else if unseenWorkPending {
                        TITLE_UNSEEN_GLYPH
                    } else if agentPanel.isActive() {
                        titleSpinner.current()
                    } else {
                        TITLE_IDLE_GLYPH
                    };
                    writeTerminalTitle(glyph, currentTopic.as_deref());
                }
                LogEvent::LspHint {
                    serverId,
                    installHint,
                } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "lsp",
                        format!("{serverId} not found"),
                        Some(format!("install hint: {installHint}")),
                        true,
                    );
                }
                LogEvent::TokenUpdate {
                    contextTokens,
                    sessionCost: sc,
                    cacheReadTokens,
                    ..
                } => {
                    tokenCount = contextTokens;
                    sessionCost = sc;
                    if cacheReadTokens > 0 {
                        lastCacheHitAt = Some(Instant::now());
                    }
                }
                LogEvent::Retrying {
                    attempt,
                    maxAttempts,
                } => {
                    agentPanel.showRetrying(attempt, maxAttempts);
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "api",
                        format!("retrying request ({attempt}/{maxAttempts})"),
                        None,
                        true,
                    );
                }
                LogEvent::Error(msg) => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Error,
                        "agent",
                        "turn error",
                        Some(msg.clone()),
                        true,
                    );
                    agentPanel.pushError(&msg);
                    // A fatal error ends the turn — stop the throbber so the
                    // user isn't misled into thinking the model is still
                    // working, and surface the retry/continue hint.
                    agentPanel.finishTurn();
                    agentPanel.errorHint = true;
                }
                LogEvent::BudgetWarning {
                    sessionCost: sc,
                    limit,
                } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "cost",
                        "session cost limit exceeded",
                        Some(format!(
                            "{} spent, {} limit",
                            construct::cost::formatCost(sc),
                            construct::cost::formatCost(limit),
                        )),
                        true,
                    );
                }
                LogEvent::CompactionStarted { stage } => {
                    tracing::info!(stage = %stage, "compaction started");
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "context",
                        format!("compaction started: {stage}"),
                        None,
                        false,
                    );
                }
                LogEvent::CompactionComplete {
                    stage,
                    reduction,
                    markerBlock,
                } => {
                    tracing::info!(stage = %stage, reduction = %reduction, "compaction complete");
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Success,
                        "context",
                        format!("compaction complete: {stage}"),
                        Some(format!("reduction: {reduction}")),
                        false,
                    );
                    if let Some(blockIdx) = markerBlock {
                        agentPanel.pushCompactionMarker(&stage, blockIdx);
                    }
                }
                LogEvent::Cleared => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "session",
                        "session cleared",
                        None,
                        true,
                    );
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    currentTopic = None;
                    // Fresh session => fresh JobPlane; the old plane's
                    // Drop kills any running tasks but the TaskStopped
                    // events may not reach us before the channel detaches,
                    // so resync the counter from a fresh ListJobs rather
                    // than trusting the per-event delta.
                    taskRunningCount = 0;
                    unreadCompletedTaskIds.clear();
                    monitorActiveCount = 0;
                    wakeSourceCount = 0;
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                    writeTerminalTitle(TITLE_IDLE_GLYPH, None);
                }
                LogEvent::Rewound { targetTurnId } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "session",
                        "conversation rewound",
                        Some(format!("target turn: {targetTurnId}")),
                        true,
                    );
                    rewindPicker = None;
                    forkPicker = None;
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    if let Some(msg) = pendingRewindMessage.take() {
                        agentPanel.textArea.setText(&msg);
                    }
                    // Restore image attachments from the rewound turn.
                    if let Some(atts) = pendingRewindAttachments.take() {
                        use base64::Engine;
                        for att in atts {
                            let data = base64::engine::general_purpose::STANDARD
                                .decode(&att.data)
                                .unwrap_or_default();
                            agentPanel.addAttachment(construct::session::Attachment {
                                mimeType: att.mimeType.clone(),
                                data,
                                label: "restored image".to_string(),
                                rgbaDimensions: None,
                            });
                        }
                    }
                    tracing::info!(target = %targetTurnId, "conversation rewound");
                }
                LogEvent::SessionRestored { turns, markers } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "session",
                        "session restored",
                        Some(format!("{} turns, {} markers", turns.len(), markers.len())),
                        true,
                    );
                    agentPanel.clearDisplay();
                    tokenCount = 0;
                    // Resumed session has a fresh JobPlane.
                    taskRunningCount = 0;
                    unreadCompletedTaskIds.clear();
                    monitorActiveCount = 0;
                    wakeSourceCount = 0;
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                    replayTranscript(agentPanel, &turns);
                    for (stage, blockIdx) in &markers {
                        agentPanel.pushCompactionMarker(stage, *blockIdx);
                    }
                }
                LogEvent::SubagentStarted {
                    sessionId,
                    agentType,
                    prompt,
                } => {
                    tracing::info!(agent = %agentType, "subagent started");
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "subagent",
                        format!("{agentType} started"),
                        Some(format!("{sessionId}: {prompt}")),
                        false,
                    );
                    agentPanel.subagentStarted(&sessionId, &agentType, &prompt);
                }
                LogEvent::SubagentEvent { sessionId, event } => match *event {
                    LogEvent::ToolAutoApproved {
                        ref name,
                        ref summary,
                    } => {
                        agentPanel.subagentToolLine(&sessionId, name, summary);
                    }
                    LogEvent::ToolStarted {
                        ref name,
                        ref summary,
                    } => {
                        agentPanel.subagentToolLine(&sessionId, name, summary);
                    }
                    LogEvent::ToolResult {
                        ref name,
                        ref output,
                    } => {
                        let brief = if output.len() > 60 {
                            format!("{}\u{2026}", &output[..output.floor_char_boundary(60)])
                        } else {
                            output.clone()
                        };
                        agentPanel.subagentToolLine(&sessionId, name, &brief);
                        agentPanel.subagentToolResult(&sessionId, name, output);
                    }
                    LogEvent::ContentDelta(ref text) => {
                        agentPanel.subagentContent(&sessionId, text);
                    }
                    LogEvent::Error(ref msg) => {
                        agentPanel.subagentToolLine(&sessionId, "error", msg);
                    }
                    _ => {}
                },
                LogEvent::SubagentShellOutput { sessionId, data } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Debug,
                        "subagent",
                        format!("{sessionId} shell output"),
                        Some(format!("{} bytes", data.len())),
                        false,
                    );
                    agentPanel.feedSubagentShell(&sessionId, &data);
                }
                LogEvent::SubagentComplete {
                    sessionId,
                    agentType,
                    turns,
                    content,
                } => {
                    tracing::info!(agent = %agentType, turns = turns, "subagent completed");
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Success,
                        "subagent",
                        format!("{agentType} completed"),
                        Some(format!("{sessionId}: {turns} turns")),
                        false,
                    );
                    agentPanel.subagentComplete(&sessionId, &agentType, turns, &content);
                }
                LogEvent::ScratchpadRecovered {
                    matchedTag,
                    snippet,
                    recoveredChars,
                } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "parser",
                        "scratchpad close recovered",
                        Some(format!(
                            "`{matchedTag}`, {recoveredChars} chars: \"{snippet}\""
                        )),
                        true,
                    );
                }
                LogEvent::JobSpawned {
                    id,
                    kind: _,
                    command,
                } => {
                    taskRunningCount += 1;
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "task",
                        format!("task #{id} spawned"),
                        Some(command),
                        true,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::JobOutput { id, line } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Debug,
                        "task",
                        format!("task #{id} output"),
                        Some(line),
                        false,
                    );
                    // Stream into the inspector if it's currently open on
                    // this task. Coalesce: keep at most one fetch in
                    // flight; if another line arrives mid-flight, mark
                    // dirty and refire when the in-flight one returns.
                    let openOnThis =
                        tasksPanel.as_ref().and_then(|p| p.inspectorTaskId()) == Some(id);
                    if openOnThis {
                        if inspectorInFlight {
                            inspectorDirty = true;
                        } else {
                            inspectorInFlight = true;
                            let sinceLine =
                                tasksPanel.as_ref().and_then(|p| p.inspectorSinceLine());
                            spawnInspectorFetch(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                id,
                                sinceLine,
                            );
                        }
                    }
                }
                LogEvent::JobComplete { id, exitCode } => {
                    taskRunningCount = taskRunningCount.saturating_sub(1);
                    // Mark as unread unless the panel is already open
                    // (in which case the user sees the row land live).
                    if tasksPanel.is_none() {
                        unreadCompletedTaskIds.insert(id);
                    }
                    let code = exitCode
                        .map(|c| format!("exit {c}"))
                        .unwrap_or_else(|| "—".into());
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Success,
                        "task",
                        format!("task #{id} completed"),
                        Some(code),
                        true,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::JobStopped { id, reason } => {
                    taskRunningCount = taskRunningCount.saturating_sub(1);
                    if tasksPanel.is_none() {
                        unreadCompletedTaskIds.insert(id);
                    }
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "task",
                        format!("task #{id} stopped"),
                        Some(reason),
                        true,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::MonitorRegistered {
                    id,
                    description,
                    filter,
                    ..
                } => {
                    monitorActiveCount += 1;
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "monitor",
                        format!("monitor #{id} registered"),
                        Some(format!("{description} · /{filter}/")),
                        true,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::MonitorEvent {
                    id,
                    line,
                    eventCount,
                } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Debug,
                        "monitor",
                        format!("monitor #{id} matched ({eventCount})"),
                        Some(line),
                        false,
                    );
                    // Per-line monitor events are noisy by design. The
                    // counter and last-event time update via the periodic
                    // /tasks refresh; we don't push a notice per line
                    // because a single noisy watcher would flood the
                    // conversation. Phase 5's wake plane is the consumer
                    // that actually acts on these.
                }
                LogEvent::MonitorAutoStopped { id, reason } => {
                    monitorActiveCount = monitorActiveCount.saturating_sub(1);
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "monitor",
                        format!("monitor #{id} auto-stopped"),
                        Some(reason),
                        true,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::MonitorStopped { id } => {
                    monitorActiveCount = monitorActiveCount.saturating_sub(1);
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "monitor",
                        format!("monitor #{id} stopped"),
                        None,
                        true,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::TerminalSpawned { name, spawnedBy } => {
                    use construct::control::TerminalSpawnedBy;
                    let by = match spawnedBy {
                        TerminalSpawnedBy::User => "user",
                        TerminalSpawnedBy::Agent => "agent",
                    };
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "terminal",
                        format!("terminal '{name}' spawned"),
                        Some(format!("spawned by {by}")),
                        true,
                    );
                }
                LogEvent::TerminalClosed { name } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "terminal",
                        format!("terminal '{name}' closed"),
                        None,
                        false,
                    );
                    // Just drop the tab. The textual confirmation is
                    // either the ToolResult (agent-initiated kill) or
                    // the user already saw the tab disappear.
                    termPane.remove(&name);
                }
                LogEvent::TerminalActiveForAgent { name } => {
                    // Agent's default target changed; deck doesn't follow
                    // the agent's focus, but we surface the change.
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "terminal",
                        "agent target terminal changed",
                        Some(name),
                        true,
                    );
                }
                LogEvent::TerminalRenamed { from, to } => {
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "terminal",
                        "terminal renamed",
                        Some(format!("{from} -> {to}")),
                        true,
                    );
                }
                LogEvent::WakeBatchInjected { count, summary } => {
                    // The session task formatted, recorded, and started
                    // a turn for this batch already — we just render a
                    // notice. Single-fire batches show the source +
                    // payload preview; multi-fire batches collapse to
                    // a counter so a noisy stampede doesn't flood the
                    // panel with one chip per match.
                    agentPanel.wakeFiredSource(&summary);
                    agentPanel.pushWakeTurn(&summary);
                    let chip = if count > 1 {
                        format!("{count} wakes")
                    } else {
                        "wake injected".to_string()
                    };
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Info,
                        "wake",
                        chip,
                        Some(summary),
                        true,
                    );
                }
                LogEvent::AutoBgWarning {
                    command,
                    elapsedSecs,
                    userTriggered,
                } => {
                    // The framework auto-respawned the command as a real
                    // bg job. The corresponding LogEvent::JobSpawned will
                    // fire separately with the new job id — no need to
                    // duplicate that here.
                    let preview = if command.len() > 80 {
                        format!("{}\u{2026}", &command[..command.floor_char_boundary(80)])
                    } else {
                        command.clone()
                    };
                    let notice = if userTriggered {
                        "shell moved to background (Ctrl+B)".to_string()
                    } else {
                        format!("shell moved to background ({elapsedSecs}s elapsed)")
                    };
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Warning,
                        "shell",
                        notice,
                        Some(preview),
                        true,
                    );
                }
                LogEvent::WakeRegistered {
                    id,
                    kind,
                    summary,
                    prompt,
                    nextFireAt,
                } => {
                    wakeSourceCount += 1;
                    agentPanel.wakeRegistered(id, kind, summary.clone(), prompt, nextFireAt);
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Debug,
                        "wake",
                        format!("wake #{id} registered"),
                        Some(format!("{} · {summary}", kind.asStr())),
                        false,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
                LogEvent::WakeDisarmed { id } => {
                    wakeSourceCount = wakeSourceCount.saturating_sub(1);
                    agentPanel.wakeDisarmed(id);
                    pushOperationalLog(
                        &mut developerLog,
                        &mut toastCenter,
                        &mut logPanel,
                        LogLevel::Debug,
                        "wake",
                        format!("wake #{id} disarmed"),
                        None,
                        false,
                    );
                    refreshTasksPanel(&tasksPanel, requestTx, deckUpdateTx);
                }
            }
        }

        // Drain session → TUI requests (permits). Each variant carries a
        // oneshot reply that is resolved when the user responds via the
        // permit prompt.
        while let Ok(req) = sessionRequestRx.try_recv() {
            needsRedraw = true;
            match req {
                SessionRequest::Permit {
                    origin,
                    name,
                    summary,
                    args,
                    diff,
                    explanation,
                    impact,
                    reply,
                } => {
                    let isSubagent =
                        matches!(origin, construct::control::PermitOrigin::Subagent { .. });
                    // Extract the subagent sessionId before `origin` is
                    // consumed by `showToolRequest` so we can pre-focus
                    // the right tab in the popup.
                    let subagentSid =
                        if let construct::control::PermitOrigin::Subagent { sessionId } = &origin {
                            Some(sessionId.clone())
                        } else {
                            None
                        };
                    agentPanel.showToolRequest(
                        &name,
                        &summary,
                        &args,
                        diff,
                        explanation,
                        impact,
                        origin,
                    );
                    pendingPermitReply = Some(reply);
                    // If permit arrived while window not focused, show warning indicator.
                    if !windowFocused {
                        unseenPermitPending = true;
                        writeTerminalTitle(TITLE_PERMIT_GLYPH, currentTopic.as_deref());
                    }
                    if isSubagent {
                        // Focus the requesting subagent's tab so the user
                        // sees whose permit they're approving. Auto-open
                        // the popup when multiple subagents are running —
                        // otherwise the inline permit prompt is shown on
                        // the main panel and the popup stays available
                        // for explicit `v`.
                        if let Some(sid) = subagentSid {
                            agentPanel.selectSubagentBySessionId(&sid);
                            if agentPanel.activeSubagents.len() > 1 && subagentPanel.is_none() {
                                subagentPanel = Some(crate::subagent_panel::SubagentPanel::live());
                            }
                        }
                    } else {
                        // Parent permits auto-close the popup so the
                        // main-panel prompt becomes visible.
                        if subagentPanel.is_some() {
                            subagentPanel = None;
                        }
                    }
                }
            }
        }

        // Drain deck-internal updates (replies to slash-command requests).
        while let Ok(update) = deckUpdateRx.try_recv() {
            needsRedraw = true;
            match update {
                DeckUpdate::McpStatus(status) => {
                    mcpPanel = Some(McpPanel::new(
                        status.servers,
                        status.totalTools,
                        status.searchMode,
                        status.configPath,
                    ));
                }
                DeckUpdate::LspStatus(status) => {
                    lspPanel = Some(LspPanel::new(status.servers));
                }
                DeckUpdate::ModelStatus(status) => {
                    modelPanel = Some(ModelPanel::new(status));
                }
                DeckUpdate::ModelCatalog { provider, result } => {
                    if let Some(panel) = modelPanel.as_mut() {
                        panel.setCatalogResult(provider, result);
                    }
                }
                DeckUpdate::PermissionsStatus(status) => {
                    permissionsPanel = Some(crate::permissions_panel::PermissionsPanel::new(
                        status.defaultMode,
                        status.rules,
                        status.source,
                        status.configPath,
                    ));
                }
                DeckUpdate::ContextDisplay(state) => {
                    agentPanel.pushContextDisplay(state);
                }
                DeckUpdate::RewindOptions(turns) => {
                    rewindPicker = Some(RewindPicker::new(&turns));
                }
                DeckUpdate::Forks(forks) => {
                    forkPicker = Some(ForkPicker::new(&forks));
                }
                DeckUpdate::ShowResult(text) => {
                    agentPanel.pushCommandResult(&text);
                }
                DeckUpdate::CommandAck(ack) => {
                    if forkPicker.is_some() && !ack.ok {
                        if let Some(ref mut picker) = forkPicker {
                            picker.switchFailed(ack.message);
                        }
                    } else if !ack.message.is_empty() {
                        agentPanel.pushCommandResult(&ack.message);
                    }
                }
                DeckUpdate::TasksList(list) => {
                    if let Some(ref mut panel) = tasksPanel {
                        panel.refresh(list);
                    } else {
                        // Panel transition closed → open: user has now
                        // seen any pending completions, so the unread
                        // set clears.
                        unreadCompletedTaskIds.clear();
                        tasksPanel = Some(crate::jobs_panel::JobsPanel::new(list));
                    }
                    needsRedraw = true;
                }
                DeckUpdate::WakesList(list) => {
                    if let Some(ref mut panel) = tasksPanel {
                        panel.refreshWakes(list);
                        needsRedraw = true;
                    }
                }
                DeckUpdate::TaskOutputOpen { id, snap } => {
                    if let Some(ref mut panel) = tasksPanel {
                        panel.applyInspectorOpen(id, snap);
                        needsRedraw = true;
                    }
                }
                DeckUpdate::TaskOutputRefresh {
                    id,
                    sinceLine,
                    snap,
                } => {
                    if let Some(ref mut panel) = tasksPanel
                        && let Some(snap) = snap
                    {
                        // Tagged-fetch race guard: if the user paged
                        // back while this fetch was in flight, the
                        // panel's `requestedSinceLine` has already
                        // moved on. Pass the fetched-against value
                        // so the panel can drop stale snapshots
                        // instead of momentarily clobbering the
                        // paged-back view.
                        if panel.applyInspectorSnapshot(id, sinceLine, snap) {
                            needsRedraw = true;
                        }
                    }
                    // Snap=None means the task is gone from the
                    // session. The next ListJobs refresh will drop
                    // the inspector via JobsPanel::refresh, so we
                    // don't need to touch the panel here.
                    // Coalescing: the in-flight fetch just returned. If
                    // TaskOutput events landed while it was flying, fire
                    // one more refresh — only on the same id, only if the
                    // inspector is still open on it.
                    inspectorInFlight = false;
                    if inspectorDirty {
                        inspectorDirty = false;
                        let stillOpen =
                            tasksPanel.as_ref().and_then(|p| p.inspectorTaskId()) == Some(id);
                        if stillOpen {
                            inspectorInFlight = true;
                            let sinceLine =
                                tasksPanel.as_ref().and_then(|p| p.inspectorSinceLine());
                            spawnInspectorFetch(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                id,
                                sinceLine,
                            );
                        }
                    }
                }
                DeckUpdate::ResumeResult(ack) => {
                    if ack.ok {
                        sessionPicker = None;
                        agentPanel
                            .entries
                            .push(crate::agent_panel::PanelEntry::SessionNotice(ack.message));
                    } else if let Some(ref mut picker) = sessionPicker {
                        picker.resumeFailed(ack.message);
                    } else {
                        agentPanel.pushError(&ack.message);
                    }
                }
            }
        }

        // Tick throbber animation (wall-clock gated).
        if agentPanel.tickThrobber() {
            needsRedraw = true;
        }
        if agentPanel.tickWakeSchedules() {
            needsRedraw = true;
        }

        // Tick animated title spinner while a turn is active. While an unseen
        // indicator is showing (envelope/warning for off-focus events), suppress
        // both the spinner tick and the spinner-end fallback so the indicator
        // survives until the user refocuses the window.
        let nowAnimating = currentTopic.is_some() && agentPanel.isActive();
        let unseenIndicatorActive = unseenWorkPending || unseenPermitPending;
        if nowAnimating {
            if titleSpinner.tick() && !unseenIndicatorActive {
                writeTerminalTitle(titleSpinner.current(), currentTopic.as_deref());
            }
        } else if titleWasAnimating && !unseenIndicatorActive {
            // Agent just finished — drop back to the static idle glyph.
            writeTerminalTitle(TITLE_IDLE_GLYPH, currentTopic.as_deref());
        }
        titleWasAnimating = nowAnimating;

        // Advance character reveal buffer.
        if agentPanel.tickReveal() {
            needsRedraw = true;
        }

        if toastCenter.tick() {
            needsRedraw = true;
        }

        // Clear the quit hint after the double-tap window expires.
        if let Some(t) = lastQuitPress
            && t.elapsed() >= Duration::from_secs(1)
        {
            lastQuitPress = None;
            needsRedraw = true;
        }

        // Handle input.
        let (quit, hadInput, wasResized) = handleInput(
            focus,
            termPane,
            agentPanel,
            selState,
            scrollLock,
            userInputTx,
            requestTx,
            deckUpdateTx,
            cancelTx,
            steerTx,
            userBgTx,
            &mut sessionPicker,
            &mut rewindPicker,
            &mut forkPicker,
            &mut pendingRewindMessage,
            &mut pendingRewindAttachments,
            &mut mcpPanel,
            &mut lspPanel,
            &mut modelPanel,
            &mut subagentPanel,
            &mut tasksPanel,
            &mut logPanel,
            &mut permissionsPanel,
            &mut developerLog,
            &mut toastCenter,
            &mut pendingPermitReply,
            &projectDir,
            &mut lastQuitPress,
            &mut helpPopupOpen,
            &mut windowFocused,
            &mut unseenWorkPending,
            &mut unseenPermitPending,
            &currentTopic,
            &titleSpinner,
            &mut statusFocus,
            &statusChipsLayout,
            &mut sessionLayout,
            &mut sessionLayoutPath,
            &mut activePresetName,
            &mut layoutPanel,
        )
        .await?;
        if quit {
            break;
        }
        if hadInput {
            needsRedraw = true;
        }
        if wasResized {
            terminal.clear()?;
        }

        tokio::task::yield_now().await;
    }

    Ok(())
}

/// Drain all pending input events. Returns (quit, hadInput).
#[allow(clippy::too_many_arguments)]
async fn handleInput(
    focus: &mut Focus,
    termPane: &mut TerminalPane,
    agentPanel: &mut AgentPanel,
    selState: &mut SelectionState,
    scrollLock: &mut ScrollAxisLock,
    userInputTx: &mpsc::Sender<construct::session::UserInput>,
    requestTx: &mpsc::Sender<TuiRequest>,
    deckUpdateTx: &mpsc::Sender<DeckUpdate>,
    cancelTx: &watch::Sender<bool>,
    steerTx: &mpsc::Sender<construct::session::UserInput>,
    userBgTx: &mpsc::Sender<()>,
    sessionPicker: &mut Option<SessionPicker>,
    rewindPicker: &mut Option<RewindPicker>,
    forkPicker: &mut Option<ForkPicker>,
    pendingRewindMessage: &mut Option<String>,
    pendingRewindAttachments: &mut Option<Vec<construct::transcript::TurnAttachment>>,
    mcpPanel: &mut Option<McpPanel>,
    lspPanel: &mut Option<LspPanel>,
    modelPanel: &mut Option<ModelPanel>,
    subagentPanel: &mut Option<crate::subagent_panel::SubagentPanel>,
    tasksPanel: &mut Option<crate::jobs_panel::JobsPanel>,
    logPanel: &mut Option<LogPanel>,
    permissionsPanel: &mut Option<crate::permissions_panel::PermissionsPanel>,
    developerLog: &mut DeveloperLog,
    toastCenter: &mut ToastCenter,
    pendingPermitReply: &mut Option<oneshot::Sender<construct::permissions::PermitResponse>>,
    projectDir: &str,
    lastQuitPress: &mut Option<Instant>,
    helpPopupOpen: &mut bool,
    windowFocused: &mut bool,
    unseenWorkPending: &mut bool,
    unseenPermitPending: &mut bool,
    currentTopic: &Option<String>,
    titleSpinner: &TitleSpinner,
    statusFocus: &mut Option<usize>,
    statusChipsLayout: &[(StatusChipKind, u16, u16, u16)],
    sessionLayout: &mut crate::layout::Layout,
    sessionLayoutPath: &mut Option<std::path::PathBuf>,
    activePresetName: &mut Option<String>,
    layoutPanel: &mut Option<crate::layout::control_panel::ControlPanel>,
) -> Result<(bool, bool, bool)> {
    // Wait up to 16ms for the first event.
    if !event::poll(Duration::from_millis(16))? {
        return Ok((false, false, false));
    }

    // Drain all queued events to avoid input lag (especially trackpad momentum).
    let mut hadInput = false;
    let mut resized = false;
    loop {
        match event::read()? {
            Event::FocusGained => {
                *windowFocused = true;
                // Completed-work indicator can always clear — the user is now
                // looking. The permit indicator only clears if the permit is
                // actually resolved; an unanswered permit should stay flagged.
                let permitStillPending = pendingPermitReply.is_some();
                let cleared = if *unseenWorkPending || (*unseenPermitPending && !permitStillPending)
                {
                    *unseenWorkPending = false;
                    if !permitStillPending {
                        *unseenPermitPending = false;
                    }
                    true
                } else {
                    false
                };
                if cleared {
                    // Pick the glyph the spinner section would draw next iteration,
                    // so refocus doesn't flash IDLE while the agent is working.
                    let glyph = if *unseenPermitPending {
                        TITLE_PERMIT_GLYPH
                    } else if agentPanel.isActive() && currentTopic.is_some() {
                        titleSpinner.current()
                    } else {
                        TITLE_IDLE_GLYPH
                    };
                    writeTerminalTitle(glyph, currentTopic.as_deref());
                }
            }
            Event::FocusLost => {
                *windowFocused = false;
            }
            Event::Key(key) => {
                if key.kind == event::KeyEventKind::Release {
                    // Poll for more without blocking.
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                    continue;
                }

                hadInput = true;

                // Clear selections on content/navigation keys, but not system
                // shortcuts (Cmd+key on macOS) so Cmd+C doesn't nuke the highlight.
                if !key.modifiers.contains(KeyModifiers::SUPER) {
                    selState.termSelection = None;
                    selState.agentSelection = None;
                    agentPanel.textArea.clearSelection();
                }

                // Global keybindings.
                // Double-tap Ctrl+Q to quit — prevents accidental exits.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('q') {
                    const DOUBLE_TAP_WINDOW: Duration = Duration::from_secs(1);
                    if let Some(prev) = *lastQuitPress
                        && prev.elapsed() < DOUBLE_TAP_WINDOW
                    {
                        return Ok((true, true, false));
                    }
                    *lastQuitPress = Some(Instant::now());
                    break;
                }

                // Ctrl+L: force full terminal redraw to fix rendering artifacts.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
                    resized = true;
                    break;
                }

                // Ctrl+O: toggle the layout control panel. Press once
                // to open, press again to dismiss. Closing this way
                // does not run the unsaved-changes confirm — same as
                // any other global keybind dropping out of a popup.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
                    if layoutPanel.is_some() {
                        *layoutPanel = None;
                    } else {
                        *layoutPanel = Some(crate::layout::control_panel::ControlPanel::new(
                            sessionLayout.clone(),
                            activePresetName.clone(),
                        ));
                    }
                    break;
                }

                // Ctrl+H: toggle the hotkey-tips popup.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('h') {
                    *helpPopupOpen = !*helpPopupOpen;
                    break;
                }

                // Any key dismisses the help popup (and does not propagate).
                if *helpPopupOpen {
                    *helpPopupOpen = false;
                    break;
                }

                // Error-mode recovery bindings. Only active when the last
                // turn fatally errored (hint is showing). Both discard the
                // error state before dispatching the command.
                if agentPanel.errorHint && key.modifiers.contains(KeyModifiers::CONTROL) {
                    match key.code {
                        KeyCode::Char('r') => {
                            agentPanel.errorHint = false;
                            let _ = cancelTx.send(false);
                            spawnAckRequest(requestTx.clone(), deckUpdateTx.clone(), |reply| {
                                TuiRequest::RetryLastTurn { reply }
                            });
                            break;
                        }
                        KeyCode::Char(' ') => {
                            agentPanel.errorHint = false;
                            let _ = cancelTx.send(false);
                            spawnAckRequest(requestTx.clone(), deckUpdateTx.clone(), |reply| {
                                TuiRequest::ContinueLastTurn { reply }
                            });
                            break;
                        }
                        _ => {}
                    }
                }

                // Subagent panel overlay. Navigation keys (Tab, scroll, close)
                // are always consumed. During a subagent permit, action keys
                // (y/n/A/D, Shift+arrows, custom-pattern chars) fall through
                // to the permit dispatcher below.
                if let Some(ref mut panel) = *subagentPanel {
                    let consumed = panel.consumedKey(&key, agentPanel);
                    let shouldClose = panel.handleKey(key, agentPanel);
                    if shouldClose {
                        *subagentPanel = None;
                        break;
                    }
                    if consumed {
                        break;
                    }
                }

                // Cancel running turn with Escape — immediate visual feedback.
                if key.code == KeyCode::Esc && agentPanel.isActive() {
                    let _ = cancelTx.send(true);
                    agentPanel.finalizeCancelled();
                    break;
                }

                if key.code == KeyCode::Tab {
                    // Don't switch focus when an overlay is active or completion menu is open.
                    if sessionPicker.is_some()
                        || rewindPicker.is_some()
                        || forkPicker.is_some()
                        || mcpPanel.is_some()
                        || lspPanel.is_some()
                        || modelPanel.is_some()
                        || permissionsPanel.is_some()
                        || tasksPanel.is_some()
                        || logPanel.is_some()
                        || layoutPanel.is_some()
                    {
                        // Let the overlay handle Tab (falls through to overlay dispatch below).
                    } else if !(*focus == Focus::Agent && agentPanel.completionActive()) {
                        *focus = match focus {
                            Focus::Terminal => Focus::Agent,
                            Focus::Agent => Focus::Terminal,
                        };
                        break;
                    }
                }

                // Cmd+C: copy active selection to clipboard.
                if key.modifiers.contains(KeyModifiers::SUPER) && key.code == KeyCode::Char('c') {
                    if let Some(text) = agentPanel.textArea.selectedText() {
                        selection::copyToClipboard(&text);
                    } else if selState.agentSelection.is_some() {
                        selState.pendingCopy = Some(PanelId::Agent);
                    } else if selState.termSelection.is_some() {
                        selState.pendingCopy = Some(PanelId::Terminal);
                    }
                    break;
                }

                // Permission prompt takes priority regardless of focus.
                if agentPanel.pendingPermit {
                    use construct::permissions::PermitResponse;

                    // Resolve the oneshot reply stashed when SessionRequest::Permit arrived.
                    // Same path for top-level and subagent permits — the origin is folded
                    // into the request variant before it reaches the TUI.
                    macro_rules! sendPermit {
                        ($resp:expr) => {
                            if let Some(tx) = pendingPermitReply.take() {
                                let _ = tx.send($resp);
                            }
                            // Clear the unseen-permit title indicator immediately so
                            // the user doesn't have to leave-and-refocus to dismiss it.
                            if *unseenPermitPending {
                                *unseenPermitPending = false;
                                let glyph = if agentPanel.isActive() && currentTopic.is_some() {
                                    titleSpinner.current()
                                } else {
                                    TITLE_IDLE_GLYPH
                                };
                                writeTerminalTitle(glyph, currentTopic.as_deref());
                            }
                        };
                    }

                    match key.code {
                        KeyCode::Char('y') => {
                            agentPanel.approvePending();
                            sendPermit!(PermitResponse::Allow);
                        }
                        // Shift+A: always allow (persist to project config).
                        // An empty pattern would persist `keyArg.contains("")`
                        // — i.e. allow-all for the tool — so guard against it.
                        KeyCode::Char('A') => match agentPanel.selectedPattern() {
                            Some(pattern) => {
                                agentPanel.approvePending();
                                sendPermit!(PermitResponse::AlwaysAllow { pattern });
                            }
                            None => {
                                agentPanel.pushNotice(
                                        "\u{26A0}\u{FE0E} cannot persist an empty pattern \u{2014} \
                                         type one in the custom field first, or press y to allow once.",
                                    );
                            }
                        },
                        KeyCode::Char('n') => {
                            agentPanel.denyPending();
                            sendPermit!(PermitResponse::Deny);
                        }
                        // Shift+D: always deny (persist to project config).
                        KeyCode::Char('D') => match agentPanel.selectedPattern() {
                            Some(pattern) => {
                                agentPanel.denyPending();
                                sendPermit!(PermitResponse::AlwaysDeny { pattern });
                            }
                            None => {
                                agentPanel.pushNotice(
                                        "\u{26A0}\u{FE0E} cannot persist an empty pattern \u{2014} \
                                         type one in the custom field first, or press n to deny once.",
                                    );
                            }
                        },
                        // Shift+Up/Down: navigate pattern selector (patterns are for persistent decisions).
                        KeyCode::Up if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            agentPanel.prevPattern();
                        }
                        KeyCode::Down if key.modifiers.contains(KeyModifiers::SHIFT) => {
                            agentPanel.nextPattern();
                        }
                        // Custom pattern text input (when custom field is selected).
                        KeyCode::Char(c) if agentPanel.isEditingCustom() => {
                            agentPanel.customPatternInsert(c);
                        }
                        KeyCode::Backspace if agentPanel.isEditingCustom() => {
                            agentPanel.customPatternBackspace();
                        }
                        KeyCode::Char('v') | KeyCode::Char('V') => {
                            // Open the subagent panel — Live mode reads
                            // transcript + shell from agentPanel.activeSubagent.
                            if !agentPanel.activeSubagents.is_empty() {
                                *subagentPanel = Some(crate::subagent_panel::SubagentPanel::live());
                            }
                        }
                        _ => {}
                    }
                    break;
                }

                // Session picker intercepts all keys when active.
                if let Some(picker) = sessionPicker {
                    match picker.handleKey(key) {
                        PickerAction::Close => {
                            *sessionPicker = None;
                        }
                        PickerAction::Select(id) => {
                            // Picker stays open — closed on ResumeResult reply.
                            let requestTxC = requestTx.clone();
                            let deckTx = deckUpdateTx.clone();
                            tokio::spawn(async move {
                                let (rTx, rRx) = oneshot::channel();
                                let _ = requestTxC
                                    .send(TuiRequest::ResumeSession {
                                        sessionId: id,
                                        reply: rTx,
                                    })
                                    .await;
                                if let Ok(ack) = rRx.await {
                                    let _ = deckTx.send(DeckUpdate::ResumeResult(ack)).await;
                                }
                            });
                        }
                        PickerAction::None => {}
                    }
                    break;
                }

                // Tasks panel intercepts all keys when active.
                if let Some(panel) = tasksPanel.as_mut() {
                    use crate::jobs_panel::PanelAction;
                    match panel.handleKey(key) {
                        PanelAction::Close => {
                            *tasksPanel = None;
                        }
                        PanelAction::Kill(id) => {
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::KillTask { id, reply },
                            );
                        }
                        PanelAction::Inspect(id) => {
                            let requestTx2 = requestTx.clone();
                            let deckUpdateTx2 = deckUpdateTx.clone();
                            tokio::spawn(async move {
                                let (rTx, rRx) = oneshot::channel();
                                let _ = requestTx2
                                    .send(TuiRequest::GetTaskOutput {
                                        id,
                                        sinceLine: None,
                                        reply: rTx,
                                    })
                                    .await;
                                if let Ok(Some(snap)) = rRx.await {
                                    let _ = deckUpdateTx2
                                        .send(DeckUpdate::TaskOutputOpen { id, snap })
                                        .await;
                                }
                            });
                        }
                        PanelAction::Refetch(id) => {
                            // Page key flipped sinceLine — fetch with the
                            // panel's current pin. Spawn directly (the
                            // dedicated task-plane handler services it
                            // even mid-turn) instead of routing through
                            // the inspectorInFlight coalescer, which is
                            // for periodic refreshes coming off live
                            // TaskOutput events.
                            let sinceLine = panel.inspectorSinceLine();
                            spawnInspectorFetch(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                id,
                                sinceLine,
                            );
                        }
                        PanelAction::None => {}
                    }
                    break;
                }

                // Developer log panel intercepts all keys when active.
                if let Some(panel) = logPanel.as_mut() {
                    use crate::log_panel::PanelAction;
                    match panel.handleKey(key) {
                        PanelAction::Close => {
                            *logPanel = None;
                        }
                        PanelAction::None => {}
                    }
                    break;
                }

                // Rewind picker intercepts all keys when active.
                if let Some(picker) = rewindPicker {
                    match picker.handleKey(key) {
                        RewindAction::Close => {
                            *rewindPicker = None;
                        }
                        RewindAction::Rewind {
                            target,
                            userMessage,
                            attachments,
                        } => {
                            *pendingRewindMessage = userMessage;
                            *pendingRewindAttachments = attachments;
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::Rewind {
                                    target,
                                    saveFork: false,
                                    reply,
                                },
                            );
                        }
                        RewindAction::ForkAndRewind {
                            target,
                            userMessage,
                            attachments,
                        } => {
                            *pendingRewindMessage = userMessage;
                            *pendingRewindAttachments = attachments;
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::Rewind {
                                    target,
                                    saveFork: true,
                                    reply,
                                },
                            );
                        }
                        RewindAction::None => {}
                    }
                    break;
                }

                // Fork picker intercepts all keys when active.
                if let Some(picker) = forkPicker {
                    match picker.handleKey(key) {
                        ForkAction::Close => {
                            *forkPicker = None;
                        }
                        ForkAction::Switch(id) => {
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SwitchFork { forkId: id, reply },
                            );
                        }
                        ForkAction::None => {}
                    }
                    break;
                }

                // MCP panel intercepts all keys when active.
                if let Some(panel) = mcpPanel {
                    match panel.handleKey(key) {
                        McpPanelAction::Close => {
                            *mcpPanel = None;
                        }
                        McpPanelAction::None => {}
                    }
                    break;
                }

                // Layout control panel (Ctrl+O) intercepts all keys when active.
                if let Some(panel) = layoutPanel.as_mut() {
                    use crate::layout::control_panel::PanelAction as LayoutAction;
                    match panel.handleKey(key) {
                        LayoutAction::Close => {
                            *layoutPanel = None;
                        }
                        LayoutAction::ApplyPreset { name, layout } => {
                            *sessionLayout = layout.clone();
                            *activePresetName = Some(name.clone());
                            panel.confirmApplied(name, layout);
                        }
                        LayoutAction::Save => {
                            let path = sessionLayoutPath.clone().unwrap_or_else(|| {
                                // No discovered file — save to ~/.config/flatline/layout.toml.
                                crate::layout::discovery::configFallbackPath().unwrap_or_else(
                                    || std::path::PathBuf::from(".flatline").join("layout.toml"),
                                )
                            });
                            match crate::layout::discovery::writeLayout(&path, sessionLayout) {
                                Ok(()) => {
                                    *sessionLayoutPath = Some(path.clone());
                                    panel.confirmSaved();
                                    pushOperationalLog(
                                        developerLog,
                                        toastCenter,
                                        logPanel,
                                        LogLevel::Success,
                                        "layout",
                                        "layout saved",
                                        Some(path.display().to_string()),
                                        true,
                                    );
                                }
                                Err(e) => {
                                    pushOperationalLog(
                                        developerLog,
                                        toastCenter,
                                        logPanel,
                                        LogLevel::Error,
                                        "layout",
                                        "layout save failed",
                                        Some(e.to_string()),
                                        true,
                                    );
                                }
                            }
                        }
                        LayoutAction::Reset => {
                            let cwd = std::env::current_dir()
                                .unwrap_or_else(|_| std::path::PathBuf::from("."));
                            let (resetLayout, resetPath) =
                                match crate::layout::discovery::discoverLayout(&cwd) {
                                    Some(d) => match d.result {
                                        Ok(l) => (l, Some(d.path)),
                                        Err(_) => {
                                            (crate::layout::Layout::defaultPhase1(), Some(d.path))
                                        }
                                    },
                                    None => (crate::layout::Layout::defaultPhase1(), None),
                                };
                            let name = crate::layout::control_panel::matchPresetName(&resetLayout);
                            *sessionLayout = resetLayout.clone();
                            *sessionLayoutPath = resetPath;
                            *activePresetName = name.clone();
                            panel.resetTo(resetLayout, name);
                        }
                        LayoutAction::None => {}
                    }
                    break;
                }

                // LSP panel intercepts all keys when active.
                if let Some(panel) = lspPanel {
                    match panel.handleKey(key) {
                        LspPanelAction::Close => {
                            *lspPanel = None;
                        }
                        LspPanelAction::Install { serverId, command } => {
                            *lspPanel = None;
                            // Run install in the shared terminal.
                            agentPanel.pushCommandResult(&format!(
                                "\u{2699}\u{FE0E} Installing {serverId}: {command}",
                            ));
                            let cmdBytes = format!("{command}\n").into_bytes();
                            termPane.sendInput(cmdBytes);
                        }
                        LspPanelAction::None => {}
                    }
                    break;
                }

                // Model profile panel intercepts all keys when active.
                if let Some(panel) = modelPanel {
                    match panel.handleKey(key) {
                        ModelPanelAction::Close => {
                            *modelPanel = None;
                        }
                        ModelPanelAction::Save {
                            scope,
                            tier,
                            profile,
                        } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SaveModelSelection {
                                    scope,
                                    tier,
                                    profile,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::Discover { provider } => {
                            let requestTx = requestTx.clone();
                            let deckUpdateTx = deckUpdateTx.clone();
                            tokio::spawn(async move {
                                let (replyTx, replyRx) = oneshot::channel();
                                let _ = requestTx
                                    .send(TuiRequest::DiscoverModels {
                                        provider: provider.clone(),
                                        reply: replyTx,
                                    })
                                    .await;
                                let result = replyRx.await.unwrap_or_else(|_| {
                                    Err("model discovery task ended".to_string())
                                });
                                let _ = deckUpdateTx
                                    .send(DeckUpdate::ModelCatalog { provider, result })
                                    .await;
                            });
                        }
                        ModelPanelAction::SaveDiscoveredModel {
                            scope,
                            profile,
                            model,
                        } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SaveDiscoveredModel {
                                    scope,
                                    profile,
                                    model,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::CreateProfile {
                            scope,
                            profile,
                            sourceProfile,
                        } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::CreateModelProfile {
                                    scope,
                                    profile,
                                    sourceProfile,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::RenameProfile {
                            scope,
                            oldProfile,
                            newProfile,
                        } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::RenameModelProfile {
                                    scope,
                                    oldProfile,
                                    newProfile,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::DeleteProfile { scope, profile } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::DeleteModelProfile {
                                    scope,
                                    profile,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::SaveContext {
                            scope,
                            profile,
                            contextWindow,
                        } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SaveModelProfileContext {
                                    scope,
                                    profile,
                                    contextWindow,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::SaveThinking {
                            scope,
                            profile,
                            promptThinking,
                            reasoningEffort,
                            reasoningSummary,
                        } => {
                            spawnSilentSettingsRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SaveModelProfileThinking {
                                    scope,
                                    profile,
                                    promptThinking,
                                    reasoningEffort,
                                    reasoningSummary,
                                    reply,
                                },
                            );
                        }
                        ModelPanelAction::None => {}
                    }
                    break;
                }

                // Permissions panel intercepts all keys when active.
                if let Some(panel) = permissionsPanel {
                    use crate::permissions_panel::PermPanelAction;
                    match panel.handleKey(key) {
                        PermPanelAction::Close => {
                            *permissionsPanel = None;
                        }
                        PermPanelAction::Save { defaultMode, rules } => {
                            *permissionsPanel = None;
                            spawnAckRequest(
                                requestTx.clone(),
                                deckUpdateTx.clone(),
                                move |reply| TuiRequest::SavePermissions {
                                    defaultMode,
                                    rules,
                                    reply,
                                },
                            );
                        }
                        PermPanelAction::None => {}
                    }
                    break;
                }

                // Status-bar chip navigation. When a chip is focused,
                // intercept arrows/Enter/Esc so the keys steer the bar
                // instead of falling through to the terminal/agent panes.
                if let Some(idx) = *statusFocus {
                    let count = statusChipsLayout.len();
                    match key.code {
                        KeyCode::Left => {
                            if idx > 0 {
                                *statusFocus = Some(idx - 1);
                            }
                            break;
                        }
                        KeyCode::Right => {
                            if idx + 1 < count {
                                *statusFocus = Some(idx + 1);
                            }
                            break;
                        }
                        KeyCode::Esc | KeyCode::Up => {
                            *statusFocus = None;
                            break;
                        }
                        KeyCode::Enter | KeyCode::Char(' ') => {
                            if let Some((kind, _, _, _)) = statusChipsLayout.get(idx) {
                                openStatusChipPanel(*kind, requestTx, deckUpdateTx);
                            }
                            *statusFocus = None;
                            break;
                        }
                        _ => {
                            // Any other key drops chip focus and falls through.
                            *statusFocus = None;
                        }
                    }
                }

                match focus {
                    Focus::Terminal => {
                        if key.modifiers.contains(KeyModifiers::SUPER) {
                            if !event::poll(Duration::ZERO)? {
                                break;
                            }
                            continue;
                        }
                        // Ctrl+T spawns a new terminal for the user.
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('t')
                        {
                            // Fire-and-forget — the new ShellIo arrives via
                            // shellIoRx and the matching tab is added on the
                            // main loop's drain pass.
                            let (replyTx, _replyRx) = oneshot::channel();
                            let _ = requestTx
                                .send(TuiRequest::SpawnTerminal {
                                    name: None,
                                    reply: replyTx,
                                })
                                .await;
                        }
                        // Ctrl+1..9 jumps to a specific tab.
                        else if key.modifiers.contains(KeyModifiers::CONTROL)
                            && matches!(key.code, KeyCode::Char(c) if c.is_ascii_digit() && c != '0')
                        {
                            if let KeyCode::Char(c) = key.code {
                                let idx = (c as u8 - b'1') as usize;
                                termPane.jumpTo(idx);
                            }
                        }
                        // Ctrl+C triggers the killchain for captured commands.
                        else if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c')
                        {
                            termPane.sendKill();
                        } else if let Some(bytes) = keyToBytes(&key) {
                            let termState = termPane.activeStateMut();
                            if termState.displayOffset() > 0 {
                                termState.scrollToBottom();
                            }
                            termPane.sendInput(bytes);
                        }
                    }
                    Focus::Agent => {
                        let mods = key.modifiers;
                        let ctrl = mods.contains(KeyModifiers::CONTROL);
                        let shift = mods.contains(KeyModifiers::SHIFT);
                        let supr = mods.contains(KeyModifiers::SUPER);
                        let alt = mods.contains(KeyModifiers::ALT);

                        // Completion menu intercepts (before textarea borrow).
                        let mut completionHandled = false;
                        if agentPanel.completionActive() {
                            completionHandled = true;
                            match key.code {
                                KeyCode::Tab => {
                                    if let Some(completed) = agentPanel.completeSelected() {
                                        agentPanel.textArea.setText(&completed);
                                    }
                                }
                                KeyCode::Up => agentPanel.selectPrev(),
                                KeyCode::Down => agentPanel.selectNext(),
                                KeyCode::Esc => agentPanel.dismissCompletion(),
                                KeyCode::Enter if !shift => {
                                    // Accept completion then fall through to execute.
                                    if let Some(completed) = agentPanel.completeSelected() {
                                        agentPanel.textArea.setText(&completed);
                                    }
                                    completionHandled = false;
                                }
                                _ => completionHandled = false,
                            }
                        }

                        let mut navigatingHistory = false;
                        if !completionHandled {
                            // Handle attachment-related keys before borrowing textArea.
                            if ctrl
                                && key.code == KeyCode::Char('d')
                                && agentPanel.attachmentCount() > 0
                            {
                                agentPanel.removeLastAttachment();
                                break;
                            }
                            if ctrl && key.code == KeyCode::Char('v') {
                                if let Ok(mut cb) = arboard::Clipboard::new() {
                                    if let Ok(imgData) = cb.get_image() {
                                        // Store raw RGBA — PNG encoding deferred to submit.
                                        agentPanel.addAttachment(construct::session::Attachment {
                                            mimeType: "image/rgba".into(),
                                            data: imgData.bytes.to_vec(),
                                            label: format!(
                                                "pasted image ({}x{})",
                                                imgData.width, imgData.height,
                                            ),
                                            rgbaDimensions: Some((
                                                imgData.width as u32,
                                                imgData.height as u32,
                                            )),
                                        });
                                    } else if let Ok(text) = cb.get_text() {
                                        let normalized =
                                            text.replace("\r\n", "\n").replace('\r', "\n");
                                        agentPanel.textArea.insertStr(&normalized);
                                    }
                                }
                                break;
                            }

                            // Pop queued message on Up before borrowing textArea.
                            if key.code == KeyCode::Up
                                && agentPanel.textArea.isAtFirstLine()
                                && agentPanel.isActive()
                                && agentPanel.queuedCount() > 0
                            {
                                if let Some(input) = agentPanel.popQueuedMessage() {
                                    agentPanel.textArea.setText(&input.text);
                                    agentPanel.restoreAttachments(input.attachments);
                                }
                                break;
                            }

                            let ta = &mut agentPanel.textArea;
                            match key.code {
                                KeyCode::Enter if shift => ta.insert('\n'),
                                KeyCode::Enter => {
                                    if let Some(msg) = ta.submit() {
                                        if let Some(output) = crate::command::tryHandle(&msg) {
                                            agentPanel.history.push(&msg);
                                            agentPanel.pushCommand(&msg);
                                            match output {
                                                crate::command::CommandOutput::Inline(text) => {
                                                    agentPanel.pushCommandResult(&text);
                                                }
                                                crate::command::CommandOutput::Action(
                                                    crate::command::CommandAction::Resume {
                                                        sessionId: None,
                                                    },
                                                ) => {
                                                    // Open the interactive picker.
                                                    *sessionPicker =
                                                        Some(SessionPicker::new(projectDir));
                                                }
                                                crate::command::CommandOutput::Action(
                                                    crate::command::CommandAction::ShowLayout,
                                                ) => {
                                                    // /layout opens the Ctrl+O panel. Path
                                                    // + applied-preset info appears in the
                                                    // panel header.
                                                    if layoutPanel.is_none() {
                                                        *layoutPanel = Some(
                                                            crate::layout::control_panel::ControlPanel::new(
                                                                sessionLayout.clone(),
                                                                activePresetName.clone(),
                                                            ),
                                                        );
                                                    }
                                                }
                                                crate::command::CommandOutput::Action(
                                                    crate::command::CommandAction::Logs,
                                                ) => {
                                                    *logPanel = Some(LogPanel::new(
                                                        developerLog.snapshot(),
                                                    ));
                                                }
                                                crate::command::CommandOutput::Action(action) => {
                                                    dispatchSlashCommand(
                                                        action,
                                                        requestTx.clone(),
                                                        deckUpdateTx.clone(),
                                                    );
                                                }
                                            }
                                        } else if agentPanel.isActive() {
                                            // Queue for mid-turn injection.
                                            agentPanel.history.push(&msg);
                                            let input = construct::session::UserInput {
                                                text: msg,
                                                attachments: agentPanel.takeAttachments(),
                                            };
                                            agentPanel.queueMessage(input.clone());
                                            let _ = steerTx.try_send(input);
                                        } else {
                                            let _ = cancelTx.send(false);
                                            agentPanel.history.push(&msg);
                                            agentPanel.pushUser(&msg);
                                            // New user input — supersedes any prior error state.
                                            agentPanel.errorHint = false;
                                            let input = construct::session::UserInput {
                                                text: msg,
                                                attachments: agentPanel.takeAttachments(),
                                            };
                                            let _ = userInputTx.send(input).await;
                                        }
                                    }
                                }
                                KeyCode::Char('a') if ctrl => ta.moveHome(),
                                KeyCode::Char('e') if ctrl => ta.moveEnd(),
                                KeyCode::Char('k') if ctrl => ta.killToEnd(),
                                KeyCode::Char('u') if ctrl => ta.killToStart(),
                                KeyCode::Char('y') if ctrl => ta.yank(),
                                KeyCode::Char('t') if ctrl => agentPanel.toggleThinking(),
                                KeyCode::Char('b') if ctrl => {
                                    // Send the in-flight shell command to background.
                                    let _ = userBgTx.try_send(());
                                }
                                KeyCode::Char(c) if !supr => ta.insert(c),
                                KeyCode::Backspace if supr => ta.killToStart(),
                                KeyCode::Backspace if alt => ta.deleteWordLeft(),
                                KeyCode::Backspace => ta.backspace(),
                                KeyCode::Delete => ta.delete(),
                                KeyCode::Left if supr => ta.moveHome(),
                                KeyCode::Right if supr => ta.moveEnd(),
                                KeyCode::Left if alt => ta.moveWordLeft(),
                                KeyCode::Right if alt => ta.moveWordRight(),
                                KeyCode::Left if ctrl => ta.moveWordLeft(),
                                KeyCode::Right if ctrl => ta.moveWordRight(),
                                KeyCode::Left => ta.moveLeft(),
                                KeyCode::Right => ta.moveRight(),
                                KeyCode::Home => ta.moveHome(),
                                KeyCode::End => ta.moveEnd(),
                                KeyCode::Up => {
                                    if ta.isAtFirstLine() {
                                        navigatingHistory = true;
                                        let currentText = ta.text().to_string();
                                        if let Some(entry) =
                                            agentPanel.history.navigateUp(&currentText)
                                        {
                                            let entry = entry.to_string();
                                            ta.setText(&entry);
                                        }
                                    } else {
                                        ta.moveUp();
                                    }
                                }
                                KeyCode::Down => {
                                    if ta.isAtLastLine() {
                                        navigatingHistory = true;
                                        if let Some(entry) = agentPanel.history.navigateDown() {
                                            let entry = entry.to_string();
                                            ta.setText(&entry);
                                        } else if !statusChipsLayout.is_empty() {
                                            // Past the end of history → land
                                            // focus on the first chip so the
                                            // bar acts like a menu reachable
                                            // by walking down.
                                            *statusFocus = Some(0);
                                        }
                                    } else {
                                        ta.moveDown();
                                    }
                                }
                                KeyCode::PageUp => agentPanel.scrollUp(10),
                                KeyCode::PageDown => agentPanel.scrollDown(10),
                                _ => {}
                            }
                        }

                        // Reset history navigation on any non-navigation key.
                        if !navigatingHistory {
                            agentPanel.history.resetCursor();
                        }

                        // Update completion after every keystroke.
                        let currentText = agentPanel.textArea.text().to_string();
                        agentPanel.updateCompletion(&currentText);
                    }
                }
            }
            Event::Mouse(mouse) => {
                // Subagent overlay: route ALL mouse events here when open.
                // Click-outside closes the popup; click-inside on the permit
                // overlay falls through to the main mouse handler so the
                // existing permit click logic (copy button etc.) still works.
                if let Some(ref mut panel) = *subagentPanel {
                    use crate::subagent_panel::SubagentMouseAction;
                    match panel.handleMouse(mouse, agentPanel) {
                        SubagentMouseAction::Handled => {
                            hadInput = true;
                        }
                        SubagentMouseAction::ClickOutside => {
                            *subagentPanel = None;
                            hadInput = true;
                        }
                    }
                    break;
                }
                // Other overlay panels consume all mouse events to prevent click-through.
                if sessionPicker.is_some()
                    || rewindPicker.is_some()
                    || forkPicker.is_some()
                    || mcpPanel.is_some()
                    || lspPanel.is_some()
                    || modelPanel.is_some()
                    || permissionsPanel.is_some()
                    || tasksPanel.is_some()
                    || logPanel.is_some()
                    || layoutPanel.is_some()
                {
                    break;
                }
                // Click on a status-bar chip opens its panel directly.
                // Hover/move highlights without opening.
                if let Some((kind, hit)) = chipHitTest(&mouse, statusChipsLayout) {
                    match mouse.kind {
                        MouseEventKind::Down(MouseButton::Left) => {
                            openStatusChipPanel(kind, requestTx, deckUpdateTx);
                            *statusFocus = None;
                            hadInput = true;
                            break;
                        }
                        MouseEventKind::Moved => {
                            *statusFocus = Some(hit);
                            hadInput = true;
                            break;
                        }
                        _ => {}
                    }
                }
                if handleMouse(
                    mouse,
                    focus,
                    agentPanel,
                    termPane,
                    selState,
                    scrollLock,
                    subagentPanel,
                    requestTx,
                ) {
                    hadInput = true;
                }
            }
            Event::Paste(text) => {
                hadInput = true;
                // Popup overlay swallows paste — main panel input is covered.
                if subagentPanel.is_some() {
                    if agentPanel.pendingPermit && agentPanel.isEditingCustom() {
                        for ch in text.chars() {
                            agentPanel.customPatternInsert(ch);
                        }
                    }
                    break;
                }
                match *focus {
                    Focus::Agent if !agentPanel.pendingPermit => {
                        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                        agentPanel.textArea.insertStr(&normalized);
                    }
                    Focus::Terminal => {
                        let bracketed = {
                            let termState = termPane.activeStateMut();
                            if termState.displayOffset() > 0 {
                                termState.scrollToBottom();
                            }
                            termState.bracketedPaste()
                        };
                        // Shells treat CR as "execute"; normalize to LF so multi-line
                        // pastes land as a single buffered command when possible.
                        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                        let payload = if bracketed {
                            let mut buf = Vec::with_capacity(normalized.len() + 12);
                            buf.extend_from_slice(b"\x1b[200~");
                            buf.extend_from_slice(normalized.as_bytes());
                            buf.extend_from_slice(b"\x1b[201~");
                            buf
                        } else {
                            normalized.into_bytes()
                        };
                        termPane.sendInput(payload);
                    }
                    _ => {}
                }
            }
            Event::Resize(cols, rows) => {
                hadInput = true;
                resized = true;
                let termCols = (cols * 3 / 5).saturating_sub(2);
                // Reserve 1 row for tab strip + 1 for status + 2 for borders.
                let termRows = rows.saturating_sub(4);
                termPane.resizeAll(termCols, termRows);
            }
        }

        // Keep draining if more events are queued.
        if !event::poll(Duration::ZERO)? {
            break;
        }
    }

    Ok((false, hadInput, resized))
}

/// Handle mouse events — selection, scroll wheel.
/// Returns true if the event modified state (needs redraw).
#[allow(clippy::too_many_arguments)]
fn handleMouse(
    mouse: event::MouseEvent,
    focus: &mut Focus,
    agentPanel: &mut AgentPanel,
    termPane: &mut TerminalPane,
    selState: &mut SelectionState,
    scrollLock: &mut ScrollAxisLock,
    subagentPanel: &mut Option<crate::subagent_panel::SubagentPanel>,
    requestTx: &mpsc::Sender<TuiRequest>,
) -> bool {
    // Resolve display offset for the given panel.
    fn panelOffset(panel: PanelId, termPane: &TerminalPane, agentPanel: &AgentPanel) -> u16 {
        match panel {
            PanelId::Terminal => termPane.activeStateRef().displayOffset() as u16,
            PanelId::Agent => agentPanel.displayOffset(),
            PanelId::Input => 0,
        }
    }

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            // Tab strip — click on a tab to switch, click [+] to spawn.
            if termPane.clickInTabBar(mouse.column, mouse.row) {
                use crate::terminal_pane::TabClick;
                match termPane.handleClick(mouse.column, mouse.row) {
                    TabClick::Switch(name) => {
                        termPane.setActive(&name);
                        *focus = Focus::Terminal;
                        return true;
                    }
                    TabClick::NewTab => {
                        let (replyTx, _replyRx) = oneshot::channel();
                        let _ = requestTx.try_send(TuiRequest::SpawnTerminal {
                            name: None,
                            reply: replyTx,
                        });
                        return true;
                    }
                    TabClick::Close(name) => {
                        let (replyTx, _replyRx) = oneshot::channel();
                        let _ = requestTx.try_send(TuiRequest::KillTerminal {
                            name,
                            reply: replyTx,
                        });
                        return true;
                    }
                    TabClick::Empty => {}
                }
            }

            // Check input area first (not part of panel hit-testing).
            if selState
                .inputContentRect
                .contains((mouse.column, mouse.row).into())
            {
                // Permission prompt: check for copy button click on code block top border.
                if agentPanel.pendingPermit {
                    let localRow = mouse.row.saturating_sub(selState.inputContentRect.y);
                    let localCol = mouse.column.saturating_sub(selState.inputContentRect.x);
                    // Top border row — check if click is on "copy" label.
                    if localRow == 0
                        && localCol + 6 >= selState.inputContentRect.width
                        && let Some(cmd) = agentPanel.pendingCommand()
                    {
                        crate::selection::copyToClipboard(cmd);
                        agentPanel.flashCopied();
                    }
                    return true;
                }
                *focus = Focus::Agent;
                let localCol = mouse.column.saturating_sub(selState.inputContentRect.x);
                let localRow = mouse.row.saturating_sub(selState.inputContentRect.y);
                let contentWidth = selState.inputContentRect.width.saturating_sub(2);
                let contentCol = localCol.saturating_sub(2);
                agentPanel
                    .textArea
                    .mouseDown(contentCol, localRow, contentWidth);
                selState.selectingIn = Some(PanelId::Input);
                selState.termSelection = None;
                selState.agentSelection = None;
                return true;
            }

            // Clear input selection when clicking in a panel.
            agentPanel.textArea.clearSelection();

            let panel = selState.hitTest(mouse.column, mouse.row);
            if let Some(panel) = panel {
                *focus = match panel {
                    PanelId::Terminal => Focus::Terminal,
                    PanelId::Agent | PanelId::Input => Focus::Agent,
                };

                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let gridLine =
                    selection::toGridLine(screenRow, panelOffset(panel, termPane, agentPanel));

                // Single click on a reasoning header toggles it.
                if panel == PanelId::Agent && agentPanel.toggleReasoningAtGridLine(gridLine) {
                    return true;
                }

                // Click on code block "copy" label copies the block content.
                if panel == PanelId::Agent && agentPanel.tryCopyCodeBlock(gridLine, col) {
                    return true;
                }

                // Click on code block top/bottom border toggles expand/collapse.
                if panel == PanelId::Agent && agentPanel.tryToggleCodeBlock(gridLine) {
                    return true;
                }

                // Click on subagent header [view] opens the overlay panel.
                if panel == PanelId::Agent && agentPanel.isSubagentHeaderLine(gridLine) {
                    if !agentPanel.activeSubagents.is_empty() {
                        // Live subagent — popup reads from agentPanel each frame.
                        // Clicking opens with the current selected tab; the
                        // user can cycle with [ / ] inside the popup.
                        *subagentPanel = Some(crate::subagent_panel::SubagentPanel::live());
                        return true;
                    } else if let Some((agentType, sid)) = agentPanel.lastSubagentSession() {
                        // Resumed session — load child transcript on demand.
                        let agentType = agentType.to_string();
                        let sid = sid.to_string();
                        if let Ok(transcript) = construct::transcript::Transcript::open(&sid)
                            && let Ok(turns) = transcript.loadAll()
                        {
                            // Build a temporary AgentPanel to replay the
                            // child transcript into PanelEntries.
                            let mut tmp = crate::agent_panel::AgentPanel::new();
                            replayTranscript(&mut tmp, &turns);
                            let entries = tmp.entries;
                            *subagentPanel = Some(crate::subagent_panel::SubagentPanel::frozen(
                                &agentType, entries,
                            ));
                            return true;
                        }
                    }
                }

                // Click on subagent content toggle (expand/collapse).
                if panel == PanelId::Agent && agentPanel.tryToggleSubagentContent(gridLine) {
                    return true;
                }

                let isAlt = mouse.modifiers.contains(KeyModifiers::ALT);

                if isAlt {
                    *selState.selectionForMut(panel) =
                        Some(selection::Selection::newRectangular(col, gridLine));
                    selState.selectingIn = Some(panel);
                } else {
                    let clickCount = selState.click.record(col, screenRow);
                    *selState.selectionForMut(panel) =
                        Some(selection::Selection::new(col, gridLine));
                    selState.selectingIn = Some(panel);

                    if clickCount >= 2 {
                        selState.pendingExpand = Some((panel, clickCount));
                    }
                }

                // Clear selection in the other panel.
                match panel {
                    PanelId::Terminal => {
                        selState.agentSelection = None;
                    }
                    PanelId::Agent => {
                        selState.termSelection = None;
                    }
                    PanelId::Input => {}
                }
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if selState.selectingIn == Some(PanelId::Input) {
                let localCol = mouse
                    .column
                    .saturating_sub(selState.inputContentRect.x)
                    .min(selState.inputContentRect.width.saturating_sub(1));
                let localRow = mouse
                    .row
                    .saturating_sub(selState.inputContentRect.y)
                    .min(selState.inputContentRect.height.saturating_sub(1));
                let contentWidth = selState.inputContentRect.width.saturating_sub(2);
                let contentCol = localCol.saturating_sub(2);
                agentPanel
                    .textArea
                    .mouseDrag(contentCol, localRow, contentWidth);
                return true;
            }
            if let Some(panel) = selState.selectingIn {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine =
                    selection::toGridLine(screenRow, panelOffset(panel, termPane, agentPanel));
                if let Some(sel) = selState.selectionForMut(panel) {
                    sel.update(col, gridLine);
                }
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            if selState.selectingIn == Some(PanelId::Input) {
                selState.selectingIn = None;
                if let Some(text) = agentPanel.textArea.selectedText() {
                    selection::copyToClipboard(&text);
                }
                return true;
            }
            if let Some(panel) = selState.selectingIn.take() {
                let (col, screenRow) = selState.toLocal(panel, mouse.column, mouse.row);
                let (col, screenRow) = selState.clampLocal(panel, col, screenRow);
                let gridLine =
                    selection::toGridLine(screenRow, panelOffset(panel, termPane, agentPanel));

                if let Some(sel) = selState.selectionForMut(panel) {
                    sel.update(col, gridLine);
                    sel.finalize();

                    if sel.isEmpty() {
                        *selState.selectionForMut(panel) = None;
                    } else {
                        selState.pendingCopy = Some(panel);
                    }
                }
            }
        }
        MouseEventKind::ScrollUp => {
            if !scrollLock.allow(ScrollAxis::Vertical) {
                return false;
            }
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => {
                    if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                        // Shift+ScrollUp = scroll code block left.
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                        let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                        if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                            agentPanel.scrollCodeBlockH(blockId, -3);
                        } else {
                            agentPanel.scrollUp(3);
                        }
                    } else {
                        agentPanel.scrollUp(3);
                    }
                }
                Some(PanelId::Terminal) => {
                    termPane.activeStateMut().scrollUp(3);
                    // Extend selection into scrollback during drag.
                    if selState.selectingIn == Some(PanelId::Terminal) {
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Terminal, mouse.column, mouse.row);
                        let offset = termPane.activeStateRef().displayOffset() as u16;
                        if let Some(sel) = &mut selState.termSelection {
                            sel.extendRow(selection::toGridLine(screenRow, offset));
                        }
                    }
                }
                Some(PanelId::Input) | None => {}
            }
        }
        MouseEventKind::ScrollDown => {
            if !scrollLock.allow(ScrollAxis::Vertical) {
                return false;
            }
            match selState.hitTest(mouse.column, mouse.row) {
                Some(PanelId::Agent) => {
                    if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                        // Shift+ScrollDown = scroll code block right.
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                        let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                        if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                            agentPanel.scrollCodeBlockH(blockId, 3);
                        } else {
                            agentPanel.scrollDown(3);
                        }
                    } else {
                        agentPanel.scrollDown(3);
                    }
                }
                Some(PanelId::Terminal) => {
                    termPane.activeStateMut().scrollDown(3);
                    if selState.selectingIn == Some(PanelId::Terminal) {
                        let (_, screenRow) =
                            selState.toLocal(PanelId::Terminal, mouse.column, mouse.row);
                        let offset = termPane.activeStateRef().displayOffset() as u16;
                        if let Some(sel) = &mut selState.termSelection {
                            sel.extendRow(selection::toGridLine(screenRow, offset));
                        }
                    }
                }
                Some(PanelId::Input) | None => {}
            }
        }
        MouseEventKind::ScrollLeft => {
            if !scrollLock.allow(ScrollAxis::Horizontal) {
                return false;
            }
            // Permission code block scroll (in input area).
            if agentPanel.pendingPermit
                && selState
                    .inputContentRect
                    .contains((mouse.column, mouse.row).into())
            {
                agentPanel.scrollPermitCode(-3);
            } else if let Some(PanelId::Agent) = selState.hitTest(mouse.column, mouse.row) {
                let (_, screenRow) = selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                    agentPanel.scrollCodeBlockH(blockId, -3);
                }
            }
        }
        MouseEventKind::ScrollRight => {
            if !scrollLock.allow(ScrollAxis::Horizontal) {
                return false;
            }
            if agentPanel.pendingPermit
                && selState
                    .inputContentRect
                    .contains((mouse.column, mouse.row).into())
            {
                agentPanel.scrollPermitCode(3);
            } else if let Some(PanelId::Agent) = selState.hitTest(mouse.column, mouse.row) {
                let (_, screenRow) = selState.toLocal(PanelId::Agent, mouse.column, mouse.row);
                let gridLine = selection::toGridLine(screenRow, agentPanel.displayOffset());
                if let Some(blockId) = agentPanel.codeBlockAtGridLine(gridLine) {
                    agentPanel.scrollCodeBlockH(blockId, 3);
                }
            }
        }
        _ => {
            return false;
        }
    }
    true
}

/// Convert a crossterm key event to raw bytes for the PTY.
fn keyToBytes(key: &event::KeyEvent) -> Option<Vec<u8>> {
    match key.code {
        KeyCode::Char(c) => {
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                let ctrl = (c as u8).wrapping_sub(b'a').wrapping_add(1);
                Some(vec![ctrl])
            } else {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                Some(s.as_bytes().to_vec())
            }
        }
        KeyCode::Enter => Some(vec![b'\r']),
        KeyCode::Backspace => Some(vec![0x7f]),
        KeyCode::Tab => Some(vec![b'\t']),
        KeyCode::Esc => Some(vec![0x1b]),
        KeyCode::Up => Some(b"\x1b[A".to_vec()),
        KeyCode::Down => Some(b"\x1b[B".to_vec()),
        KeyCode::Right => Some(b"\x1b[C".to_vec()),
        KeyCode::Left => Some(b"\x1b[D".to_vec()),
        KeyCode::Home => Some(b"\x1b[H".to_vec()),
        KeyCode::End => Some(b"\x1b[F".to_vec()),
        KeyCode::Delete => Some(b"\x1b[3~".to_vec()),
        _ => None,
    }
}

/// Status-bar cache-heat indicator — just the glyph.
///
/// Returns a single glyph span, color-coded by how long it's been since the
/// last cache hit. Empty vec when caching is disabled. The caller appends
/// the literal word " cache" so the user can't mistake the glyph for a
/// sampling-temperature indicator.
fn cacheHeatSpans(
    enabled: bool,
    lastHit: Option<Instant>,
    barBg: Color,
    _barFg: Color,
) -> Vec<Span<'static>> {
    if !enabled {
        return Vec::new();
    }

    let glyphSpan = if let Some(ts) = lastHit {
        let elapsed = ts.elapsed().as_secs();
        let (glyph, fg, dim) = if elapsed < 60 {
            ("\u{2668}\u{FE0E}", Color::Red, false)
        } else if elapsed < 180 {
            ("\u{2668}\u{FE0E}", Color::LightRed, false)
        } else if elapsed < 300 {
            ("\u{2668}\u{FE0E}", Color::Yellow, true)
        } else {
            ("\u{2744}\u{FE0E}", Color::Cyan, false)
        };
        let mut style = Style::default().bg(barBg).fg(fg);
        if dim {
            style = style.add_modifier(Modifier::DIM);
        }
        Span::styled(glyph.to_string(), style)
    } else {
        // Caching configured but no hit yet this session.
        Span::styled(
            "\u{25CB}".to_string(),
            Style::default().bg(barBg).fg(Color::Cyan),
        )
    };

    vec![glyphSpan]
}

/// Short "ctx 3%" string for the status bar. Empty when no context consumed yet.
fn formatContextPct(tokens: usize, window: usize) -> String {
    if tokens == 0 || window == 0 {
        return String::new();
    }
    let pct = (tokens as f64 / window as f64 * 100.0).round() as usize;
    format!("ctx {pct}%")
}

/// Render the hotkey-tips popup — a centered dialog listing every bound
/// shortcut. Dismisses on any keypress (handled in input loop).
fn renderHelpPopup(area: Rect, buf: &mut ratatui::buffer::Buffer) {
    const ROWS: &[(&str, &str)] = &[
        ("Tab", "Switch focus between terminal and agent"),
        ("Esc", "Cancel running turn / close overlay"),
        ("Ctrl+T", "Spawn a new terminal"),
        ("Ctrl+B", "Hand a running shell to the agent (background)"),
        ("Ctrl+1..9", "Jump to terminal tab N"),
        ("Ctrl+Q \u{00d7}2", "Quit flatline (double-tap)"),
        ("Ctrl+L", "Force terminal redraw"),
        ("Ctrl+H", "Toggle this help"),
        (
            "\u{2191} / \u{2193}",
            "Scroll / history navigation in agent panel",
        ),
        (
            "\u{2193} (after history)",
            "Focus the status bar; \u{2190}/\u{2192} cycle, Enter opens",
        ),
    ];

    // Center a 56x(rows+4) box inside the given area.
    let innerW: u16 = 56;
    let innerH: u16 = ROWS.len() as u16 + 4;
    let w = innerW.min(area.width.saturating_sub(2));
    let h = innerH.min(area.height.saturating_sub(2));
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let popup = Rect {
        x,
        y,
        width: w,
        height: h,
    };

    // Clear underlying cells so the popup reads cleanly.
    Clear.render(popup, buf);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" hotkeys \u{2014} any key to dismiss ")
        .style(Style::default().bg(Color::Black).fg(Color::White));
    let inner = block.inner(popup);
    block.render(popup, buf);

    let lines: Vec<Line<'static>> = ROWS
        .iter()
        .map(|(key, desc)| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{:<10}", key),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(*desc),
            ])
        })
        .collect();
    Paragraph::new(lines).render(inner, buf);
}

/// Extract text from the terminal selection, rejoining soft-wrapped lines.
///
/// Uses alacritty's WRAPLINE flag on the last cell of each row to detect
/// lines that were soft-wrapped by the terminal emulator.
fn extractTerminalUnwrapped(
    sel: &selection::Selection,
    area: ratatui::layout::Rect,
    buf: &ratatui::buffer::Buffer,
    displayOffset: u16,
    termState: &TerminalState,
) -> String {
    if sel.isEmpty() {
        return String::new();
    }

    let ((sc, sr), (ec, er)) = sel.sorted();
    let mut segments: Vec<(String, bool)> = Vec::new();

    for gridLine in sr..=er {
        let screenRow = match selection::toScreenRow(gridLine, displayOffset, area.height) {
            Some(r) => r,
            None => continue,
        };

        let colStart = if gridLine == sr { sc } else { 0 };
        let colEnd = if gridLine == er { ec } else { area.width };

        let mut line = String::new();
        for col in colStart..colEnd {
            if col >= area.width {
                break;
            }
            if let Some(cell) = buf.cell((area.x + col, area.y + screenRow)) {
                line.push_str(cell.symbol());
            }
        }
        let trimmed = line.trim_end().to_string();

        // Check if the PREVIOUS line was soft-wrapped (making this a continuation).
        let isCont = gridLine > sr && termState.isLineWrapped(gridLine - 1);
        segments.push((trimmed, isCont));
    }

    // Remove trailing empty lines.
    while segments.last().is_some_and(|(l, _)| l.is_empty()) {
        segments.pop();
    }

    let mut result = String::new();
    for (i, (line, isCont)) in segments.iter().enumerate() {
        if i > 0 {
            if *isCont {
                // Soft-wrapped continuation — join without newline.
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

/// Build a one-line replay notice from a wake-turn transcript content.
/// The content is the `<wakes count="N">…</wakes>` envelope written by
/// `Session::injectWakeBatch`; we just want a compact summary line.
fn wakeNoticeFromContent(content: &str) -> String {
    let count = content
        .split_once("count=\"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .and_then(|(n, _)| n.parse::<usize>().ok())
        .unwrap_or(0);
    let firstSource = content
        .split_once("source=\"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .map(|(s, _)| s.to_string());

    match (count, firstSource) {
        (n, Some(src)) if n > 1 => format!("\u{2299} {n} wakes (first: {src})"),
        (_, Some(src)) => format!("\u{2299} wake \u{00B7} {src}"),
        (n, None) if n > 0 => format!("\u{2299} {n} wakes"),
        _ => "\u{2299} wake".to_string(),
    }
}

/// Convert deck's CommandAction to construct's CommandAction.
/// Replay transcript turns into the agent panel for a resumed session.
fn replayTranscript(panel: &mut AgentPanel, turns: &[construct::transcript::Turn]) {
    use construct::transcript::TurnRole;

    // Track pending task (subagent) calls: toolCallId -> entry index.
    let mut pendingTasks: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for turn in turns {
        match turn.role {
            TurnRole::User => {
                let display = match &turn.attachments {
                    Some(atts) if !atts.is_empty() => {
                        let n = atts.len();
                        let suffix = if n == 1 {
                            "1 image".to_string()
                        } else {
                            format!("{n} images")
                        };
                        format!("{}\n[+{suffix} attached]", turn.content)
                    }
                    _ => turn.content.clone(),
                };
                panel
                    .entries
                    .push(crate::agent_panel::PanelEntry::User(display));
            }
            TurnRole::Wake => {
                // Wake turns replay as a notice, not a user bubble. The
                // model's history still sees the user-shaped content via
                // context::reconstruct; this is purely display.
                let summary = wakeNoticeFromContent(&turn.content);
                panel
                    .entries
                    .push(crate::agent_panel::PanelEntry::SessionNotice(summary));
            }
            TurnRole::Assistant => {
                if let Some(ref reasoning) = turn.reasoning
                    && !reasoning.is_empty()
                {
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::Reasoning {
                            text: reasoning.clone(),
                            expanded: false,
                        });
                }
                if !turn.content.is_empty() {
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::Assistant(
                            turn.content.clone(),
                        ));
                }
            }
            TurnRole::ToolCall => {
                let name = turn.tool.as_deref().unwrap_or("tool");

                if name == "task" {
                    // Reconstruct SubagentBlock from the task tool call.
                    let prompt = turn
                        .args
                        .as_ref()
                        .and_then(|a| a["prompt"].as_str())
                        .unwrap_or("")
                        .to_string();
                    let agentType = turn
                        .args
                        .as_ref()
                        .and_then(|a| a["agent"].as_str())
                        .unwrap_or("general")
                        .to_string();

                    let entryIdx = panel.entries.len();
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::SubagentBlock {
                            agentType,
                            prompt,
                            toolLines: Vec::new(),
                            done: true,
                            turns: 0,
                            content: None,
                            contentExpanded: false,
                            sessionId: None, // Filled when ToolResult arrives.
                        });

                    if let Some(ref callId) = turn.toolCallId {
                        pendingTasks.insert(callId.clone(), entryIdx);
                    }
                } else {
                    let summary = turn
                        .args
                        .as_ref()
                        .map(|a| {
                            let s = a.to_string();
                            if s.len() > 80 {
                                format!("{}\u{2026}", &s[..s.floor_char_boundary(79)])
                            } else {
                                s
                            }
                        })
                        .unwrap_or_default();
                    panel
                        .entries
                        .push(crate::agent_panel::PanelEntry::ToolApproved {
                            name: format!("{name}: {summary}"),
                        });
                }
            }
            TurnRole::ToolResult => {
                // Check if this result belongs to a pending task.
                if let Some(ref callId) = turn.toolCallId
                    && let Some(entryIdx) = pendingTasks.remove(callId)
                {
                    // Extract child session ID and content from
                    // "[subagent session: {id}]\n\n{content}".
                    let raw = &turn.content;
                    let (childSessionId, content) =
                        if let Some(start) = raw.find("[subagent session: ") {
                            let idStart = start + "[subagent session: ".len();
                            if let Some(end) = raw[idStart..].find(']') {
                                let sid = raw[idStart..idStart + end].to_string();
                                let bodyStart = idStart + end + 1;
                                let body = if raw[bodyStart..].starts_with("\n\n") {
                                    raw[bodyStart + 2..].to_string()
                                } else {
                                    raw[bodyStart..].to_string()
                                };
                                (Some(sid), body)
                            } else {
                                (None, raw.clone())
                            }
                        } else {
                            (None, raw.clone())
                        };

                    // Load the child transcript to reconstruct tool lines and turn count.
                    let (childToolLines, childTurns) = if let Some(ref csid) = childSessionId {
                        loadChildToolLines(csid)
                    } else {
                        (Vec::new(), 0)
                    };

                    if let Some(crate::agent_panel::PanelEntry::SubagentBlock {
                        content: c,
                        sessionId: sid,
                        toolLines: tl,
                        turns: t,
                        ..
                    }) = panel.entries.get_mut(entryIdx)
                    {
                        *sid = childSessionId;
                        *tl = childToolLines;
                        *t = childTurns;
                        if !content.is_empty() && content != "Task completed (no text output)." {
                            *c = Some(content);
                        }
                    }
                    // Skip pushing a ToolResult — SubagentBlock handles display.
                    continue;
                }

                let name = turn.tool.as_deref().unwrap_or("tool");
                panel
                    .entries
                    .push(crate::agent_panel::PanelEntry::ToolResult {
                        name: name.to_string(),
                        output: turn.content.clone(),
                    });
            }
            // System turns are ephemeral — never appear in transcript chains.
            TurnRole::System => {}
        }
    }
}

/// Load tool lines and turn count from a child session transcript.
fn loadChildToolLines(sessionId: &str) -> (Vec<(String, String)>, usize) {
    use construct::transcript::TurnRole;

    let transcript = match construct::transcript::Transcript::open(sessionId) {
        Ok(t) => t,
        Err(_) => return (Vec::new(), 0),
    };
    let turns = match transcript.loadAll() {
        Ok(t) => t,
        Err(_) => return (Vec::new(), 0),
    };

    let mut toolLines: Vec<(String, String)> = Vec::new();
    let mut turnCount: usize = 0;

    for turn in &turns {
        match turn.role {
            TurnRole::Assistant => {
                turnCount += 1;
            }
            TurnRole::ToolCall => {
                let name = turn.tool.as_deref().unwrap_or("tool");
                let argsJson = turn
                    .args
                    .as_ref()
                    .map(|a| a.to_string())
                    .unwrap_or_default();
                let summary = match construct::tool::parse(name, &argsJson) {
                    Ok(action) => construct::tool::summarize(&action),
                    Err(_) => format!("{name} (parse error)"),
                };
                toolLines.push((name.to_string(), summary));
            }
            TurnRole::ToolResult => {
                let name = turn.tool.as_deref().unwrap_or("tool");
                let output = &turn.content;
                let brief = if output.len() > 60 {
                    format!("{}\u{2026}", &output[..output.floor_char_boundary(60)])
                } else {
                    output.clone()
                };
                toolLines.push((name.to_string(), brief));
            }
            _ => {}
        }
    }

    (toolLines, turnCount)
}

/// Send a TuiRequest whose reply is a `CommandAck`, and forward the ack
/// onto the deck update channel.
fn spawnAckRequest<F>(
    requestTx: mpsc::Sender<TuiRequest>,
    deckUpdateTx: mpsc::Sender<DeckUpdate>,
    build: F,
) where
    F: FnOnce(oneshot::Sender<construct::control::CommandAck>) -> TuiRequest + Send + 'static,
{
    tokio::spawn(async move {
        let (rTx, rRx) = oneshot::channel();
        let req = build(rTx);
        let _ = requestTx.send(req).await;
        if let Ok(ack) = rRx.await {
            let _ = deckUpdateTx.send(DeckUpdate::CommandAck(ack)).await;
        }
    });
}

/// Send a settings mutation and surface only failures in the command log.
fn spawnSilentSettingsRequest<F>(
    requestTx: mpsc::Sender<TuiRequest>,
    deckUpdateTx: mpsc::Sender<DeckUpdate>,
    build: F,
) where
    F: FnOnce(oneshot::Sender<construct::control::CommandAck>) -> TuiRequest + Send + 'static,
{
    tokio::spawn(async move {
        let (rTx, rRx) = oneshot::channel();
        let req = build(rTx);
        let _ = requestTx.send(req).await;
        if let Ok(ack) = rRx.await
            && !ack.ok
        {
            let _ = deckUpdateTx.send(DeckUpdate::CommandAck(ack)).await;
        }
    });
}

/// Return the chip under a mouse event, plus its index in the layout,
/// if the event coordinates fall on a tracked status-bar chip.
fn chipHitTest(
    mouse: &event::MouseEvent,
    chips: &[(StatusChipKind, u16, u16, u16)],
) -> Option<(StatusChipKind, usize)> {
    for (i, (kind, xs, xe, y)) in chips.iter().enumerate() {
        if mouse.row == *y && mouse.column >= *xs && mouse.column < *xe {
            return Some((*kind, i));
        }
    }
    None
}

/// Open the overlay bound to a status-bar item. Mirrors the wiring of
/// the equivalent slash commands so the same panels appear regardless
/// of how the user reached them.
fn openStatusChipPanel(
    kind: StatusChipKind,
    requestTx: &mpsc::Sender<TuiRequest>,
    deckUpdateTx: &mpsc::Sender<DeckUpdate>,
) {
    let requestTx = requestTx.clone();
    let deckUpdateTx = deckUpdateTx.clone();
    match kind {
        StatusChipKind::Jobs => {
            tokio::spawn(async move {
                let (jTx, jRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ListJobs { reply: jTx }).await;
                if let Ok(list) = jRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::TasksList(list)).await;
                }
                let (wTx, wRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ListWakes { reply: wTx }).await;
                if let Ok(list) = wRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::WakesList(list)).await;
                }
            });
        }
        StatusChipKind::Cost => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ShowCost { reply: rTx }).await;
                if let Ok(text) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ShowResult(text)).await;
                }
            });
        }
        StatusChipKind::Context => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ShowContext { reply: rTx }).await;
                if let Ok(state) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ContextDisplay(state)).await;
                }
            });
        }
    }
}

/// Dispatch a parsed slash-command action. Each variant constructs a
/// `TuiRequest` with an appropriate oneshot reply channel and spawns a
/// task to forward the reply to the deck update channel.
fn dispatchSlashCommand(
    action: crate::command::CommandAction,
    requestTx: mpsc::Sender<TuiRequest>,
    deckUpdateTx: mpsc::Sender<DeckUpdate>,
) {
    match action {
        crate::command::CommandAction::ShowContext => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ShowContext { reply: rTx }).await;
                if let Ok(state) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ContextDisplay(state)).await;
                }
            });
        }
        crate::command::CommandAction::Undo => {
            spawnAckRequest(requestTx, deckUpdateTx, |reply| TuiRequest::Undo { reply });
        }
        crate::command::CommandAction::Rewind { target } => {
            if target.is_empty() {
                // No target → open the picker.
                tokio::spawn(async move {
                    let (rTx, rRx) = oneshot::channel();
                    let _ = requestTx
                        .send(TuiRequest::GetRewindOptions { reply: rTx })
                        .await;
                    if let Ok(turns) = rRx.await {
                        let _ = deckUpdateTx.send(DeckUpdate::RewindOptions(turns)).await;
                    }
                });
            } else {
                // Direct rewind to the named turn id. Does NOT save a
                // fork — matches the picker's destructive rewind path.
                spawnAckRequest(requestTx, deckUpdateTx, move |reply| TuiRequest::Rewind {
                    target,
                    saveFork: false,
                    reply,
                });
            }
        }
        crate::command::CommandAction::Forks { forkId: None } => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetForks { reply: rTx }).await;
                if let Ok(forks) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::Forks(forks)).await;
                }
            });
        }
        crate::command::CommandAction::Forks { forkId: Some(id) } => {
            spawnAckRequest(requestTx, deckUpdateTx, move |reply| {
                TuiRequest::SwitchFork { forkId: id, reply }
            });
        }
        crate::command::CommandAction::Resume {
            sessionId: Some(id),
        } => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::ResumeSession {
                        sessionId: id,
                        reply: rTx,
                    })
                    .await;
                if let Ok(ack) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ResumeResult(ack)).await;
                }
            });
        }
        crate::command::CommandAction::Resume { sessionId: None } => {
            // Opening the interactive picker is handled inline; this branch
            // is unreachable but handled defensively with a list fallback.
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::ListSessions { reply: rTx })
                    .await;
                if let Ok(text) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ShowResult(text)).await;
                }
            });
        }
        crate::command::CommandAction::Clear => {
            spawnAckRequest(requestTx, deckUpdateTx, |reply| TuiRequest::Clear { reply });
        }
        crate::command::CommandAction::Mcp => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetMcp { reply: rTx }).await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::McpStatus(status)).await;
                }
            });
        }
        crate::command::CommandAction::Lsp => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetLsp { reply: rTx }).await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::LspStatus(status)).await;
                }
            });
        }
        crate::command::CommandAction::Permissions => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx
                    .send(TuiRequest::GetPermissions { reply: rTx })
                    .await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx
                        .send(DeckUpdate::PermissionsStatus(status))
                        .await;
                }
            });
        }
        crate::command::CommandAction::Model => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::GetModels { reply: rTx }).await;
                if let Ok(status) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ModelStatus(status)).await;
                }
            });
        }
        crate::command::CommandAction::ShowCost => {
            tokio::spawn(async move {
                let (rTx, rRx) = oneshot::channel();
                let _ = requestTx.send(TuiRequest::ShowCost { reply: rTx }).await;
                if let Ok(text) = rRx.await {
                    let _ = deckUpdateTx.send(DeckUpdate::ShowResult(text)).await;
                }
            });
        }
        crate::command::CommandAction::Tasks => {
            // /tasks (alias /jobs) opens the same panel as the status
            // chip — fetch jobs + wakes and let the TasksList /
            // WakesList handlers create the panel.
            openStatusChipPanel(StatusChipKind::Jobs, &requestTx, &deckUpdateTx);
        }
        crate::command::CommandAction::Logs => {
            tracing::warn!("/logs reached dispatchSlashCommand — should have been handled inline");
        }
        crate::command::CommandAction::ShowLayout => {
            // /layout reads the local sessionLayoutPath at the input
            // handler site — should never reach here. Render a notice
            // so a future caller that forgets to intercept gets a hint
            // rather than a silent no-op.
            tracing::warn!(
                "/layout reached dispatchSlashCommand — should have been handled inline"
            );
        }
    }
}
