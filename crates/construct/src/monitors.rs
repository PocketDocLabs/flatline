//! Monitor plane — line-streamed watchers backed by `JobPlane` bash
//! tasks. Each [`Monitor`] owns a compiled regex; lines that match are
//! counted as events and emit [`crate::control::LogEvent::MonitorEvent`].
//! A rolling EWMA flood guard stops a monitor that exceeds
//! `autoStopThreshold` matches/sec for `floodWindowSecs`. The filter
//! regex is mandatory at the schema layer, so unfiltered floods cannot
//! reach the guard by construction.
//!
//! Wakes (sub-phase B) consume the same `MonitorEvent` stream and fire
//! a synthetic `<wake>` message to the model on each match.
//!
//! # Public API
//! - [`MonitorPlane`] — registry of monitors keyed by [`MonitorId`]
//! - [`Monitor`], [`MonitorState`], [`MonitorInfo`]
//!
//! # Dependencies
//! `regex`, [`crate::jobs`], [`crate::control`]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use regex::Regex;
use tokio::sync::mpsc;

use crate::control::LogEvent;
use crate::jobs::{LineCallback, JobId, JobPlane};

/// Monotonically increasing per-session monitor id.
pub type MonitorId = u64;

/// Default threshold for the flood-guard EWMA, in matched-events/second.
/// 50/s was too low for any real log incident (the original value);
/// 500/s gives an incident-grade headroom — a real production crashloop
/// emits hundreds of ERROR lines per second and we still want those
/// surfaced. Filter is required at the tool layer so unfiltered raw
/// log throughput never reaches this guard.
pub const DEFAULT_AUTOSTOP_EPS: f64 = 500.0;

/// Window over which sustained excess events/sec causes auto-stop.
pub const FLOOD_WINDOW_SECS: f64 = 5.0;

/// Lifecycle state of a monitor (mirrors but is independent of the
/// underlying [`crate::jobs::JobState`] — a task can be `Completed`
/// while the monitor is still `Running` until explicitly stopped).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonitorState {
    Running,
    Stopped,
    AutoStopped(String),
}

impl MonitorState {
    pub fn isTerminal(&self) -> bool {
        !matches!(self, MonitorState::Running)
    }
}

/// Inner state shared between the registry and the per-line callback.
struct MonitorInner {
    state: MonitorState,
    eventCount: u64,
    lastEventAt: Option<Instant>,
    /// Exponentially-weighted events/sec — only sampled on a match.
    ewmaEps: f64,
    floodStart: Option<Instant>,
}

/// A registered monitor.
pub struct Monitor {
    pub id: MonitorId,
    /// Id of the backing bash task in [`JobPlane`].
    pub taskId: JobId,
    /// Short human-readable label ("errors in deploy.log"). Shown in
    /// every notification, F2 control panel row, and the inline MonitorChip.
    pub description: String,
    pub command: String,
    /// Raw filter pattern for display (compiled regex lives in the
    /// per-line callback). Required at the tool-schema layer.
    pub filter: String,
    pub autoStopThresholdEps: f64,
    pub createdAt: Instant,
    inner: Arc<Mutex<MonitorInner>>,
    /// Wake context. The per-line callback reads this slot to push a
    /// `WakeFire` into the registry's batcher; the batcher coalesces
    /// and forwards. Holding the unbounded sender (a cheap clone)
    /// means the sync callback never has to take an async lock.
    wakeCtx: Arc<Mutex<Option<MonitorWakeCtx>>>,
}

/// Per-monitor wake plumbing — id of the passive wake source, the
/// shared wake registry (used by auto-stop to unregister the passive
/// source), and the batcher fire sender. The session sets this
/// immediately after register so the callback sees it from the first
/// match.
#[derive(Clone)]
pub struct MonitorWakeCtx {
    pub wakeId: crate::wakes::WakeId,
    pub registry: Arc<tokio::sync::Mutex<crate::wakes::WakeRegistry>>,
    pub fireTx: tokio::sync::mpsc::UnboundedSender<crate::wakes::WakeFire>,
}

impl Monitor {
    pub fn state(&self) -> MonitorState {
        self.inner.lock().unwrap().state.clone()
    }

    pub fn eventCount(&self) -> u64 {
        self.inner.lock().unwrap().eventCount
    }

    pub fn lastEventAt(&self) -> Option<Instant> {
        self.inner.lock().unwrap().lastEventAt
    }

    pub fn info(&self) -> MonitorInfo {
        let g = self.inner.lock().unwrap();
        MonitorInfo {
            id: self.id,
            taskId: self.taskId,
            description: self.description.clone(),
            command: self.command.clone(),
            filter: self.filter.clone(),
            state: g.state.clone(),
            eventCount: g.eventCount,
            lastEventAt: g.lastEventAt,
            createdAt: self.createdAt,
        }
    }
}

/// Lightweight snapshot for `monitorList` and the F2 control panel panel.
#[derive(Debug, Clone)]
pub struct MonitorInfo {
    pub id: MonitorId,
    pub taskId: JobId,
    pub description: String,
    pub command: String,
    pub filter: String,
    pub state: MonitorState,
    pub eventCount: u64,
    pub lastEventAt: Option<Instant>,
    pub createdAt: Instant,
}

/// Per-session registry of monitors.
pub struct MonitorPlane {
    monitors: HashMap<MonitorId, Monitor>,
    /// Insertion order for stable listing.
    order: Vec<MonitorId>,
    nextId: MonitorId,
}

impl MonitorPlane {
    pub fn new() -> Self {
        Self { monitors: HashMap::new(), order: Vec::new(), nextId: 1 }
    }

    /// Reserve the next monitor id without registering anything. Used
    /// when the caller needs to register a passive wake source against
    /// a known id BEFORE the monitor is registered (closes the
    /// attach-after-spawn race for per-line wake fires). Pair with
    /// `registerWithId`.
    pub fn reserveMonitorId(&mut self) -> MonitorId {
        let id = self.nextId;
        self.nextId += 1;
        id
    }

    /// Register a new monitor. Spawns the backing bash task via
    /// [`JobPlane::spawnMonitor`], wires up the per-line filter
    /// callback, and emits `MonitorRegistered`. The task and the
    /// monitor share lifecycles only through the callback — stopping
    /// one does not automatically stop the other; callers use
    /// [`MonitorPlane::stop`] to kill both cleanly.
    pub fn register(
        &mut self,
        description: String,
        command: String,
        filter: String,
        autoStopThresholdEps: f64,
        tasks: &Arc<Mutex<JobPlane>>,
        logTx: mpsc::Sender<LogEvent>,
    ) -> Result<MonitorId> {
        let id = self.reserveMonitorId();
        self.registerWithId(
            id,
            description,
            command,
            filter,
            autoStopThresholdEps,
            tasks,
            logTx,
            None,
        )
    }

    /// Like `register`, but with a pre-reserved id and an optional
    /// passive-wake context resolved up front. The per-line callback
    /// captures the wake slot before the backing bash task spawns —
    /// the first match cannot precede the attach.
    pub fn registerWithId(
        &mut self,
        id: MonitorId,
        description: String,
        command: String,
        filter: String,
        autoStopThresholdEps: f64,
        tasks: &Arc<Mutex<JobPlane>>,
        logTx: mpsc::Sender<LogEvent>,
        wakeCtx: Option<MonitorWakeCtx>,
    ) -> Result<MonitorId> {
        self.registerImpl(
            id, description, command, filter, autoStopThresholdEps,
            FLOOD_WINDOW_SECS, tasks, logTx, wakeCtx,
        )
    }

    /// Implementation backing both `register` and `registerWithId`. The
    /// `floodWindowSecs` parameter lets tests drive the flood-guard
    /// auto-stop path quickly without waiting the production 5s window.
    #[allow(clippy::too_many_arguments)]
    fn registerImpl(
        &mut self,
        id: MonitorId,
        description: String,
        command: String,
        filter: String,
        autoStopThresholdEps: f64,
        floodWindowSecs: f64,
        tasks: &Arc<Mutex<JobPlane>>,
        logTx: mpsc::Sender<LogEvent>,
        wakeCtx: Option<MonitorWakeCtx>,
    ) -> Result<MonitorId> {
        // Description is required — the tool layer enforces non-empty
        // strings, but accept defensively here too.
        if description.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "monitor description is required (short label like 'errors in deploy.log')",
            ));
        }
        if filter.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "monitor filter is required \u{2014} pass a regex that matches only the lines you want to be notified about",
            ));
        }
        // Compile regex up-front so bad patterns reject at registration
        // time (rather than silently never matching).
        let compiled: Regex = Regex::new(&filter).map_err(|e| {
            anyhow::anyhow!("invalid regex filter `{filter}`: {e}")
        })?;

        let inner = Arc::new(Mutex::new(MonitorInner {
            state: MonitorState::Running,
            eventCount: 0,
            lastEventAt: None,
            ewmaEps: 0.0,
            floodStart: None,
        }));

        // Backing taskId is unknown until `spawnMonitor` returns. Stash
        // a slot the callback can read; we fill it in below before any
        // line could plausibly trigger the flood guard (5s window).
        let backingTaskIdSlot: Arc<Mutex<Option<JobId>>> = Arc::new(Mutex::new(None));

        // Wake context: pre-populated when the caller hands one in
        // via `registerWithId`, so the per-line callback enqueues into
        // the registry's batcher from the very first match. Kept behind
        // a Mutex so monitorStop can take() it and unregister the
        // passive source.
        let wakeCtxSlot: Arc<Mutex<Option<MonitorWakeCtx>>> =
            Arc::new(Mutex::new(wakeCtx));

        // The per-line callback runs on the bash drainer's hot path.
        // Keep it lock-fast and side-effect-only.
        let cbInner = inner.clone();
        let cbLogTx = logTx.clone();
        let cbId = id;
        let cbThreshold = autoStopThresholdEps;
        let cbTasks = tasks.clone();
        let cbTaskIdSlot = backingTaskIdSlot.clone();
        let cbWakeCtxSlot = wakeCtxSlot.clone();
        let callback: LineCallback = Box::new(move |line: &str| {
            if !compiled.is_match(line) {
                return;
            }
            let now = Instant::now();
            let mut g = cbInner.lock().unwrap();
            if g.state.isTerminal() {
                return;
            }
            g.eventCount += 1;
            let dtSecs = g
                .lastEventAt
                .map(|t| now.saturating_duration_since(t).as_secs_f64().max(1e-6))
                .unwrap_or(1.0);
            let instantEps = 1.0 / dtSecs;
            // EWMA: alpha = 0.2 gives smooth-but-responsive averaging.
            g.ewmaEps = 0.8 * g.ewmaEps + 0.2 * instantEps;
            g.lastEventAt = Some(now);

            // Flood guard: track how long the EWMA has been above the
            // threshold. If sustained for FLOOD_WINDOW_SECS, auto-stop
            // AND kill the backing bash task so the runaway process
            // doesn't keep pumping output past the auto-stop event.
            if g.ewmaEps > cbThreshold {
                let started = *g.floodStart.get_or_insert(now);
                if now.duration_since(started).as_secs_f64() >= floodWindowSecs {
                    let reason = format!(
                        "flood: {:.0} events/s sustained over {:.0}s (threshold {:.0})",
                        g.ewmaEps, floodWindowSecs, cbThreshold,
                    );
                    g.state = MonitorState::AutoStopped(reason.clone());
                    drop(g);
                    if let Some(tid) = *cbTaskIdSlot.lock().unwrap() {
                        let _ = cbTasks.lock().unwrap().stop(tid);
                    }
                    // Take the wake context out of the slot — future
                    // matches won't fire — and unregister the passive
                    // source on the async registry via a spawned task
                    // (we're on a sync drainer hot-path, can't await
                    // here directly).
                    if let Some(ctx) = cbWakeCtxSlot.lock().unwrap().take() {
                        let logTxClone = cbLogTx.clone();
                        tokio::spawn(async move {
                            ctx.registry
                                .lock()
                                .await
                                .unregisterPassive(ctx.wakeId, &logTxClone);
                        });
                    }
                    let _ = cbLogTx.try_send(LogEvent::MonitorAutoStopped {
                        id: cbId,
                        reason,
                    });
                    return;
                }
            } else {
                g.floodStart = None;
            }
            let eventCount = g.eventCount;
            drop(g);

            let _ = cbLogTx.try_send(LogEvent::MonitorEvent {
                id: cbId,
                line: line.to_string(),
                eventCount,
            });
            // Enqueue a `WakeFire` through the registry's batcher. The
            // batcher coalesces fires within the debounce window, so a
            // stampede of matches becomes one `WakeBatch` → one model
            // turn, instead of one turn per line.
            if let Some(ctx) = cbWakeCtxSlot.lock().unwrap().as_ref() {
                let _ = ctx.fireTx.send(crate::wakes::WakeFire {
                    wakeId: ctx.wakeId,
                    source: format!("monitor#{cbId}"),
                    kind: crate::control::WakeKind::MonitorMatch,
                    payload: line.to_string(),
                    firedAt: now,
                });
            }
        });

        // Spawn the backing bash task with the callback in place. The
        // callback can now consult `backingTaskIdSlot` to kill the task
        // when the flood guard trips.
        let taskId = tasks.lock().unwrap().spawnMonitor(
            description.clone(),
            command.clone(),
            filter.clone(),
            callback,
            logTx.clone(),
        )?;
        *backingTaskIdSlot.lock().unwrap() = Some(taskId);

        let monitor = Monitor {
            id,
            taskId,
            description: description.clone(),
            command: command.clone(),
            filter: filter.clone(),
            autoStopThresholdEps,
            createdAt: Instant::now(),
            inner,
            wakeCtx: wakeCtxSlot,
        };
        self.monitors.insert(id, monitor);
        self.order.push(id);

        let _ = logTx.try_send(LogEvent::MonitorRegistered {
            id,
            taskId,
            description,
            command,
            filter,
        });

        Ok(id)
    }

    /// Take the attached wake id (if any) so the caller can unregister
    /// the passive source on monitor stop. Clears the slot — subsequent
    /// matches will no longer route to the batcher until reattached.
    pub fn takeWakeId(&self, id: MonitorId) -> Option<crate::wakes::WakeId> {
        self.monitors.get(&id).and_then(|m| {
            m.wakeCtx.lock().unwrap().take().map(|ctx| ctx.wakeId)
        })
    }

    /// Stop a monitor cooperatively. Always kills the backing bash task
    /// — even on already-terminal monitors, so a user `monitorStop`
    /// after an auto-stop can still reap a runaway process if the
    /// in-callback kill didn't take (e.g. process group leader exited
    /// but a `disown`ed grandchild kept the stdout pipe open).
    pub fn stop(&self, id: MonitorId, tasks: &Arc<Mutex<JobPlane>>) -> Result<()> {
        let m = self
            .monitors
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("no monitor #{id}"))?;
        {
            let mut g = m.inner.lock().unwrap();
            if !g.state.isTerminal() {
                g.state = MonitorState::Stopped;
            }
        }
        // Always attempt the kill — JobPlane::stop is a no-op for
        // already-terminal tasks, so this is safe to retry.
        let _ = tasks.lock().unwrap().stop(m.taskId);
        Ok(())
    }

    pub fn list(&self) -> Vec<MonitorInfo> {
        self.order
            .iter()
            .filter_map(|id| self.monitors.get(id).map(|m| m.info()))
            .collect()
    }

    pub fn lookup(&self, id: MonitorId) -> Option<&Monitor> {
        self.monitors.get(&id)
    }

    pub fn len(&self) -> usize {
        self.monitors.len()
    }

    pub fn isEmpty(&self) -> bool {
        self.monitors.is_empty()
    }

    /// Snapshot of currently-running monitor ids — useful for the
    /// status-strip live counter.
    pub fn runningCount(&self) -> usize {
        self.monitors
            .values()
            .filter(|m| matches!(m.state(), MonitorState::Running))
            .count()
    }
}

impl Default for MonitorPlane {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(mut rx: mpsc::Receiver<LogEvent>) -> tokio::task::JoinHandle<Vec<LogEvent>> {
        tokio::spawn(async move {
            let mut events = Vec::new();
            while let Some(ev) = rx.recv().await {
                events.push(ev);
            }
            events
        })
    }

    #[tokio::test]
    async fn registerAndMatchEmitsMonitorEvent() {
        let (tx, rx) = mpsc::channel(256);
        let drainer = drain(rx);
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        let monitorId = plane
            .register(
                "errors".into(),
                "for i in 1 2 3 4; do echo line$i; done; echo ERROR final".into(),
                "ERROR".into(),
                DEFAULT_AUTOSTOP_EPS,
                &tasks,
                tx.clone(),
            )
            .unwrap();

        // Wait for the underlying task to finish (it's a one-shot).
        for _ in 0..100 {
            let info = plane.lookup(monitorId).unwrap().info();
            if info.eventCount >= 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        // Give drainers a moment to flush.
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;

        let info = plane.lookup(monitorId).unwrap().info();
        assert_eq!(info.eventCount, 1, "only `ERROR final` should match");

        drop(tx);
        drop(plane);
        drop(tasks);
        let events = drainer.await.unwrap();
        let registered = events
            .iter()
            .filter(|e| matches!(e, LogEvent::MonitorRegistered { id, .. } if *id == monitorId))
            .count();
        let matched = events
            .iter()
            .filter(|e| matches!(e, LogEvent::MonitorEvent { id, .. } if *id == monitorId))
            .count();
        assert_eq!(registered, 1);
        assert_eq!(matched, 1);
    }

    #[tokio::test]
    async fn matchAllRegexCountsEveryLine() {
        // Filter is required, so to exercise the count-every-line path
        // we pass a regex that matches everything.
        let (tx, rx) = mpsc::channel(256);
        let drainer = drain(rx);
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        let id = plane
            .register(
                "every line".into(),
                "for i in 1 2 3 4 5; do echo l$i; done".into(),
                ".".into(),
                DEFAULT_AUTOSTOP_EPS,
                &tasks,
                tx.clone(),
            )
            .unwrap();
        for _ in 0..100 {
            let info = plane.lookup(id).unwrap().info();
            if info.eventCount >= 5 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let info = plane.lookup(id).unwrap().info();
        assert_eq!(info.eventCount, 5);
        drop(tx);
        drop(plane);
        drop(tasks);
        let _ = drainer.await.unwrap();
    }

    #[tokio::test]
    async fn missingFilterRejected() {
        let (tx, _rx) = mpsc::channel(16);
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        let err = plane
            .register(
                "no filter test".into(),
                "echo ok".into(),
                "".into(),
                DEFAULT_AUTOSTOP_EPS,
                &tasks,
                tx.clone(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("filter is required"));
        assert_eq!(plane.len(), 0);
    }

    #[tokio::test]
    async fn missingDescriptionRejected() {
        let (tx, _rx) = mpsc::channel(16);
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        let err = plane
            .register(
                "".into(),
                "echo ok".into(),
                "ok".into(),
                DEFAULT_AUTOSTOP_EPS,
                &tasks,
                tx.clone(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("description is required"));
        assert_eq!(plane.len(), 0);
    }

    #[tokio::test]
    async fn invalidRegexRejected() {
        let (tx, _rx) = mpsc::channel(16);
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        // `[` is unterminated character class.
        let err = plane
            .register(
                "broken regex".into(),
                "echo ok".into(),
                "[unclosed".into(),
                DEFAULT_AUTOSTOP_EPS,
                &tasks,
                tx.clone(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("invalid regex"));
        assert_eq!(plane.len(), 0);
    }

    #[tokio::test]
    async fn autoStopUnregistersPassiveWakeSource() {
        // Slice 6a fix: when the flood-guard auto-stops a monitor, the
        // passive wake row in WakeRegistry must be removed. Otherwise
        // /jobs schedules keeps a phantom source forever after the
        // monitor has gone away.
        use crate::wakes::WakeRegistry;

        let (tx, _rx) = mpsc::channel(1024);
        let (regArc, _batchRx) = WakeRegistry::new();
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        let monitorId = plane.reserveMonitorId();

        let (wakeId, fireTx) = {
            let mut g = regArc.lock().await;
            let wid = g.registerMonitor(monitorId, &tx);
            (wid, g.fireSender())
        };
        let wakeCtx = MonitorWakeCtx {
            wakeId,
            registry: regArc.clone(),
            fireTx,
        };

        // Drive the flood guard hard: threshold 0.0001 eps (any match
        // exceeds it), 0.3s flood window (vs production 5s). Bash loop
        // emits a matching line every 20ms so the EWMA stays well above
        // threshold across the window.
        plane
            .registerImpl(
                monitorId,
                "burst".into(),
                "while true; do echo HIT; sleep 0.02; done".into(),
                "HIT".into(),
                0.0001,
                0.3,
                &tasks,
                tx.clone(),
                Some(wakeCtx),
            )
            .unwrap();

        // Wait until the auto-stop fires. Budget = flood window +
        // tokio::spawn slack for the unregister.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        loop {
            let state = plane.lookup(monitorId).unwrap().state();
            if matches!(state, MonitorState::AutoStopped(_)) {
                break;
            }
            if std::time::Instant::now() >= deadline {
                panic!("monitor never auto-stopped (state still {state:?})");
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // The unregister happens via tokio::spawn from the sync
        // callback — give it a beat to run before asserting.
        for _ in 0..50 {
            if regArc.lock().await.len() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(
            regArc.lock().await.len(),
            0,
            "passive wake source must be unregistered after auto-stop",
        );

        // The monitor row itself stays in the registry as AutoStopped —
        // /monitors still surfaces it so the user/agent can see why it
        // died — but the wake source is gone.
        let info = plane.lookup(monitorId).unwrap().info();
        assert!(matches!(info.state, MonitorState::AutoStopped(_)));

        drop(plane);
        drop(tasks);
        drop(tx);
    }

    #[tokio::test]
    async fn stopMarksMonitorAndKillsBackingTask() {
        let (tx, _rx) = mpsc::channel(256);
        let tasks = Arc::new(Mutex::new(JobPlane::new(None)));
        let mut plane = MonitorPlane::new();
        let id = plane
            .register(
                "long sleep".into(),
                "sleep 30".into(),
                ".".into(),
                DEFAULT_AUTOSTOP_EPS,
                &tasks,
                tx.clone(),
            )
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        plane.stop(id, &tasks).unwrap();
        let info = plane.lookup(id).unwrap().info();
        assert_eq!(info.state, MonitorState::Stopped);
        // Second stop is a no-op (no error).
        plane.stop(id, &tasks).unwrap();
    }
}
