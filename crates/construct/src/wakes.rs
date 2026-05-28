//! Wake plane — central registry of agent wake sources.
//!
//! A "wake" is a synthetic user-shaped message the agent receives in
//! response to some external event. Today's sources:
//!
//! - [`WakeKind::Delay`]: one-shot, fires N seconds after registration
//! - [`WakeKind::Cron`]: recurring, fires on a 5-field cron schedule
//! - [`WakeKind::FileWatch`]: fires on fs events under a path
//! - [`WakeKind::MonitorMatch`]: fires per matched line from a Monitor
//! - [`WakeKind::TaskComplete`]: fires when a backgrounded task exits —
//!   either a `shell(runInBackground)` bash job or a
//!   `task(runInBackground)` subagent
//!
//! Wake fires are coalesced into [`WakeBatch`] values. The session
//! records each batch as a `TurnRole::Wake` transcript event, injects it
//! into model context, and emits `LogEvent::WakeBatchInjected` for deck
//! display.
//!
//! # Lifecycle
//! - [`WakeRegistry::armDelay`], [`WakeRegistry::armCron`], etc. register
//!   a source and return a [`WakeId`]. Delay/Cron/FileWatch spawn an
//!   internal tokio task that schedules the fire(s); the task drops
//!   cleanly when the source is disarmed.
//! - MonitorMatch and TaskComplete have no scheduler — they're
//!   enqueued by `MonitorPlane` / `JobPlane` at the appropriate moment.
//! - [`WakeRegistry::disarm`] cancels the source's scheduler (if any)
//!   and removes it from the registry.
//!
//! # Public API
//! - [`WakeRegistry`], [`WakeKind`], [`WakeId`], [`WakeSourceInfo`]
//!
//! # Dependencies
//! `cron`, `chrono`, `notify`, [`crate::control`]

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{Mutex, mpsc, watch};

use crate::control::{LogEvent, WakeKind as ControlWakeKind};

/// Debounce window for coalescing wake fires into a batch.
///
/// Picked to swallow log stampedes (everyone-fires-at-once after a
/// `tail -F` opens a hot file) while still feeling instantaneous on
/// genuine single events. 300 ms is below the threshold where the
/// agent's "I just woke up" feels delayed; above that and a slow human
/// would notice the gap.
pub const WAKE_BATCH_WINDOW: Duration = Duration::from_millis(300);

/// Upper bound on fires the batcher will hold in a single batch. Past
/// this point we close the batch and start a new one — combined with the
/// debounce, this caps the per-batch payload size while still preserving
/// ordering.
pub const WAKE_BATCH_MAX_FIRES: usize = 256;

/// One wake fire — pre-batch. Carries everything the consumer needs to
/// format the `<wake>` element later. Constructed at the fire site and
/// pushed through the batcher's queue.
#[derive(Debug, Clone)]
pub struct WakeFire {
    pub wakeId: WakeId,
    pub source: String,
    pub kind: ControlWakeKind,
    pub payload: String,
    pub firedAt: Instant,
}

/// A coalesced group of wake fires. Emitted by the batcher and consumed
/// by the session task — one batch produces one synthetic user-shaped
/// message, one transcript entry, one model turn.
#[derive(Debug, Clone)]
pub struct WakeBatch {
    pub fires: Vec<WakeFire>,
    /// Wall-clock instant when the batch was closed.
    pub closedAt: Instant,
}

/// Monotonically increasing per-session wake source id.
pub type WakeId = u64;

/// Why a wake fires. The control-plane `WakeKind` is the on-the-wire
/// shape (used in the synthetic `<wake>` message); this is the
/// registry's internal shape with the scheduling state attached.
#[derive(Debug, Clone)]
pub enum WakeKind {
    /// One-shot delay. `fireAt` is computed at registration.
    Delay { fireAt: Instant, prompt: String },
    /// Recurring cron schedule. `spec` is a 5- or 6-field cron string.
    /// When `recurring` is false the source disarms after the first fire.
    Cron {
        spec: String,
        recurring: bool,
        prompt: String,
    },
    /// Watch a filesystem path for events. Fires the wake for each
    /// matching event the OS reports.
    FileWatch { path: PathBuf, prompt: String },
    /// External event source — a Monitor regex matched a line. Has no
    /// scheduler; `MonitorPlane` calls `fireMonitorMatch` directly.
    MonitorMatch { monitorId: u64 },
    /// External event source — a backgrounded task exited (bash job or
    /// subagent). `JobPlane` calls into the registry directly.
    TaskComplete { taskId: u64 },
}

impl WakeKind {
    #[allow(dead_code)]
    fn controlKind(&self) -> ControlWakeKind {
        match self {
            WakeKind::Delay { .. } => ControlWakeKind::Delay,
            WakeKind::Cron { .. } => ControlWakeKind::Cron,
            WakeKind::FileWatch { .. } => ControlWakeKind::FileWatch,
            WakeKind::MonitorMatch { .. } => ControlWakeKind::MonitorMatch,
            WakeKind::TaskComplete { .. } => ControlWakeKind::TaskComplete,
        }
    }
}

/// Lightweight snapshot for `cronList`, `wakeList`, and the /tasks schedules panel.
#[derive(Debug, Clone)]
pub struct WakeSourceInfo {
    pub id: WakeId,
    pub kind: ControlWakeKind,
    /// Human-readable summary ("every weekday at 09:00", "5 minutes",
    /// "/var/log/app.log").
    pub summary: String,
    /// Optional model-supplied prompt; the wake's payload when it fires.
    pub prompt: Option<String>,
    pub createdAt: Instant,
    pub firesSoFar: u64,
}

struct WakeSource {
    info: WakeSourceInfo,
    /// Send `()` to cancel the source's internal scheduler. None for
    /// externally-driven sources (MonitorMatch, TaskComplete).
    cancelTx: Option<watch::Sender<bool>>,
}

/// Central wake-source registry.
pub struct WakeRegistry {
    sources: HashMap<WakeId, WakeSource>,
    order: Vec<WakeId>,
    nextId: WakeId,
    /// Unbounded sender used by every fire site to enqueue a `WakeFire`.
    /// The batcher actor drains the corresponding receiver and emits
    /// debounced `WakeBatch` values on `batchTx`. Unbounded is safe here
    /// because the batcher drains continuously, the monitor flood guard
    /// kills runaway sources at 500 eps, and a batch caps at
    /// `WAKE_BATCH_MAX_FIRES`.
    fireTx: mpsc::UnboundedSender<WakeFire>,
    /// Cancel handle for the batcher task — flipped on `disarmAll`.
    batcherCancelTx: watch::Sender<bool>,
    /// True after `disarmAll`. The batcher checks this just before send so
    /// an already-queued fire cannot leak from an old session into a new one.
    closed: bool,
}

impl WakeRegistry {
    /// Build a registry, wrap it in `Arc<Mutex<…>>`, and spawn the
    /// batcher actor. Returns both the shared handle and the receiver
    /// for coalesced `WakeBatch` values — the session task selects on
    /// the receiver and runs one turn per batch.
    pub fn new() -> (Arc<Mutex<Self>>, mpsc::Receiver<WakeBatch>) {
        let (fireTx, fireRx) = mpsc::unbounded_channel::<WakeFire>();
        let (batchTx, batchRx) = mpsc::channel::<WakeBatch>(32);
        let (cancelTx, cancelRx) = watch::channel(false);
        let reg = Self {
            sources: HashMap::new(),
            order: Vec::new(),
            nextId: 1,
            fireTx,
            batcherCancelTx: cancelTx,
            closed: false,
        };
        let arc = Arc::new(Mutex::new(reg));
        tokio::spawn(batcherActor(fireRx, batchTx, cancelRx, arc.clone()));
        (arc, batchRx)
    }

    /// Snapshot the fire-channel sender. Used by sync sites like the
    /// monitor per-line callback (which can't await `.send`). The
    /// resulting sender bypasses `firesSoFar` accounting — the batcher
    /// performs the per-source bump when it drains the queue.
    pub fn fireSender(&self) -> mpsc::UnboundedSender<WakeFire> {
        self.fireTx.clone()
    }

    pub fn list(&self) -> Vec<WakeSourceInfo> {
        self.order
            .iter()
            .filter_map(|id| self.sources.get(id).map(|s| s.info.clone()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Disarm every wake AND stop the batcher. Use this before swapping
    /// the registry slot on `/clear` / `/resume` so the previous
    /// session's schedulers and batcher actor stop firing into the new
    /// session. Without the batcher kill the actor would keep the
    /// registry Arc alive (it holds a clone) and outlive the swap.
    pub fn disarmAll(&mut self) {
        self.closed = true;
        for (_, src) in self.sources.drain() {
            if let Some(tx) = src.cancelTx {
                let _ = tx.send(true);
            }
        }
        self.order.clear();
        let _ = self.batcherCancelTx.send(true);
    }

    /// Disarm a wake source. No-op if the id isn't registered.
    /// External event sources (MonitorMatch, TaskComplete) are
    /// auto-removed when the underlying monitor/task ends, but can be
    /// explicitly disarmed too.
    pub fn disarm(&mut self, id: WakeId, logTx: &mpsc::Sender<LogEvent>) -> bool {
        if let Some(src) = self.sources.remove(&id) {
            self.order.retain(|x| *x != id);
            if let Some(tx) = src.cancelTx {
                let _ = tx.send(true);
            }
            let _ = logTx.try_send(LogEvent::WakeDisarmed { id });
            true
        } else {
            false
        }
    }

    /// Arm a one-shot delay. Returns the new wake id. The internal
    /// scheduler task enqueues a wake after `duration` and then removes
    /// the source from the registry (one-shot).
    pub fn armDelay(
        registryArc: &Arc<Mutex<WakeRegistry>>,
        duration: Duration,
        prompt: String,
        logTx: mpsc::Sender<LogEvent>,
    ) -> WakeId {
        let mut g = registryArc.blocking_lock();
        let id = g.nextId;
        g.nextId += 1;
        let fireAt = Instant::now() + duration;
        let summary = formatDuration(duration);
        let promptForEvent = prompt.clone();
        let (cancelTx, cancelRx) = watch::channel(false);
        let registryArc2 = registryArc.clone();
        tokio::spawn(delayScheduler(
            id,
            fireAt,
            prompt.clone(),
            cancelRx,
            logTx.clone(),
            registryArc2,
        ));
        let info = WakeSourceInfo {
            id,
            kind: ControlWakeKind::Delay,
            summary: summary.clone(),
            prompt: Some(prompt),
            createdAt: Instant::now(),
            firesSoFar: 0,
        };
        g.sources.insert(
            id,
            WakeSource {
                info,
                cancelTx: Some(cancelTx),
            },
        );
        g.order.push(id);
        let _ = logTx.try_send(LogEvent::WakeRegistered {
            id,
            kind: ControlWakeKind::Delay,
            summary,
            prompt: Some(promptForEvent),
            nextFireAt: Some(fireAt),
        });
        id
    }

    /// Arm a cron schedule. `spec` is a 5- or 6-field cron string in
    /// local time. `recurring` controls whether the source persists
    /// after the first fire.
    pub fn armCron(
        registryArc: &Arc<Mutex<WakeRegistry>>,
        spec: String,
        recurring: bool,
        prompt: String,
        logTx: mpsc::Sender<LogEvent>,
    ) -> Result<WakeId> {
        // The `cron` crate uses 6-field with seconds in field 0. Accept
        // standard 5-field input by prepending "0 ".
        let normalized = if spec.split_whitespace().count() == 5 {
            format!("0 {spec}")
        } else {
            spec.clone()
        };
        let schedule = cron::Schedule::from_str(&normalized)
            .map_err(|e| anyhow::anyhow!("invalid cron spec `{spec}`: {e}"))?;
        let nextFireAt = schedule
            .upcoming(chrono::Local)
            .next()
            .and_then(|dt| (dt - chrono::Local::now()).to_std().ok())
            .map(|delta| Instant::now() + delta);
        let mut g = registryArc.blocking_lock();
        let id = g.nextId;
        g.nextId += 1;
        let (cancelTx, cancelRx) = watch::channel(false);
        let registryArc2 = registryArc.clone();
        let summary = format!("cron: {spec}{}", if recurring { "" } else { " (one-shot)" });
        let promptForEvent = prompt.clone();
        tokio::spawn(cronScheduler(
            id,
            schedule,
            recurring,
            prompt.clone(),
            cancelRx,
            logTx.clone(),
            registryArc2,
        ));
        let info = WakeSourceInfo {
            id,
            kind: ControlWakeKind::Cron,
            summary: summary.clone(),
            prompt: Some(prompt),
            createdAt: Instant::now(),
            firesSoFar: 0,
        };
        g.sources.insert(
            id,
            WakeSource {
                info,
                cancelTx: Some(cancelTx),
            },
        );
        g.order.push(id);
        let _ = logTx.try_send(LogEvent::WakeRegistered {
            id,
            kind: ControlWakeKind::Cron,
            summary,
            prompt: Some(promptForEvent),
            nextFireAt,
        });
        Ok(id)
    }

    /// Arm a filesystem watch. Each fs event under `path` (created,
    /// modified, removed) fires a wake. Returns the new wake id.
    pub fn armFileWatch(
        registryArc: &Arc<Mutex<WakeRegistry>>,
        path: PathBuf,
        prompt: String,
        logTx: mpsc::Sender<LogEvent>,
    ) -> Result<WakeId> {
        use notify::{RecursiveMode, Watcher};

        if !path.exists() {
            return Err(anyhow::anyhow!("path does not exist: {}", path.display()));
        }

        // Create + start the watcher up front so any error surfaces as a
        // failed Result instead of an "armed" entry whose scheduler never
        // actually delivers events.
        let (evTx, evRx) = mpsc::channel::<notify::Result<notify::Event>>(64);
        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = evTx.blocking_send(res);
        })
        .map_err(|e| anyhow::anyhow!("fileWatch init failed: {e}"))?;
        watcher
            .watch(&path, RecursiveMode::Recursive)
            .map_err(|e| anyhow::anyhow!("fileWatch watch({}) failed: {e}", path.display()))?;

        let mut g = registryArc.blocking_lock();
        let id = g.nextId;
        g.nextId += 1;
        let (cancelTx, cancelRx) = watch::channel(false);
        let registryArc2 = registryArc.clone();
        let summary = format!("watch: {}", path.display());
        tokio::spawn(fileWatchScheduler(
            id,
            path.clone(),
            prompt.clone(),
            watcher,
            evRx,
            cancelRx,
            logTx.clone(),
            registryArc2,
        ));
        let promptForEvent = prompt.clone();
        let info = WakeSourceInfo {
            id,
            kind: ControlWakeKind::FileWatch,
            summary: summary.clone(),
            prompt: Some(prompt),
            createdAt: Instant::now(),
            firesSoFar: 0,
        };
        g.sources.insert(
            id,
            WakeSource {
                info,
                cancelTx: Some(cancelTx),
            },
        );
        g.order.push(id);
        let _ = logTx.try_send(LogEvent::WakeRegistered {
            id,
            kind: ControlWakeKind::FileWatch,
            summary,
            prompt: Some(promptForEvent),
            nextFireAt: None,
        });
        Ok(id)
    }

    /// Register a passive MonitorMatch wake. Returns a wake id used by
    /// the MonitorPlane to label fires; there's no scheduler — fires
    /// are driven by the monitor's per-line callback.
    pub fn registerMonitor(&mut self, monitorId: u64, logTx: &mpsc::Sender<LogEvent>) -> WakeId {
        let id = self.nextId;
        self.nextId += 1;
        let summary = format!("monitor#{monitorId}");
        let info = WakeSourceInfo {
            id,
            kind: ControlWakeKind::MonitorMatch,
            summary: summary.clone(),
            prompt: None,
            createdAt: Instant::now(),
            firesSoFar: 0,
        };
        self.sources.insert(
            id,
            WakeSource {
                info,
                cancelTx: None,
            },
        );
        self.order.push(id);
        let _ = logTx.try_send(LogEvent::WakeRegistered {
            id,
            kind: ControlWakeKind::MonitorMatch,
            summary,
            prompt: None,
            nextFireAt: None,
        });
        id
    }

    /// Register a passive TaskComplete wake. Same shape as
    /// MonitorMatch — fires are driven by JobPlane when a backgrounded
    /// task (bash job or subagent) exits.
    pub fn registerTaskComplete(&mut self, taskId: u64, logTx: &mpsc::Sender<LogEvent>) -> WakeId {
        let id = self.nextId;
        self.nextId += 1;
        let summary = format!("task#{taskId}");
        let info = WakeSourceInfo {
            id,
            kind: ControlWakeKind::TaskComplete,
            summary: summary.clone(),
            prompt: None,
            createdAt: Instant::now(),
            firesSoFar: 0,
        };
        self.sources.insert(
            id,
            WakeSource {
                info,
                cancelTx: None,
            },
        );
        self.order.push(id);
        let _ = logTx.try_send(LogEvent::WakeRegistered {
            id,
            kind: ControlWakeKind::TaskComplete,
            summary,
            prompt: None,
            nextFireAt: None,
        });
        id
    }

    /// Enqueue a `WakeFire` through the batcher. Returns true if the
    /// source is still registered. Call sites (delay/cron/fileWatch
    /// schedulers, JobPlane bg-task watcher) use this while holding
    /// the registry lock; `firesSoFar` is bumped by the batcher when
    /// the queued fire is drained, so concurrent stampedes don't race
    /// on the counter under the registry lock.
    pub fn enqueueFire(&self, id: WakeId, source: String, payload: String) -> bool {
        if self.closed {
            return false;
        }
        let Some(src) = self.sources.get(&id) else {
            return false;
        };
        let kind = src.info.kind;
        self.fireTx
            .send(WakeFire {
                wakeId: id,
                source,
                kind,
                payload,
                firedAt: Instant::now(),
            })
            .is_ok()
    }

    /// Sync-callable handle for the per-line monitor callback. The
    /// callback can't take an async lock on the registry, so we let it
    /// build a `WakeFire` against a known wake id and push it directly.
    /// `firesSoFar` is updated later by the batcher when the fire is
    /// pulled off the queue.
    pub fn enqueueFromCallback(sender: &mpsc::UnboundedSender<WakeFire>, fire: WakeFire) {
        let _ = sender.send(fire);
    }

    /// Remove a passive source (called by MonitorPlane on monitorStop
    /// and JobPlane on task removal). Emits `WakeDisarmed` so consumers
    /// (status-strip wake counter, /jobs schedules section) stay in sync
    /// with the registry. Silently no-op if the id isn't registered.
    pub fn unregisterPassive(&mut self, id: WakeId, logTx: &mpsc::Sender<LogEvent>) {
        if self.sources.remove(&id).is_some() {
            self.order.retain(|x| *x != id);
            let _ = logTx.try_send(LogEvent::WakeDisarmed { id });
        }
    }
}

/// Batcher actor — single consumer of `fireRx`. Coalesces fires within a
/// `WAKE_BATCH_WINDOW` debounce and emits one `WakeBatch` per window.
///
/// Ordering guarantees: fires are pushed in arrival order on a single
/// unbounded mpsc and pulled in that same order here, so the resulting
/// `WakeBatch.fires` vec preserves the order in which the underlying
/// callbacks ran. There is no spawned-task race like the old direct-
/// emission path.
async fn batcherActor(
    mut fireRx: mpsc::UnboundedReceiver<WakeFire>,
    batchTx: mpsc::Sender<WakeBatch>,
    mut cancelRx: watch::Receiver<bool>,
    registry: Arc<Mutex<WakeRegistry>>,
) {
    loop {
        // Wait for the first fire — no batch in flight, no timeout.
        let first = tokio::select! {
            _ = cancelRx.changed() => return,
            fire = fireRx.recv() => match fire {
                Some(f) => f,
                None => return, // fireTx dropped — registry gone.
            },
        };
        let mut batch: Vec<WakeFire> = Vec::with_capacity(8);
        batch.push(first);
        // Accumulate additional fires until the debounce window closes
        // or the batch hits its size cap.
        let deadline = tokio::time::sleep(WAKE_BATCH_WINDOW);
        tokio::pin!(deadline);
        let mut cancelled = false;
        loop {
            if batch.len() >= WAKE_BATCH_MAX_FIRES {
                break;
            }
            tokio::select! {
                _ = &mut deadline => break,
                _ = cancelRx.changed() => { cancelled = true; break; }
                fire = fireRx.recv() => match fire {
                    Some(f) => batch.push(f),
                    None => break,
                },
            }
        }
        if cancelled {
            return;
        }
        if !batch.is_empty() {
            // Bump per-source counters in one critical section, then
            // hand the batch off. The lock is held only across the
            // counter updates — no awaits while held.
            {
                let mut g = registry.lock().await;
                if g.closed {
                    return;
                }
                for f in &batch {
                    if let Some(src) = g.sources.get_mut(&f.wakeId) {
                        src.info.firesSoFar += 1;
                    }
                }
            }
            let _ = batchTx
                .send(WakeBatch {
                    fires: batch,
                    closedAt: Instant::now(),
                })
                .await;
        }
    }
}

fn formatDuration(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{}s", s / 60, s % 60)
    } else {
        format!("{}h{}m", s / 3600, (s % 3600) / 60)
    }
}

async fn delayScheduler(
    id: WakeId,
    fireAt: Instant,
    prompt: String,
    mut cancelRx: watch::Receiver<bool>,
    logTx: mpsc::Sender<LogEvent>,
    registry: Arc<Mutex<WakeRegistry>>,
) {
    let sleep = tokio::time::sleep_until(fireAt.into());
    tokio::pin!(sleep);
    tokio::select! {
        _ = &mut sleep => {
            // Enqueue via the batcher (bumps firesSoFar atomically with
            // the queue insertion) then disarm the one-shot source.
            {
                let mut g = registry.lock().await;
                g.enqueueFire(id, format!("delay#{id}"), prompt);
                g.sources.remove(&id);
                g.order.retain(|x| *x != id);
            }
            let _ = logTx.try_send(LogEvent::WakeDisarmed { id });
        }
        _ = cancelRx.changed() => {
            // Disarmed externally; nothing to do (disarm() already cleaned up).
        }
    }
}

async fn cronScheduler(
    id: WakeId,
    schedule: cron::Schedule,
    recurring: bool,
    prompt: String,
    mut cancelRx: watch::Receiver<bool>,
    logTx: mpsc::Sender<LogEvent>,
    registry: Arc<Mutex<WakeRegistry>>,
) {
    loop {
        // Compute next fire from wall-clock so macOS sleep doesn't
        // permanently shift the schedule.
        let next = match schedule.upcoming(chrono::Local).next() {
            Some(dt) => dt,
            None => return, // Spec has no future fires.
        };
        let now = chrono::Local::now();
        let delta = (next - now).to_std().unwrap_or(Duration::from_secs(1));
        let sleep = tokio::time::sleep(delta);
        tokio::pin!(sleep);
        tokio::select! {
            _ = &mut sleep => {
                let disarmedNow = {
                    let mut g = registry.lock().await;
                    g.enqueueFire(id, format!("cron#{id}"), prompt.clone());
                    if !recurring {
                        g.sources.remove(&id);
                        g.order.retain(|x| *x != id);
                        true
                    } else {
                        false
                    }
                };
                if disarmedNow {
                    let _ = logTx.try_send(LogEvent::WakeDisarmed { id });
                    return;
                }
                // Recurring: loop and recompute next fire.
            }
            _ = cancelRx.changed() => {
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn fileWatchScheduler(
    id: WakeId,
    path: PathBuf,
    prompt: String,
    // Hold the watcher for the lifetime of the task — notify drops the
    // watch when the watcher is dropped, so we cannot let it leave scope.
    _watcher: notify::RecommendedWatcher,
    mut rx: mpsc::Receiver<notify::Result<notify::Event>>,
    mut cancelRx: watch::Receiver<bool>,
    _logTx: mpsc::Sender<LogEvent>,
    registry: Arc<Mutex<WakeRegistry>>,
) {
    let _ = path;
    loop {
        tokio::select! {
            event = rx.recv() => {
                let ev = match event {
                    Some(Ok(ev)) => ev,
                    Some(Err(e)) => {
                        tracing::warn!(wakeId = id, "fileWatch error: {e}");
                        continue;
                    }
                    None => return,
                };
                // Summarize the event compactly — the payload tells the
                // model what fired without dumping the full notify Event.
                let kindStr = format!("{:?}", ev.kind);
                let paths: Vec<String> = ev.paths.iter().map(|p| p.display().to_string()).collect();
                let payload = format!(
                    "{prompt}\n\nfs event: {kindStr}\npaths: {}",
                    paths.join(", "),
                );
                {
                    let g = registry.lock().await;
                    g.enqueueFire(id, format!("fileWatch#{id}"), payload);
                }
            }
            _ = cancelRx.changed() => {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a registry + log channel pair ready for tests. Returns
    /// `(registry, logTx, logRx, wakeBatchRx)`.
    fn buildTestRegistry() -> (
        Arc<Mutex<WakeRegistry>>,
        mpsc::Sender<LogEvent>,
        mpsc::Receiver<LogEvent>,
        mpsc::Receiver<WakeBatch>,
    ) {
        let (tx, rx) = mpsc::channel(16);
        let (regArc, batchRx) = WakeRegistry::new();
        (regArc, tx, rx, batchRx)
    }

    #[tokio::test]
    async fn delayFiresOnce() {
        let (reg, tx, mut rx, mut batchRx) = buildTestRegistry();
        let id = tokio::task::spawn_blocking({
            let reg = reg.clone();
            let tx = tx.clone();
            move || WakeRegistry::armDelay(&reg, Duration::from_millis(100), "ping".into(), tx)
        })
        .await
        .unwrap();
        assert!(id > 0);
        // First log event is WakeRegistered.
        let reg1 = rx.recv().await.unwrap();
        assert!(matches!(
            reg1,
            LogEvent::WakeRegistered {
                kind: ControlWakeKind::Delay,
                ..
            }
        ));
        // After ~100ms + the batch window the wake arrives as a batch.
        let batch = tokio::time::timeout(
            Duration::from_millis(100) + WAKE_BATCH_WINDOW + Duration::from_millis(300),
            batchRx.recv(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(batch.fires.len(), 1);
        assert!(matches!(batch.fires[0].kind, ControlWakeKind::Delay));
        // The one-shot disarm log event lands on the log channel.
        let dis = tokio::time::timeout(Duration::from_millis(200), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(dis, LogEvent::WakeDisarmed { .. }));
        assert_eq!(reg.lock().await.len(), 0);
    }

    #[tokio::test]
    async fn disarmCancelsDelayBeforeFire() {
        let (reg, tx, mut rx, mut batchRx) = buildTestRegistry();
        let id = tokio::task::spawn_blocking({
            let reg = reg.clone();
            let tx = tx.clone();
            move || WakeRegistry::armDelay(&reg, Duration::from_secs(60), "ping".into(), tx)
        })
        .await
        .unwrap();
        let _ = rx.recv().await; // WakeRegistered
        assert!(reg.lock().await.disarm(id, &tx));
        let ev = tokio::time::timeout(Duration::from_millis(200), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(ev, LogEvent::WakeDisarmed { .. }));
        // No batch should be produced.
        assert!(
            tokio::time::timeout(
                WAKE_BATCH_WINDOW + Duration::from_millis(100),
                batchRx.recv(),
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn batcherCoalescesBurst() {
        // Register a monitor-style passive source and shove ten fires
        // at it back-to-back. The batcher should hand back one batch
        // with all ten in arrival order.
        let (reg, tx, _rx, mut batchRx) = buildTestRegistry();
        let id = reg.lock().await.registerMonitor(7, &tx);
        for i in 0..10 {
            reg.lock()
                .await
                .enqueueFire(id, "monitor#7".into(), format!("line {i}"));
        }
        let batch = tokio::time::timeout(
            WAKE_BATCH_WINDOW + Duration::from_millis(200),
            batchRx.recv(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(batch.fires.len(), 10);
        for (i, f) in batch.fires.iter().enumerate() {
            assert_eq!(f.payload, format!("line {i}"));
            assert!(matches!(f.kind, ControlWakeKind::MonitorMatch));
        }
    }

    #[tokio::test]
    async fn disarmAllDropsLateCallbackFire() {
        // `/clear` and resume swap the registry after calling disarmAll.
        // Monitor callbacks hold a fire sender clone, so a line that arrives
        // just after disarmAll can still hit the old batcher. The closed
        // registry must drop that fire instead of injecting it.
        let (reg, tx, _rx, mut batchRx) = buildTestRegistry();
        let id = reg.lock().await.registerMonitor(7, &tx);
        let fireTx = reg.lock().await.fireSender();
        reg.lock().await.disarmAll();
        let _ = fireTx.send(WakeFire {
            wakeId: id,
            source: "monitor#7".into(),
            kind: ControlWakeKind::MonitorMatch,
            payload: "line after clear".into(),
            firedAt: Instant::now(),
        });
        let received = tokio::time::timeout(
            WAKE_BATCH_WINDOW + Duration::from_millis(200),
            batchRx.recv(),
        )
        .await;
        assert!(
            matches!(received, Err(_) | Ok(None)),
            "closed registry must not emit fires from stale callback senders",
        );
    }

    #[test]
    fn cronAcceptsFiveAndSixFieldSpec() {
        // 5-field: minute hour dom month dow
        assert!(cron::Schedule::from_str("0 0 9 * * 1-5").is_ok()); // 6-field, with seconds
        let normalized = format!("0 {}", "0 9 * * 1-5");
        assert!(cron::Schedule::from_str(&normalized).is_ok());
    }

    #[tokio::test]
    async fn invalidCronRejected() {
        let (tx, _rx) = mpsc::channel(16);
        let (reg, _batchRx) = WakeRegistry::new();
        let err = WakeRegistry::armCron(&reg, "nope".into(), true, "x".into(), tx).unwrap_err();
        assert!(err.to_string().contains("invalid cron"));
    }
}
