//! Background-job plane: agent-spawned subprocesses that run outside
//! the foreground turn loop.
//!
//! Unlike the regular `shell` tool, a [`BgJob`] does not block the
//! agent's turn — `bashSpawn` returns a handle immediately. Output is
//! line-buffered into a bounded ring so the agent can poll via
//! `jobOutput`; the deck mirrors task state for the /tasks panel.
//!
//! Tasks live until the session ends. There is no auto-cleanup of
//! completed tasks in phase 2; user can `jobStop` running ones.
//!
//! # Public API
//! - [`JobKind`], [`JobState`], [`JobInfo`], [`JobOutputSnapshot`]
//!
//! # Dependencies
//! `tokio` (process + io + sync), [`crate::control`]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::mpsc;

use crate::control::LogEvent;

#[cfg(test)]
use std::process::Stdio;
#[cfg(test)]
use tokio::io::{AsyncBufReadExt, BufReader};
#[cfg(test)]
use tokio::process::Command;

/// Identifier for a background job, monotonically increasing per session.
pub type JobId = u64;

pub type JobResult<T> = std::result::Result<T, JobError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    NotFound { id: JobId },
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobError::NotFound { id } => write!(f, "no task #{id}"),
        }
    }
}

impl std::error::Error for JobError {}

/// Cap on lines kept in a task's stdout ring buffer. Earlier lines are
/// dropped with a "[truncated N earlier lines]" marker on retrieval.
pub const MAX_BUFFERED_LINES: usize = 5000;

/// Cap on `jobOutput` response lines. Model can page with `sinceLine`.
pub const MAX_RESPONSE_LINES: usize = 500;

/// Side-effect callback invoked for every line emitted by a bash drainer.
/// Used by [`MonitorPlane`] to filter+count without duplicating the
/// drainer logic. Must be cheap (it's called on the hot path of each
/// stdout/stderr line).
#[cfg(test)]
type LineCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Side-effect callback invoked once when a bash-backed task reaches a
/// terminal state. Used by `MonitorPlane` to mirror natural backing-task
/// exits into monitor lifecycle without polling.
#[cfg(test)]
type ExitCallback = Box<dyn Fn(JobId, JobState) + Send + Sync>;

/// Kind of background job.
#[derive(Debug, Clone)]
pub enum JobKind {
    /// `shell(runInBackground: true)` — fork a child process under
    /// `bash -c`.
    Bash,
    /// `task(runInBackground: true)` — a child session running on its
    /// own tokio task. `agentType` is `"explore"`, `"general"`, etc.
    /// `prompt` is captured so the tasks panel can show what the agent is
    /// working on without polling.
    Subagent { agentType: String, prompt: String },
}

/// Lifecycle state of a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    Running,
    Completed { exitCode: i32 },
    Killed,
    Errored(String),
}

impl JobState {
    pub fn isTerminal(&self) -> bool {
        !matches!(self, JobState::Running)
    }
}

/// Lightweight snapshot for the `jobList` tool and /tasks panel.
#[derive(Debug, Clone)]
pub struct JobInfo {
    pub id: JobId,
    pub kind: JobKind,
    pub command: String,
    pub state: JobState,
    pub spawnedAt: Instant,
    pub completedAt: Option<Instant>,
    /// Total lines emitted (not just buffered).
    pub totalLines: u64,
}

/// Snapshot of a task's output for `jobOutput`.
pub struct JobOutputSnapshot {
    pub lines: Vec<String>,
    /// Completed subagents have a message-shaped final answer in addition
    /// to their live line stream. Bash-style jobs leave this empty.
    pub finalOutput: Option<String>,
    /// First line index this snapshot includes (caller uses to compute
    /// next `sinceLine`).
    pub firstLine: u64,
    pub totalLines: u64,
    pub state: JobState,
    pub command: String,
    /// Earliest line index still buffered in the ring. Any line below
    /// this number has been evicted; lines from `earliestBuffered..firstLine`
    /// are still retrievable via `taskOutput(sinceLine: earliestBuffered)`.
    pub earliestBuffered: u64,
}

/// Bounded line ring with a monotonic line counter.
struct LineRing {
    items: VecDeque<String>,
    capacity: usize,
    /// Total lines ever pushed — `items[0]` corresponds to
    /// `totalPushed - items.len()`.
    totalPushed: u64,
}

impl LineRing {
    fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
            totalPushed: 0,
        }
    }

    fn push(&mut self, line: String) {
        if self.items.len() >= self.capacity {
            self.items.pop_front();
        }
        self.items.push_back(line);
        self.totalPushed += 1;
    }

    /// Lines from `sinceLine` onwards (clamped to what's still buffered),
    /// capped at `maxLines`. When `sinceLine` is `None`, returns the
    /// *latest* `maxLines` lines — this matches the agent-facing contract
    /// ("most recent tail"), not the head of the ring.
    fn since(&self, sinceLine: Option<u64>, maxLines: usize) -> (Vec<String>, u64) {
        let earliest = self.totalPushed.saturating_sub(self.items.len() as u64);
        let start = match sinceLine {
            Some(s) => s.max(earliest),
            None => {
                // Tail: start `maxLines` before the end (or the earliest
                // we still hold, whichever is later).
                let tailStart = self.totalPushed.saturating_sub(maxLines as u64);
                tailStart.max(earliest)
            }
        };
        let skip = (start - earliest) as usize;
        let lines: Vec<String> = self
            .items
            .iter()
            .skip(skip)
            .take(maxLines)
            .cloned()
            .collect();
        (lines, start)
    }
}

/// Inner state shared between the registry and the spawned drainer task.
struct BgJobInner {
    state: JobState,
    completedAt: Option<Instant>,
    stdoutTail: LineRing,
    finalOutput: Option<String>,
    /// Wake context attached AFTER spawn. The drainer reads this on
    /// completion and pushes a `WakeFire` through the registry's batcher
    /// — coalesced with any concurrent wakes into a single `WakeBatch`.
    /// `None` means no wake was registered for this job.
    wakeCtx: Option<TaskWakeCtx>,
}

/// Per-task wake plumbing — id of the passive wake source, the shared
/// registry handle (used once on completion to unregister the passive
/// source), and the batcher's fire sender (used to enqueue the
/// completion fire). Shared by bash bg jobs and subagent jobs.
#[derive(Clone)]
pub(crate) struct TaskWakeCtx {
    pub wakeId: crate::wakes::WakeId,
    pub registry: Arc<tokio::sync::Mutex<crate::wakes::WakeRegistry>>,
    pub fireTx: tokio::sync::mpsc::UnboundedSender<crate::wakes::WakeFire>,
}

/// A background job: command + drainer handle + shared state.
pub(crate) struct BgJob {
    pub id: JobId,
    pub kind: JobKind,
    pub command: String,
    pub spawnedAt: Instant,
    inner: Arc<Mutex<BgJobInner>>,
    /// Idempotent kill closure. Each task kind plugs in its own cancel
    /// mechanism (bash drainer's mpsc, subagent runner's watch). Calling
    /// twice is safe — implementations swallow send errors.
    kill: Arc<dyn Fn() + Send + Sync>,
}

impl BgJob {
    pub fn state(&self) -> JobState {
        self.inner.lock().unwrap().state.clone()
    }

    pub fn info(&self) -> JobInfo {
        let inner = self.inner.lock().unwrap();
        JobInfo {
            id: self.id,
            kind: self.kind.clone(),
            command: self.command.clone(),
            state: inner.state.clone(),
            spawnedAt: self.spawnedAt,
            completedAt: inner.completedAt,
            totalLines: inner.stdoutTail.totalPushed,
        }
    }

    pub fn snapshot(&self, sinceLine: Option<u64>, maxLines: usize) -> JobOutputSnapshot {
        let inner = self.inner.lock().unwrap();
        let (lines, firstLine) = inner.stdoutTail.since(sinceLine, maxLines);
        let earliestBuffered = inner
            .stdoutTail
            .totalPushed
            .saturating_sub(inner.stdoutTail.items.len() as u64);
        JobOutputSnapshot {
            lines,
            finalOutput: inner.finalOutput.clone(),
            firstLine,
            totalLines: inner.stdoutTail.totalPushed,
            state: inner.state.clone(),
            command: self.command.clone(),
            earliestBuffered,
        }
    }

    /// Signal the runner to cancel its work and exit.
    pub fn kill(&self) {
        (self.kill)();
    }
}

/// Handle a subagent runner uses to feed its progress into the
/// JobPlane. Owned by the runner; dropping it doesn't terminate the
/// task — only an explicit `complete`/`error` call moves it out of
/// `Running`.
pub(crate) struct SubagentJobHandle {
    pub id: JobId,
    inner: Arc<Mutex<BgJobInner>>,
    logTx: mpsc::Sender<LogEvent>,
    /// Watch flag the runner polls to notice `jobStop` requests.
    pub cancelRx: tokio::sync::watch::Receiver<bool>,
}

impl SubagentJobHandle {
    /// Append a line of streamed output into the task's ring buffer
    /// and emit a `TaskOutput` event for live UIs (e.g. the inspector).
    #[cfg(test)]
    pub fn pushLine(&self, line: impl Into<String>) {
        let line = line.into();
        self.inner.lock().unwrap().stdoutTail.push(line.clone());
        let _ = self
            .logTx
            .try_send(LogEvent::JobOutput { id: self.id, line });
    }

    /// Mark the subagent complete and attach its final answer for the
    /// completion wake and post-completion `jobOutput` calls.
    pub async fn completeWithOutput(&self, exitCode: i32, finalOutput: String) {
        let (wakeCtx, outputPreview) = {
            let mut g = self.inner.lock().unwrap();
            g.state = JobState::Completed { exitCode };
            g.completedAt = Some(Instant::now());
            if !finalOutput.is_empty() {
                g.finalOutput = Some(finalOutput.clone());
            }
            let joined = if !finalOutput.is_empty() {
                finalOutput
            } else {
                let (lines, _) = g.stdoutTail.since(None, 20);
                if lines.is_empty() {
                    "(no output)".to_string()
                } else {
                    lines.join("\n")
                }
            };
            (g.wakeCtx.clone(), joined)
        };
        let _ = self
            .logTx
            .send(LogEvent::JobComplete {
                id: self.id,
                exitCode: Some(exitCode),
            })
            .await;
        self.fireWake(
            wakeCtx,
            format!(
                "Subagent task #{id} exited with code {exitCode}.\n\
             Final output:\n{outputPreview}",
                id = self.id,
            ),
        )
        .await;
    }

    /// Mark the subagent errored and attach any partial/final text that
    /// the child produced before failing.
    pub async fn erroredWithOutput(&self, reason: String, finalOutput: String) {
        let (wakeCtx, outputPreview) = {
            let mut g = self.inner.lock().unwrap();
            g.state = JobState::Errored(reason.clone());
            g.completedAt = Some(Instant::now());
            if !finalOutput.is_empty() {
                g.finalOutput = Some(finalOutput.clone());
            }
            let joined = if !finalOutput.is_empty() {
                finalOutput
            } else {
                let (lines, _) = g.stdoutTail.since(None, 20);
                if lines.is_empty() {
                    "(no output)".to_string()
                } else {
                    lines.join("\n")
                }
            };
            (g.wakeCtx.clone(), joined)
        };
        let _ = self
            .logTx
            .send(LogEvent::JobStopped {
                id: self.id,
                reason: reason.clone(),
            })
            .await;
        self.fireWake(
            wakeCtx,
            format!(
                "Subagent task #{id} errored: {reason}.\n\
             Final output:\n{outputPreview}",
                id = self.id,
            ),
        )
        .await;
    }

    /// Mark the subagent killed and retain any text it produced before
    /// cancellation for later inspection.
    /// No wake fires for kills — the parent already knows it asked.
    pub async fn killedWithOutput(&self, finalOutput: String) {
        let wakeCtx = {
            let mut g = self.inner.lock().unwrap();
            g.state = JobState::Killed;
            g.completedAt = Some(Instant::now());
            if !finalOutput.is_empty() {
                g.finalOutput = Some(finalOutput);
            }
            g.wakeCtx.clone()
        };
        let _ = self
            .logTx
            .send(LogEvent::JobStopped {
                id: self.id,
                reason: "killed".into(),
            })
            .await;
        if let Some(ctx) = wakeCtx {
            ctx.registry
                .lock()
                .await
                .unregisterPassive(ctx.wakeId, &self.logTx);
        }
    }

    /// Fire the TaskComplete wake (and unregister the passive source).
    async fn fireWake(&self, wakeCtx: Option<TaskWakeCtx>, payload: String) {
        let Some(ctx) = wakeCtx else { return };
        let _ = ctx.fireTx.send(crate::wakes::WakeFire {
            wakeId: ctx.wakeId,
            source: format!("task#{}", self.id),
            kind: crate::control::WakeKind::TaskComplete,
            payload,
            firedAt: Instant::now(),
        });
        ctx.registry
            .lock()
            .await
            .unregisterPassive(ctx.wakeId, &self.logTx);
    }

    /// True once `jobStop` (or session shutdown) has signaled cancel.
    pub fn cancelRequested(&self) -> bool {
        *self.cancelRx.borrow()
    }

    /// Return a cheap clone-able pusher for streaming lines into the
    /// task's ring buffer from another tokio task. Useful when the runner
    /// hands off line production to a separate forwarder.
    pub fn lineSender(&self) -> SubagentLineSender {
        SubagentLineSender {
            id: self.id,
            inner: self.inner.clone(),
            logTx: self.logTx.clone(),
        }
    }
}

/// Cheap clone-able line pusher for a subagent task. Detached from the
/// handle so the runner's log forwarder can own it without holding the
/// full handle (which would otherwise block the completion path).
#[derive(Clone)]
pub(crate) struct SubagentLineSender {
    id: JobId,
    inner: Arc<Mutex<BgJobInner>>,
    logTx: mpsc::Sender<LogEvent>,
}

impl SubagentLineSender {
    pub fn push(&self, line: impl Into<String>) {
        let line = line.into();
        self.inner.lock().unwrap().stdoutTail.push(line.clone());
        let _ = self
            .logTx
            .try_send(LogEvent::JobOutput { id: self.id, line });
    }
}

/// Per-session registry of background jobs.
pub(crate) struct JobPlane {
    jobs: HashMap<JobId, BgJob>,
    /// Insertion order for stable listing.
    order: Vec<JobId>,
    nextId: JobId,
    /// CWD newly-spawned tasks inherit. Falls back to current_dir at
    /// spawn time if None.
    #[cfg(test)]
    cwd: Option<std::path::PathBuf>,
}

impl Drop for JobPlane {
    /// Kill every still-running task when the plane drops. Prevents
    /// orphan subprocesses when the session ends.
    fn drop(&mut self) {
        for task in self.jobs.values() {
            task.kill();
        }
    }
}

impl JobPlane {
    pub fn new(cwd: Option<std::path::PathBuf>) -> Self {
        #[cfg(not(test))]
        let _ = &cwd;
        Self {
            jobs: HashMap::new(),
            order: Vec::new(),
            nextId: 1,
            #[cfg(test)]
            cwd,
        }
    }

    /// Reserve the next job id without spawning anything. Used when
    /// the caller needs to register a passive wake source against a known id.
    pub fn reserveJobId(&mut self) -> JobId {
        let id = self.nextId;
        self.nextId += 1;
        id
    }

    /// Spawn a new bash task. The command runs via `bash -c` so shell
    /// features (pipes, redirects, env expansion, bashisms) work.
    /// Stdout and stderr are interleaved into the same ring buffer in
    /// line order of arrival.
    #[cfg(test)]
    pub fn spawnBash(
        &mut self,
        command: String,
        logTx: mpsc::Sender<LogEvent>,
    ) -> JobResult<JobId> {
        let id = self.reserveJobId();
        self.spawnBashInner(id, command, JobKind::Bash, "bash", None, None, logTx, None)
    }

    /// Shared body: build the bg task entry, emit `TaskSpawned`, then
    /// kick off the bash drainer with an optional per-line side effect.
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    fn spawnBashInner(
        &mut self,
        id: JobId,
        command: String,
        kind: JobKind,
        kindLabel: &str,
        onLine: Option<LineCallback>,
        onExit: Option<ExitCallback>,
        logTx: mpsc::Sender<LogEvent>,
        wakeCtx: Option<TaskWakeCtx>,
    ) -> JobResult<JobId> {
        let cwd = self.cwd.clone().unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        });

        let inner = Arc::new(Mutex::new(BgJobInner {
            state: JobState::Running,
            completedAt: None,
            stdoutTail: LineRing::new(MAX_BUFFERED_LINES),
            finalOutput: None,
            wakeCtx,
        }));

        let (killTx, killRx) = mpsc::channel::<()>(1);
        let killClosure: Arc<dyn Fn() + Send + Sync> = {
            let killTx = killTx.clone();
            Arc::new(move || {
                let _ = killTx.try_send(());
            })
        };

        let task = BgJob {
            id,
            kind,
            command: command.clone(),
            spawnedAt: Instant::now(),
            inner: inner.clone(),
            kill: killClosure,
        };

        self.jobs.insert(id, task);
        self.order.push(id);

        // Emit TaskSpawned BEFORE the drainer starts (see spawnBash for
        // the rationale on counter-race avoidance).
        let _ = logTx.try_send(LogEvent::JobSpawned {
            id,
            kind: kindLabel.into(),
            command: command.clone(),
        });

        // Both bash bg jobs and subagent jobs fire a `TaskComplete` wake
        // on exit. Monitors fire wakes per-match instead.
        let fireWakeOnComplete = kindLabel == "bash";

        let drainerInner = inner.clone();
        let drainerLog = logTx.clone();
        tokio::spawn(async move {
            runBashTask(
                id,
                command,
                cwd,
                drainerInner,
                killRx,
                drainerLog,
                onLine,
                onExit,
                fireWakeOnComplete,
            )
            .await;
        });

        Ok(id)
    }

    /// Register a subagent task. Unlike `spawnBash` this does NOT spawn
    /// a child process — the caller (session.rs) owns the actual child
    /// session and drives it. We just allocate an id, set up the ring
    /// buffer, and return a handle the caller pushes lines/completion
    /// status into. The handle's `cancelRx` lets the runner notice
    /// `jobStop` requests cooperatively.
    #[cfg(test)]
    pub fn spawnSubagent(
        &mut self,
        agentType: String,
        prompt: String,
        logTx: mpsc::Sender<LogEvent>,
    ) -> SubagentJobHandle {
        let id = self.reserveJobId();
        self.spawnSubagentWithId(id, agentType, prompt, logTx, None)
    }

    /// Spawn a subagent task with a pre-reserved id and a passive wake
    /// context already resolved. Used by session.rs to register a
    /// `TaskComplete` wake against the same id before the runner starts.
    pub fn spawnSubagentWithId(
        &mut self,
        id: JobId,
        agentType: String,
        prompt: String,
        logTx: mpsc::Sender<LogEvent>,
        wakeCtx: Option<TaskWakeCtx>,
    ) -> SubagentJobHandle {
        let inner = Arc::new(Mutex::new(BgJobInner {
            state: JobState::Running,
            completedAt: None,
            stdoutTail: LineRing::new(MAX_BUFFERED_LINES),
            finalOutput: None,
            wakeCtx,
        }));

        // Cooperative cancel: the runner watches this signal.
        let (cancelTx, cancelRx) = tokio::sync::watch::channel(false);
        let killClosure: Arc<dyn Fn() + Send + Sync> = {
            let cancelTx = cancelTx.clone();
            Arc::new(move || {
                let _ = cancelTx.send(true);
            })
        };

        // For tasks-panel list rendering the "command" string is the prompt
        // preview — same column as bash uses for the command.
        let summary = format!("task[{agentType}]: {prompt}");

        let task = BgJob {
            id,
            kind: JobKind::Subagent {
                agentType: agentType.clone(),
                prompt: prompt.clone(),
            },
            command: summary.clone(),
            spawnedAt: Instant::now(),
            inner: inner.clone(),
            kill: killClosure,
        };

        self.jobs.insert(id, task);
        self.order.push(id);

        // Emit spawn event for the deck mirror.
        let _ = logTx.try_send(LogEvent::JobSpawned {
            id,
            kind: "subagent".into(),
            command: summary,
        });

        SubagentJobHandle {
            id,
            inner,
            logTx,
            cancelRx,
        }
    }

    /// Output snapshot for a task.
    pub fn output(
        &self,
        id: JobId,
        sinceLine: Option<u64>,
        maxLines: usize,
    ) -> JobResult<JobOutputSnapshot> {
        let task = self.jobs.get(&id).ok_or(JobError::NotFound { id })?;
        Ok(task.snapshot(sinceLine, maxLines.min(MAX_RESPONSE_LINES)))
    }

    /// Kill a running task. Returns `Ok(())` for both successful kills
    /// and tasks that have already finished — the agent-facing contract
    /// is "no-op on already-terminal tasks", which means callers should
    /// not see an error for the harmless case of stopping a task that
    /// has just exited on its own. Only unknown task ids return Err.
    pub fn stop(&self, id: JobId) -> JobResult<()> {
        let task = self.jobs.get(&id).ok_or(JobError::NotFound { id })?;
        if task.state().isTerminal() {
            return Ok(());
        }
        task.kill();
        Ok(())
    }

    /// Kill every running task. Returns the ids that received a signal.
    /// Skips already-terminal tasks. Used by the `/killall` slash command.
    pub fn list(&self) -> Vec<JobInfo> {
        self.order
            .iter()
            .filter_map(|id| self.jobs.get(id).map(|t| t.info()))
            .collect()
    }
}

/// Drainer body: spawn the child, fan out stdout/stderr into the shared
/// ring buffer, race the child against `killRx`. On any termination,
/// update the task's state and emit `TaskComplete` / `TaskStopped`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
async fn runBashTask(
    id: JobId,
    command: String,
    cwd: std::path::PathBuf,
    inner: Arc<Mutex<BgJobInner>>,
    mut killRx: mpsc::Receiver<()>,
    logTx: mpsc::Sender<LogEvent>,
    onLine: Option<LineCallback>,
    onExit: Option<ExitCallback>,
    fireWakeOnComplete: bool,
) {
    // The tool is named `bashSpawn` and the schema promises bash — use
    // bash via PATH so common installs (`/bin/bash` on macOS/Linux,
    // `/opt/homebrew/bin/bash` for newer macOS users, $PATH bash for
    // containers) all resolve. The agent can rely on bashisms like
    // `[[ ... ]]`, process substitution, and `pipefail`.
    let mut cmd = Command::new("bash");
    cmd.arg("-c")
        .arg(&command)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Put the child in its own process group so `jobStop` can kill the
    // whole tree, not just `/bin/sh`. Without this, a child like
    // `npm run dev` survives when only the shell wrapper receives SIGKILL.
    #[cfg(unix)]
    cmd.process_group(0);

    let spawnResult = cmd.spawn();

    let mut child = match spawnResult {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("failed to spawn: {e}");
            {
                let mut g = inner.lock().unwrap();
                g.state = JobState::Errored(msg.clone());
                g.completedAt = Some(Instant::now());
            }
            let _ = logTx.send(LogEvent::JobStopped { id, reason: msg }).await;
            return;
        }
    };

    // Process group id — same as the leader's pid on unix (process_group(0)).
    // Captured here so kill paths can target the entire tree.
    #[cfg(unix)]
    let pgid = child.id().map(|p| p as i32);
    #[cfg(not(unix))]
    let pgid: Option<i32> = None;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Drain stdout + stderr concurrently. Lines go into the ring buffer
    // and emit a TaskOutput event so the deck can stream them. We retain
    // the JoinHandles so we can await both drainers before emitting
    // TaskComplete — otherwise the UI/agent can observe "completed" with
    // final lines not yet in the ring.
    // The optional per-line side effect (e.g. MonitorPlane filter/count)
    // is wrapped in `Arc` so both drainers share one closure. It must
    // outlive both drainer tasks, hence `Arc<Option<...>>`.
    let onLineShared: Arc<Option<LineCallback>> = Arc::new(onLine);

    let stdoutJoin = stdout.map(|s| {
        let innerStdout = inner.clone();
        let logStdout = logTx.clone();
        let onLineStdout = onLineShared.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(s).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                innerStdout.lock().unwrap().stdoutTail.push(line.clone());
                if let Some(cb) = onLineStdout.as_ref() {
                    cb(&line);
                }
                let _ = logStdout.try_send(LogEvent::JobOutput { id, line });
            }
        })
    });
    let stderrJoin = stderr.map(|s| {
        let innerStderr = inner.clone();
        let logStderr = logTx.clone();
        let onLineStderr = onLineShared.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(s).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                innerStderr.lock().unwrap().stdoutTail.push(line.clone());
                if let Some(cb) = onLineStderr.as_ref() {
                    cb(&line);
                }
                let _ = logStderr.try_send(LogEvent::JobOutput { id, line });
            }
        })
    });

    // Wait for child exit OR kill signal.
    enum Outcome {
        Completed(Result<std::process::ExitStatus, std::io::Error>),
        Killed,
    }
    let outcome = tokio::select! {
        result = child.wait() => Outcome::Completed(result),
        _ = killRx.recv() => {
            // Signal the whole tree first; fall back to just the child
            // when we can't reach the process group (e.g. windows).
            killProcessTree(pgid, &mut child).await;
            Outcome::Killed
        }
    };

    // On natural exit ALSO reap the process group so detached
    // grandchildren (`bash -c 'yes &'` is the canonical example)
    // can't continue running past the task's TaskComplete event.
    // The child is gone, but the kernel keeps the pgid alive as long
    // as any member still exists; SIGTERM the group, sleep briefly,
    // then SIGKILL anything still alive. With process_group(0) the
    // leader IS the pgid, so any escapee is in the same group.
    if matches!(outcome, Outcome::Completed(_)) {
        #[cfg(unix)]
        if let Some(pgid) = pgid {
            unsafe {
                libc::kill(-pgid, libc::SIGTERM);
            }
        }
    }

    // Drain the readers before reporting terminal state so the ring
    // buffer has the final stdout/stderr lines from the child. We bound
    // the wait with a short grace period: a child that exits while
    // leaving grandchildren attached to its stdout (e.g. `bash -c '... &'`)
    // would otherwise keep these handles open indefinitely.
    // `tokio::time::timeout` only drops the inner future on elapsed —
    // for a JoinHandle that *detaches* the underlying task instead of
    // aborting it. To actually terminate the drainer (and release the
    // pipe fd that's keeping yes/cat/etc. alive on stdout), we hold
    // the JoinHandle and call `.abort()` on timeout.
    let drainGrace = std::time::Duration::from_millis(200);
    async fn awaitWithAbort(j: tokio::task::JoinHandle<()>, grace: std::time::Duration) {
        let mut j = j;
        tokio::select! {
            _ = &mut j => {}
            _ = tokio::time::sleep(grace) => {
                j.abort();
                let _ = j.await;
            }
        }
    }
    if let Some(j) = stdoutJoin {
        awaitWithAbort(j, drainGrace).await;
    }
    if let Some(j) = stderrJoin {
        awaitWithAbort(j, drainGrace).await;
    }

    // After draining (or aborting the drainer) any pgroup escapees
    // still alive get a forceful SIGKILL. Belt-and-suspenders: the
    // SIGTERM above may have been ignored, and the readers might have
    // exited on our abort rather than EOF.
    #[cfg(unix)]
    if matches!(outcome, Outcome::Completed(_))
        && let Some(pgid) = pgid
    {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        unsafe {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }

    // Snapshot a short payload preview + capture any attached wake
    // context before firing. The drainer fires via WakeRegistry when
    // ctx is set (bumps firesSoFar, unregisters the passive source);
    // otherwise it falls back to a direct LogEvent emission.
    let (tailPreview, wakeCtx) = if fireWakeOnComplete {
        let g = inner.lock().unwrap();
        let (lines, _) = g.stdoutTail.since(None, 20);
        let joined = if lines.is_empty() {
            "(no output)".to_string()
        } else {
            lines.join("\n")
        };
        (Some(joined), g.wakeCtx.clone())
    } else {
        (None, None)
    };
    let source = format!("task#{id}");
    // Completion fire path: enqueue through the registry's batcher so
    // the per-task wake is coalesced with any concurrent fires, then
    // unregister the passive source so it disappears from `/jobs`.
    let fireWake = |payload: String| {
        let wakeCtxClone = wakeCtx.clone();
        let sourceClone = source.clone();
        let logTxClone = logTx.clone();
        async move {
            if let Some(ctx) = wakeCtxClone {
                let _ = ctx.fireTx.send(crate::wakes::WakeFire {
                    wakeId: ctx.wakeId,
                    source: sourceClone,
                    kind: crate::control::WakeKind::TaskComplete,
                    payload,
                    firedAt: Instant::now(),
                });
                ctx.registry
                    .lock()
                    .await
                    .unregisterPassive(ctx.wakeId, &logTxClone);
            }
        }
    };

    match outcome {
        Outcome::Completed(Ok(status)) => {
            let code = status.code().unwrap_or(-1);
            let terminalState = JobState::Completed { exitCode: code };
            {
                let mut g = inner.lock().unwrap();
                g.state = terminalState.clone();
                g.completedAt = Some(Instant::now());
            }
            let _ = logTx
                .send(LogEvent::JobComplete {
                    id,
                    exitCode: Some(code),
                })
                .await;
            if let Some(preview) = &tailPreview {
                fireWake(format!(
                    "Background task #{id} ({command}) exited with code {code}.\n\
                     Tail of output:\n{preview}"
                ))
                .await;
            }
            if let Some(cb) = onExit.as_ref() {
                cb(id, terminalState);
            }
        }
        Outcome::Completed(Err(e)) => {
            let msg = format!("wait failed: {e}");
            let terminalState = JobState::Errored(msg.clone());
            {
                let mut g = inner.lock().unwrap();
                g.state = terminalState.clone();
                g.completedAt = Some(Instant::now());
            }
            let _ = logTx
                .send(LogEvent::JobStopped {
                    id,
                    reason: msg.clone(),
                })
                .await;
            if let Some(preview) = &tailPreview {
                fireWake(format!(
                    "Background task #{id} ({command}) errored: {msg}.\n\
                     Tail of output:\n{preview}"
                ))
                .await;
            }
            if let Some(cb) = onExit.as_ref() {
                cb(id, terminalState);
            }
        }
        Outcome::Killed => {
            // Reap the zombie.
            let _ = child.wait().await;
            let terminalState = JobState::Killed;
            {
                let mut g = inner.lock().unwrap();
                g.state = terminalState.clone();
                g.completedAt = Some(Instant::now());
            }
            let _ = logTx
                .send(LogEvent::JobStopped {
                    id,
                    reason: "killed".into(),
                })
                .await;
            // No wake fired for kills — the user/agent initiated the
            // stop and already knows about it. But we DO still need to
            // unregister the passive wake source so /jobs schedules
            // doesn't keep showing a phantom row for a dead job.
            if let Some(ctx) = wakeCtx {
                ctx.registry
                    .lock()
                    .await
                    .unregisterPassive(ctx.wakeId, &logTx);
            }
            if let Some(cb) = onExit.as_ref() {
                cb(id, terminalState);
            }
        }
    }
}

/// Kill the entire process tree rooted at `pgid` if known; falls back to
/// killing just the leader. On unix, sends SIGTERM to the process group
/// (graceful), waits briefly, then SIGKILLs anything still alive.
#[cfg(test)]
async fn killProcessTree(pgid: Option<i32>, child: &mut tokio::process::Child) {
    #[cfg(unix)]
    {
        if let Some(pgid) = pgid {
            // SAFETY: passing a negative pid to `kill(2)` targets the
            // process group; we only call this with a pgid we know exists
            // from `child.id()`. Even if the group has already exited the
            // worst case is ESRCH which we ignore.
            unsafe {
                libc::kill(-pgid, libc::SIGTERM);
            }
            // Give the tree ~250ms to exit gracefully before SIGKILL.
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
            return;
        }
    }
    let _ = child.kill().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const TEST_EVENT_TIMEOUT: Duration = Duration::from_secs(3);

    async fn recvLogMatching(
        rx: &mut mpsc::Receiver<LogEvent>,
        matchesEvent: impl Fn(&LogEvent) -> bool,
        label: &str,
    ) -> LogEvent {
        tokio::time::timeout(TEST_EVENT_TIMEOUT, async move {
            loop {
                let ev = rx
                    .recv()
                    .await
                    .unwrap_or_else(|| panic!("log channel closed before {label}"));
                if matchesEvent(&ev) {
                    return ev;
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {label}"))
    }

    async fn waitForJobTerminal(rx: &mut mpsc::Receiver<LogEvent>, id: JobId) -> LogEvent {
        recvLogMatching(
            rx,
            |ev| {
                matches!(
                    ev,
                    LogEvent::JobComplete { id: eventId, .. }
                        | LogEvent::JobStopped { id: eventId, .. }
                        if *eventId == id
                )
            },
            "job terminal event",
        )
        .await
    }

    #[tokio::test]
    async fn spawnAndComplete() {
        let (tx, mut rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let id = plane.spawnBash("echo hello".into(), tx.clone()).unwrap();

        let ev = waitForJobTerminal(&mut rx, id).await;
        assert!(matches!(
            ev,
            LogEvent::JobComplete {
                id: eventId,
                exitCode: Some(0),
            } if eventId == id
        ));
        let snap = plane.output(id, None, 100).unwrap();
        assert!(matches!(snap.state, JobState::Completed { exitCode: 0 }));
        assert!(snap.lines.iter().any(|l| l == "hello"));

        drop(tx);
        drop(plane);
    }

    #[tokio::test]
    async fn killStopsRunning() {
        let (tx, mut rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let id = plane.spawnBash("sleep 30".into(), tx.clone()).unwrap();

        assert_eq!(plane.jobs[&id].state(), JobState::Running);
        plane.stop(id).unwrap();

        let ev = waitForJobTerminal(&mut rx, id).await;
        assert!(matches!(
            ev,
            LogEvent::JobStopped {
                id: eventId,
                reason,
            } if eventId == id && reason == "killed"
        ));
        assert_eq!(plane.jobs[&id].state(), JobState::Killed);

        drop(tx);
        drop(plane);
    }

    #[tokio::test]
    async fn detachedGrandchildDoesNotKeepTaskHanging() {
        // `yes hello &` detaches a producer that inherits stdout. The
        // bash leader exits immediately. Without process-group cleanup
        // + drainer-abort, `yes` would keep the stdout pipe open and
        // the drainer would never see EOF, so the task would hang past
        // child.wait(). With the fix, the process group is reaped and
        // the drainer is aborted on the grace timeout, so the task
        // reaches a terminal state quickly.
        let (tx, mut rx) = mpsc::channel(1024);
        let mut plane = JobPlane::new(None);
        let id = plane.spawnBash("yes hello &".into(), tx.clone()).unwrap();

        let _ = waitForJobTerminal(&mut rx, id).await;
        // Should report as completed (not killed) — the leader exited
        // on its own. The producer was reaped as cleanup, not as a
        // user-initiated kill.
        match plane.jobs[&id].state() {
            JobState::Completed { .. } => {}
            other => panic!("expected Completed, got {other:?}"),
        }

        drop(tx);
        drop(plane);
    }

    #[tokio::test]
    async fn subagentHandleStreamsAndCompletes() {
        // Verifies the agent-facing contract: registering a subagent
        // returns a handle; pushed lines land in the ring buffer in
        // order; complete() moves state to Completed and emits
        // TaskComplete with the right id.
        let (tx, mut rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let handle = plane.spawnSubagent(
            "explore".into(),
            "find the auth handlers".into(),
            tx.clone(),
        );
        let id = handle.id;
        assert!(matches!(
            plane.jobs[&id].kind,
            JobKind::Subagent { ref agentType, .. } if agentType == "explore",
        ));

        // Stream a few lines.
        handle.pushLine("starting search");
        handle.pushLine("found 3 candidates");
        handle.pushLine("returning summary");

        // Snapshot before completion: ring has the lines, state is Running.
        let snap = plane.output(id, None, 100).unwrap();
        assert_eq!(snap.totalLines, 3);
        assert_eq!(
            snap.lines,
            vec![
                "starting search".to_string(),
                "found 3 candidates".into(),
                "returning summary".into(),
            ]
        );
        assert_eq!(snap.state, JobState::Running);

        handle.completeWithOutput(0, String::new()).await;
        let snap = plane.output(id, None, 100).unwrap();
        assert_eq!(snap.state, JobState::Completed { exitCode: 0 });

        // Drain the event stream and assert we saw Spawned/Output*N/Complete.
        // Drop both the test-held sender AND the handle (whose logTx is a
        // clone of tx) so the channel actually closes and `rx.recv()`
        // returns None.
        drop(tx);
        drop(handle);
        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        let spawned = events.iter().any(|e| {
            matches!(
                e, LogEvent::JobSpawned { id: i, kind, .. }
                if *i == id && kind == "subagent"
            )
        });
        assert!(spawned, "TaskSpawned event missing: {events:?}");
        let outputCount = events
            .iter()
            .filter(|e| matches!(e, LogEvent::JobOutput { id: i, .. } if *i == id))
            .count();
        assert_eq!(outputCount, 3);
        let complete = events.iter().any(|e| {
            matches!(
                e, LogEvent::JobComplete { id: i, exitCode: Some(0) }
                if *i == id
            )
        });
        assert!(complete, "TaskComplete event missing: {events:?}");
    }

    #[tokio::test]
    async fn subagentCompleteFiresTaskCompleteWake() {
        // Pre-Phase-6 P1 fix: backgrounded subagents must fire a
        // TaskComplete wake on exit, mirroring the bash drainer path.
        // This guards the behavioral contract — earlier the wakeCtx
        // was always None, so completion only emitted a JobComplete
        // event and the parent agent never resumed.
        use crate::control::WakeKind as ControlWakeKind;
        use crate::wakes::WakeRegistry;

        let (tx, _rx) = mpsc::channel(64);
        let (regArc, mut batchRx) = WakeRegistry::new();

        let mut plane = JobPlane::new(None);
        let taskId = plane.reserveJobId();

        // Register the passive wake source and snapshot the fire
        // sender BEFORE the subagent runs — mirrors what session.rs
        // does on the bg-task path.
        let (wakeId, fireTx) = {
            let mut g = regArc.lock().await;
            let wid = g.registerTaskComplete(taskId, &tx);
            (wid, g.fireSender())
        };
        let wakeCtx = TaskWakeCtx {
            wakeId,
            registry: regArc.clone(),
            fireTx,
        };

        let handle = plane.spawnSubagentWithId(
            taskId,
            "explore".into(),
            "find the auth handlers".into(),
            tx.clone(),
            Some(wakeCtx),
        );

        handle.pushLine("scanning files");
        handle.completeWithOutput(0, String::new()).await;

        // The batcher coalesces fires within WAKE_BATCH_WINDOW; wait a
        // little longer than that for the batch to land.
        let batch = tokio::time::timeout(
            crate::wakes::WAKE_BATCH_WINDOW + std::time::Duration::from_millis(300),
            batchRx.recv(),
        )
        .await
        .expect("wake batch did not arrive")
        .expect("batch channel closed");
        assert_eq!(batch.fires.len(), 1);
        let fire = &batch.fires[0];
        assert!(matches!(fire.kind, ControlWakeKind::TaskComplete));
        assert_eq!(fire.source, format!("task#{taskId}"));
        assert!(
            fire.payload.contains("exited with code 0"),
            "payload missing exit info: {}",
            fire.payload,
        );
        assert!(
            fire.payload.contains("scanning files"),
            "payload missing tail line: {}",
            fire.payload,
        );

        // The passive wake source must be unregistered so /jobs
        // schedules don't keep a phantom row after exit.
        assert_eq!(regArc.lock().await.len(), 0);
    }

    #[tokio::test]
    async fn subagentCompletionWakeUsesFinalOutputNotProgressTail() {
        // Background subagents stream progress into the task buffer while
        // running, but the completion wake should carry the completed
        // answer, not whichever progress lines happen to be in the tail.
        use crate::control::WakeKind as ControlWakeKind;
        use crate::wakes::WakeRegistry;

        let (tx, _rx) = mpsc::channel(64);
        let (regArc, mut batchRx) = WakeRegistry::new();

        let mut plane = JobPlane::new(None);
        let taskId = plane.reserveJobId();
        let (wakeId, fireTx) = {
            let mut g = regArc.lock().await;
            let wid = g.registerTaskComplete(taskId, &tx);
            (wid, g.fireSender())
        };
        let wakeCtx = TaskWakeCtx {
            wakeId,
            registry: regArc.clone(),
            fireTx,
        };

        let handle = plane.spawnSubagentWithId(
            taskId,
            "explore".into(),
            "summarize auth".into(),
            tx.clone(),
            Some(wakeCtx),
        );

        handle.pushLine("progress: scanning files");
        handle.pushLine("progress: reading handlers");
        let finalOutput = "Final answer:\nAuth is routed through src/auth.rs.";
        handle.completeWithOutput(0, finalOutput.into()).await;

        let snap = plane.output(taskId, None, 100).unwrap();
        assert_eq!(snap.finalOutput.as_deref(), Some(finalOutput));

        let batch = tokio::time::timeout(
            crate::wakes::WAKE_BATCH_WINDOW + std::time::Duration::from_millis(300),
            batchRx.recv(),
        )
        .await
        .expect("wake batch did not arrive")
        .expect("batch channel closed");
        assert_eq!(batch.fires.len(), 1);
        let fire = &batch.fires[0];
        assert!(matches!(fire.kind, ControlWakeKind::TaskComplete));
        assert!(
            fire.payload.contains(finalOutput),
            "payload missing final output: {}",
            fire.payload,
        );
        assert!(
            !fire.payload.contains("progress: scanning files"),
            "payload should not use progress tail: {}",
            fire.payload,
        );
    }

    #[tokio::test]
    async fn subagentKilledUnregistersWithoutFiringWake() {
        // Kills are parent-initiated — the agent already knows it
        // asked, so firing a TaskComplete wake would just add noise.
        // But the passive source still needs to be removed so the
        // wake schedule stays clean.
        use crate::wakes::WakeRegistry;

        let (tx, _rx) = mpsc::channel(64);
        let (regArc, mut batchRx) = WakeRegistry::new();

        let mut plane = JobPlane::new(None);
        let taskId = plane.reserveJobId();
        let (wakeId, fireTx) = {
            let mut g = regArc.lock().await;
            let wid = g.registerTaskComplete(taskId, &tx);
            (wid, g.fireSender())
        };
        let wakeCtx = TaskWakeCtx {
            wakeId,
            registry: regArc.clone(),
            fireTx,
        };

        let handle = plane.spawnSubagentWithId(
            taskId,
            "general".into(),
            "go do the thing".into(),
            tx.clone(),
            Some(wakeCtx),
        );

        handle.killedWithOutput(String::new()).await;

        // No batch should arrive within the debounce window.
        let timed = tokio::time::timeout(
            crate::wakes::WAKE_BATCH_WINDOW + std::time::Duration::from_millis(300),
            batchRx.recv(),
        )
        .await;
        assert!(timed.is_err(), "expected no wake batch for killed subagent");

        assert_eq!(regArc.lock().await.len(), 0);
    }

    #[tokio::test]
    async fn subagentCancelFlipsHandleFlag() {
        // taskStop on a subagent must flip the handle's cancelRequested
        // bit (the runner polls this to abort cooperatively). Earlier
        // bash drainer logic used an mpsc, so this guards the unified
        // closure-based kill path for subagents.
        let (tx, _rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let handle = plane.spawnSubagent("general".into(), "go do the thing".into(), tx.clone());
        let id = handle.id;
        assert!(!handle.cancelRequested());
        plane.stop(id).unwrap();
        // Watch propagation is synchronous on send, so cancelRequested
        // should observe `true` immediately.
        assert!(handle.cancelRequested());
    }

    #[tokio::test]
    async fn stopOnTerminalTaskIsNoOp() {
        // The tool-facing contract says taskStop is a no-op on
        // already-terminal tasks. The agent must not see an error for
        // stopping a task that has just exited on its own.
        let (tx, mut rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let id = plane.spawnBash("true".into(), tx.clone()).unwrap();

        let _ = waitForJobTerminal(&mut rx, id).await;
        assert!(plane.jobs[&id].state().isTerminal());

        // Stopping the now-completed task must succeed silently.
        plane
            .stop(id)
            .expect("stop on terminal task should be no-op");

        // Stopping an unknown id is still an error.
        assert_eq!(plane.stop(9999), Err(JobError::NotFound { id: 9999 }));
        match plane.output(9999, None, 100) {
            Err(err) => assert_eq!(err, JobError::NotFound { id: 9999 }),
            Ok(_) => panic!("expected missing job error"),
        }

        drop(tx);
        drop(plane);
    }

    #[test]
    fn sinceNoneReturnsLatestTail() {
        let mut ring = LineRing::new(1000);
        for i in 0..1000 {
            ring.push(format!("line{i}"));
        }
        // Default (None, max 200) should return the *last* 200 lines, not
        // the head. This is the agent-facing contract: omitting sinceLine
        // means "give me the most recent tail".
        let (lines, firstLine) = ring.since(None, 200);
        assert_eq!(lines.len(), 200);
        assert_eq!(firstLine, 800);
        assert_eq!(lines.first().map(String::as_str), Some("line800"));
        assert_eq!(lines.last().map(String::as_str), Some("line999"));

        // When the buffer holds fewer lines than maxLines, we get them all.
        let mut small = LineRing::new(100);
        for i in 0..5 {
            small.push(format!("s{i}"));
        }
        let (lines, firstLine) = small.since(None, 50);
        assert_eq!(lines.len(), 5);
        assert_eq!(firstLine, 0);
    }

    #[tokio::test]
    async fn snapshotReportsEarliestBufferedDistinctFromFirstLine() {
        // Tail of a noisy task: firstLine > earliestBuffered means earlier
        // lines are still in the ring and can be fetched with an explicit
        // sinceLine. This is the value the session-side formatter uses to
        // distinguish "evicted" from "just not in this slice."
        let (tx, mut rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let id = plane
            .spawnBash(
                "for i in $(seq 1 50); do echo line$i; done".into(),
                tx.clone(),
            )
            .unwrap();

        let _ = waitForJobTerminal(&mut rx, id).await;

        let snap = plane.output(id, None, 10).unwrap();
        assert_eq!(snap.totalLines, 50);
        assert_eq!(snap.lines.len(), 10);
        assert_eq!(snap.firstLine, 40); // tail of 10 from 50 total
        // Nothing evicted: ring is 5000, only 50 lines pushed.
        assert_eq!(snap.earliestBuffered, 0);
        // The 40 earlier lines are still recoverable, not lost.
        assert!(snap.firstLine > snap.earliestBuffered);

        drop(tx);
        drop(plane);
    }

    #[tokio::test]
    async fn sinceLinePagingWorks() {
        let (tx, mut rx) = mpsc::channel(64);
        let mut plane = JobPlane::new(None);
        let id = plane
            .spawnBash(
                "for i in 1 2 3 4 5; do echo line$i; done".into(),
                tx.clone(),
            )
            .unwrap();

        let _ = waitForJobTerminal(&mut rx, id).await;

        let all = plane.output(id, None, 100).unwrap();
        assert_eq!(all.lines.len(), 5);
        assert_eq!(all.firstLine, 0);
        assert_eq!(all.totalLines, 5);

        let tail = plane.output(id, Some(3), 100).unwrap();
        assert_eq!(tail.lines, vec!["line4".to_string(), "line5".into()]);
        assert_eq!(tail.firstLine, 3);

        drop(tx);
        drop(plane);
    }
}
