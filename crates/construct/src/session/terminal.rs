use anyhow::Result;
use tokio::sync::mpsc;

use super::{Session, ShellResolveError, unixNow};
use crate::control::LogEvent;
use crate::shell::Shell;
use crate::shells::SpawnedBy;
use crate::tool;

fn terminalRunStatusForExecution(
    exec: &crate::shell::CommandExecution,
) -> crate::storage::TerminalRunStatus {
    if exec.output.starts_with("Terminal is busy") {
        crate::storage::TerminalRunStatus::Rejected
    } else if exec.timedOut && exec.exitCode.is_none() {
        crate::storage::TerminalRunStatus::TimedOut
    } else if exec.exitCode.unwrap_or(0) == 0 {
        crate::storage::TerminalRunStatus::Completed
    } else {
        crate::storage::TerminalRunStatus::Failed
    }
}

impl Session {
    /// Resolve the target shell for a shell-using action. Returns an
    /// error message — including the agent's current target and the
    /// list of available terminals — if the named terminal doesn't
    /// exist. The verbose error is the model's recovery path; without
    /// it, a single typo can derail several turns.
    pub(super) async fn resolveShell(
        &self,
        action: &tool::ToolAction,
    ) -> std::result::Result<Shell, ShellResolveError> {
        let guard = self.shells.lock().await;
        match action.terminal() {
            Some(name) => {
                guard
                    .shellFor(Some(name))
                    .ok_or_else(|| ShellResolveError::MissingNamed {
                        name: name.to_string(),
                        available: guard.names().to_vec(),
                        target: guard.activeForAgent().to_string(),
                    })
            }
            None => guard.shellFor(None).ok_or(ShellResolveError::NoAgentTarget),
        }
    }

    /// Handle the terminal-management tools (Spawn/Switch/Kill/List).
    /// These mutate `self.shells` and emit terminal lifecycle events on
    /// the log channel so the deck can update its tab strip.
    pub(super) async fn executeTerminalTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        match action {
            tool::ToolAction::TerminalSpawn { name } => {
                // NOTE: agent-spawned terminals do NOT emit TerminalSpawned
                // log events — the surrounding ToolStarted/ToolResult pair
                // already represents the action in the panel. Emitting a
                // separate notice between Started and Result causes the
                // panel to fail to pop the ToolActive entry (it pops only
                // when the trailing entry is still ToolActive), leaving
                // an orphan throbber. User-initiated spawns DO emit the
                // event (they have no surrounding tool lifecycle).
                // Hold the lock across `spawn`'s await — fine with
                // tokio::sync::Mutex; the user-side terminal handler
                // simply waits for this to finish.
                let spawnResult = {
                    let mut guard = self.shells.lock().await;
                    guard.spawn(name.clone(), SpawnedBy::Agent).await
                };
                match spawnResult {
                    Ok(resolved) => format!(
                        "Spawned terminal '{resolved}'. Use shell with \
                         terminal:'{resolved}' to run commands there."
                    ),
                    Err(e) => format!("Failed to spawn terminal: {e}"),
                }
            }
            tool::ToolAction::TerminalSwitch { name } => {
                // Same reasoning as TerminalSpawn — no separate notice.
                let switchResult = {
                    let mut guard = self.shells.lock().await;
                    guard.setActiveForAgent(name)
                };
                match switchResult {
                    Ok(()) => format!("Agent target terminal is now '{name}'."),
                    Err(e) => format!("Failed to switch terminal: {e}"),
                }
            }
            tool::ToolAction::TerminalKill { name } => {
                let killResult = {
                    let mut guard = self.shells.lock().await;
                    guard.kill(name)
                };
                match killResult {
                    Ok(()) => {
                        self.stopMonitorsForTerminal(name, logTx).await;
                        // TerminalClosed is needed regardless — the deck must
                        // know to drop the tab.
                        let _ = logTx
                            .send(LogEvent::TerminalClosed { name: name.clone() })
                            .await;
                        format!("Terminal '{name}' killed.")
                    }
                    Err(e) => format!("Failed to kill terminal: {e}"),
                }
            }
            tool::ToolAction::TerminalList => {
                let infos = self.shells.lock().await.list();
                if infos.is_empty() {
                    return "No terminals.".into();
                }
                let mut out = String::from("Terminals:\n");
                for info in infos {
                    let active = if info.activeForAgent { " (active)" } else { "" };
                    let busy = if info.busy { " busy" } else { "" };
                    let by = match info.spawnedBy {
                        SpawnedBy::User => "user",
                        SpawnedBy::Agent => "agent",
                    };
                    out.push_str(&format!(
                        "  {} — by {}, age {}s{}{}\n",
                        info.name, by, info.ageSecs, active, busy
                    ));
                }
                out
            }
            tool::ToolAction::TerminalRunList => {
                let runs = match self.listTerminalRuns() {
                    Ok(runs) => runs,
                    Err(e) => return format!("Failed to list terminal runs: {e}"),
                };
                if runs.is_empty() {
                    return "No terminal runs archived yet.".into();
                }
                let mut out = String::from("Terminal runs:\n");
                for run in runs.into_iter().take(50) {
                    let exit = run
                        .exitCode
                        .map(|c| format!(" exit {c}"))
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "  {} [{}{}] {} · terminal={} · impact={} · lines={}\n",
                        run.runId,
                        run.status,
                        exit,
                        run.purpose,
                        run.terminalName,
                        run.impact,
                        run.lineCount,
                    ));
                }
                out
            }
            tool::ToolAction::TerminalRunStop { runId } => {
                let conn = match crate::storage::openSessionDb(self.transcript.sessionDir()) {
                    Ok(conn) => conn,
                    Err(e) => return format!("Failed to open terminal-run archive: {e}"),
                };
                let Some(run) = (match crate::storage::getTerminalRun(&conn, runId) {
                    Ok(run) => run,
                    Err(e) => return format!("Failed to read terminal run {runId}: {e}"),
                }) else {
                    return format!("No terminal run with id {runId}.");
                };
                if run.status != crate::storage::TerminalRunStatus::Running {
                    return format!(
                        "Terminal run {runId} is already {} — no signal sent.",
                        run.status
                    );
                }
                let shell = {
                    let guard = self.shells.lock().await;
                    guard.shellFor(Some(&run.terminalName))
                };
                let Some(shell) = shell else {
                    return format!(
                        "Terminal run {runId} is marked running, but terminal '{}' is no longer live.",
                        run.terminalName
                    );
                };
                shell.interrupt();
                format!(
                    "Sent interrupt to terminal run {runId} in terminal '{}'.",
                    run.terminalName
                )
            }
            _ => unreachable!("non-registry action passed to executeTerminalTool"),
        }
    }

    /// Spawn a visible terminal-backed async shell run. This replaces the
    /// old `JobPlane` bash path for `shell(runInBackground: true)`.
    pub(super) async fn spawnTerminalRun(
        &mut self,
        command: String,
        purpose: String,
        impact: crate::tool::ShellImpact,
        timeout: Option<u64>,
        requestedTerminal: Option<String>,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        let runId = crate::transcript::randomHexId("run");
        let startedAt = unixNow();
        let sessionDir = self.transcript.sessionDir().to_path_buf();

        let (terminalName, shell, ephemeral) = match requestedTerminal {
            Some(name) => {
                let guard = self.shells.lock().await;
                let Some(shell) = guard.shellFor(Some(&name)) else {
                    return format!(
                        "No terminal named '{name}'. Use terminalList to see available terminals."
                    );
                };
                if shell.isBusy() {
                    return format!(
                        "Terminal '{name}' is busy; wait for it to finish or choose another terminal."
                    );
                }
                (name, shell, false)
            }
            None => {
                let spawnResult = {
                    let mut guard = self.shells.lock().await;
                    guard.spawn(None, SpawnedBy::Agent).await
                };
                let name = match spawnResult {
                    Ok(name) => name,
                    Err(e) => return format!("Failed to spawn ephemeral terminal: {e}"),
                };
                let shell = {
                    let guard = self.shells.lock().await;
                    guard.shellFor(Some(&name))
                };
                let Some(shell) = shell else {
                    return format!("Failed to resolve ephemeral terminal '{name}' after spawn.");
                };
                (name, shell, true)
            }
        };

        let impact = crate::storage::TerminalRunImpact::from(&impact);
        let initialRecord = crate::storage::TerminalRunRecord {
            runId: runId.clone(),
            terminalName: terminalName.clone(),
            command: command.clone(),
            purpose: if purpose.trim().is_empty() {
                command.clone()
            } else {
                purpose.clone()
            },
            impact,
            ephemeral,
            startedAt,
            endedAt: None,
            status: crate::storage::TerminalRunStatus::Running,
            exitCode: None,
            lineCount: 0,
            replayBlob: Vec::new(),
        };
        if let Ok(conn) = crate::storage::openSessionDb(&sessionDir)
            && let Err(e) = crate::storage::upsertTerminalRun(&conn, &initialRecord)
        {
            tracing::warn!("failed to record terminal run start: {e}");
        }

        let (wakeId, fireTx) = {
            let mut g = self.wakes.lock().await;
            let wid = g.registerTerminalRun(&runId, logTx);
            (wid, g.fireSender())
        };

        let wakes = self.wakes.clone();
        let shells = self.shells.clone();
        let monitors = self.monitors.clone();
        let logTxClone = logTx.clone();
        let runIdForTask = runId.clone();
        let terminalForTask = terminalName.clone();
        let commandForTask = command.clone();
        let purposeForTask = initialRecord.purpose.clone();
        tokio::spawn(async move {
            let dur = timeout.map(std::time::Duration::from_secs);
            tracing::debug!(
                runId = %runIdForTask,
                terminal = %terminalForTask,
                hasTimeout = timeout.is_some(),
                "spawnTerminalRun: execution started"
            );
            // Respect explicit model-provided timeout; otherwise no timeout.
            // Background runs should not inherit the 30s foreground default.
            let exec = if let Some(d) = dur {
                shell.executeDetailed(&commandForTask, Some(d)).await
            } else {
                shell.executeDetailedNoTimeout(&commandForTask).await
            };
            tracing::debug!(
                runId = %runIdForTask,
                status = %terminalRunStatusForExecution(&exec),
                exitCode = ?exec.exitCode,
                timedOut = exec.timedOut,
                lineCount = exec.lineCount,
                "spawnTerminalRun: execution finished"
            );
            let status = terminalRunStatusForExecution(&exec);
            let completed = crate::storage::TerminalRunRecord {
                runId: runIdForTask.clone(),
                terminalName: terminalForTask.clone(),
                command: commandForTask.clone(),
                purpose: purposeForTask.clone(),
                impact,
                ephemeral,
                startedAt,
                endedAt: Some(unixNow()),
                status,
                exitCode: exec.exitCode,
                lineCount: exec.lineCount,
                replayBlob: exec.replayBytes.clone(),
            };
            if let Ok(conn) = crate::storage::openSessionDb(&sessionDir)
                && let Err(e) = crate::storage::upsertTerminalRun(&conn, &completed)
            {
                tracing::warn!("failed to record terminal run completion: {e}");
            }

            let payload = format!(
                "terminal run {runIdForTask} in {terminalForTask} finished with status {status}{}.\n{}",
                exec.exitCode
                    .map(|c| format!(" (exit code {c})"))
                    .unwrap_or_default(),
                if exec.output.trim().is_empty() {
                    "(no output)".to_string()
                } else {
                    exec.output.lines().take(20).collect::<Vec<_>>().join("\n")
                },
            );
            let _ = fireTx.send(crate::wakes::WakeFire {
                wakeId,
                source: format!("terminalRun#{runIdForTask}"),
                kind: crate::control::WakeKind::TaskComplete,
                payload,
                firedAt: std::time::Instant::now(),
            });
            wakes.lock().await.unregisterPassive(wakeId, &logTxClone);

            if ephemeral {
                let killed = {
                    let mut guard = shells.lock().await;
                    guard.kill(&terminalForTask)
                };
                if killed.is_ok() {
                    let stopped = {
                        let plane = monitors.lock().unwrap();
                        plane.stopForTerminal(&terminalForTask)
                    };
                    for (id, wakeId) in stopped {
                        if let Some(wid) = wakeId {
                            wakes.lock().await.unregisterPassive(wid, &logTxClone);
                        }
                        let _ = logTxClone.send(LogEvent::MonitorStopped { id }).await;
                    }
                    let _ = logTxClone
                        .send(LogEvent::TerminalClosed {
                            name: terminalForTask,
                        })
                        .await;
                }
            }
        });

        format!(
            "Started terminal run {runId} in terminal '{terminalName}': {command}\n\n\
             It is running asynchronously in a visible terminal. You'll be notified when it completes; do not poll. Use terminalRunList to inspect archived output."
        )
    }

    /// Terminal-owned monitor cleanup: attach-only monitors subscribe to a
    /// visible terminal's output stream, so closing that terminal stops and
    /// disarms every monitor attached to it.
    async fn stopMonitorsForTerminal(&self, terminalName: &str, logTx: &mpsc::Sender<LogEvent>) {
        let stopped = {
            let plane = self.monitors.lock().unwrap();
            plane.stopForTerminal(terminalName)
        };
        for (id, wakeId) in stopped {
            if let Some(wid) = wakeId {
                self.wakes.lock().await.unregisterPassive(wid, logTx);
            }
            let _ = logTx.send(LogEvent::MonitorStopped { id }).await;
        }
    }

    /// Detach an already-running visible terminal command into the terminal
    /// run archive. This is the timeout/Ctrl+B path: the command keeps
    /// running in the same terminal and the model turn gets the run id
    /// immediately, with no hidden respawn.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn detachTerminalRunJoin(
        &mut self,
        command: String,
        purpose: String,
        impact: crate::tool::ShellImpact,
        terminalName: String,
        startedAt: u64,
        execTask: tokio::task::JoinHandle<crate::shell::CommandExecution>,
        logTx: &mpsc::Sender<LogEvent>,
        trigger: String,
    ) -> String {
        let runId = crate::transcript::randomHexId("run");
        let sessionDir = self.transcript.sessionDir().to_path_buf();
        let impact = crate::storage::TerminalRunImpact::from(&impact);
        let purpose = if purpose.trim().is_empty() {
            command.clone()
        } else {
            purpose
        };

        let initialRecord = crate::storage::TerminalRunRecord {
            runId: runId.clone(),
            terminalName: terminalName.clone(),
            command: command.clone(),
            purpose: purpose.clone(),
            impact,
            ephemeral: false,
            startedAt,
            endedAt: None,
            status: crate::storage::TerminalRunStatus::Running,
            exitCode: None,
            lineCount: 0,
            replayBlob: Vec::new(),
        };
        if let Ok(conn) = crate::storage::openSessionDb(&sessionDir)
            && let Err(e) = crate::storage::upsertTerminalRun(&conn, &initialRecord)
        {
            tracing::warn!("failed to record detached terminal run start: {e}");
        }

        let (wakeId, fireTx) = {
            let mut g = self.wakes.lock().await;
            let wid = g.registerTerminalRun(&runId, logTx);
            (wid, g.fireSender())
        };

        let wakes = self.wakes.clone();
        let logTxClone = logTx.clone();
        let runIdForTask = runId.clone();
        let terminalForTask = terminalName.clone();
        let commandForTask = command.clone();
        let purposeForTask = purpose.clone();
        tokio::spawn(async move {
            let exec = match execTask.await {
                Ok(exec) => {
                    tracing::debug!(
                        runId = %runIdForTask,
                        status = %terminalRunStatusForExecution(&exec),
                        exitCode = ?exec.exitCode,
                        timedOut = exec.timedOut,
                        "detachTerminalRunJoin: execution finished"
                    );
                    exec
                }
                Err(e) => {
                    tracing::warn!(
                        runId = %runIdForTask,
                        error = %e,
                        "detachTerminalRunJoin: task join failed"
                    );
                    crate::shell::CommandExecution {
                        command: commandForTask.clone(),
                        output: format!("Terminal run task failed to join: {e}"),
                        exitCode: None,
                        lineCount: 1,
                        replayBytes: Vec::new(),
                        timedOut: false,
                    }
                }
            };
            let status = terminalRunStatusForExecution(&exec);
            let completed = crate::storage::TerminalRunRecord {
                runId: runIdForTask.clone(),
                terminalName: terminalForTask.clone(),
                command: commandForTask.clone(),
                purpose: purposeForTask,
                impact,
                ephemeral: false,
                startedAt,
                endedAt: Some(unixNow()),
                status,
                exitCode: exec.exitCode,
                lineCount: exec.lineCount,
                replayBlob: exec.replayBytes.clone(),
            };
            if let Ok(conn) = crate::storage::openSessionDb(&sessionDir)
                && let Err(e) = crate::storage::upsertTerminalRun(&conn, &completed)
            {
                tracing::warn!("failed to record detached terminal run completion: {e}");
            }

            let payload = format!(
                "terminal run {runIdForTask} in {terminalForTask} finished with status {status}{}.\n{}",
                exec.exitCode
                    .map(|c| format!(" (exit code {c})"))
                    .unwrap_or_default(),
                if exec.output.trim().is_empty() {
                    "(no output)".to_string()
                } else {
                    exec.output.lines().take(20).collect::<Vec<_>>().join("\n")
                },
            );
            let _ = fireTx.send(crate::wakes::WakeFire {
                wakeId,
                source: format!("terminalRun#{runIdForTask}"),
                kind: crate::control::WakeKind::TaskComplete,
                payload,
                firedAt: std::time::Instant::now(),
            });
            wakes.lock().await.unregisterPassive(wakeId, &logTxClone);
        });

        format!(
            "DETACHED_TERMINAL_RUN: {trigger}. The command is still running in terminal '{terminalName}' as run {runId}.\n\n\
             You will be notified when it completes; do not poll. Use terminalRunList to inspect archived output."
        )
    }

    pub fn listTerminalRuns(&self) -> Result<Vec<crate::storage::TerminalRunRecord>> {
        let conn = crate::storage::openSessionDb(self.transcript.sessionDir())?;
        crate::storage::listTerminalRuns(&conn)
    }
}
