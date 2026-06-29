use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use super::Session;
use super::format::{formatJobList, formatJobOutput, formatMonitorList, formatWakeList};
use crate::control::LogEvent;
use crate::{tool, web};

impl Session {
    /// Handle the task-plane tools: backgrounded `shell` calls
    /// (`runInBackground: true`) and the lifecycle tools TaskOutput /
    /// TaskStop / TaskList.
    pub(super) async fn executeJobTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
        cancelRx: &mut watch::Receiver<bool>,
    ) -> String {
        tracing::debug!(action = ?action, "executeJobTool");
        match action {
            tool::ToolAction::Shell {
                command,
                explanation,
                impact,
                timeout,
                terminal,
                runInBackground: true,
            } => {
                self.spawnTerminalRun(
                    command.clone(),
                    explanation.clone(),
                    impact.clone(),
                    *timeout,
                    terminal.clone(),
                    logTx,
                )
                .await
            }
            tool::ToolAction::JobOutput {
                jobId,
                sinceLine,
                maxLines,
            } => {
                let cap = maxLines.unwrap_or(200);
                match self.jobs.lock().unwrap().output(*jobId, *sinceLine, cap) {
                    Ok(snap) => formatJobOutput(*jobId, &snap, *sinceLine),
                    Err(e) => format!("{e}. Use jobList to see available job ids."),
                }
            }
            tool::ToolAction::JobStop { jobId } => {
                let preState = self
                    .jobs
                    .lock()
                    .unwrap()
                    .list()
                    .into_iter()
                    .find(|t| t.id == *jobId)
                    .map(|t| t.state);
                match self.jobs.lock().unwrap().stop(*jobId) {
                    Ok(()) => match preState {
                        Some(s) if s.isTerminal() => {
                            format!("Job #{jobId} was already {:?} \u{2014} no signal sent.", s,)
                        }
                        _ => format!("Sent kill signal to job #{jobId}."),
                    },
                    Err(e) => format!("Failed to stop job: {e}"),
                }
            }
            tool::ToolAction::JobList => formatJobList(&self.jobs.lock().unwrap().list()),
            tool::ToolAction::WaitForSubagent { jobId } => {
                // Only subagent jobs can be waited on — bash background
                // tasks stream output while running, there's no "completion"
                // to wait for.
                {
                    let kind = self
                        .jobs
                        .lock()
                        .unwrap()
                        .list()
                        .into_iter()
                        .find(|t| t.id == *jobId)
                        .map(|t| t.kind);
                    match kind {
                        Some(crate::jobs::JobKind::Bash) => {
                            return format!(
                                "Job #{jobId} is a background shell task, not a subagent. \
                                 Use jobOutput(jobId: {jobId}) to stream its output."
                            );
                        }
                        Some(crate::jobs::JobKind::Subagent { .. }) => {}
                        None => {
                            return format!(
                                "No task #{jobId}. Use jobList to see available job ids."
                            );
                        }
                    }
                }
                // Poll until the subagent reaches a terminal state, checking
                // for cancellation between polls.
                let start = std::time::Instant::now();
                let pollInterval = std::time::Duration::from_millis(500);
                let maxWait = std::time::Duration::from_secs(600);
                loop {
                    let state = self
                        .jobs
                        .lock()
                        .unwrap()
                        .list()
                        .into_iter()
                        .find(|t| t.id == *jobId)
                        .map(|t| t.state);
                    let Some(s) = state else {
                        return format!("No task #{jobId}. Use jobList to see available job ids.");
                    };
                    if s.isTerminal() {
                        let cap = crate::jobs::MAX_RESPONSE_LINES;
                        // Prevent the TaskComplete wake from delivering the
                        // same output again as a synthetic turn.
                        self.consumedTaskWakes.insert(*jobId);
                        return match self.jobs.lock().unwrap().output(*jobId, None, cap) {
                            Ok(snap) => formatJobOutput(*jobId, &snap, None),
                            Err(e) => format!("{e}. Use jobList to see available job ids."),
                        };
                    }
                    if start.elapsed() > maxWait {
                        return format!(
                            "Subagent #{jobId} is still running after 10 minutes. \
                             Continue work in parallel and it will notify you on completion, \
                             or call jobStop({jobId}) to cancel."
                        );
                    }
                    // Check for cancellation during the sleep.
                    tokio::select! {
                        _ = tokio::time::sleep(pollInterval) => {}
                        _ = cancelRx.changed() => {
                            if *cancelRx.borrow() {
                                return format!("Wait for subagent #{jobId} cancelled.");
                            }
                        }
                    }
                }
            }
            _ => unreachable!("non-job action passed to executeJobTool"),
        }
    }

    /// Handle the monitor-plane tools (Monitor / MonitorStop / MonitorList).
    pub(super) async fn executeMonitorTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        tracing::debug!(action = ?action, "executeMonitorTool");
        match action {
            tool::ToolAction::Monitor {
                description,
                terminal,
                filter,
            } => {
                let (terminalName, shell) = {
                    let guard = self.shells.lock().await;
                    let name = terminal
                        .clone()
                        .unwrap_or_else(|| guard.activeForAgent().to_string());
                    let Some(shell) = guard.shellFor(Some(&name)) else {
                        return format!(
                            "No terminal named '{name}'. Use terminalList to see available terminals."
                        );
                    };
                    (name, shell)
                };

                let id = self.monitors.lock().unwrap().reserveMonitorId();
                let (wakeId, fireTx) = {
                    let mut g = self.wakes.lock().await;
                    let wid = g.registerMonitor(id, logTx);
                    (wid, g.fireSender())
                };
                let result = {
                    let mut plane = self.monitors.lock().unwrap();
                    plane.registerWithId(
                        id,
                        description.clone(),
                        terminalName.clone(),
                        filter.clone(),
                        crate::monitors::DEFAULT_AUTOSTOP_EPS,
                        shell,
                        logTx.clone(),
                        Some(crate::monitors::MonitorWakeCtx {
                            wakeId,
                            registry: self.wakes.clone(),
                            fireTx,
                        }),
                    )
                };
                match result {
                    Ok(_) => format!(
                        "Registered monitor #{id} \"{description}\" with filter /{filter}/.\n\n\
                         Watching terminal: {terminalName}\n\n\
                         You'll be notified when matches arrive (do not poll). \
                         Use monitorList to check event counts, monitorStop({id}) to stop."
                    ),
                    Err(e) => {
                        self.wakes.lock().await.unregisterPassive(wakeId, logTx);
                        format!("Failed to register monitor: {e}")
                    }
                }
            }
            tool::ToolAction::MonitorStop { monitorId } => {
                let wakeId = self.monitors.lock().unwrap().takeWakeId(*monitorId);
                let stopResult = {
                    let plane = self.monitors.lock().unwrap();
                    plane.stop(*monitorId)
                };
                match stopResult {
                    Ok(()) => {
                        if let Some(wid) = wakeId {
                            self.wakes.lock().await.unregisterPassive(wid, logTx);
                        }
                        let _ = logTx
                            .send(LogEvent::MonitorStopped { id: *monitorId })
                            .await;
                        format!("Stopped monitor #{monitorId}.")
                    }
                    Err(e) => format!("Failed to stop monitor: {e}"),
                }
            }
            tool::ToolAction::MonitorList => {
                let snapshot = self.monitors.lock().unwrap().list();
                formatMonitorList(&snapshot)
            }
            _ => unreachable!("non-monitor action passed to executeMonitorTool"),
        }
    }

    /// Handle the wake-registry tools: scheduleWakeup, cronCreate,
    /// cronList, cronDelete, fileWatch.
    pub(super) async fn executeWakeTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
        tracing::debug!(action = ?action, "executeWakeTool");
        match action {
            tool::ToolAction::ScheduleWakeup {
                delaySeconds,
                prompt,
            } => {
                if *delaySeconds == 0 {
                    return "delaySeconds must be at least 1".into();
                }
                let regArc = self.wakes.clone();
                let promptOwned = prompt.clone();
                let logTxClone = logTx.clone();
                let secs = *delaySeconds;
                let id = tokio::task::spawn_blocking(move || {
                    crate::wakes::WakeRegistry::armDelay(
                        &regArc,
                        Duration::from_secs(secs),
                        promptOwned,
                        logTxClone,
                    )
                })
                .await
                .unwrap_or(0);
                format!(
                    "Armed wake #{id} \u{2014} will fire in {secs}s with prompt: {prompt}.\n\
                     You'll receive a <wake source=\"delay#{id}\" kind=\"Delay\"> message at that time. \
                     Cancel with cronDelete({id})."
                )
            }
            tool::ToolAction::CronCreate {
                spec,
                prompt,
                recurring,
            } => {
                let regArc = self.wakes.clone();
                let specOwned = spec.clone();
                let promptOwned = prompt.clone();
                let logTxClone = logTx.clone();
                let recurringFlag = *recurring;
                let result = tokio::task::spawn_blocking(move || {
                    crate::wakes::WakeRegistry::armCron(
                        &regArc,
                        specOwned,
                        recurringFlag,
                        promptOwned,
                        logTxClone,
                    )
                })
                .await;
                match result {
                    Ok(Ok(id)) => format!(
                        "Armed wake #{id} \u{2014} cron `{spec}`{}.\n\
                         You'll receive a <wake source=\"cron#{id}\" kind=\"Cron\"> message on each fire. \
                         Cancel with cronDelete({id}).",
                        if *recurring {
                            " (recurring)"
                        } else {
                            " (one-shot)"
                        },
                    ),
                    Ok(Err(e)) => format!("Failed to arm cron: {e}"),
                    Err(e) => format!("Failed to arm cron: join error: {e}"),
                }
            }
            tool::ToolAction::CronList => {
                let sources = self.wakes.lock().await.list();
                formatWakeList(&sources)
            }
            tool::ToolAction::CronDelete { wakeId } => {
                let removed = self.wakes.lock().await.disarm(*wakeId, logTx);
                if removed {
                    format!("Disarmed wake #{wakeId}.")
                } else {
                    format!("No wake source #{wakeId} (use cronList to see active ids).")
                }
            }
            tool::ToolAction::FileWatch { path, prompt } => {
                let regArc = self.wakes.clone();
                let pathBuf = PathBuf::from(path);
                let promptOwned = prompt.clone();
                let logTxClone = logTx.clone();
                let result = tokio::task::spawn_blocking(move || {
                    crate::wakes::WakeRegistry::armFileWatch(
                        &regArc,
                        pathBuf,
                        promptOwned,
                        logTxClone,
                    )
                })
                .await;
                match result {
                    Ok(Ok(id)) => format!(
                        "Armed wake #{id} \u{2014} watching {path}.\n\
                         Each fs event under that path fires a <wake source=\"fileWatch#{id}\"> message. \
                         Cancel with cronDelete({id})."
                    ),
                    Ok(Err(e)) => format!("Failed to arm fileWatch: {e}"),
                    Err(e) => format!("Failed to arm fileWatch: join error: {e}"),
                }
            }
            _ => unreachable!("non-wake action passed to executeWakeTool"),
        }
    }

    pub(super) fn executeTranscriptTool(&self, action: &tool::ToolAction) -> String {
        tracing::debug!(action = ?action, "executeTranscriptTool");
        match action {
            tool::ToolAction::HistoryFetch { blockId } => match self.transcript.loadAll() {
                Ok(turns) => {
                    let blockTurns: Vec<_> =
                        turns.iter().filter(|t| t.blockId == *blockId).collect();

                    if blockTurns.is_empty() {
                        return format!("No block found with ID \"{blockId}\".");
                    }

                    let mut output = format!("## Block {blockId}\n\n");
                    for turn in &blockTurns {
                        let roleLabel = match turn.role {
                            crate::transcript::TurnRole::User => "User",
                            crate::transcript::TurnRole::Assistant => "Assistant",
                            crate::transcript::TurnRole::ToolCall => "Tool Call",
                            crate::transcript::TurnRole::ToolResult => "Tool Result",
                            crate::transcript::TurnRole::System => "System",
                            crate::transcript::TurnRole::Wake => "Wake",
                        };

                        output.push_str(&format!("### [{roleLabel}] {}\n", turn.id));

                        if let Some(ref toolName) = turn.tool {
                            output.push_str(&format!("Tool: {toolName}\n"));
                        }
                        if let Some(ref args) = turn.args {
                            output.push_str(&format!("Args: {args}\n"));
                        }

                        if !turn.content.is_empty() {
                            output.push_str(&turn.content);
                            output.push('\n');
                        }

                        if let Some(ref atts) = turn.attachments
                            && !atts.is_empty()
                        {
                            output.push_str(&format!("[+{} image(s) attached]\n", atts.len()));
                        }
                        output.push('\n');
                    }
                    output
                }
                Err(e) => format!("Failed to load transcript: {e}"),
            },
            tool::ToolAction::HistorySearch { query, mediaType } => {
                match self.transcript.loadAll() {
                    Ok(turns) => {
                        let rawHits = crate::transcript_search::search(&turns, query, 40);

                        // Post-filter by mediaType if requested, then cap at 20.
                        let hits: Vec<_> = rawHits
                            .into_iter()
                            .filter(|hit| {
                                if let Some(mt) = mediaType {
                                    let turn = &turns[hit.turnIndex];
                                    turn.attachments.as_ref().is_some_and(|atts| {
                                        atts.iter().any(|a| a.mimeType.starts_with(mt.as_str()))
                                    })
                                } else {
                                    true
                                }
                            })
                            .take(20)
                            .collect();

                        if hits.is_empty() {
                            return format!("No matches found for \"{query}\".");
                        }

                        let mut output =
                            format!("Found {} results for \"{query}\":\n\n", hits.len());
                        for hit in &hits {
                            let turn = &turns[hit.turnIndex];
                            let imageNote = turn
                                .attachments
                                .as_ref()
                                .filter(|a| !a.is_empty())
                                .map(|a| format!(" [+{} image(s)]", a.len()))
                                .unwrap_or_default();
                            output.push_str(&format!(
                                "- **{}** {} ({}){imageNote}: ...{}...\n",
                                hit.blockId, hit.turnId, hit.role, hit.snippet,
                            ));
                        }
                        output
                    }
                    Err(e) => format!("Failed to load transcript: {e}"),
                }
            }
            _ => "Not a transcript tool.".into(),
        }
    }

    /// Execute a web tool (webSearch, webFetch, webSimilar).
    pub(super) async fn executeWebTool(&mut self, action: &tool::ToolAction) -> String {
        tracing::debug!(action = ?action, "executeWebTool");
        let exa = match &self.exaClient {
            Some(c) => c,
            None => return web::notConfiguredError(),
        };

        match action {
            tool::ToolAction::WebSearch {
                query,
                allowedDomains,
                blockedDomains,
                maxResults,
            } => {
                web::executeSearch(
                    exa,
                    query,
                    allowedDomains.as_deref(),
                    blockedDomains.as_deref(),
                    *maxResults,
                )
                .await
            }
            tool::ToolAction::WebFetch {
                url,
                prompt,
                subpages,
            } => {
                web::executeFetch(
                    exa,
                    &mut self.urlCache,
                    &self.client,
                    &self.config,
                    url,
                    prompt.as_deref(),
                    *subpages,
                )
                .await
            }
            tool::ToolAction::WebSimilar {
                url,
                allowedDomains,
                blockedDomains,
                maxResults,
            } => {
                web::executeSimilar(
                    exa,
                    url,
                    allowedDomains.as_deref(),
                    blockedDomains.as_deref(),
                    *maxResults,
                )
                .await
            }
            _ => "Not a web tool.".into(),
        }
    }

    /// Execute an MCP tool action.
    pub(super) async fn executeMcpTool(&self, action: &tool::ToolAction) -> String {
        tracing::debug!(action = ?action, "executeMcpTool");
        let mgr = match &self.mcpManager {
            Some(m) => m,
            None => return "MCP not configured.".into(),
        };

        match action {
            tool::ToolAction::Mcp {
                qualifiedName,
                args,
            } => {
                if qualifiedName == "mcpToolSearch" {
                    mgr.executeSearch(args).await
                } else {
                    mgr.routeToolCall(qualifiedName, args).await
                }
            }
            _ => "Not an MCP tool.".into(),
        }
    }

    /// Execute an LSP diagnostics tool call.
    pub(super) async fn executeLspTool(&mut self, action: &tool::ToolAction) -> String {
        tracing::debug!(action = ?action, "executeLspTool");
        let tool::ToolAction::Diagnostics { path, severity } = action else {
            return "Not an LSP tool.".into();
        };

        let minSeverity = match severity.as_str() {
            "warning" => async_lsp::lsp_types::DiagnosticSeverity::WARNING,
            _ => async_lsp::lsp_types::DiagnosticSeverity::ERROR,
        };

        self.lspManager
            .getDiagnosticsForTool(path, minSeverity, Duration::from_secs(15))
            .await
    }
}
