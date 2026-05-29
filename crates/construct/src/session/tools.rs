use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::mpsc;

use super::Session;
use super::format::{
    extractSnippet, formatJobList, formatJobOutput, formatMonitorList, formatWakeList,
};
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
    ) -> String {
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
            _ => unreachable!("non-job action passed to executeJobTool"),
        }
    }

    /// Handle the monitor-plane tools (Monitor / MonitorStop / MonitorList).
    pub(super) async fn executeMonitorTool(
        &mut self,
        action: &tool::ToolAction,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> String {
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
                        let queryLower = query.to_lowercase();
                        let mut matches: Vec<(String, String, String)> = Vec::new();

                        for turn in &turns {
                            if let Some(mt) = mediaType {
                                let hasMatchingMedia =
                                    turn.attachments.as_ref().is_some_and(|atts| {
                                        atts.iter().any(|a| a.mimeType.starts_with(mt.as_str()))
                                    });
                                if !hasMatchingMedia {
                                    continue;
                                }
                            }

                            if turn.content.to_lowercase().contains(&queryLower) {
                                let snippet = extractSnippet(&turn.content, &queryLower);
                                let roleLabel = match turn.role {
                                    crate::transcript::TurnRole::User => "user",
                                    crate::transcript::TurnRole::Assistant => "assistant",
                                    crate::transcript::TurnRole::ToolCall => "tool_call",
                                    crate::transcript::TurnRole::ToolResult => "tool_result",
                                    crate::transcript::TurnRole::System => "system",
                                    crate::transcript::TurnRole::Wake => "wake",
                                };
                                let imageNote = turn
                                    .attachments
                                    .as_ref()
                                    .filter(|a| !a.is_empty())
                                    .map(|a| format!(" [+{} image(s)]", a.len()))
                                    .unwrap_or_default();
                                matches.push((
                                    turn.blockId.clone(),
                                    format!("{} ({}){imageNote}", turn.id, roleLabel),
                                    snippet,
                                ));
                            }
                        }

                        if matches.is_empty() {
                            return format!("No matches found for \"{query}\".");
                        }

                        let totalMatches = matches.len();
                        let shown = matches.len().min(20);
                        let mut output =
                            format!("Found {totalMatches} matches for \"{query}\":\n\n");
                        for (blockId, turnInfo, snippet) in &matches[..shown] {
                            output.push_str(&format!(
                                "- **{blockId}** {turnInfo}: ...{snippet}...\n"
                            ));
                        }
                        if totalMatches > shown {
                            output.push_str(&format!(
                                "\n({} more matches not shown)\n",
                                totalMatches - shown
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
