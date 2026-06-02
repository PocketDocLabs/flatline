use tokio::sync::{mpsc, oneshot, watch};

use super::{Session, UserInput, toolDefsForPermitMode};
use crate::control::{LogEvent, PermitOrigin, SessionRequest};
use crate::tool;

#[allow(clippy::too_many_arguments)]
async fn runSubagentTaskInBackground(
    mut child: Session,
    childMainIo: crate::shell::ShellIo,
    mut childIoRx: mpsc::Receiver<(String, crate::shell::ShellIo, crate::shells::SpawnedBy)>,
    handle: crate::jobs::SubagentJobHandle,
    prompt: String,
    agentType: String,
    childSessionId: String,
    parentLogTx: mpsc::Sender<LogEvent>,
    parentSessionRequestTx: mpsc::Sender<SessionRequest>,
) {
    let shellForwardTx = parentLogTx.clone();
    let shellForwardId = childSessionId.clone();
    tokio::spawn(async move {
        let mut mainRx = childMainIo.outputRx;
        loop {
            tokio::select! {
                Some(data) = mainRx.recv() => {
                    let _ = shellForwardTx
                        .send(LogEvent::SubagentShellOutput {
                            sessionId: shellForwardId.clone(),
                            data,
                        })
                        .await;
                }
                Some((_n, _io, _by)) = childIoRx.recv() => {
                    // Subagents don't currently spawn extra shells. Drop io.
                }
                else => break,
            }
        }
    });

    let _ = parentLogTx
        .send(LogEvent::SubagentStarted {
            sessionId: childSessionId.clone(),
            agentType: agentType.clone(),
            prompt: prompt.clone(),
        })
        .await;

    tracing::info!(
        agent = %agentType,
        childSession = %childSessionId,
        taskId = handle.id,
        "background subagent spawned"
    );

    let (childLogTx, mut childLogRx) = mpsc::channel::<LogEvent>(256);
    let (childRequestTx, mut childRequestRx) = mpsc::channel::<SessionRequest>(16);

    let (childCancelTx, mut childCancelRx) = watch::channel(false);
    let mut handleCancelRx = handle.cancelRx.clone();
    let cancelBridge = tokio::spawn(async move {
        loop {
            if handleCancelRx.changed().await.is_err() {
                break;
            }
            if *handleCancelRx.borrow() {
                let _ = childCancelTx.send(true);
                break;
            }
        }
    });

    let logSessionId = childSessionId.clone();
    let logParentTx = parentLogTx.clone();
    let logHandleId = handle.id;
    let lineSender = handle.lineSender();
    let logHandle = tokio::spawn(async move {
        let mut content = String::new();
        let mut turns: usize = 0;
        let mut deltaCarry = String::new();
        while let Some(event) = childLogRx.recv().await {
            match &event {
                LogEvent::ContentDelta(text) => {
                    content.push_str(text);
                    deltaCarry.push_str(text);
                    while let Some(pos) = deltaCarry.find('\n') {
                        let line = deltaCarry[..pos].to_string();
                        deltaCarry.drain(..=pos);
                        lineSender.push(line);
                    }
                }
                LogEvent::TurnComplete => turns += 1,
                _ => {}
            }
            match &event {
                LogEvent::ContentDelta(_)
                | LogEvent::ReasoningDelta(_)
                | LogEvent::ToolStarted { .. }
                | LogEvent::ToolAutoReviewStarted { .. }
                | LogEvent::ToolAutoApproved { .. }
                | LogEvent::ToolResult { .. }
                | LogEvent::ToolDenied { .. }
                | LogEvent::ToolAutoDenied { .. }
                | LogEvent::TurnAborted { .. }
                | LogEvent::TurnComplete
                | LogEvent::TurnCancelled
                | LogEvent::Error(_) => {
                    let _ = logParentTx
                        .send(LogEvent::SubagentEvent {
                            sessionId: logSessionId.clone(),
                            event: Box::new(event),
                        })
                        .await;
                }
                _ => {}
            }
        }
        if !deltaCarry.is_empty() {
            lineSender.push(deltaCarry);
        }
        let _ = logHandleId;
        (content, turns)
    });

    let permitSessionId = childSessionId.clone();
    let permitParentTx = parentSessionRequestTx.clone();
    let permitHandle = tokio::spawn(async move {
        while let Some(req) = childRequestRx.recv().await {
            match req {
                SessionRequest::Permit {
                    origin: _,
                    name,
                    summary,
                    args,
                    diff,
                    explanation,
                    impact,
                    review,
                    reply: childReply,
                } => {
                    let (parentReplyTx, parentReplyRx) = oneshot::channel();
                    if permitParentTx
                        .send(SessionRequest::Permit {
                            origin: PermitOrigin::Subagent {
                                sessionId: permitSessionId.clone(),
                            },
                            name,
                            summary,
                            args,
                            diff,
                            explanation,
                            impact,
                            review,
                            reply: parentReplyTx,
                        })
                        .await
                        .is_err()
                    {
                        let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                        continue;
                    }
                    match parentReplyRx.await {
                        Ok(response) => {
                            let _ = childReply.send(response);
                        }
                        Err(_) => {
                            let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                        }
                    }
                }
            }
        }
    });

    let childInput = UserInput::from(prompt.clone());
    let (_childSteerTx, mut childSteerRx) = mpsc::channel::<UserInput>(1);
    let (_childUserBgTx, mut childUserBgRx) = mpsc::channel::<()>(1);

    let sendResult = child
        .send(
            &childInput,
            &childLogTx,
            &childRequestTx,
            &mut childCancelRx,
            &mut childSteerRx,
            &mut childUserBgRx,
        )
        .await;

    drop(childLogTx);
    drop(childRequestTx);

    let (rawContent, turns) = logHandle.await.unwrap_or_default();
    let _ = permitHandle.await;
    cancelBridge.abort();

    enum Outcome {
        Completed,
        Killed,
        Errored(String),
    }
    let outcome = if handle.cancelRequested() {
        Outcome::Killed
    } else if let Err(e) = &sendResult {
        Outcome::Errored(e.to_string())
    } else {
        Outcome::Completed
    };
    let displayContent = match &outcome {
        Outcome::Completed => rawContent.clone(),
        Outcome::Killed => {
            if rawContent.is_empty() {
                "[subagent cancelled by user]".into()
            } else {
                format!("[subagent cancelled by user]\n\n{rawContent}")
            }
        }
        Outcome::Errored(e) => {
            if rawContent.is_empty() {
                format!("[subagent errored: {e}]")
            } else {
                format!("[subagent errored: {e}]\n\n{rawContent}")
            }
        }
    };

    let _ = parentLogTx
        .send(LogEvent::SubagentComplete {
            sessionId: childSessionId.clone(),
            agentType: agentType.clone(),
            content: displayContent.clone(),
            turns,
        })
        .await;

    match outcome {
        Outcome::Killed => handle.killedWithOutput(displayContent).await,
        Outcome::Errored(e) => {
            handle
                .erroredWithOutput(format!("subagent failed: {e}"), displayContent)
                .await
        }
        Outcome::Completed => handle.completeWithOutput(0, displayContent).await,
    }
}

fn buildChildPermissions(
    parent: &crate::permissions::Permissions,
    preset: &crate::runner::AgentPreset,
) -> crate::permissions::Permissions {
    use crate::permissions::{Permissions, Rule};
    use crate::tool::ToolSet;

    let mut rules: Vec<Rule> = Vec::new();

    if matches!(preset.toolSet, ToolSet::ReadOnly) {
        for tool in [
            "shell",
            "writeFile",
            "editFile",
            "multiEdit",
            "copyFile",
            "moveFile",
            "deleteFile",
            "makeDirs",
            "terminalSpawn",
            "terminalSwitch",
            "terminalKill",
            "monitor",
            "monitorStop",
            "scheduleWakeup",
            "cronCreate",
            "cronDelete",
            "fileWatch",
        ] {
            rules.push(Rule {
                tool: tool.into(),
                pattern: None,
                allow: false,
            });
        }
    }

    rules.extend(preset.permissions.rules.iter().cloned());
    rules.extend(parent.rules.iter().cloned());

    Permissions {
        defaultMode: preset.permissions.defaultMode.clone(),
        rules,
        source: parent.source,
    }
}

impl Session {
    /// Execute a subagent task.
    ///
    /// Spawns a child session with its own context, shell, and tool set,
    /// runs the task to completion, and returns the child's final text.
    /// Child log events are wrapped as `SubagentEvent` and forwarded on the
    /// parent log channel; child permit requests are rewrapped with
    /// `PermitOrigin::Subagent` and forwarded on the parent request channel.
    pub(super) async fn executeTask(
        &mut self,
        prompt: &str,
        agentType: &str,
        parentLogTx: &mpsc::Sender<LogEvent>,
        parentSessionRequestTx: &mpsc::Sender<SessionRequest>,
        parentCancelRx: &mut watch::Receiver<bool>,
    ) -> String {
        use crate::runner;

        let preset = runner::agentPreset(agentType);

        let mut childConfig = self.config.clone();
        childConfig.heavy = match preset.tier {
            runner::AgentTier::Heavy => childConfig.heavy.clone(),
            runner::AgentTier::Light => childConfig.light.clone(),
            runner::AgentTier::Utility => childConfig.utility.clone(),
        };

        let (childIoTx, mut childIoRx) =
            mpsc::channel::<(String, crate::shell::ShellIo, crate::shells::SpawnedBy)>(8);
        let (childRegistry, childMainIo) =
            match crate::shells::ShellRegistry::newWithMain(120, 40, childIoTx) {
                Ok(r) => r,
                Err(e) => return format!("Failed to spawn subagent shell: {e}"),
            };

        let childPermissions = buildChildPermissions(&self.permissions, &preset);

        let childRegistryArc = std::sync::Arc::new(tokio::sync::Mutex::new(childRegistry));
        let mut child = match Session::new(
            &childConfig,
            childPermissions,
            childRegistryArc,
            preset.interface,
            &[crate::prompt::DomainModule::Swe],
        ) {
            Ok(s) => s,
            Err(e) => return format!("Failed to create subagent session: {e}"),
        };

        let childDefs = toolDefsForPermitMode(&child.permissions.defaultMode);
        let filtered = tool::filterDefs(&childDefs, &preset.toolSet);
        child.setTools(filtered);

        let childSessionId = child.sessionId().to_string();

        let shellForwardTx = parentLogTx.clone();
        let shellForwardId = childSessionId.clone();
        tokio::spawn(async move {
            let mut mainRx = childMainIo.outputRx;
            loop {
                tokio::select! {
                    Some(data) = mainRx.recv() => {
                        let _ = shellForwardTx
                            .send(LogEvent::SubagentShellOutput {
                                sessionId: shellForwardId.clone(),
                                data,
                            })
                            .await;
                    }
                    Some((_name, _io, _by)) = childIoRx.recv() => {
                        // Phase 1: subagents don't spawn additional shells.
                    }
                    else => break,
                }
            }
        });

        let _ = parentLogTx
            .send(LogEvent::SubagentStarted {
                sessionId: childSessionId.clone(),
                agentType: agentType.into(),
                prompt: prompt.into(),
            })
            .await;

        tracing::info!(
            agent = %agentType,
            childSession = %childSessionId,
            "subagent spawned"
        );

        let (childLogTx, mut childLogRx) = mpsc::channel::<LogEvent>(256);
        let (childRequestTx, mut childRequestRx) = mpsc::channel::<SessionRequest>(16);
        let mut childCancelRx = parentCancelRx.clone();

        let logSessionId = childSessionId.clone();
        let logParentTx = parentLogTx.clone();
        let logHandle = tokio::spawn(async move {
            let mut content = String::new();
            let mut turns: usize = 0;
            while let Some(event) = childLogRx.recv().await {
                match &event {
                    LogEvent::ContentDelta(text) => content.push_str(text),
                    LogEvent::TurnComplete => turns += 1,
                    _ => {}
                }
                match &event {
                    LogEvent::ContentDelta(_)
                    | LogEvent::ReasoningDelta(_)
                    | LogEvent::ToolStarted { .. }
                    | LogEvent::ToolAutoReviewStarted { .. }
                    | LogEvent::ToolAutoApproved { .. }
                    | LogEvent::ToolResult { .. }
                    | LogEvent::ToolDenied { .. }
                    | LogEvent::ToolAutoDenied { .. }
                    | LogEvent::TurnAborted { .. }
                    | LogEvent::TurnComplete
                    | LogEvent::TurnCancelled
                    | LogEvent::Error(_) => {
                        let _ = logParentTx
                            .send(LogEvent::SubagentEvent {
                                sessionId: logSessionId.clone(),
                                event: Box::new(event),
                            })
                            .await;
                    }
                    _ => {}
                }
            }
            (content, turns)
        });

        let permitSessionId = childSessionId.clone();
        let permitParentTx = parentSessionRequestTx.clone();
        let permitHandle = tokio::spawn(async move {
            while let Some(req) = childRequestRx.recv().await {
                match req {
                    SessionRequest::Permit {
                        origin: _,
                        name,
                        summary,
                        args,
                        diff,
                        explanation,
                        impact,
                        review,
                        reply: childReply,
                    } => {
                        let (parentReplyTx, parentReplyRx) = oneshot::channel();
                        if permitParentTx
                            .send(SessionRequest::Permit {
                                origin: PermitOrigin::Subagent {
                                    sessionId: permitSessionId.clone(),
                                },
                                name,
                                summary,
                                args,
                                diff,
                                explanation,
                                impact,
                                review,
                                reply: parentReplyTx,
                            })
                            .await
                            .is_err()
                        {
                            let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                            continue;
                        }
                        match parentReplyRx.await {
                            Ok(response) => {
                                let _ = childReply.send(response);
                            }
                            Err(_) => {
                                let _ = childReply.send(crate::permissions::PermitResponse::Deny);
                            }
                        }
                    }
                }
            }
        });

        let childInput = UserInput::from(prompt.to_string());
        let (_childSteerTx, mut childSteerRx) = mpsc::channel::<UserInput>(1);
        let (_childUserBgTx, mut childUserBgRx) = mpsc::channel::<()>(1);
        let sendResult = child
            .send(
                &childInput,
                &childLogTx,
                &childRequestTx,
                &mut childCancelRx,
                &mut childSteerRx,
                &mut childUserBgRx,
            )
            .await;

        drop(childLogTx);
        drop(childRequestTx);

        let (content, turns) = match logHandle.await {
            Ok(r) => r,
            Err(e) => {
                let _ = parentLogTx
                    .send(LogEvent::SubagentComplete {
                        sessionId: childSessionId.clone(),
                        agentType: agentType.into(),
                        content: String::new(),
                        turns: 0,
                    })
                    .await;
                return format!("Subagent forwarding failed: {e}");
            }
        };
        let _ = permitHandle.await;

        let _ = parentLogTx
            .send(LogEvent::SubagentComplete {
                sessionId: childSessionId.clone(),
                agentType: agentType.into(),
                content: content.clone(),
                turns,
            })
            .await;

        if let Err(e) = sendResult {
            return format!("Subagent failed: {e}");
        }

        tracing::info!(
            agent = %agentType,
            childSession = %childSessionId,
            turns = turns,
            "subagent completed"
        );

        if content.is_empty() {
            format!("[subagent session: {childSessionId}]\n\nTask completed (no text output).")
        } else {
            format!("[subagent session: {childSessionId}]\n\n{content}")
        }
    }

    /// Background variant of [`Self::executeTask`]. Registers a subagent
    /// task in [`crate::jobs::JobPlane`] and spawns the child-session runner
    /// so the parent's tool-call returns immediately with the new task id.
    pub(super) async fn executeTaskBackground(
        &mut self,
        prompt: &str,
        agentType: &str,
        parentLogTx: &mpsc::Sender<LogEvent>,
        parentSessionRequestTx: &mpsc::Sender<SessionRequest>,
        _parentCancelRx: &mut watch::Receiver<bool>,
    ) -> String {
        use crate::runner;

        let preset = runner::agentPreset(agentType);
        let mut childConfig = self.config.clone();
        childConfig.heavy = match preset.tier {
            runner::AgentTier::Heavy => childConfig.heavy.clone(),
            runner::AgentTier::Light => childConfig.light.clone(),
            runner::AgentTier::Utility => childConfig.utility.clone(),
        };

        let (childIoTx, childIoRx) =
            mpsc::channel::<(String, crate::shell::ShellIo, crate::shells::SpawnedBy)>(8);
        let (childRegistry, childMainIo) =
            match crate::shells::ShellRegistry::newWithMain(120, 40, childIoTx) {
                Ok(r) => r,
                Err(e) => return format!("Failed to spawn subagent shell: {e}"),
            };

        let childPermissions = buildChildPermissions(&self.permissions, &preset);

        let childRegistryArc = std::sync::Arc::new(tokio::sync::Mutex::new(childRegistry));
        let mut child = match Session::new(
            &childConfig,
            childPermissions,
            childRegistryArc,
            preset.interface,
            &[crate::prompt::DomainModule::Swe],
        ) {
            Ok(s) => s,
            Err(e) => return format!("Failed to create subagent session: {e}"),
        };

        let childDefs = toolDefsForPermitMode(&child.permissions.defaultMode);
        let filtered = tool::filterDefs(&childDefs, &preset.toolSet);
        child.setTools(filtered);

        let taskId = self.jobs.lock().unwrap().reserveJobId();
        let (wakeId, fireTx) = {
            let mut g = self.wakes.lock().await;
            let wid = g.registerTaskComplete(taskId, parentLogTx);
            (wid, g.fireSender())
        };
        let wakeCtx = crate::jobs::TaskWakeCtx {
            wakeId,
            registry: self.wakes.clone(),
            fireTx,
        };
        let handle = self.jobs.lock().unwrap().spawnSubagentWithId(
            taskId,
            agentType.to_string(),
            prompt.to_string(),
            parentLogTx.clone(),
            Some(wakeCtx),
        );

        let childSessionId = child.sessionId().to_string();
        let promptOwned = prompt.to_string();
        let agentTypeOwned = agentType.to_string();
        let parentLogTxOwned = parentLogTx.clone();
        let parentRequestTxOwned = parentSessionRequestTx.clone();

        tokio::spawn(async move {
            runSubagentTaskInBackground(
                child,
                childMainIo,
                childIoRx,
                handle,
                promptOwned,
                agentTypeOwned,
                childSessionId.clone(),
                parentLogTxOwned,
                parentRequestTxOwned,
            )
            .await;
        });

        format!(
            "Spawned subagent #{taskId} ({agentType}). The agent is running in the \
             background \u{2014} call jobOutput(jobId: {taskId}) for streaming \
             progress or to read its final answer, jobList to see status, and \
             jobStop(jobId: {taskId}) to cancel."
        )
    }
}
