use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;

use super::Session;
use crate::control::LogEvent;
use crate::{compaction_trigger, context};

/// Outcome of running a single compaction stage.
///
/// The distinction between `Exhausted` and `Failed` is load-bearing: a
/// stage that genuinely has nothing left to do is marked exhausted so
/// the trigger escalates to the next stage, but a stage that hit a
/// transient error (e.g. utility model API failure) must NOT be marked
/// exhausted — otherwise one failed call permanently wedges compaction
/// until session restart. `Failed` leaves exhaustion untouched and bails
/// the current pass so the stage is retried on the next turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageOutcome {
    /// Context was reduced.
    DidWork,
    /// Nothing eligible to compact — mark exhausted and escalate.
    Exhausted,
    /// Transient failure — do not poison exhaustion; retry next turn.
    Failed,
}

impl Session {
    /// Check compaction trigger and run the appropriate stage.
    ///
    /// Loops on exhaustion: if a stage exhausts without reducing context,
    /// re-evaluates and tries the next cheapest stage. Stops when a stage
    /// does work, nothing is returned, or all stages are exhausted.
    pub(super) async fn checkCompactionTrigger(&mut self, logTx: &mpsc::Sender<LogEvent>) {
        loop {
            let tokens = self.compactionTracker.lastTokens();
            let stage = match self.compactionTracker.evaluate(tokens) {
                Some(s) => s,
                None => return,
            };

            let ratio = self.compactionTracker.usageRatio();
            tracing::info!(
                stage = ?stage,
                tokens,
                ratio = format!("{:.1}%", ratio * 100.0),
                "compaction trigger fired"
            );

            let stageStr = format!("{stage}");
            let _ = logTx
                .send(LogEvent::CompactionStarted {
                    stage: stageStr.clone(),
                })
                .await;

            let outcome = match stage {
                compaction_trigger::StagePick::S1 => self.runS1(&stageStr, logTx).await,
                compaction_trigger::StagePick::S2 => self.runS2(&stageStr, logTx).await,
                compaction_trigger::StagePick::S3 => self.runS3(&stageStr, logTx).await,
                compaction_trigger::StagePick::S4 => self.runS4Trigger(&stageStr, logTx).await,
            };

            match outcome {
                // Context reduced — stop, the next API response re-evaluates.
                StageOutcome::DidWork => return,
                // Transient failure — bail this pass WITHOUT marking the
                // stage exhausted, so it's retried on the next turn instead
                // of permanently wedging compaction.
                StageOutcome::Failed => return,
                // Genuinely nothing to do — exhaustion was recorded by the
                // stage; keep escalating unless everything is exhausted.
                StageOutcome::Exhausted => {
                    if self.compactionTracker.allExhausted() {
                        return;
                    }
                }
            }
        }
    }

    /// Run S1 mechanical pruning. S1 makes no API calls, so it can only
    /// reduce context (`DidWork`) or find nothing left (`Exhausted`).
    async fn runS1(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> StageOutcome {
        let blockHints = match self.transcript.loadAll() {
            Ok(turns) => {
                let mut map = HashMap::new();
                for t in &turns {
                    if let Some(tcid) = &t.toolCallId {
                        map.insert(tcid.clone(), t.blockId.clone());
                    }
                }
                map
            }
            Err(e) => {
                tracing::warn!("failed to load transcript for block hints: {e}");
                HashMap::new()
            }
        };

        let alreadyProcessed = match self.compactionLog.loadAll() {
            Ok(ops) => {
                let mut set = HashSet::new();
                for op in &ops {
                    match op {
                        crate::compaction::CompactionOp::FileDedup { targetIds, .. } => {
                            set.extend(targetIds.iter().cloned());
                        }
                        crate::compaction::CompactionOp::MiddleOut { targetIds, .. } => {
                            set.extend(targetIds.iter().cloned());
                        }
                        _ => {}
                    }
                }
                set
            }
            Err(_) => HashSet::new(),
        };

        let s1Result = crate::s1::run(
            &mut self.history,
            crate::s1::DEFAULT_MIDDLE_OUT_THRESHOLD,
            &blockHints,
            &alreadyProcessed,
        );
        if s1Result.didWork {
            let afterTurn = self.headTurnId.clone().unwrap_or_default();
            if !s1Result.dedupedCallIds.is_empty()
                && let Err(e) = self
                    .compactionLog
                    .recordFileDedup(s1Result.dedupedCallIds.clone(), &afterTurn)
            {
                tracing::warn!("compaction log write failed: {e}");
            }
            if !s1Result.middleOutCallIds.is_empty()
                && let Err(e) = self.compactionLog.recordMiddleOut(
                    s1Result.middleOutCallIds.clone(),
                    &afterTurn,
                    s1Result.middleOutThreshold,
                )
            {
                tracing::warn!("compaction log write failed: {e}");
            }
            for path in &s1Result.invalidatedFiles {
                self.filesRead.remove(path);
            }
            self.compactionTracker.clearExhaustion();
            let reduction = format!(
                "deduped {} reads, truncated {} outputs",
                s1Result.dedupedCallIds.len(),
                s1Result.middleOutCallIds.len()
            );
            let _ = logTx
                .send(LogEvent::CompactionComplete {
                    stage: stageStr.to_string(),
                    reduction,
                    markerBlock: None,
                })
                .await;
            StageOutcome::DidWork
        } else {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S1);
            tracing::debug!("S1 exhausted \u{2014} nothing to prune");
            StageOutcome::Exhausted
        }
    }

    /// Run S2 block compaction.
    async fn runS2(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> StageOutcome {
        let headTurn = self.headTurnId.clone().unwrap_or_default();
        // S2 protects newest 40% of budget. Zone covers the rest.
        let budget = self.compactionTracker.compactLimit().max(1) as f64;
        let current = self.compactionTracker.lastTokens().max(1) as f64;
        let zoneFraction = (1.0 - (0.40 * budget / current)).clamp(0.0, 1.0);
        let s2Result = match crate::s2::run(
            &self.transcript,
            &self.compactionLog,
            &headTurn,
            &self.client,
            &self.config.utility.model,
            self.config.heavy.contextWindow,
            zoneFraction,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S2 compaction failed: {e}");
                let _ = logTx
                    .send(LogEvent::CompactionFailed {
                        stage: stageStr.to_string(),
                        reason: e.to_string(),
                    })
                    .await;
                return StageOutcome::Failed;
            }
        };
        if let Some(cost) = s2Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s2Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S2);
            tracing::debug!("S2 exhausted \u{2014} no blocks to compact");
            return StageOutcome::Exhausted;
        }
        let afterTurn = self.headTurnId.clone().unwrap_or_default();
        let blockCount = s2Result.compacted.len();
        for block in &s2Result.compacted {
            if let Err(e) = self.compactionLog.recordBlockCompact(
                &block.blockId,
                &block.summary,
                block.sourceIds.clone(),
                &afterTurn,
            ) {
                tracing::warn!("compaction log write failed for {}: {e}", block.blockId);
            }
            for path in &block.invalidatedFiles {
                self.filesRead.remove(path);
            }
        }
        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId, 0, 0) {
            Ok(r) => self.history = r.messages,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S2: {e}");
                let _ = logTx
                    .send(LogEvent::CompactionFailed {
                        stage: stageStr.to_string(),
                        reason: format!("history reconstruction failed: {e}"),
                    })
                    .await;
                return StageOutcome::Failed;
            }
        }
        self.compactionTracker.clearExhaustion();
        let reduction = format!("compressed {blockCount} blocks");
        let _ = logTx
            .send(LogEvent::CompactionComplete {
                stage: stageStr.to_string(),
                reduction,
                markerBlock: Some(0),
            })
            .await;
        StageOutcome::DidWork
    }

    /// Run S3 topic compaction.
    async fn runS3(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> StageOutcome {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        // S3 protects newest 70% of budget. Zone covers the rest.
        let budget = self.compactionTracker.compactLimit().max(1) as f64;
        let current = self.compactionTracker.lastTokens().max(1) as f64;
        let zoneFraction = (1.0 - (0.70 * budget / current)).clamp(0.0, 1.0);
        let s3Result = match crate::s3::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            self.topicTracker.topics(),
            &self.client,
            &self.config.utility.model,
            self.config.heavy.contextWindow,
            zoneFraction,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S3 compaction failed: {e}");
                let _ = logTx
                    .send(LogEvent::CompactionFailed {
                        stage: stageStr.to_string(),
                        reason: e.to_string(),
                    })
                    .await;
                return StageOutcome::Failed;
            }
        };
        if let Some(cost) = s3Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s3Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S3);
            tracing::debug!("S3 exhausted \u{2014} no topics to compact");
            return StageOutcome::Exhausted;
        }
        let afterTurn = self.headTurnId.clone().unwrap_or_default();
        let topicCount = s3Result.compacted.len();
        for topic in &s3Result.compacted {
            if let Err(e) = self.compactionLog.recordTopicCompact(
                &topic.topicLabel,
                &topic.summary,
                topic.sourceBlockIds.clone(),
                &afterTurn,
            ) {
                tracing::warn!("compaction log write failed for {}: {e}", topic.topicId);
            }
            for path in &topic.invalidatedFiles {
                self.filesRead.remove(path);
            }
        }
        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId, 0, 0) {
            Ok(r) => self.history = r.messages,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S3: {e}");
                let _ = logTx
                    .send(LogEvent::CompactionFailed {
                        stage: stageStr.to_string(),
                        reason: format!("history reconstruction failed: {e}"),
                    })
                    .await;
                return StageOutcome::Failed;
            }
        }
        self.compactionTracker.clearExhaustion();
        let reduction = format!("compressed {topicCount} topics");
        let _ = logTx
            .send(LogEvent::CompactionComplete {
                stage: stageStr.to_string(),
                reduction,
                markerBlock: Some(0),
            })
            .await;
        StageOutcome::DidWork
    }

    /// Run S4 deep recompaction. Merges the latest active S4 briefing,
    /// fresh S3 topic summaries, and orphan S2 summaries from outside
    /// the protected recent band into a single handoff briefing.
    async fn runS4Trigger(
        &mut self,
        stageStr: &str,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> StageOutcome {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        // Use 80% of the utility model's context window for S4 input,
        // leaving room for system prompt and output.
        let maxInputChars = self.config.heavy.contextWindow * 2; // tokens → ~chars
        let s4Result = match crate::s4::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            &self.client,
            &self.config.utility.model,
            maxInputChars,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("S4 compaction failed: {e}");
                let _ = logTx
                    .send(LogEvent::CompactionFailed {
                        stage: stageStr.to_string(),
                        reason: e.to_string(),
                    })
                    .await;
                return StageOutcome::Failed;
            }
        };

        if let Some(cost) = s4Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s4Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S4);
            tracing::debug!("S4 exhausted \u{2014} no S3/S4 content to merge");
            return StageOutcome::Exhausted;
        }

        let afterTurn = self.headTurnId.clone().unwrap_or_default();
        let blockCount = s4Result.sourceBlockIds.len();
        let summaryLen = s4Result.summary.len();
        if let Err(e) = self.compactionLog.recordFullCompact(
            &s4Result.summary,
            s4Result.sourceBlockIds,
            &afterTurn,
        ) {
            tracing::warn!("compaction log write failed: {e}");
        }

        self.filesRead.clear();

        let headId = self.headTurnId.as_deref().unwrap_or("");
        match context::reconstruct(&self.transcript, &self.compactionLog, headId, 0, 0) {
            Ok(r) => self.history = r.messages,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S4: {e}");
                let _ = logTx
                    .send(LogEvent::CompactionFailed {
                        stage: stageStr.to_string(),
                        reason: format!("history reconstruction failed: {e}"),
                    })
                    .await;
                return StageOutcome::Failed;
            }
        }

        self.compactionTracker.clearExhaustion();
        let reduction =
            format!("merged {blockCount} source blocks into briefing ({summaryLen} chars)");
        let _ = logTx
            .send(LogEvent::CompactionComplete {
                stage: stageStr.to_string(),
                reduction,
                markerBlock: Some(0),
            })
            .await;
        StageOutcome::DidWork
    }

    /// Remove the last S4 FullCompact from the compaction log and
    /// reconstruct the live history. Returns an ack describing what
    /// happened.
    pub fn undoFullCompact(&mut self) -> crate::control::CommandAck {
        match self.compactionLog.removeLastFullCompact() {
            Ok(true) => {
                let headId = self.headTurnId.as_deref().unwrap_or("");
                match crate::context::reconstruct(
                    &self.transcript,
                    &self.compactionLog,
                    headId,
                    0,
                    0,
                ) {
                    Ok(r) => {
                        self.history = r.messages;
                        self.compactionTracker.clearExhaustion();
                        self.filesRead.clear();
                        crate::control::CommandAck::ok(
                            "Removed last S4 compaction. Context reconstructed from S3/S2 summaries.",
                        )
                    }
                    Err(e) => crate::control::CommandAck::err(format!(
                        "S4 removed but history reconstruction failed: {e}"
                    )),
                }
            }
            Ok(false) => crate::control::CommandAck::err("No S4 compaction to undo."),
            Err(e) => crate::control::CommandAck::err(format!("Failed to undo S4: {e}")),
        }
    }

    /// Recalibrate the context by reconstructing with the current
    /// budget. Applies cached compressions bottom-up, then generates
    /// new S2/S3/S4 ops for uncached content until under budget.
    pub async fn recalibrateContext(
        &mut self,
        logTx: &mpsc::Sender<LogEvent>,
    ) -> crate::control::CommandAck {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        if headId.is_empty() {
            return crate::control::CommandAck::err("No turns to compact.");
        }
        let budget = self.compactionTracker.compactLimit();
        let toolDefsChars = serde_json::to_string(&self.tools)
            .map(|s| s.len())
            .unwrap_or(0);
        let overheadChars = self.systemPrompt.len() + toolDefsChars;
        let overheadTokens = overheadChars * 2 / 5;

        // Phase 1: Apply cached compressions.
        let r = match crate::context::reconstruct(
            &self.transcript,
            &self.compactionLog,
            headId,
            budget,
            overheadTokens,
        ) {
            Ok(r) => r,
            Err(e) => {
                return crate::control::CommandAck::err(format!("Recalibration failed: {e}"));
            }
        };
        self.history = r.messages;
        let historyChars = serde_json::to_string(&self.history)
            .map(|s| s.len())
            .unwrap_or(0);
        let mut estimatedTotal = (historyChars + overheadChars) * 2 / 5;
        tracing::debug!(
            estimatedTotal,
            budget,
            historyChars,
            overheadChars,
            "recalibrate: phase 1 estimate"
        );

        if estimatedTotal <= budget {
            return crate::control::CommandAck::ok(format!(
                "Context recalibrated (budget {budget}, ~{estimatedTotal} tokens). Under budget from cache alone."
            ));
        }

        // Unified cascade: S1→S2→S3→S4, after any lower stage does
        // work, restart from S1. S4 fires only when S1/S2/S3 all
        // exhaust AND over budget. After S4, restart from S1 — but if
        // S1/S2/S3 all exhaust again without new work, stop. S4 only
        // gets one shot per cycle of lower-stage exhaustion.
        self.compactionTracker.updateTokens(estimatedTotal);
        self.compactionTracker.clearExhaustion();

        for _ in 0..30 {
            let s1 = self.runS1("S1", logTx).await;
            if matches!(s1, StageOutcome::DidWork) {
                self.compactionTracker.clearExhaustion();
                continue;
            }
            let s2 = self.runS2("S2", logTx).await;
            if matches!(s2, StageOutcome::DidWork) {
                self.compactionTracker.clearExhaustion();
                continue;
            }
            let s3 = self.runS3("S3", logTx).await;
            if matches!(s3, StageOutcome::DidWork) {
                self.compactionTracker.clearExhaustion();
                let historyChars = serde_json::to_string(&self.history)
                    .map(|s| s.len())
                    .unwrap_or(0);
                estimatedTotal = (historyChars + overheadChars) * 2 / 5;
                self.compactionTracker.updateTokens(estimatedTotal);
                continue;
            }
            // S1/S2/S3 all exhausted — reconstruct and check budget.
            let headId = self.headTurnId.as_deref().unwrap_or("");
            match crate::context::reconstruct(
                &self.transcript,
                &self.compactionLog,
                headId,
                budget,
                overheadTokens,
            ) {
                Ok(r) => {
                    self.history = r.messages;
                    let historyChars = serde_json::to_string(&self.history)
                        .map(|s| s.len())
                        .unwrap_or(0);
                    estimatedTotal = (historyChars + overheadChars) * 2 / 5;
                }
                Err(_) => break,
            }
            if estimatedTotal <= budget {
                tracing::debug!(
                    estimatedTotal,
                    budget,
                    "recalibrate: under budget, stopping"
                );
                break;
            }
            tracing::debug!(
                estimatedTotal,
                budget,
                "recalibrate: still over budget, firing S4"
            );
            let s4 = self.runS4Trigger("S4", logTx).await;
            if !matches!(s4, StageOutcome::DidWork) {
                break;
            }
            self.compactionTracker.clearExhaustion();
            let historyChars = serde_json::to_string(&self.history)
                .map(|s| s.len())
                .unwrap_or(0);
            estimatedTotal = (historyChars + overheadChars) * 2 / 5;
            self.compactionTracker.updateTokens(estimatedTotal);
        }

        // Phase 3: Reconstruct again with the expanded cache.
        let headId = self.headTurnId.as_deref().unwrap_or("");
        match crate::context::reconstruct(
            &self.transcript,
            &self.compactionLog,
            headId,
            budget,
            overheadTokens,
        ) {
            Ok(r) => {
                self.history = r.messages;
                let historyChars = serde_json::to_string(&self.history)
                    .map(|s| s.len())
                    .unwrap_or(0);
                let finalTokens = (historyChars + overheadChars) * 2 / 5;
                self.compactionTracker.clearExhaustion();
                let status = if finalTokens <= budget {
                    "Under budget from cache alone."
                } else {
                    "Still over budget — may need more S2/S3 ops."
                };
                crate::control::CommandAck::ok(format!(
                    "Context recalibrated (budget {budget}, ~{finalTokens} tokens). {status}"
                ))
            }
            Err(e) => crate::control::CommandAck::err(format!("Final reconstruction failed: {e}")),
        }
    }
}
