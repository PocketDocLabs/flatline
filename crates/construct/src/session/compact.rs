use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;

use super::Session;
use crate::control::LogEvent;
use crate::{compaction_trigger, context};

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

            let didWork = match stage {
                compaction_trigger::StagePick::S1 => self.runS1(&stageStr, logTx).await,
                compaction_trigger::StagePick::S2 => self.runS2(&stageStr, logTx).await,
                compaction_trigger::StagePick::S3 => self.runS3(&stageStr, logTx).await,
                compaction_trigger::StagePick::S4 => self.runS4Trigger(&stageStr, logTx).await,
            };

            if didWork || self.compactionTracker.allExhausted() {
                return;
            }
        }
    }

    /// Run S1 mechanical pruning. Returns true if context was reduced.
    async fn runS1(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
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
            true
        } else {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S1);
            tracing::debug!("S1 exhausted \u{2014} nothing to prune");
            false
        }
    }

    /// Run S2 block compaction. Returns true if context was reduced.
    async fn runS2(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        let headTurn = self.headTurnId.clone().unwrap_or_default();
        let s2Result = match crate::s2::run(
            &self.transcript,
            &self.compactionLog,
            &headTurn,
            &self.client,
            &self.config.utility.model,
            self.config.heavy.contextWindow,
            self.config.compactRatio,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S2 compaction failed: {e}");
                self.compactionTracker
                    .markExhausted(compaction_trigger::StagePick::S2);
                return false;
            }
        };
        if let Some(cost) = s2Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s2Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S2);
            tracing::debug!("S2 exhausted \u{2014} no blocks to compact");
            return false;
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
        match context::reconstruct(&self.transcript, &self.compactionLog, headId) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S2: {e}");
                return false;
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
        true
    }

    /// Run S3 topic compaction. Returns true if context was reduced.
    async fn runS3(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        let s3Result = match crate::s3::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            self.topicTracker.topics(),
            &self.client,
            &self.config.utility.model,
            self.config.heavy.contextWindow,
            self.config.compactRatio,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("S3 compaction failed: {e}");
                self.compactionTracker
                    .markExhausted(compaction_trigger::StagePick::S3);
                return false;
            }
        };
        if let Some(cost) = s3Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s3Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S3);
            tracing::debug!("S3 exhausted \u{2014} no topics to compact");
            return false;
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
        match context::reconstruct(&self.transcript, &self.compactionLog, headId) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S3: {e}");
                return false;
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
        true
    }

    /// Run S4 deep recompaction. Merges the latest active S4 briefing,
    /// fresh S3 topic summaries, and orphan S2 summaries from outside
    /// the protected recent band into a single handoff briefing.
    /// Returns true if context was reduced.
    async fn runS4Trigger(&mut self, stageStr: &str, logTx: &mpsc::Sender<LogEvent>) -> bool {
        let headId = self.headTurnId.as_deref().unwrap_or("");
        let s4Result = match crate::s4::run(
            &self.transcript,
            &self.compactionLog,
            headId,
            &self.client,
            &self.config.utility.model,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("S4 compaction failed: {e}");
                self.compactionTracker
                    .markExhausted(compaction_trigger::StagePick::S4);
                return false;
            }
        };

        if let Some(cost) = s4Result.cost {
            self.costTracker.record(cost, &self.config.utility.model);
        }
        if !s4Result.didWork {
            self.compactionTracker
                .markExhausted(compaction_trigger::StagePick::S4);
            tracing::debug!("S4 exhausted \u{2014} no S3/S4 content to merge");
            return false;
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
        match context::reconstruct(&self.transcript, &self.compactionLog, headId) {
            Ok(h) => self.history = h,
            Err(e) => {
                tracing::error!("failed to reconstruct history after S4: {e}");
                return false;
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
        true
    }
}
