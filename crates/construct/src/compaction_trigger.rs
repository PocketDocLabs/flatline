//! Compaction trigger evaluation — decides when and which stage to fire.
//!
//! `compactLimit = contextWindow * compactRatio` is the ceiling — no
//! compaction fires above it.  Thresholds are fractions of compactLimit:
//! - 80%: S1 (mechanical pruning), then S2 (per-block LLM) after S1 exhausted
//! - 90%: S3 (per-topic LLM, requires S2 exhausted)
//! - 100%: S4 (full briefing, requires S3 exhausted)
//!
//! Stages escalate within each band until context is reduced.
//!
//! # Public API
//! - [`StagePick`] — which compaction stage to run
//! - [`Tracker`] — stateful trigger evaluator
//!
//! # Dependencies
//! None (pure logic).

use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StagePick {
    S1,
    S2,
    S3,
    S4,
}

impl fmt::Display for StagePick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StagePick::S1 => write!(f, "S1"),
            StagePick::S2 => write!(f, "S2"),
            StagePick::S3 => write!(f, "S3"),
            StagePick::S4 => write!(f, "S4"),
        }
    }
}

pub struct Tracker {
    contextWindow: usize,
    compactRatio: f64,
    lastTokens: usize,
    exhausted: HashSet<StagePick>,
}

impl Tracker {
    pub fn new(contextWindow: usize, compactRatio: f64) -> Self {
        Self {
            contextWindow,
            compactRatio,
            lastTokens: 0,
            exhausted: HashSet::new(),
        }
    }

    pub fn updateTokens(&mut self, tokens: usize) {
        self.lastTokens = tokens;
    }

    pub fn lastTokens(&self) -> usize {
        self.lastTokens
    }

    /// The compact limit (absolute token count that triggers compaction).
    pub fn compactLimit(&self) -> usize {
        (self.contextWindow as f64 * self.compactRatio) as usize
    }

    /// Current usage as a ratio of the compact limit (0.0–1.0+).
    pub fn usageRatio(&self) -> f64 {
        let limit = self.compactLimit();
        if limit == 0 {
            return 0.0;
        }
        self.lastTokens as f64 / limit as f64
    }

    /// Evaluate whether compaction should fire.
    ///
    /// Tries the cheapest applicable stage first within the current
    /// threshold band. Returns None if no stage should fire (either
    /// below all thresholds, or all applicable stages are exhausted).
    pub fn evaluate(&self, tokens: usize) -> Option<StagePick> {
        let limit = self.compactLimit();
        if limit == 0 {
            return None;
        }

        let s1s2Threshold = limit * 80 / 100;
        let s3Threshold = limit * 90 / 100;

        if tokens < s1s2Threshold {
            return None;
        }

        // 80%+: S1 first (cheap mechanical), then S2 after S1 exhausted.
        if !self.exhausted.contains(&StagePick::S1) {
            return Some(StagePick::S1);
        }
        if !self.exhausted.contains(&StagePick::S2) {
            return Some(StagePick::S2);
        }

        // 90%+: S3 after S2 exhausted.
        if tokens >= s3Threshold && !self.exhausted.contains(&StagePick::S3) {
            return Some(StagePick::S3);
        }

        // 100%: S4 after S3 exhausted.
        if tokens >= limit && !self.exhausted.contains(&StagePick::S4) {
            return Some(StagePick::S4);
        }

        None
    }

    pub fn markExhausted(&mut self, stage: StagePick) {
        self.exhausted.insert(stage);
    }

    pub fn clearExhaustion(&mut self) {
        self.exhausted.clear();
    }

    /// Check if a specific stage is exhausted.
    pub fn isExhausted(&self, stage: StagePick) -> bool {
        self.exhausted.contains(&stage)
    }

    /// Check if all stages are exhausted (nothing more we can do).
    pub fn allExhausted(&self) -> bool {
        self.exhausted.contains(&StagePick::S1)
            && self.exhausted.contains(&StagePick::S2)
            && self.exhausted.contains(&StagePick::S3)
            && self.exhausted.contains(&StagePick::S4)
    }
}
