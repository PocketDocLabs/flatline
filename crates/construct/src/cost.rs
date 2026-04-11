//! Cost tracking for sessions and rolling windows.
//!
//! Public types:
//! - [`CostTracker`] — per-session cost accumulator.
//!
//! Public functions:
//! - [`rollingWindowCost`] — sum cost across recent sessions.
//!
//! Dependencies: `crate::transcript`

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

const ROLLING_WINDOW_SECS: u64 = 16 * 3600;

/// Per-session cost accumulator.
#[derive(Debug, Clone)]
pub struct CostTracker {
    /// Total USD accumulated this session.
    sessionCost: f64,
    /// Cost of the most recent API call.
    lastTurnCost: f64,
    /// Per-model cost breakdown: model_id -> total USD.
    perModel: HashMap<String, f64>,
    /// Whether the budget warning has fired (prevents repeated warnings).
    budgetWarned: bool,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self {
            sessionCost: 0.0,
            lastTurnCost: 0.0,
            perModel: HashMap::new(),
            budgetWarned: false,
        }
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cost from an API response.
    pub fn record(&mut self, cost: f64, model: &str) {
        self.lastTurnCost = cost;
        self.sessionCost += cost;
        *self.perModel.entry(model.to_string()).or_default() += cost;
    }

    pub fn sessionCost(&self) -> f64 {
        self.sessionCost
    }

    pub fn lastTurnCost(&self) -> f64 {
        self.lastTurnCost
    }

    pub fn perModel(&self) -> &HashMap<String, f64> {
        &self.perModel
    }

    /// Seed from a previously persisted total (session resume).
    pub fn seed(&mut self, total: f64) {
        self.sessionCost = total;
    }

    /// Check if cost has crossed the warning threshold for the first time.
    /// Returns true exactly once when sessionCost >= limit.
    pub fn checkWarning(&mut self, limit: f64) -> bool {
        if !self.budgetWarned && self.sessionCost >= limit {
            self.budgetWarned = true;
            return true;
        }
        false
    }
}

/// Compute total cost across sessions updated within the last 16 hours.
pub fn rollingWindowCost(projectDir: Option<&str>) -> f64 {
    let sessions = match crate::transcript::listSessions(projectDir) {
        Ok(s) => s,
        Err(_) => return 0.0,
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let cutoff = now.saturating_sub(ROLLING_WINDOW_SECS);

    sessions
        .iter()
        .filter(|m| m.updatedAt >= cutoff)
        .map(|m| m.totalCost)
        .sum()
}

/// Format a USD cost for display.
/// Uses 2 decimal places when >= $0.50, 4 otherwise.
pub fn formatCost(cost: f64) -> String {
    if cost >= 0.50 {
        format!("${:.2}", cost)
    } else {
        format!("${:.4}", cost)
    }
}
