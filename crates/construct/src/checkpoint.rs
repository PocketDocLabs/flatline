//! Shadow git checkpoints — per-project file state snapshots.
//!
//! # Status: stub — real implementation in task #7.

use anyhow::Result;

#[derive(Clone)]
pub struct CheckpointManager {
    // TODO(pocketdoc, 2026-03-11): Shadow git repo handle.
}

impl CheckpointManager {
    /// Initialize the checkpoint manager for a project directory.
    pub async fn init(_projectDir: &str) -> Result<Self> {
        // TODO(pocketdoc, 2026-03-11): Create/open shadow git repo.
        Ok(Self {})
    }

    /// Take a snapshot of current file state, tagged with a turn ID.
    pub async fn snapshot(&self, _turnId: &str) -> Result<()> {
        // TODO(pocketdoc, 2026-03-11): git add + commit in shadow repo.
        Ok(())
    }

    /// Undo to the previous checkpoint. Returns the turn ID restored to.
    pub async fn undo(&self) -> Result<String> {
        anyhow::bail!("Checkpoint system not yet implemented.")
    }

    /// Restore to a specific checkpoint by turn ID or target string.
    pub async fn restoreTo(&self, _target: &str) -> Result<()> {
        anyhow::bail!("Checkpoint system not yet implemented.")
    }
}
