//! Input history with user-level file persistence.
//!
//! Stores submitted messages and supports up/down navigation
//! with draft preservation. Persists to ~/.config/flatline/history.json.
//!
//! # Public API
//! - [`History`] — history state and navigation
//!
//! # Dependencies
//! `serde_json`, `dirs`

use std::fs;
use std::path::PathBuf;

const MAX_ENTRIES: usize = 500;

/// Input history with cursor-based navigation.
pub struct History {
    entries: Vec<String>,
    /// Index into entries during navigation. None = not navigating.
    cursor: Option<usize>,
    /// Saved draft (the text that was in the textarea before navigating).
    draft: String,
    /// File path for persistence.
    filePath: Option<PathBuf>,
}

impl History {
    /// Create a new history, loading from the user config file if it exists.
    pub fn new() -> Self {
        let filePath: Option<PathBuf> =
            dirs::config_dir().map(|d| d.join("flatline").join("history.json"));
        let entries = filePath
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();

        Self {
            entries,
            cursor: None,
            draft: String::new(),
            filePath,
        }
    }

    /// Record a submitted message. Deduplicates consecutive entries.
    pub fn push(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        // Deduplicate consecutive.
        if self.entries.last().map(|s| s.as_str()) == Some(text) {
            self.cursor = None;
            return;
        }
        self.entries.push(text.to_string());
        // Cap at MAX_ENTRIES.
        if self.entries.len() > MAX_ENTRIES {
            let excess = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(..excess);
        }
        self.cursor = None;
        self.save();
    }

    /// Navigate to the previous (older) entry.
    ///
    /// Args:
    ///     currentText: The current textarea content (saved as draft on first call).
    ///
    /// Returns:
    ///     Option<&str>: The history entry to display, or None if at the beginning.
    pub fn navigateUp(&mut self, currentText: &str) -> Option<&str> {
        if self.entries.is_empty() {
            return None;
        }

        let idx = match self.cursor {
            None => {
                // First navigation — save current text as draft.
                self.draft = currentText.to_string();
                self.entries.len() - 1
            }
            Some(0) => return None,
            Some(i) => i - 1,
        };

        self.cursor = Some(idx);
        Some(&self.entries[idx])
    }

    /// Navigate to the next (newer) entry, or restore the draft.
    ///
    /// Returns:
    ///     Option<&str>: The entry or draft to display, or None if not navigating.
    pub fn navigateDown(&mut self) -> Option<&str> {
        let idx = self.cursor?;

        if idx + 1 >= self.entries.len() {
            // Past the end — restore draft.
            self.cursor = None;
            Some(&self.draft)
        } else {
            self.cursor = Some(idx + 1);
            Some(&self.entries[idx + 1])
        }
    }

    /// Reset navigation state (called on any edit to the textarea).
    pub fn resetCursor(&mut self) {
        self.cursor = None;
    }

    /// Whether we're currently navigating history.
    pub fn isNavigating(&self) -> bool {
        self.cursor.is_some()
    }

    /// Persist entries to the config file.
    fn save(&self) {
        let Some(path) = &self.filePath else { return };
        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string(&self.entries) {
            let _ = fs::write(path, json);
        }
    }
}
