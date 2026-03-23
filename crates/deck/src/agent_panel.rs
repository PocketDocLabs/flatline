//! Agent panel — renders conversation, permission prompts, and input.
//!
//! Displays streamed assistant responses with markdown rendering,
//! tool request approvals, and a text input line.
//!
//! # Public API
//! - [`AgentPanel`] — panel state and rendering
//!
//! # Dependencies
//! `ratatui`

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::command::{self, CommandDef};
use crate::history::History;
use crate::markdown::{self, RenderedBlock};
use crate::selection::{self, Selection};
use crate::text_area::{TextArea, unicode_display_width};
use crate::throbber::Throbber;

/// A single entry in the agent panel display.
#[derive(Debug, Clone)]
pub enum PanelEntry {
    User(String),
    Assistant(String),
    Reasoning { text: String, expanded: bool },
    ToolRequest { summary: String, diff: Option<String>, command: Option<String> },
    ToolApproved { name: String },
    ToolDenied { name: String },
    /// A tool was auto-denied by a permission rule (shown with red background).
    ToolAutoDenied { name: String, summary: String },
    ToolResult { name: String, output: String },
    /// A tool is currently executing (shown with throbber + elapsed time).
    ToolActive { name: String, summary: String },
    /// Subagent activity — renders as a single bordered panel.
    SubagentBlock {
        agentType: String,
        prompt: String,
        /// One-liner tool activity entries: (name, summary).
        toolLines: Vec<(String, String)>,
        /// Whether the subagent has finished.
        done: bool,
        /// Turn count (set on completion).
        turns: usize,
        /// Final content from the subagent (set on completion, rendered as markdown).
        content: Option<String>,
        /// Whether the content section is expanded.
        contentExpanded: bool,
        /// Child session ID (for loading transcript on "view" after resume).
        sessionId: Option<String>,
    },
    Error(String),
    CommandResult(String),
    /// Non-copyable informational notice (e.g. session resumed).
    SessionNotice(String),
    /// Visual marker showing where compaction replaced content.
    CompactionMarker { stage: String },
    Cancelled,
}

/// Tracked state for a running subagent (for the overlay panel).
pub struct ActiveSubagent {
    pub sessionId: String,
    pub agentType: String,
    pub prompt: String,
    pub startTime: Instant,
    /// Structured transcript entries for the overlay panel.
    pub transcript: Vec<PanelEntry>,
    /// Raw PTY output bytes for the shell tab.
    pub shellScrollback: Vec<u8>,
}

/// Slash command completion state.
struct CompletionState {
    /// Filtered matches for the current prefix.
    matches: Vec<&'static CommandDef>,
    /// Index of the selected item in `matches`.
    selected: usize,
}

/// Maximum visible content lines before a code block is collapsed.
const MAX_CODE_BLOCK_LINES: usize = 6;

/// Metadata for a rendered code block in the visual line buffer.
struct CodeBlockRange {
    /// First visual line index (inclusive, includes top border).
    startLine: usize,
    /// Last visual line index (inclusive, includes bottom border).
    endLine: usize,
    /// Identity: (entry index, code block ordinal within entry).
    blockId: (usize, usize),
    /// Maximum content width in display columns.
    maxContentWidth: usize,
    /// Available inner width (box width minus 2 border characters).
    innerWidth: usize,
    /// Raw highlighted spans per content line (for drag-select copy).
    contentLines: Vec<Vec<Span<'static>>>,
    /// Original code text (for click-to-copy).
    rawCode: String,
    /// Screen column where the "copy" label starts on the top border.
    copyLabelCol: u16,
    /// True if this block has more lines than MAX_CODE_BLOCK_LINES.
    collapsible: bool,
    /// Total content line count (before truncation).
    totalLines: usize,
}

/// Agent panel state.
pub struct AgentPanel {
    pub(crate) entries: Vec<PanelEntry>,
    /// Active slash command completion, if any.
    completion: Option<CompletionState>,
    streamingContent: String,
    /// Buffered incoming content waiting to be revealed character-by-character.
    pendingContent: String,
    /// Fade progress of the next character to reveal (0.0 = invisible, 1.0 = fully bright).
    fadeProgress: f32,
    /// Last time the reveal ticker advanced.
    lastRevealTick: Instant,
    /// When buffered content started arriving (for spool-up delay).
    revealStart: Option<Instant>,
    /// When the last content chunk was received (for stream rate calculation).
    lastReceiveTime: Instant,
    /// Total chars received this turn (for stream rate calculation).
    totalCharsReceived: usize,
    /// Finalization deferred until pending reveal buffer drains.
    deferredFinalize: bool,
    streamingReasoning: String,
    isStreaming: bool,
    /// True from pushUser() until TurnComplete/TurnCancelled.
    turnActive: bool,
    pub textArea: TextArea,
    pub history: History,
    pub pendingPermit: bool,
    pendingToolName: String,
    /// Human-readable summary of the tool call (key arg + description).
    pendingToolSummary: String,
    /// Model-provided explanation (shell commands only).
    pendingToolExplanation: Option<String>,
    /// Impact tier for visual treatment (color/symbol).
    pendingToolImpact: construct::tool::ShellImpact,
    /// Suggested "always allow" patterns for the pending permission prompt.
    permitPatterns: Vec<String>,
    /// Currently selected pattern index (last = custom).
    permitSelectedPattern: usize,
    /// Editable custom pattern text.
    permitCustomPattern: String,
    /// Whether the custom pattern field is being edited.
    permitEditingCustom: bool,
    /// Horizontal scroll offset for the permission code block.
    permitCodeScrollX: u16,
    /// Throbber animation for inline thinking indicator.
    throbber: Throbber,
    /// When reasoning started (for elapsed time display).
    thinkingStartTime: Option<Instant>,
    /// Whether the currently-streaming reasoning block is expanded.
    thinkingExpanded: bool,
    /// Whether reasoning is actively streaming right now.
    reasoningActive: bool,
    /// Whether a tool is currently executing (for throbber).
    toolActive: bool,
    /// When the current tool started executing (for elapsed time display).
    toolStartTime: Option<Instant>,
    /// Active subagent state (for the overlay panel).
    pub activeSubagent: Option<ActiveSubagent>,
    /// Last wall-clock time the throbber was ticked.
    lastThrobberTick: Instant,
    /// Scroll offset from the bottom (in visual lines).
    pub scrollOffset: u16,
    /// ScrollY value from the last render (for visual-line lookups).
    lastScrollY: u16,
    /// Chat area width from the last render (for wrap estimation).
    lastChatWidth: u16,
    /// Previous maxScroll value (for stable scroll during streaming).
    lastMaxScroll: u16,
    /// Which visual lines are wrap continuations (not real line breaks).
    lastContinuationMap: Vec<bool>,
    /// Visual line index of each reasoning header (entry index, line index).
    /// `None` entry index means streaming reasoning.
    lastReasoningHeaders: Vec<(Option<usize>, usize)>,
    /// Plain text of each visual line from the last buildLines (for scrollback copy).
    lastLineTexts: Vec<String>,
    /// Visual line indices that should copy as empty (e.g. session notices).
    nonCopyableLines: HashSet<usize>,
    /// Input area rect from the last render (for mouse hit-testing).
    pub lastInputRect: Rect,
    /// Horizontal scroll offset per code block: (entryIndex, codeBlockOrdinal) -> scrollX.
    codeScrollX: HashMap<(usize, usize), u16>,
    /// Code block metadata from the last buildLines (for hit-testing and copy).
    lastCodeBlockRanges: Vec<CodeBlockRange>,
    /// Click-to-copy visual feedback: (blockId, when copied).
    copiedFlash: Option<((usize, usize), Instant)>,
    /// Code blocks toggled to expanded (shows all lines).
    codeExpanded: HashSet<(usize, usize)>,
    /// Visual line index of the last subagent header (for click-to-view).
    lastSubagentHeaderLine: std::cell::Cell<Option<usize>>,
    /// Visual line index + entry index of the subagent content toggle border.
    lastSubagentToggleLine: std::cell::Cell<Option<(usize, usize)>>,
    /// Transient retry status shown in throbber instead of "thinking".
    retryStatus: Option<String>,
    /// Image attachments queued for the next message submission.
    attachments: Vec<construct::session::Attachment>,
}

impl AgentPanel {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            completion: None,
            streamingContent: String::new(),
            pendingContent: String::new(),
            fadeProgress: 1.0,
            lastRevealTick: Instant::now(),
            revealStart: None,
            lastReceiveTime: Instant::now(),
            totalCharsReceived: 0,
            deferredFinalize: false,
            streamingReasoning: String::new(),
            isStreaming: false,
            turnActive: false,
            textArea: TextArea::new(),
            history: History::new(),
            pendingPermit: false,
            pendingToolName: String::new(),
            pendingToolSummary: String::new(),
            pendingToolExplanation: None,
            pendingToolImpact: construct::tool::ShellImpact::Read,
            permitPatterns: Vec::new(),
            permitSelectedPattern: 0,
            permitCustomPattern: String::new(),
            permitEditingCustom: false,
            permitCodeScrollX: 0,
            throbber: Throbber::new(),
            thinkingStartTime: None,
            thinkingExpanded: false,
            reasoningActive: false,
            toolActive: false,
            toolStartTime: None,
            activeSubagent: None,
            lastThrobberTick: Instant::now(),
            scrollOffset: 0,
            lastScrollY: 0,
            lastChatWidth: 0,
            lastMaxScroll: 0,
            lastContinuationMap: Vec::new(),
            lastReasoningHeaders: Vec::new(),
            lastLineTexts: Vec::new(),
            nonCopyableLines: HashSet::new(),
            lastInputRect: Rect::default(),
            codeScrollX: HashMap::new(),
            lastCodeBlockRanges: Vec::new(),
            copiedFlash: None,
            codeExpanded: HashSet::new(),
            lastSubagentHeaderLine: std::cell::Cell::new(None),
            lastSubagentToggleLine: std::cell::Cell::new(None),
            retryStatus: None,
            attachments: Vec::new(),
        }
    }

    /// Reset the panel to a clean state (new session).
    pub fn clearDisplay(&mut self) {
        self.entries.clear();
        self.streamingContent.clear();
        self.pendingContent.clear();
        self.fadeProgress = 1.0;
        self.revealStart = None;
        self.lastReceiveTime = Instant::now();
        self.totalCharsReceived = 0;
        self.deferredFinalize = false;
        self.streamingReasoning.clear();
        self.isStreaming = false;
        self.turnActive = false;
        self.pendingPermit = false;
        self.pendingToolName.clear();
        self.thinkingStartTime = None;
        self.thinkingExpanded = false;
        self.reasoningActive = false;
        self.scrollOffset = 0;
        self.lastScrollY = 0;
        self.lastMaxScroll = 0;
        self.lastContinuationMap.clear();
        self.lastReasoningHeaders.clear();
        self.lastLineTexts.clear();
        self.nonCopyableLines.clear();
        self.codeScrollX.clear();
        self.lastCodeBlockRanges.clear();
        self.copiedFlash = None;
        self.codeExpanded.clear();
        self.attachments.clear();
    }

    /// Add an image attachment for the next submission.
    pub fn addAttachment(&mut self, att: construct::session::Attachment) {
        self.attachments.push(att);
    }

    /// Drain and return all queued attachments.
    pub fn takeAttachments(&mut self) -> Vec<construct::session::Attachment> {
        std::mem::take(&mut self.attachments)
    }

    /// Number of queued attachments.
    pub fn attachmentCount(&self) -> usize {
        self.attachments.len()
    }

    /// Remove the most recently added attachment.
    pub fn removeLastAttachment(&mut self) {
        self.attachments.pop();
    }

    pub fn pushUser(&mut self, text: &str) {
        let display = if self.attachments.is_empty() {
            text.to_string()
        } else {
            let n = self.attachments.len();
            let suffix = if n == 1 { "1 image" } else { &format!("{n} images") };
            format!("{text}\n[+{suffix} attached]")
        };
        self.entries.push(PanelEntry::User(display));
        self.scrollOffset = 0;
        self.turnActive = true;
        // Start thinking indicator immediately for responsiveness.
        self.isStreaming = true;
        self.thinkingStartTime = Some(Instant::now());
    }

    /// Display a slash command without activating the turn or throbber.
    pub fn pushCommand(&mut self, text: &str) {
        self.entries.push(PanelEntry::User(text.into()));
        self.scrollOffset = 0;
    }

    pub fn appendContent(&mut self, text: &str) {
        if !self.turnActive { return; }
        self.isStreaming = true;
        self.retryStatus = None;
        // Content streaming means reasoning phase is over.
        self.reasoningActive = false;
        let now = Instant::now();
        // On first content arrival, start the spool-up timer and reset the
        // reveal clock so the first tick doesn't see a stale elapsed time.
        if self.revealStart.is_none() {
            self.revealStart = Some(now);
            self.lastRevealTick = now;
            self.fadeProgress = 0.0;
        }
        self.lastReceiveTime = now;
        self.totalCharsReceived += text.chars().count();
        self.pendingContent.push_str(text);
    }

    pub fn appendReasoning(&mut self, text: &str) {
        if !self.turnActive { return; }
        self.isStreaming = true;
        self.retryStatus = None;
        self.reasoningActive = true;
        if self.thinkingStartTime.is_none() {
            self.thinkingStartTime = Some(Instant::now());
        }
        self.streamingReasoning.push_str(text);
    }

    pub fn finalizeStreaming(&mut self) {
        if !self.streamingReasoning.is_empty() {
            let clean = construct::text::sanitizeVariationSelectors(&self.streamingReasoning);
            self.streamingReasoning.clear();
            self.entries.push(PanelEntry::Reasoning {
                text: clean,
                expanded: false,
            });
        }
        // If there's still pending content being revealed, defer finalization
        // so the reveal can finish at its natural pace instead of snapping.
        if !self.pendingContent.is_empty() {
            self.deferredFinalize = true;
            return;
        }
        self.doFinalize();
    }

    /// Actually commit streaming content to entries and reset state.
    /// Called directly when pending is empty, or by tickReveal after deferred drain.
    fn doFinalize(&mut self) {
        self.deferredFinalize = false;
        self.fadeProgress = 1.0;
        self.revealStart = None;
        self.totalCharsReceived = 0;
        if !self.streamingContent.is_empty() {
            let clean = construct::text::sanitizeVariationSelectors(&self.streamingContent);
            self.streamingContent.clear();
            self.entries.push(PanelEntry::Assistant(clean));
            // Transfer code block scroll state from streaming (usize::MAX) to real entry index.
            let realIdx = self.entries.len() - 1;
            let streamingKeys: Vec<(usize, usize)> = self.codeScrollX
                .keys()
                .filter(|(ei, _)| *ei == usize::MAX)
                .copied()
                .collect();
            for (_, ordinal) in streamingKeys {
                if let Some(scrollX) = self.codeScrollX.remove(&(usize::MAX, ordinal)) {
                    self.codeScrollX.insert((realIdx, ordinal), scrollX);
                }
            }
        }
        self.isStreaming = false;
        self.reasoningActive = false;
        self.thinkingStartTime = None;
        self.thinkingExpanded = false;
    }

    /// Whether a turn is currently in progress.
    pub fn isActive(&self) -> bool {
        self.turnActive
    }

    /// Finalize the turn completely (TurnComplete).
    pub fn finishTurn(&mut self) {
        if !self.turnActive { return; }
        self.finalizeStreaming();
        self.turnActive = false;
        self.toolActive = false;
        self.toolStartTime = None;
        self.retryStatus = None;
    }

    /// Finalize streaming state after a cancellation.
    pub fn finalizeCancelled(&mut self) {
        if !self.turnActive { return; }
        self.finalizeStreaming();
        self.turnActive = false;
        self.toolActive = false;
        self.toolStartTime = None;
        self.retryStatus = None;
        self.activeSubagent = None;
        self.pendingPermit = false;
        self.pendingToolName.clear();
        // Mark any active subagent block as done.
        for entry in self.entries.iter_mut().rev() {
            if let PanelEntry::SubagentBlock { done, .. } = entry {
                if !*done {
                    *done = true;
                    break;
                }
            }
        }
        self.entries.push(PanelEntry::Cancelled);
    }

    pub fn showToolRequest(
        &mut self,
        name: &str,
        summary: &str,
        args: &str,
        diff: Option<String>,
        explanation: Option<String>,
        impact: construct::tool::ShellImpact,
    ) {
        if !self.turnActive { return; }
        self.finalizeStreaming();
        // Extract raw command for shell tools (used for code block preview).
        let command = construct::tool::parse(name, args).ok().and_then(|action| {
            match action {
                construct::tool::ToolAction::Shell { command, .. } => Some(command),
                _ => None,
            }
        });

        self.entries
            .push(PanelEntry::ToolRequest { summary: summary.into(), diff, command });
        self.pendingPermit = true;
        self.pendingToolName = name.into();
        self.pendingToolSummary = summary.into();
        self.pendingToolExplanation = explanation;
        self.pendingToolImpact = impact;
        // Generate pattern suggestions (re-parse — the action was consumed above).
        self.permitPatterns = construct::tool::parse(name, args)
            .map(|a| construct::permissions::suggestPatterns(&a))
            .unwrap_or_default();
        self.permitSelectedPattern = 0;
        // Pre-fill custom with the most specific pattern.
        self.permitCustomPattern = self.permitPatterns
            .first()
            .cloned()
            .unwrap_or_default();
        self.permitEditingCustom = false;
        self.permitCodeScrollX = 0;

        self.scrollOffset = 0;
    }

    pub fn approvePending(&mut self) {
        self.pendingPermit = false;
        let name = std::mem::take(&mut self.pendingToolName);
        self.permitPatterns.clear();
        self.permitEditingCustom = false;
        // Don't push ToolApproved for task — SubagentStarted renders the panel.
        if name != "task" {
            self.entries.push(PanelEntry::ToolApproved { name });
        }
    }

    pub fn denyPending(&mut self) {
        self.pendingPermit = false;
        let name = std::mem::take(&mut self.pendingToolName);
        self.permitPatterns.clear();
        self.permitEditingCustom = false;
        self.entries.push(PanelEntry::ToolDenied { name });
    }

    /// Get the currently selected "always allow" pattern.
    pub fn selectedPattern(&self) -> String {
        if self.permitSelectedPattern >= self.permitPatterns.len() {
            // Custom field selected.
            self.permitCustomPattern.clone()
        } else {
            self.permitPatterns.get(self.permitSelectedPattern)
                .cloned()
                .unwrap_or_default()
        }
    }

    /// Move selection to previous pattern.
    pub fn prevPattern(&mut self) {
        if self.permitSelectedPattern > 0 {
            self.permitSelectedPattern -= 1;
            self.permitEditingCustom = false;
        }
    }

    /// Move selection to next pattern (or to custom field).
    pub fn nextPattern(&mut self) {
        // Allow one past the end for the custom field.
        if self.permitSelectedPattern < self.permitPatterns.len() {
            self.permitSelectedPattern += 1;
            if self.permitSelectedPattern >= self.permitPatterns.len() {
                self.permitEditingCustom = true;
            }
        }
    }

    /// Toggle custom pattern editing.
    /// Get the pending shell command (if the active ToolRequest has one).
    pub fn pendingCommand(&self) -> Option<&String> {
        if !self.pendingPermit {
            return None;
        }
        self.entries.last().and_then(|e| match e {
            PanelEntry::ToolRequest { command, .. } => command.as_ref(),
            _ => None,
        })
    }

    /// Flash "copied" on the permission code block's top border.
    pub fn flashCopied(&mut self) {
        self.copiedFlash = Some(((usize::MAX, 0), std::time::Instant::now()));
    }

    /// Scroll the permission code block horizontally.
    pub fn scrollPermitCode(&mut self, delta: i16) {
        if delta < 0 {
            self.permitCodeScrollX = self.permitCodeScrollX.saturating_sub((-delta) as u16);
        } else {
            // Clamp to prevent scrolling past content.
            self.permitCodeScrollX = self.permitCodeScrollX.saturating_add(delta as u16);
        }
    }

    /// Whether the custom pattern field is being edited.
    pub fn isEditingCustom(&self) -> bool {
        self.permitEditingCustom
    }

    /// Insert a character into the custom pattern.
    pub fn customPatternInsert(&mut self, c: char) {
        self.permitCustomPattern.push(c);
    }

    /// Delete last character from the custom pattern.
    pub fn customPatternBackspace(&mut self) {
        self.permitCustomPattern.pop();
    }

    // -- Subagent lifecycle --

    pub fn subagentStarted(&mut self, sessionId: &str, agentType: &str, prompt: &str) {
        if !self.turnActive { return; }
        self.finalizeStreaming();
        // Remove the preceding ToolRequest for the task tool — the SubagentBlock replaces it.
        if let Some(PanelEntry::ToolRequest { .. }) = self.entries.last() {
            self.entries.pop();
        }
        self.activeSubagent = Some(ActiveSubagent {
            sessionId: sessionId.into(),
            agentType: agentType.into(),
            prompt: prompt.into(),
            startTime: Instant::now(),
            transcript: Vec::new(),
            shellScrollback: Vec::new(),
        });
        self.entries.push(PanelEntry::SubagentBlock {
            agentType: agentType.into(),
            prompt: prompt.into(),
            toolLines: Vec::new(),
            done: false,
            turns: 0,
            content: None,
            contentExpanded: false,
            sessionId: Some(sessionId.into()),
        });
        self.scrollOffset = 0;
    }

    pub fn subagentToolLine(&mut self, name: &str, summary: &str) {
        if !self.turnActive { return; }
        // Find the active SubagentBlock and append.
        if let Some(PanelEntry::SubagentBlock { toolLines, .. }) = self.entries.iter_mut().rev()
            .find(|e| matches!(e, PanelEntry::SubagentBlock { done: false, .. }))
        {
            toolLines.push((name.into(), summary.into()));
        }
        if let Some(ref mut sub) = self.activeSubagent {
            sub.transcript.push(PanelEntry::ToolApproved {
                name: format!("{name}: {summary}"),
            });
        }
        self.scrollOffset = 0;
    }

    /// Push a full tool result to the subagent's transcript (for the overlay).
    pub fn subagentToolResult(&mut self, name: &str, output: &str) {
        if let Some(ref mut sub) = self.activeSubagent {
            sub.transcript.push(PanelEntry::ToolResult {
                name: name.into(),
                output: output.into(),
            });
        }
    }

    pub fn subagentContent(&mut self, text: &str) {
        // Content goes to transcript for the overlay.
        if let Some(ref mut sub) = self.activeSubagent {
            // Append to existing Assistant entry or create new one.
            if let Some(PanelEntry::Assistant(existing)) = sub.transcript.last_mut() {
                existing.push_str(text);
            } else {
                sub.transcript.push(PanelEntry::Assistant(text.into()));
            }
        }
    }

    pub fn subagentComplete(&mut self, agentType: &str, turns: usize, finalContent: &str) {
        if !self.turnActive { return; }
        if let Some(PanelEntry::SubagentBlock { done, turns: t, content, contentExpanded, .. }) = self.entries.iter_mut().rev()
            .find(|e| matches!(e, PanelEntry::SubagentBlock { done: false, .. }))
        {
            *done = true;
            *t = turns;
            if !finalContent.is_empty() {
                *content = Some(finalContent.into());
            }
        }
    }

    /// Toggle the content section of a subagent block (expand/collapse).
    pub fn toggleSubagentContent(&mut self, entryIndex: usize) {
        if let Some(PanelEntry::SubagentBlock { contentExpanded, done: true, .. }) = self.entries.get_mut(entryIndex) {
            *contentExpanded = !*contentExpanded;
        }
    }

    /// Try toggling subagent content expand/collapse at the given grid line.
    /// Returns true if the click was handled.
    pub fn tryToggleSubagentContent(&mut self, gridLine: i32) -> bool {
        if let Some((visualLine, entryIdx)) = self.lastSubagentToggleLine.get() {
            let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
            let clickVisual = (gridLine + maxScroll) as usize;
            if clickVisual == visualLine {
                self.toggleSubagentContent(entryIdx);
                return true;
            }
        }
        false
    }

    /// Check if a grid line (scroll-adjusted) is a subagent header (for click-to-view).
    pub fn isSubagentHeaderLine(&self, gridLine: i32) -> bool {
        if let Some(headerLine) = self.lastSubagentHeaderLine.get() {
            let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
            let visualLine = (gridLine + maxScroll) as usize;
            headerLine == visualLine
        } else {
            false
        }
    }

    /// Find the last SubagentBlock's session info (for lazy-loading the overlay).
    pub fn lastSubagentSession(&self) -> Option<(&str, &str)> {
        self.entries.iter().rev()
            .find_map(|e| match e {
                PanelEntry::SubagentBlock { agentType, sessionId: Some(sid), .. } => {
                    Some((agentType.as_str(), sid.as_str()))
                }
                _ => None,
            })
    }

    pub fn toolApproved(&mut self, name: &str) {
        if !self.turnActive { return; }
        self.finalizeStreaming();
        self.entries
            .push(PanelEntry::ToolApproved { name: name.into() });
    }

    pub fn toolStarted(&mut self, name: &str, summary: &str) {
        if !self.turnActive { return; }
        self.finalizeStreaming();
        self.toolActive = true;
        self.toolStartTime = Some(Instant::now());
        self.entries.push(PanelEntry::ToolActive {
            name: name.into(),
            summary: summary.into(),
        });
        self.scrollOffset = 0;
    }

    pub fn toolDenied(&mut self, name: &str) {
        if !self.turnActive { return; }
        self.toolActive = false;
        self.toolStartTime = None;
        self.entries
            .push(PanelEntry::ToolDenied { name: name.into() });
    }

    pub fn toolAutoDenied(&mut self, name: &str, summary: &str) {
        if !self.turnActive { return; }
        self.toolActive = false;
        self.toolStartTime = None;
        self.entries
            .push(PanelEntry::ToolAutoDenied { name: name.into(), summary: summary.into() });
    }

    pub fn pushToolResult(&mut self, name: &str, output: &str) {
        if !self.turnActive { return; }
        self.toolActive = false;
        self.toolStartTime = None;
        // Replace the ToolActive entry with the result.
        if let Some(last) = self.entries.last() {
            if matches!(last, PanelEntry::ToolActive { .. }) {
                self.entries.pop();
            }
        }
        // Commit any in-flight streaming before the tool result.
        self.finalizeStreaming();
        self.entries.push(PanelEntry::ToolResult {
            name: name.into(),
            output: output.into(),
        });
        self.scrollOffset = 0;
        // Restart thinking indicator — model will be called again after tool results.
        self.isStreaming = true;
        self.thinkingStartTime = Some(Instant::now());
    }

    /// Show a transient retry indicator in the throbber area.
    pub fn showRetrying(&mut self, attempt: u32, maxAttempts: u32) {
        self.retryStatus = Some(format!("retrying ({attempt}/{maxAttempts})"));
        // Reset thinking timer so elapsed restarts from the retry.
        self.thinkingStartTime = Some(Instant::now());
    }

    pub fn pushError(&mut self, msg: &str) {
        self.entries.push(PanelEntry::Error(msg.into()));
    }

    /// Push a command result and reset turn state.
    ///
    /// Called after pushUser() for slash commands, so we undo the
    /// turnActive/isStreaming flags that pushUser sets.
    pub fn pushCommandResult(&mut self, text: &str) {
        self.entries.push(PanelEntry::CommandResult(text.into()));
        self.turnActive = false;
        self.isStreaming = false;
        self.thinkingStartTime = None;
        self.scrollOffset = 0;
    }

    /// Insert a compaction marker at the block where compressed content begins.
    ///
    /// `blockIdx` is the 0-based block index (each block starts with a User
    /// entry). Removes any existing marker for the same stage first, since the
    /// zone may expand across repeated compaction runs.
    pub fn pushCompactionMarker(&mut self, stage: &str, blockIdx: usize) {
        // Remove existing marker for this stage.
        self.entries.retain(|e| {
            !matches!(e, PanelEntry::CompactionMarker { stage: s } if s == stage)
        });

        // Find the insert position by counting User entries.
        let mut userCount = 0;
        let mut insertAt = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            if matches!(entry, PanelEntry::User(_)) {
                if userCount == blockIdx {
                    insertAt = i;
                    break;
                }
                userCount += 1;
            }
        }

        self.entries.insert(insertAt, PanelEntry::CompactionMarker { stage: stage.into() });
    }

    // --- Completion ---

    /// Update completion state based on current input text.
    ///
    /// Activates when text starts with `/` and has no spaces (still
    /// typing the command name). Dismisses otherwise.
    pub fn updateCompletion(&mut self, text: &str) {
        let trimmed = text.trim_start();
        if !trimmed.starts_with('/') || trimmed.contains(' ') {
            self.completion = None;
            return;
        }

        let prefix = &trimmed[1..];
        let matches = command::completions(prefix);
        if matches.is_empty() {
            self.completion = None;
            return;
        }

        // Preserve selection index if still valid.
        let prevSelected = self.completion.as_ref().map(|c| c.selected).unwrap_or(0);
        let selected = if prevSelected < matches.len() { prevSelected } else { 0 };
        self.completion = Some(CompletionState { matches, selected });
    }

    /// Accept the currently selected completion.
    ///
    /// Returns:
    ///     Option<String>: The full `/commandname ` string, or None if no completion active.
    pub fn completeSelected(&self) -> Option<String> {
        let state = self.completion.as_ref()?;
        let cmd = state.matches.get(state.selected)?;
        Some(format!("/{} ", cmd.name))
    }

    /// Move selection down in the completion menu.
    pub fn selectNext(&mut self) {
        if let Some(state) = &mut self.completion {
            if state.selected + 1 < state.matches.len() {
                state.selected += 1;
            }
        }
    }

    /// Move selection up in the completion menu.
    pub fn selectPrev(&mut self) {
        if let Some(state) = &mut self.completion {
            state.selected = state.selected.saturating_sub(1);
        }
    }

    /// Dismiss the completion menu.
    pub fn dismissCompletion(&mut self) {
        self.completion = None;
    }

    /// Whether completion is currently active.
    pub fn completionActive(&self) -> bool {
        self.completion.is_some()
    }

    pub fn scrollUp(&mut self, amount: u16) {
        self.scrollOffset = self.scrollOffset.saturating_add(amount);
    }

    pub fn scrollDown(&mut self, amount: u16) {
        self.scrollOffset = self.scrollOffset.saturating_sub(amount);
    }

    /// Scroll offset from the bottom (analogous to terminal displayOffset).
    pub fn displayOffset(&self) -> u16 {
        self.scrollOffset
    }

    /// Scroll a code block horizontally.
    pub fn scrollCodeBlockH(&mut self, blockId: (usize, usize), delta: i16) {
        let range = match self.lastCodeBlockRanges.iter().find(|r| r.blockId == blockId) {
            Some(r) => r,
            None => return,
        };
        let maxScroll = range.maxContentWidth.saturating_sub(range.innerWidth) as u16;
        if maxScroll == 0 {
            return;
        }
        let current = self.codeScrollX.get(&blockId).copied().unwrap_or(0);
        let new = if delta > 0 {
            current.saturating_add(delta as u16).min(maxScroll)
        } else {
            current.saturating_sub((-delta) as u16)
        };
        if new == 0 {
            self.codeScrollX.remove(&blockId);
        } else {
            self.codeScrollX.insert(blockId, new);
        }
    }

    /// Find the code block at a given grid line (for mouse hit-testing).
    pub fn codeBlockAtGridLine(&self, gridLine: i32) -> Option<(usize, usize)> {
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as usize;
        for range in &self.lastCodeBlockRanges {
            if visualLine >= range.startLine && visualLine <= range.endLine {
                return Some(range.blockId);
            }
        }
        None
    }

    /// Try to copy a code block's content via the "copy" label hit-test.
    ///
    /// Args:
    ///     gridLine: Grid line of the click.
    ///     col: Column relative to agentContentRect (prefix excluded).
    ///
    /// Returns true if a copy occurred (caller should skip selection).
    pub fn tryCopyCodeBlock(&mut self, gridLine: i32, col: u16) -> bool {
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as usize;
        for range in &self.lastCodeBlockRanges {
            // Only match the top border line.
            if visualLine != range.startLine {
                continue;
            }
            // The stored copyLabelCol includes the prefix width.
            // Content-local col needs the prefix added back for comparison.
            let contentCol = col + 2;
            if contentCol >= range.copyLabelCol {
                selection::copyToClipboard(&range.rawCode);
                self.copiedFlash = Some((range.blockId, Instant::now()));
                return true;
            }
        }
        false
    }

    /// Toggle a code block between collapsed and expanded.
    ///
    /// Expand: click on top border (startLine) of a collapsed block.
    /// Collapse: click on top border or bottom border of an expanded block.
    /// Compensates scroll offset so the header stays at its current screen position.
    ///
    /// Returns true if a toggle occurred (caller should skip selection).
    pub fn tryToggleCodeBlock(&mut self, gridLine: i32) -> bool {
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as usize;
        for range in &self.lastCodeBlockRanges {
            if !range.collapsible {
                continue;
            }
            let isExpanded = self.codeExpanded.contains(&range.blockId);
            let hiddenLines = range.totalLines.saturating_sub(MAX_CODE_BLOCK_LINES) as u16;

            if !isExpanded && visualLine == range.startLine {
                self.codeExpanded.insert(range.blockId);
                self.scrollOffset = self.scrollOffset.saturating_add(hiddenLines);
                self.lastMaxScroll = self.lastMaxScroll.saturating_add(hiddenLines);
                return true;
            }
            if isExpanded && (visualLine == range.startLine || visualLine == range.endLine) {
                self.codeExpanded.remove(&range.blockId);
                self.scrollOffset = self.scrollOffset.saturating_sub(hiddenLines);
                self.lastMaxScroll = self.lastMaxScroll.saturating_sub(hiddenLines);
                return true;
            }
        }
        false
    }

    /// Advance the throbber animation on a wall-clock schedule (~8 FPS).
    /// Returns true if the throbber actually ticked (caller needs redraw).
    pub fn tickThrobber(&mut self) -> bool {
        let waiting = self.isStreaming
            && self.streamingContent.is_empty()
            && self.pendingContent.is_empty()
            && self.streamingReasoning.is_empty();
        let subagentRunning = self.activeSubagent.is_some()
            && !self.entries.iter().rev().any(|e| matches!(e, PanelEntry::SubagentBlock { done: true, .. }));
        if !(waiting || self.reasoningActive || self.toolActive || subagentRunning) {
            return false;
        }
        let now = Instant::now();
        if now.duration_since(self.lastThrobberTick) >= Duration::from_millis(125) {
            self.lastThrobberTick = now;
            self.throbber.tick();
            true
        } else {
            false
        }
    }

    /// Advance the character reveal animation. Each character fades in fully
    /// before the next one begins. Speed is calculated directly from the
    /// stream's throughput so the reveal matches the actual rate of arrival.
    /// Returns true if visual state changed (caller needs redraw).
    pub fn tickReveal(&mut self) -> bool {
        if self.pendingContent.is_empty() && self.fadeProgress >= 1.0 {
            // If reveal just finished and finalization was deferred, do it now.
            if self.deferredFinalize {
                self.doFinalize();
                return true;
            }
            return false;
        }

        // Spool-up delay: let the buffer fill for 150ms before starting reveal.
        const SPOOL_DELAY: Duration = Duration::from_millis(150);
        if let Some(start) = self.revealStart {
            if start.elapsed() < SPOOL_DELAY {
                return false;
            }
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.lastRevealTick).as_secs_f32();
        self.lastRevealTick = now;
        // Guard against stale timestamps producing a huge first tick.
        let elapsed = elapsed.min(0.05);

        // Pure calculated speed: total chars received / time elapsed.
        // During streaming, use wall clock so the rate naturally dips during
        // token gaps (prevents draining the buffer then stuttering).
        // After the stream ends, freeze to the stream's own duration so the
        // remaining buffer drains at the established pace.
        let start = self.revealStart.unwrap_or(now);
        let denom = if self.deferredFinalize {
            self.lastReceiveTime.duration_since(start).as_secs_f32()
        } else {
            now.duration_since(start).as_secs_f32()
        };
        let fadesPerSec = (self.totalCharsReceived as f32 / denom.max(0.1)).max(15.0);

        self.fadeProgress += elapsed * fadesPerSec;

        // Each time fadeProgress crosses 1.0, that char is fully revealed.
        while self.fadeProgress >= 1.0 && !self.pendingContent.is_empty() {
            let firstLen = self.pendingContent
                .chars().next().unwrap().len_utf8();
            let ch: String = self.pendingContent.drain(..firstLen).collect();
            self.streamingContent.push_str(&ch);
            self.fadeProgress -= 1.0;
        }

        // Nothing left — clamp.
        if self.pendingContent.is_empty() {
            self.fadeProgress = self.fadeProgress.max(1.0);
        }

        true
    }

    /// Build the display string for streaming content: fully revealed text
    /// plus the currently-fading character (if any).
    fn displayContent(&self) -> String {
        if self.pendingContent.is_empty() || self.fadeProgress <= 0.0 {
            return self.streamingContent.clone();
        }
        // Include the next pending char that's mid-fade.
        let mut s = self.streamingContent.clone();
        if let Some(ch) = self.pendingContent.chars().next() {
            s.push(ch);
        }
        s
    }

    /// Toggle the most recent reasoning block (streaming or finalized).
    pub fn toggleThinking(&mut self) {
        if self.isStreaming && !self.streamingReasoning.is_empty() {
            self.thinkingExpanded = !self.thinkingExpanded;
        } else {
            for entry in self.entries.iter_mut().rev() {
                if let PanelEntry::Reasoning { expanded, .. } = entry {
                    *expanded = !*expanded;
                    break;
                }
            }
        }
    }

    /// Toggle a reasoning block if the given grid line is its header.
    ///
    /// Returns true if a toggle occurred (caller should skip selection).
    pub fn toggleReasoningAtGridLine(&mut self, gridLine: i32) -> bool {
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as usize;
        let w = self.lastChatWidth.max(1);

        for &(entryIdx, lineIdx) in &self.lastReasoningHeaders {
            if lineIdx == visualLine {
                match entryIdx {
                    Some(idx) => {
                        if let PanelEntry::Reasoning { text, expanded } = &mut self.entries[idx] {
                            let delta = countReasoningLines(text, w);
                            *expanded = !*expanded;
                            if *expanded {
                                self.scrollOffset = self.scrollOffset.saturating_add(delta);
                                // Preempt streaming compensation so it doesn't double-adjust.
                                self.lastMaxScroll = self.lastMaxScroll.saturating_add(delta);
                            } else {
                                self.scrollOffset = self.scrollOffset.saturating_sub(delta);
                                self.lastMaxScroll = self.lastMaxScroll.saturating_sub(delta);
                            }
                        }
                    }
                    None => {
                        let delta = countReasoningLines(&self.streamingReasoning, w);
                        self.thinkingExpanded = !self.thinkingExpanded;
                        if self.thinkingExpanded {
                            self.scrollOffset = self.scrollOffset.saturating_add(delta);
                            self.lastMaxScroll = self.lastMaxScroll.saturating_add(delta);
                        } else {
                            self.scrollOffset = self.scrollOffset.saturating_sub(delta);
                            self.lastMaxScroll = self.lastMaxScroll.saturating_sub(delta);
                        }
                    }
                }
                return true;
            }
        }

        false
    }

    /// Extract text from the agent panel selection, rejoining wrapped lines.
    ///
    /// Uses the continuation map to detect lines added by word-wrapping
    /// and joins them back together so the clipboard gets unwrapped text.
    /// For code blocks, extracts full untruncated content from stored spans
    /// rather than from the (potentially scrolled/clipped) Buffer.
    pub fn extractUnwrappedText(
        &self,
        sel: &Selection,
        area: Rect,
        buf: &Buffer,
        displayOffset: u16,
    ) -> String {
        if sel.isEmpty() {
            return String::new();
        }

        let ((sc, sr), (ec, er)) = sel.sorted();
        let maxScroll = (self.lastScrollY + self.scrollOffset) as i32;
        // Display column offset for the 2-char prefix that's excluded from content rect.
        let prefixCols: u16 = 2;

        let mut segments: Vec<(String, bool)> = Vec::new();

        for gridLine in sr..=er {
            let visualIdx = (gridLine as i32 + maxScroll) as usize;
            let colStart = if gridLine == sr { sc } else { 0 };
            let colEnd = if gridLine == er { ec } else { area.width };

            // Non-copyable lines (session notices) produce empty text.
            if self.nonCopyableLines.contains(&visualIdx) {
                segments.push((String::new(), false));
                continue;
            }

            // Check if this visual line falls within a code block.
            let codeBlockHit = self.lastCodeBlockRanges.iter().find(|r| {
                visualIdx >= r.startLine && visualIdx <= r.endLine
            });

            if let Some(range) = codeBlockHit {
                // Inside a code block — extract from stored content, not Buffer.
                let isBorder = visualIdx == range.startLine || visualIdx == range.endLine;
                if isBorder {
                    // Skip border lines entirely.
                    segments.push((String::new(), false));
                } else {
                    // Content line: index into stored contentLines.
                    let contentIdx = visualIdx - range.startLine - 1;
                    let text = if contentIdx < range.contentLines.len() {
                        range.contentLines[contentIdx]
                            .iter()
                            .map(|s| s.content.as_ref())
                            .collect::<String>()
                    } else {
                        String::new()
                    };
                    segments.push((text, false));
                }
            } else {
                let text = if let Some(screenRow) =
                    selection::toScreenRow(gridLine, displayOffset, area.height)
                {
                    // Visible — read from Buffer.
                    let mut line = String::new();
                    for col in colStart..colEnd {
                        if col >= area.width {
                            break;
                        }
                        if let Some(cell) = buf.cell((area.x + col, area.y + screenRow)) {
                            line.push_str(cell.symbol());
                        }
                    }
                    line.trim_end().to_string()
                } else if visualIdx < self.lastLineTexts.len() {
                    // Off-screen — extract from cached line text.
                    sliceByDisplayColumn(
                        &self.lastLineTexts[visualIdx],
                        prefixCols + colStart,
                        prefixCols + colEnd,
                    )
                } else {
                    String::new()
                };

                let isCont = visualIdx < self.lastContinuationMap.len()
                    && self.lastContinuationMap[visualIdx];

                segments.push((text, isCont));
            }
        }

        // Remove trailing empty lines.
        while segments.last().is_some_and(|(l, _)| l.is_empty()) {
            segments.pop();
        }

        // Join lines, merging wrap continuations.
        let mut result = String::new();
        for (i, (line, isCont)) in segments.iter().enumerate() {
            if i > 0 {
                if *isCont {
                    // Continuation from word-wrapping — join without newline.
                    if !result.ends_with(' ') && !result.is_empty() && !line.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(line);
                    continue;
                } else {
                    result.push('\n');
                }
            }
            result.push_str(line);
        }

        result
    }

    /// Render the panel. Returns the chat content area Rect.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer, focused: bool) -> Rect {
        let borderStyle = if focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" agent ");

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 3 {
            return Rect::default();
        }

        // Dynamic input height based on content.
        // Width accounts for: 2 right padding + 2 prompt prefix.
        let inputHeight = if self.pendingPermit {
            // code block (top border + content, no bottom) + header + explanation + blank + 2 keys + blank + patterns + custom.
            let hasCmd = self.pendingCommand().is_some();
            let codeBlockLines: u16 = if hasCmd { 2 } else { 0 };
            // Header is in the input area only for shell (code block present).
            // Non-shell header is in the separator slot.
            let headerLine: u16 = if hasCmd { 1 } else { 0 };
            let availW = inner.width.saturating_sub(6) as usize;
            let explanationLines: u16 = self.pendingToolExplanation.as_ref().map_or(0, |e| {
                if availW == 0 { 1 } else { (e.len() / availW + 1) as u16 }
            });
            let patternLines = self.permitPatterns.len() as u16 + 1; // +1 for custom field.
            // code block + header + explanation + blank + 2 keys + blank + patterns.
            (codeBlockLines + headerLine + explanationLines + 1 + 2 + 1 + patternLines)
                .min(inner.height * 2 / 3).max(5)
        } else {
            let baseHeight = self.textArea.desiredHeight(inner.width.saturating_sub(4)).min(8).max(1);
            // Add space for attachment bar: entries + separator.
            let attLines = if self.attachments.is_empty() { 0 } else { self.attachments.len() as u16 + 1 };
            baseHeight + attLines
        };

        // Split: chat area + separator + input.
        let chunks = Layout::default()
            .constraints([
                Constraint::Min(1),
                Constraint::Length(1),
                Constraint::Length(inputHeight),
            ])
            .split(inner);
        let chatArea = chunks[0];
        let separatorArea = chunks[1];
        let inputArea = chunks[2];

        // Separator. Non-shell permit: header in separator slot. Shell permit: hidden
        // (header is in input area after the code block). Normal: plain separator.
        if self.pendingPermit && self.pendingCommand().is_none() {
            let mut headerLines: Vec<Line> = Vec::new();
            self.renderPermitHeaderInline(separatorArea.width, &mut headerLines);
            if let Some(line) = headerLines.into_iter().next() {
                Paragraph::new(line).render(separatorArea, buf);
            }
        } else if !self.pendingPermit {
            let sep = "\u{2500}".repeat(separatorArea.width as usize);
            Paragraph::new(sep)
                .style(Style::default().fg(Color::DarkGray))
                .render(separatorArea, buf);
        }

        let inputRightPad: u16 = if self.pendingPermit { 1 } else { 2 };
        let paddedInput = Rect {
            x: inputArea.x,
            y: inputArea.y,
            width: inputArea.width.saturating_sub(inputRightPad),
            height: inputArea.height,
        };
        self.lastInputRect = paddedInput;
        self.renderInput(paddedInput, buf, focused);

        // Right padding so content doesn't touch border.
        let paddedChat = Rect {
            x: chatArea.x,
            y: chatArea.y,
            width: chatArea.width.saturating_sub(2),
            height: chatArea.height,
        };

        // Build all display lines (pre-wrapped to fit paddedChat width).
        let (lines, contMap, reasoningHeaders, cbRanges, nonCopyable) = self.buildLines(paddedChat.width);
        self.lastContinuationMap = contMap;
        self.lastReasoningHeaders = reasoningHeaders;
        self.lastCodeBlockRanges = cbRanges;
        // Clear stale copied flash after 2 seconds.
        if self.copiedFlash.as_ref().is_some_and(|(_, t)| t.elapsed().as_secs() >= 2) {
            self.copiedFlash = None;
        }
        // Cache plain text of each visual line for scrollback copy.
        self.lastLineTexts = lines.iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        self.nonCopyableLines = nonCopyable;

        // buildLines already wraps text to fit paddedChat.width — don't let
        // Paragraph re-wrap, as that shifts visual line indices and breaks
        // click targets (code block scroll/copy, subagent toggle, etc.).
        let totalLines = lines.len() as u16;
        let paragraph = Paragraph::new(lines);
        let totalWrapped = totalLines;
        let visible = paddedChat.height;
        let maxScroll = totalWrapped.saturating_sub(visible);
        // Keep view pinned when scrolled up and new content arrives.
        if self.scrollOffset > 0 && maxScroll > self.lastMaxScroll {
            self.scrollOffset = self.scrollOffset.saturating_add(maxScroll - self.lastMaxScroll);
        }
        self.lastMaxScroll = maxScroll;
        // Clamp to prevent scroll accumulation past content top.
        self.scrollOffset = self.scrollOffset.min(maxScroll);
        let scrollY = maxScroll.saturating_sub(self.scrollOffset);

        self.lastScrollY = scrollY;
        self.lastChatWidth = paddedChat.width;

        paragraph
            .scroll((scrollY, 0))
            .render(paddedChat, buf);

        // Completion dropdown overlays the chat area above the input.
        self.renderCompletionDropdown(paddedInput, buf);

        paddedChat
    }

    /// Render just the chat transcript area without input, separator, or border.
    /// Used by the subagent overlay panel for read-only transcript display.
    pub fn renderChatOnly(&mut self, area: Rect, buf: &mut Buffer) {
        let padded = Rect {
            x: area.x,
            y: area.y,
            width: area.width.saturating_sub(2),
            height: area.height,
        };
        if padded.width < 4 || padded.height < 1 { return; }

        let (chatLines, contMap, reasoningHeaders, cbRanges, nonCopyable) =
            self.buildLines(padded.width);
        self.lastContinuationMap = contMap;
        self.lastReasoningHeaders = reasoningHeaders;
        self.lastCodeBlockRanges = cbRanges;
        self.lastLineTexts = chatLines.iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        self.nonCopyableLines = nonCopyable;

        let totalLines = chatLines.len() as u16;
        let paragraph = Paragraph::new(chatLines);
        let totalWrapped = totalLines;
        let visible = padded.height;
        let maxScroll = totalWrapped.saturating_sub(visible);
        self.scrollOffset = self.scrollOffset.min(maxScroll);
        let scrollY = maxScroll.saturating_sub(self.scrollOffset);
        self.lastScrollY = scrollY;
        self.lastChatWidth = padded.width;

        paragraph
            .scroll((scrollY, 0))
            .render(padded, buf);
    }



    /// Build the permission header line into a Vec<Line> (for renderInput).
    fn renderPermitHeaderInline(&self, w: u16, lines: &mut Vec<Line<'static>>) {
        use unicode_width::UnicodeWidthStr;
        let w = w as usize;
        let (impactColor, impactSymbol) = match self.pendingToolImpact {
            construct::tool::ShellImpact::Delete => (Color::Red, "\u{2620}\u{FE0E}"),
            construct::tool::ShellImpact::MajorMod => (Color::Rgb(200, 140, 40), "\u{26A0}\u{FE0E}"),
            construct::tool::ShellImpact::MinorMod => (Color::Rgb(180, 160, 80), "\u{2691}\u{FE0E}"),
            construct::tool::ShellImpact::Read => (Color::Rgb(80, 160, 200), "\u{2315}"),
        };
        let toolLabel = format!(" {} {} ", self.pendingToolName, impactSymbol);

        // Timeout from summary.
        let timeoutLabel = if let Some(start) = self.pendingToolSummary.find('(') {
            if let Some(end) = self.pendingToolSummary.find("s):") {
                let secs = &self.pendingToolSummary[start + 1..end];
                if secs.parse::<u64>().is_ok() {
                    format!(" \u{23F2}\u{FE0E} {secs}s ")
                } else { String::new() }
            } else { String::new() }
        } else { String::new() };

        let spanWidth = |spans: &[Span]| -> usize {
            spans.iter().map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum()
        };

        let mut spans: Vec<Span> = Vec::new();
        let hasBlock = self.pendingCommand().is_some();

        // Left ┴ at column 1 (aligns with code block left │, shifted right by margin).
        if hasBlock {
            spans.push(Span::styled("\u{2500}\u{2534}", Style::default().fg(Color::DarkGray)));
        }

        // Tool name + symbol.
        spans.push(Span::styled(
            format!("\u{2500}{toolLabel}"),
            Style::default().fg(impactColor).add_modifier(Modifier::BOLD),
        ));

        // Fill between tool label and timeout.
        let used = spanWidth(&spans);
        let timeoutW = UnicodeWidthStr::width(timeoutLabel.as_str());
        let rightW: usize = if hasBlock { 1 } else { 0 }; // ┴
        let fill = w.saturating_sub(used + timeoutW + rightW);
        spans.push(Span::styled("\u{2500}".repeat(fill), Style::default().fg(Color::DarkGray)));

        // Timeout.
        if !timeoutLabel.is_empty() {
            spans.push(Span::styled(timeoutLabel, Style::default().fg(Color::DarkGray)));
        }

        // Right ┴ at last column (aligns with code block right │).
        if hasBlock {
            spans.push(Span::styled("\u{2534}", Style::default().fg(Color::DarkGray)));
        }

        lines.push(Line::from(spans));
    }

    fn renderInput(&mut self, area: Rect, buf: &mut Buffer, focused: bool) {
        if self.pendingPermit {
            let mut lines: Vec<Line> = Vec::new();

            // Code block: render command preview — just the content lines with │ borders,
            // no top/bottom border (header line below serves as the bottom).
            if let Some(cmd) = self.pendingCommand() {
                let codeLines = vec![vec![Span::raw(cmd.clone())]];
                // Width minus 1 for the left margin shift.
                let codeWidth = area.width.saturating_sub(1);
                let mcw = crate::markdown::highlight::maxContentWidth(&codeLines);
                let innerWidth = codeWidth.saturating_sub(2) as usize;
                // Clamp scroll to prevent overflow.
                let maxScroll = mcw.saturating_sub(innerWidth) as u16;
                if self.permitCodeScrollX > maxScroll {
                    self.permitCodeScrollX = maxScroll;
                }
                let mut codeBlock = crate::markdown::highlight::renderCodeBlock(
                    &codeLines,
                    Some("sh"),
                    codeWidth,
                    self.permitCodeScrollX,
                    mcw,
                    self.copiedFlash.as_ref().is_some_and(|(bid, t)| bid.0 == usize::MAX && t.elapsed().as_secs() < 2),
                    None,
                    None,
                );
                // Remove bottom border — the header line replaces it.
                if codeBlock.len() > 1 {
                    codeBlock.pop();
                }
                // Shift right by 1 char (left margin).
                for codeLine in codeBlock {
                    let mut spans = vec![Span::raw(" ")];
                    spans.extend(codeLine.spans);
                    lines.push(Line::from(spans));
                }
            }

            // Header line: only in input area for shell (with code block).
            // Non-shell header is rendered in the separator slot.
            if self.pendingCommand().is_some() {
                self.renderPermitHeaderInline(area.width, &mut lines);
            }

            // Explanation (shell commands only), wrapped to fit.
            if let Some(ref explanation) = self.pendingToolExplanation {
                let style = Style::default().fg(Color::DarkGray);
                let maxW = area.width.saturating_sub(4) as usize;
                let spanLine = Line::from(Span::styled(explanation.clone(), style));
                let wrapped = wrapSpannedLine(spanLine, maxW);
                for wLine in wrapped {
                    let mut spans = vec![Span::styled("  ", style)];
                    spans.extend(wLine.spans);
                    lines.push(Line::from(spans));
                }
                lines.push(Line::from(""));
            }

            // 2x2 key grid: allow/deny on rows, once/always on columns.
            lines.push(Line::from(vec![
                Span::styled("  [y]", Style::default().fg(Color::Green)),
                Span::styled(" allow  ", Style::default().fg(Color::DarkGray)),
                Span::styled("[A]", Style::default().fg(Color::Rgb(80, 80, 120))),
                Span::styled(" always allow", Style::default().fg(Color::Rgb(80, 80, 80))),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  [n]", Style::default().fg(Color::Red)),
                Span::styled(" deny   ", Style::default().fg(Color::DarkGray)),
                Span::styled("[D]", Style::default().fg(Color::Rgb(120, 80, 80))),
                Span::styled(" always deny", Style::default().fg(Color::Rgb(80, 80, 80))),
            ]));

            // Blank line between keys and scope patterns.
            lines.push(Line::from(""));

            // Pattern choices (scope for [A]/[D]).
            for (i, pattern) in self.permitPatterns.iter().enumerate() {
                let selected = i == self.permitSelectedPattern && !self.permitEditingCustom;
                let marker = if selected { " \u{25b8}" } else { "  " };
                let style = if selected {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                lines.push(Line::from(vec![
                    Span::styled(marker, style),
                    Span::styled(pattern.clone(), style),
                ]));
            }

            // Custom field — auto-editable when selected.
            let customSelected = self.permitSelectedPattern >= self.permitPatterns.len();
            let customMarker = if customSelected { " \u{25b8}" } else { "  " };
            let customStyle = if customSelected {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            let cursor = if customSelected { "\u{2502}" } else { "" };
            lines.push(Line::from(vec![
                Span::styled(customMarker, customStyle),
                Span::styled("custom: ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{}{cursor}", self.permitCustomPattern), customStyle),
            ]));

            // Track the header line row for gap fill.
            let headerRow = if self.pendingCommand().is_some() {
                // Code block top border + content lines (1 each) = 2, header is line index 2.
                Some(area.y + 2)
            } else {
                None
            };

            Paragraph::new(lines).render(area, buf);

            // Fill the 1-col gap between the header line and the panel border.
            if let Some(row) = headerRow {
                let gapCol = area.x + area.width;
                if let Some(cell) = buf.cell_mut((gapCol, row)) {
                    cell.set_char('\u{2500}');
                    cell.set_style(Style::default().fg(Color::DarkGray));
                }
            }
        } else {
            // Attachment bar — shown above text area when images are queued.
            let attCount = self.attachments.len() as u16;
            let inputArea = if attCount > 0 {
                // Entries + separator line. Ensure at least 1 line for text area.
                let barHeight = (attCount + 1).min(area.height.saturating_sub(1));
                let attRect = Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: barHeight,
                };
                let inputRect = Rect {
                    x: area.x,
                    y: area.y + barHeight,
                    width: area.width,
                    height: area.height.saturating_sub(barHeight),
                };

                let labelStyle = Style::default().fg(Color::White);
                let dimStyle = Style::default().fg(Color::DarkGray);

                let mut lines: Vec<Line> = Vec::new();

                // One line per attachment with size info.
                for (i, att) in self.attachments.iter().enumerate() {
                    let sizeStr = formatBytes(att.data.len());
                    let mut spans = vec![
                        Span::styled(" \u{2398}\u{FE0E} ", labelStyle),
                        Span::styled(&att.label, labelStyle),
                        Span::styled(format!("  {sizeStr}"), dimStyle),
                    ];
                    // Show Ctrl+D hint on the last attachment line.
                    if i == self.attachments.len() - 1 {
                        let usedLen = att.label.len() + sizeStr.len() + 6; // " ⎘︎ " + "  "
                        let pad = (area.width as usize).saturating_sub(usedLen + 8);
                        spans.push(Span::raw(" ".repeat(pad)));
                        spans.push(Span::styled("[Ctrl+D]", dimStyle));
                    }
                    lines.push(Line::from(spans));
                }

                // Render attachment entries.
                let entriesRect = Rect {
                    x: attRect.x,
                    y: attRect.y,
                    width: attRect.width,
                    height: attCount,
                };
                Paragraph::new(lines).render(entriesRect, buf);

                // Separator — rendered at full inner width (undo the 2-char right padding
                // so it matches the panel's own chat/input separator).
                let sepRect = Rect {
                    x: attRect.x,
                    y: attRect.y + attCount,
                    width: attRect.width + 2,
                    height: 1,
                };
                let sep = "\u{2500}".repeat(sepRect.width as usize);
                Paragraph::new(sep)
                    .style(Style::default().fg(Color::DarkGray))
                    .render(sepRect, buf);

                inputRect
            } else {
                area
            };

            self.textArea.render(inputArea, buf, focused);
            // Ghost text overlay for completion.
            if let Some(state) = &self.completion {
                if let Some(cmd) = state.matches.get(state.selected) {
                    let typed = self.textArea.text().trim_start();
                    let prefix = if typed.starts_with('/') { &typed[1..] } else { "" };
                    if cmd.name.starts_with(prefix) && cmd.name.len() > prefix.len() {
                        let suffix = &cmd.name[prefix.len()..];
                        // Render ghost text at cursor position.
                        if let Some((cx, cy)) = self.textArea.cursorScreenPos {
                            let ghostStyle = Style::default().fg(Color::DarkGray);
                            for (i, ch) in suffix.chars().enumerate() {
                                let col = cx + i as u16;
                                if col < area.x + area.width {
                                    if let Some(cell) = buf.cell_mut((col, cy)) {
                                        cell.set_char(ch);
                                        cell.set_style(ghostStyle);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Render the completion dropdown above the input area.
    fn renderCompletionDropdown(&self, inputArea: Rect, buf: &mut Buffer) {
        let state = match &self.completion {
            Some(s) if !s.matches.is_empty() => s,
            _ => return,
        };

        let count = state.matches.len().min(8);
        let menuHeight = count as u16 + 2; // +2 for border
        let menuWidth = inputArea.width;

        // Position above the input area.
        let menuY = inputArea.y.saturating_sub(menuHeight);
        let menuArea = Rect {
            x: inputArea.x,
            y: menuY,
            width: menuWidth,
            height: menuHeight,
        };

        // Clear the entire menu area to a solid background.
        let bgStyle = Style::default().bg(Color::Rgb(20, 20, 30));
        for row in menuArea.y..menuArea.y + menuArea.height {
            for col in menuArea.x..menuArea.x + menuArea.width {
                if let Some(cell) = buf.cell_mut((col, row)) {
                    cell.set_char(' ');
                    cell.set_style(bgStyle);
                }
            }
        }

        let borderStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(20, 20, 30));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle);
        let inner = block.inner(menuArea);
        block.render(menuArea, buf);

        // Render each command row.
        let normalStyle = Style::default().fg(Color::White).bg(Color::Rgb(20, 20, 30));
        let selectedStyle = Style::default().fg(Color::White).bg(Color::Rgb(40, 40, 80));
        let descStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(20, 20, 30));
        let selectedDescStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(40, 40, 80));

        for (i, cmd) in state.matches.iter().take(count).enumerate() {
            let row = inner.y + i as u16;
            if row >= inner.y + inner.height {
                break;
            }
            let isSelected = i == state.selected;
            let (nameStyle, dStyle) = if isSelected {
                (selectedStyle, selectedDescStyle)
            } else {
                (normalStyle, descStyle)
            };

            let rowArea = Rect {
                x: inner.x,
                y: row,
                width: inner.width,
                height: 1,
            };

            // Clear row background for selected item.
            if isSelected {
                for col in rowArea.x..rowArea.x + rowArea.width {
                    if let Some(cell) = buf.cell_mut((col, row)) {
                        cell.set_char(' ');
                        cell.set_style(selectedStyle);
                    }
                }
            }

            let line = Line::from(vec![
                Span::styled(format!(" /{}", cmd.name), nameStyle),
                Span::styled(format!(" \u{2014} {}", cmd.description), dStyle),
            ]);
            // Render truncated to row width.
            Paragraph::new(line).render(rowArea, buf);
        }
    }

    /// Find the visual line range of the message at the given grid line.
    ///
    /// Grid line = screenRow - scrollY. Groups consecutive Reasoning +
    /// Assistant entries into one message since they come from the same
    /// LLM turn. Returns (startGridLine, endGridLine) inclusive.
    pub fn entryBoundsAtGridLine(&self, gridLine: i32) -> Option<(i32, i32)> {
        let w = self.lastChatWidth.max(1) as usize;
        let maxScroll = self.scrollOffset as i32 + self.lastScrollY as i32;
        let visualLine = (gridLine + maxScroll) as u32;

        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let mut cursor: u32 = 0;

        let mut dummyRanges: Vec<CodeBlockRange> = Vec::new();
        for (idx, entry) in self.entries.iter().enumerate() {
            let mut entryLines: Vec<Line<'static>> = Vec::new();
            let mut entryCont: Vec<bool> = Vec::new();
            self.renderEntry(entry, &mut entryLines, &mut entryCont, w as u16, idx, &mut dummyRanges);

            let entryStart = cursor;
            for line in &entryLines {
                let lineWidth = line.width();
                if lineWidth == 0 {
                    cursor += 1;
                } else {
                    cursor += ((lineWidth + w - 1) / w) as u32;
                }
            }
            ranges.push((entryStart, cursor));
            cursor += 1;
        }

        let mut matchIdx: Option<usize> = None;
        for (i, &(_start, end)) in ranges.iter().enumerate() {
            if visualLine < end + 1 {
                matchIdx = Some(i);
                break;
            }
        }

        // Handle streaming content as a single entry.
        if matchIdx.is_none() && self.isStreaming {
            let streamStart = cursor;
            let waiting = self.streamingContent.is_empty()
                && self.pendingContent.is_empty()
                && self.streamingReasoning.is_empty();

            // Throbber: shown while waiting or during reasoning.
            if waiting || !self.streamingReasoning.is_empty() {
                cursor += 2; // blob rows
                if self.thinkingExpanded {
                    for line in self.streamingReasoning.lines() {
                        let span = Span::raw(format!("  {line}"));
                        let lineWidth = span.width();
                        cursor += if lineWidth == 0 {
                            1
                        } else {
                            ((lineWidth + w - 1) / w) as u32
                        };
                    }
                }
                cursor += 1; // separator
            }

            let displayContent = self.displayContent();
            if !displayContent.is_empty() {
                // NOTE: Subtract prefix width ("◆ " = 2 cols) so blocks size correctly.
                let blocks = markdown::render(&displayContent, (w as u16).saturating_sub(2));
                let md = flattenRenderedBlocks(blocks, (w as u16).saturating_sub(2));
                for line in &md {
                    let lineWidth = line.width();
                    cursor += if lineWidth == 0 {
                        1
                    } else {
                        ((lineWidth + w - 1) / w) as u32
                    };
                }
            }
            if visualLine >= streamStart && visualLine < cursor {
                let startGrid = streamStart as i32 - maxScroll;
                let endGrid = cursor as i32 - 1 - maxScroll;
                if endGrid < startGrid {
                    return None;
                }
                return Some((startGrid, endGrid));
            }
            return None;
        }

        let idx = matchIdx?;

        let (groupStart, groupEnd) = self.messageGroup(idx, &ranges);

        let startGrid = groupStart as i32 - maxScroll;
        let endGrid = groupEnd as i32 - 1 - maxScroll;
        if endGrid < startGrid {
            return None;
        }
        Some((startGrid, endGrid))
    }

    /// Find the visual line range of the message group containing entry `idx`.
    ///
    /// Consecutive Reasoning + Assistant entries are one LLM message.
    fn messageGroup(&self, idx: usize, ranges: &[(u32, u32)]) -> (u32, u32) {
        let mut startIdx = idx;
        let mut endIdx = idx;

        if matches!(self.entries[idx], PanelEntry::Assistant(_)) {
            if idx > 0 && matches!(self.entries[idx - 1], PanelEntry::Reasoning { .. }) {
                startIdx = idx - 1;
            }
        }

        if matches!(self.entries[idx], PanelEntry::Reasoning { .. }) {
            if idx + 1 < self.entries.len()
                && matches!(self.entries[idx + 1], PanelEntry::Assistant(_))
            {
                endIdx = idx + 1;
            }
        }

        (ranges[startIdx].0, ranges[endIdx].1)
    }

    fn buildLines(&self, width: u16) -> (Vec<Line<'static>>, Vec<bool>, Vec<(Option<usize>, usize)>, Vec<CodeBlockRange>, HashSet<usize>) {
        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut cont: Vec<bool> = Vec::new();
        let mut reasoningHeaders: Vec<(Option<usize>, usize)> = Vec::new();
        let mut codeBlockRanges: Vec<CodeBlockRange> = Vec::new();

        let mut nonCopyable = HashSet::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if matches!(entry, PanelEntry::Reasoning { .. }) {
                reasoningHeaders.push((Some(i), lines.len()));
            }
            let isNotice = matches!(entry, PanelEntry::SessionNotice(_) | PanelEntry::CompactionMarker { .. });
            let linesBefore = lines.len();
            self.renderEntry(entry, &mut lines, &mut cont, width, i, &mut codeBlockRanges);
            if isNotice {
                for idx in linesBefore..lines.len() {
                    nonCopyable.insert(idx);
                }
            }
            lines.push(Line::from(""));
            cont.push(false);
        }

        // Streaming content.
        if self.isStreaming {
            let waiting = self.streamingContent.is_empty()
                && self.pendingContent.is_empty()
                && self.streamingReasoning.is_empty();
            let showThrobber = waiting || self.reasoningActive;
            let hasReasoning = !self.streamingReasoning.is_empty();

            if showThrobber {
                // Record header position for click-to-toggle.
                if hasReasoning {
                    reasoningHeaders.push((None, lines.len()));
                }
                // Animated throbber with elapsed time.
                let blobLines = self.throbber.renderLines();
                let elapsed = self.thinkingStartTime
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                let suffix = if let Some(ref status) = self.retryStatus {
                    format!(" {status}")
                } else if hasReasoning {
                    let icon = if self.thinkingExpanded { "\u{25BE}" } else { "\u{25B8}" };
                    format!(" thinking ({elapsed}s)  {icon}")
                } else {
                    format!(" thinking ({elapsed}s)")
                };

                lines.push(Line::from(vec![
                    blobLines[0].spans[0].clone(),
                    Span::styled(suffix, Style::default().fg(Color::DarkGray)),
                ]));
                cont.push(false);
                lines.push(blobLines[1].clone());
                cont.push(false);
            } else if hasReasoning {
                // Record header position for click-to-toggle.
                reasoningHeaders.push((None, lines.len()));
                // Reasoning finished but text exists — show static collapse header.
                let icon = if self.thinkingExpanded { "\u{25BE}" } else { "\u{25B8}" };
                lines.push(Line::from(Span::styled(
                    format!("{icon} reasoning"),
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
            }

            if self.thinkingExpanded && hasReasoning {
                let style = Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC);
                let textWidth = (width as usize).saturating_sub(2);
                for logicalLine in self.streamingReasoning.lines() {
                    let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
                    let wrapped = wrapSpannedLine(spanLine, textWidth);
                    for (idx, wLine) in wrapped.into_iter().enumerate() {
                        cont.push(idx > 0);
                        let mut spans = vec![Span::styled("  ".to_string(), style)];
                        spans.extend(wLine.spans);
                        lines.push(Line::from(spans));
                    }
                }
            }

            if showThrobber || hasReasoning {
                lines.push(Line::from(""));
                cont.push(false);
            }

            let displayContent = self.displayContent();
            let hasFadingChar = !self.pendingContent.is_empty() && self.fadeProgress < 1.0;
            if !displayContent.is_empty() {
                let linesBeforeMd = lines.len();
                let mdBlocks = markdown::render(&displayContent, width.saturating_sub(2));
                prefixRenderedBlocks(
                    &mut lines, &mut cont, mdBlocks,
                    "\u{25C6} ", Style::default().fg(Color::White), width,
                    usize::MAX, &self.codeScrollX, &self.codeExpanded,
                    &mut codeBlockRanges, &self.copiedFlash,
                );
                // Apply fade to the last character if one is mid-reveal.
                if hasFadingChar {
                    applyFadeToLastChar(&mut lines[linesBeforeMd..], self.fadeProgress);
                }
            }
        }


        (lines, cont, reasoningHeaders, codeBlockRanges, nonCopyable)
    }

    fn renderEntry(&self, entry: &PanelEntry, lines: &mut Vec<Line<'static>>, cont: &mut Vec<bool>, width: u16, entryIndex: usize, codeBlockRanges: &mut Vec<CodeBlockRange>) {
        match entry {
            PanelEntry::User(text) => {
                let style = Style::default().fg(Color::Cyan);
                let dimStyle = Style::default().fg(Color::DarkGray);
                let prefixWidth: usize = 2; // "› " = 2 display columns.
                let textWidth = (width as usize).saturating_sub(prefixWidth);
                let mut isFirst = true;

                for logicalLine in text.lines() {
                    // Attachment indicator gets dimmed styling.
                    let isDim = logicalLine.starts_with("[+") && logicalLine.ends_with("attached]");
                    let lineStyle = if isDim { dimStyle } else { style };
                    let spanLine = Line::from(Span::styled(logicalLine.to_string(), lineStyle));
                    let wrapped = wrapSpannedLine(spanLine, textWidth);
                    for (idx, wLine) in wrapped.into_iter().enumerate() {
                        cont.push(idx > 0);
                        let prefix = if isFirst {
                            "\u{203A} "
                        } else {
                            "  "
                        };
                        isFirst = false;
                        let mut spans = vec![Span::styled(prefix.to_string(), lineStyle)];
                        spans.extend(wLine.spans);
                        lines.push(Line::from(spans));
                    }
                }
                if text.ends_with('\n') {
                    lines.push(Line::from(Span::styled("  ", style)));
                    cont.push(false);
                }
            }
            PanelEntry::Assistant(text) => {
                // NOTE: Subtract prefix width ("◆ " = 2 cols) so blocks size correctly.
                let mdBlocks = markdown::render(text, width.saturating_sub(2));
                prefixRenderedBlocks(
                    lines, cont, mdBlocks,
                    "\u{25C6} ", Style::default().fg(Color::White), width,
                    entryIndex, &self.codeScrollX, &self.codeExpanded,
                    codeBlockRanges, &self.copiedFlash,
                );
            }
            PanelEntry::Reasoning { text, expanded } => {
                let icon = if *expanded { "\u{25BE}" } else { "\u{25B8}" };
                lines.push(Line::from(Span::styled(
                    format!("{icon} reasoning"),
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
                if *expanded {
                    let style = Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC);
                    let textWidth = (width as usize).saturating_sub(2);
                    for logicalLine in text.lines() {
                        let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
                        let wrapped = wrapSpannedLine(spanLine, textWidth);
                        for (idx, wLine) in wrapped.into_iter().enumerate() {
                            cont.push(idx > 0);
                            let mut spans = vec![Span::styled("  ".to_string(), style)];
                            spans.extend(wLine.spans);
                            lines.push(Line::from(spans));
                        }
                    }
                }
            }
            PanelEntry::ToolRequest { summary, diff, command } => {
                if let Some(diffText) = diff {
                    // File edit: summary line with (+N -M) stats + diff code block.
                    let (added, removed) = diffStats(diffText);
                    let summaryBlock = RenderedBlock::Text(vec![Line::from(vec![
                        Span::styled(
                            summary.clone(),
                            Style::default().fg(Color::Yellow),
                        ),
                        Span::raw(" "),
                        Span::styled(
                            format!("+{added}"),
                            Style::default().fg(Color::Green),
                        ),
                        Span::raw(" "),
                        Span::styled(
                            format!("-{removed}"),
                            Style::default().fg(Color::Red),
                        ),
                    ])]);
                    let diffBlock = RenderedBlock::Code {
                        lang: Some("diff".to_string()),
                        lines: crate::markdown::highlight::diffLines(diffText),
                        code: diffText.clone(),
                    };
                    prefixRenderedBlocks(
                        lines, cont, vec![summaryBlock, diffBlock],
                        "\u{2699}\u{FE0E} ", Style::default().fg(Color::Yellow), width,
                        entryIndex, &self.codeScrollX, &self.codeExpanded,
                        codeBlockRanges, &self.copiedFlash,
                    );
                } else if command.is_some() {
                    // Shell command: skip rendering in chat when prompt is active
                    // (code block + info is in the input area). Show compact summary
                    // only for historical/approved entries.
                    let isPending = self.pendingPermit
                        && entryIndex == self.entries.len() - 1;
                    if !isPending {
                        let style = Style::default().fg(Color::Yellow);
                        let content = vec![Line::from(Span::styled(summary.clone(), style))];
                        prefixFirstLine(lines, cont, content, "\u{2699}\u{FE0E} ", style, width);
                    }
                } else {
                    let style = Style::default().fg(Color::Yellow);
                    let content = vec![Line::from(Span::styled(summary.clone(), style))];
                    prefixFirstLine(lines, cont, content, "\u{2699}\u{FE0E} ", style, width);
                }
            }
            PanelEntry::ToolApproved { name } => {
                let style = Style::default().fg(Color::Green);
                let content = vec![Line::from(Span::styled(name.clone(), style))];
                prefixFirstLine(lines, cont, content, "\u{2713}\u{FE0E} ", style, width);
            }
            PanelEntry::ToolDenied { name } => {
                let style = Style::default().fg(Color::Red);
                let content = vec![Line::from(Span::styled(
                    format!("{name} (denied)"),
                    style,
                ))];
                prefixFirstLine(lines, cont, content, "\u{2717}\u{FE0E} ", style, width);
            }
            PanelEntry::ToolAutoDenied { name, summary } => {
                // Red-tinted background for rule-blocked tool calls.
                let style = Style::default()
                    .fg(Color::Red)
                    .bg(Color::Rgb(60, 20, 20));
                let content = vec![Line::from(Span::styled(
                    format!("{name}: {summary} (blocked by rule)"),
                    style,
                ))];
                prefixFirstLine(lines, cont, content, "\u{2717}\u{FE0E} ", style, width);
            }
            PanelEntry::ToolActive { name: _, summary } => {
                let elapsed = self.toolStartTime
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                let style = Style::default().fg(Color::Yellow);
                let blobLines = self.throbber.renderLines();
                let statusText = format!("{summary}  ({elapsed}s)");
                // First line: throbber char + summary + elapsed.
                lines.push(Line::from(vec![
                    blobLines[0].spans.first().cloned().unwrap_or_default(),
                    Span::raw(" "),
                    Span::styled(statusText, style),
                ]));
                cont.push(false);
                // Second line: throbber continuation.
                lines.push(blobLines[1].clone());
                cont.push(true);
            }
            PanelEntry::SubagentBlock { agentType, prompt, toolLines, done, turns, content, contentExpanded, .. } => {
                let prefixPad: usize = 2; // Match other entries' prefix indent.
                let w = (width as usize).saturating_sub(prefixPad);
                let innerW = w.saturating_sub(2); // Inside the left+right borders.
                let indent = " ".repeat(prefixPad);
                let borderStyle = Style::default().fg(Color::DarkGray);
                let headerStyle = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
                let toolStyle = Style::default().fg(Color::Cyan);

                // Track header line for click-to-view.
                self.lastSubagentHeaderLine.set(Some(lines.len()));

                // Top border: ╭ explore ─── prompt... ─── view ╮
                use unicode_width::UnicodeWidthStr as UWS;
                let label = format!("{agentType} ");
                let labelW = UWS::width(label.as_str());
                let viewLabel = "view";
                let viewW = UWS::width(viewLabel);
                let viewStyle = Style::default().fg(Color::Yellow);
                let fixedParts = labelW + 1 + viewW + 1;
                let promptSpace = innerW.saturating_sub(fixedParts + 2);
                let promptW = UWS::width(prompt.as_str());
                let promptText = if promptW > promptSpace {
                    truncateToWidth(prompt, promptSpace)
                } else {
                    prompt.clone()
                };
                let promptW = UWS::width(promptText.as_str());
                let ruleTotal = innerW.saturating_sub(labelW + 1 + promptW + 1 + 1 + viewW);
                let leftRule = ruleTotal / 2;
                let rightRule = ruleTotal.saturating_sub(leftRule);
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled("\u{256D}", borderStyle),
                    Span::styled(label, headerStyle),
                    Span::styled("\u{2500}".repeat(leftRule), borderStyle),
                    Span::styled(format!(" {promptText} "), Style::default().fg(Color::Cyan)),
                    Span::styled("\u{2500}".repeat(rightRule), borderStyle),
                    Span::styled(format!(" {viewLabel}"), viewStyle),
                    Span::styled("\u{256E}", borderStyle),
                ]));
                cont.push(false);

                // Helper: build a fixed-width bordered content line.
                // Uses truncateToWidth to guarantee content fits exactly in innerW columns.
                let indentStr = indent.clone();
                let borderedLine = move |content: &str, contentStyle: Style| -> Line<'static> {
                    // Truncate content to fit, then pad remainder with spaces.
                    let truncated = truncateToWidth(content, innerW);
                    let truncW = unicode_width::UnicodeWidthStr::width(truncated.as_str());
                    let pad = innerW.saturating_sub(truncW);
                    // Combine content + padding into one string so wrap can't split them.
                    let body = format!("{truncated}{}", " ".repeat(pad));
                    Line::from(vec![
                        Span::raw(indentStr.clone()),
                        Span::styled("\u{2502}".to_string(), borderStyle),
                        Span::styled(body, contentStyle),
                        Span::styled("\u{2502}".to_string(), borderStyle),
                    ])
                };

                // Show only the last 3 tool lines.
                let maxVisible = 3;
                let skipCount = toolLines.len().saturating_sub(maxVisible);
                if skipCount > 0 {
                    lines.push(borderedLine(
                        &format!(" ... {skipCount} earlier"),
                        Style::default().fg(Color::DarkGray),
                    ));
                    cont.push(false);
                }
                for (name, summary) in toolLines.iter().skip(skipCount) {
                    let prefix = format!(" {name}: ");
                    let prefixW = UWS::width(prefix.as_str());
                    let maxSummaryW = innerW.saturating_sub(prefixW);
                    let truncSummary = truncateToWidth(summary, maxSummaryW);
                    let content = format!("{prefix}{truncSummary}");
                    lines.push(borderedLine(&content, toolStyle));
                    cont.push(false);
                }

                // Show placeholder only while still running.
                if toolLines.is_empty() && !*done {
                    lines.push(borderedLine(
                        " waiting\u{2026}",
                        Style::default().fg(Color::DarkGray),
                    ));
                    cont.push(false);
                }

                use unicode_width::UnicodeWidthStr;

                // Helper: render a horizontal border line with a centered label.
                let makeBorderLine = |leftCorner: &str, rightCorner: &str,
                                      label: &str, labelStyle: Style,
                                      indent: &str| -> Line<'static> {
                    let labelW = UnicodeWidthStr::width(label);
                    let ruleLen = innerW.saturating_sub(labelW);
                    let lRule = ruleLen / 2;
                    let rRule = ruleLen.saturating_sub(lRule);
                    Line::from(vec![
                        Span::raw(indent.to_string()),
                        Span::styled(leftCorner.to_string(), borderStyle),
                        Span::styled("\u{2500}".repeat(lRule), borderStyle),
                        Span::styled(label.to_string(), labelStyle),
                        Span::styled("\u{2500}".repeat(rRule), borderStyle),
                        Span::styled(rightCorner.to_string(), borderStyle),
                    ])
                };

                if *done {
                    let footerLabel = format!(" \u{2713}\u{FE0E} {agentType} ({turns} turns) ");

                    if let Some(contentText) = content {
                        // Divider: ├─ ✓ explore (N turns) ─┤
                        lines.push(makeBorderLine("\u{251C}", "\u{2524}",
                            &footerLabel, Style::default().fg(Color::Green), &indent));
                        cont.push(false);

                        let toggleEntryIdx = entryIndex;

                        // Render markdown blocks, then wrap each in │ ... │ borders.
                        // Code blocks get full renderCodeBlock() treatment (copy, scroll, expand).
                        let contentInnerW = innerW.saturating_sub(2); // 1 space pad each side.
                        let mdBlocks = crate::markdown::render(contentText, contentInnerW as u16);

                        let mut contentLines: Vec<Line<'static>> = Vec::new();
                        let mut contentCont: Vec<bool> = Vec::new();
                        let mut codeOrdinal: usize = 0;
                        // Track where new CodeBlockRanges start for global offset fixup.
                        let codeBlockRangeStart = codeBlockRanges.len();

                        for block in mdBlocks {
                            match block {
                                crate::markdown::RenderedBlock::Text(textLines) => {
                                    for textLine in textLines {
                                        let wrapped = wrapSpannedLine(textLine, contentInnerW);
                                        for wLine in wrapped {
                                            // Measure and pad.
                                            let lineText: String = wLine.spans.iter()
                                                .map(|s| s.content.as_ref()).collect();
                                            let lineW = UnicodeWidthStr::width(lineText.as_str());
                                            let pad = contentInnerW.saturating_sub(lineW);
                                            let mut spans = vec![
                                                Span::styled("\u{2502} ", borderStyle),
                                            ];
                                            spans.extend(wLine.spans);
                                            spans.push(Span::styled(
                                                format!("{} \u{2502}", " ".repeat(pad)),
                                                borderStyle,
                                            ));
                                            contentLines.push(Line::from(spans));
                                            contentCont.push(false);
                                        }
                                    }
                                }
                                crate::markdown::RenderedBlock::Code { lang, lines: codeLines, code } => {
                                    let blockId = (entryIndex, codeOrdinal);
                                    codeOrdinal += 1;
                                    let isExpanded = self.codeExpanded.contains(&blockId);
                                    let totalCodeLines = codeLines.len();
                                    let cbCollapsible = totalCodeLines > MAX_CODE_BLOCK_LINES;

                                    let (visibleCodeLines, topExtra, bottomLabel): (
                                        &[Vec<Span<'static>>], Option<String>, Option<&str>,
                                    ) = if cbCollapsible && !isExpanded {
                                        let hidden = totalCodeLines - MAX_CODE_BLOCK_LINES;
                                        (&codeLines[..MAX_CODE_BLOCK_LINES], Some(format!("\u{25BE}{hidden}")), None)
                                    } else if cbCollapsible {
                                        (&codeLines, Some("\u{25B4}".to_string()), Some("\u{25B4}"))
                                    } else {
                                        (&codeLines, None, None)
                                    };

                                    let mcw = crate::markdown::highlight::maxContentWidth(&codeLines);
                                    let scrollX = self.codeScrollX.get(&blockId).copied().unwrap_or(0);
                                    let showCopied = self.copiedFlash
                                        .as_ref()
                                        .is_some_and(|(bid, t)| *bid == blockId && t.elapsed().as_secs() < 2);

                                    let codeBlockWidth = contentInnerW as u16;
                                    let startLine = contentLines.len();
                                    let rendered = crate::markdown::renderCodeBlock(
                                        visibleCodeLines, lang.as_deref(), codeBlockWidth,
                                        scrollX, mcw, showCopied,
                                        topExtra.as_deref(), bottomLabel,
                                    );

                                    // Wrap each code block line in outer │ ... │ borders.
                                    for codeLine in &rendered {
                                        let codeText: String = codeLine.spans.iter()
                                            .map(|s| s.content.as_ref()).collect();
                                        let codeW = UnicodeWidthStr::width(codeText.as_str());
                                        let pad = contentInnerW.saturating_sub(codeW);
                                        let mut spans = vec![
                                            Span::styled("\u{2502} ", borderStyle),
                                        ];
                                        spans.extend(codeLine.spans.clone());
                                        spans.push(Span::styled(
                                            format!("{} \u{2502}", " ".repeat(pad)),
                                            borderStyle,
                                        ));
                                        contentLines.push(Line::from(spans));
                                        contentCont.push(false);
                                    }
                                    let endLine = contentLines.len().saturating_sub(1);

                                    let cbInnerW = (codeBlockWidth as usize).saturating_sub(2);
                                    let copyLabelCol = codeBlockWidth.saturating_sub(
                                        if showCopied { 6 } else { 4 } + 1
                                    );

                                    codeBlockRanges.push(CodeBlockRange {
                                        startLine,
                                        endLine,
                                        blockId,
                                        maxContentWidth: mcw,
                                        innerWidth: cbInnerW,
                                        contentLines: codeLines.clone(),
                                        rawCode: code,
                                        // +4 for indent(2) + outer "│ "(2).
                                        copyLabelCol: copyLabelCol + 4,
                                        collapsible: cbCollapsible,
                                        totalLines: totalCodeLines,
                                    });
                                }
                            }
                        }

                        let totalLines = contentLines.len();
                        let collapsible = totalLines > MAX_CODE_BLOCK_LINES;
                        let visibleCount = if collapsible && !*contentExpanded {
                            MAX_CODE_BLOCK_LINES
                        } else {
                            totalLines
                        };

                        // Push visible content lines with outer indent.
                        let globalBase = lines.len();
                        for (i, contentLine) in contentLines[..visibleCount].iter().enumerate() {
                            let mut spans = vec![Span::raw(indent.clone())];
                            spans.extend(contentLine.spans.clone());
                            lines.push(Line::from(spans));
                            cont.push(contentCont.get(i).copied().unwrap_or(false));
                        }

                        // Fix CodeBlockRange indices: convert local contentLines offsets
                        // to global lines offsets, prune ranges outside visible area.
                        let visibleEnd = lines.len();
                        let mut i = codeBlockRangeStart;
                        while i < codeBlockRanges.len() {
                            codeBlockRanges[i].startLine += globalBase;
                            codeBlockRanges[i].endLine += globalBase;
                            if codeBlockRanges[i].startLine >= visibleEnd {
                                codeBlockRanges.remove(i);
                            } else {
                                if codeBlockRanges[i].endLine >= visibleEnd {
                                    codeBlockRanges[i].endLine = visibleEnd.saturating_sub(1);
                                }
                                i += 1;
                            }
                        }

                        // Bottom border with collapse/expand.
                        if collapsible && !*contentExpanded {
                            let hidden = totalLines - MAX_CODE_BLOCK_LINES;
                            lines.push(makeBorderLine("\u{2570}", "\u{256F}",
                                &format!(" \u{25BE}{hidden} "), Style::default().fg(Color::Gray), &indent));
                        } else if collapsible {
                            lines.push(makeBorderLine("\u{2570}", "\u{256F}",
                                " \u{25B4} ", Style::default().fg(Color::Gray), &indent));
                        } else {
                            lines.push(makeBorderLine("\u{2570}", "\u{256F}",
                                "", borderStyle, &indent));
                        }
                        cont.push(false);
                        // Set toggle line AFTER the border is pushed.
                        self.lastSubagentToggleLine.set(Some((lines.len() - 1, toggleEntryIdx)));
                    } else {
                        // No content — simple bottom border.
                        lines.push(makeBorderLine("\u{2570}", "\u{256F}",
                            &footerLabel, Style::default().fg(Color::Green), &indent));
                        cont.push(false);
                    }
                } else {
                    let elapsed = self.activeSubagent
                        .as_ref()
                        .map(|s| s.startTime.elapsed().as_secs())
                        .unwrap_or(0);
                    let blobLines = self.throbber.renderLines();
                    let throbberChar = blobLines[0].spans.first()
                        .map(|s| s.content.to_string())
                        .unwrap_or_else(|| "\u{25cc}".into());
                    let footerLabel = format!(" {throbberChar} running ({elapsed}s) ");
                    lines.push(makeBorderLine("\u{2570}", "\u{256F}",
                        &footerLabel, Style::default().fg(Color::Yellow), &indent));
                    cont.push(false);
                }
            }
            PanelEntry::ToolResult { name, output } => {
                let codeBlock = RenderedBlock::Code {
                    lang: Some(name.clone()),
                    lines: output.lines()
                        .map(|l| vec![Span::styled(l.to_string(), Style::default().fg(Color::Green))])
                        .collect(),
                    code: output.clone(),
                };
                prefixRenderedBlocks(
                    lines, cont, vec![codeBlock],
                    "\u{25C7} ", Style::default().fg(Color::DarkGray), width,
                    entryIndex, &self.codeScrollX, &self.codeExpanded,
                    codeBlockRanges, &self.copiedFlash,
                );
            }
            PanelEntry::CommandResult(text) => {
                // NOTE: Subtract prefix width ("ℹ︎ " = 2 cols) so blocks size correctly.
                let mdBlocks = markdown::render(text, width.saturating_sub(2));
                prefixRenderedBlocks(
                    lines, cont, mdBlocks,
                    "\u{2139}\u{FE0E} ", Style::default().fg(Color::DarkGray), width,
                    entryIndex, &self.codeScrollX, &self.codeExpanded,
                    codeBlockRanges, &self.copiedFlash,
                );
            }
            PanelEntry::Error(msg) => {
                let style = Style::default().fg(Color::Red);
                let content = vec![Line::from(Span::styled(msg.clone(), style))];
                prefixFirstLine(lines, cont, content, "\u{26A0}\u{FE0E} ", style, width);
            }
            PanelEntry::SessionNotice(text) => {
                let pillColor = Color::Rgb(30, 50, 90);
                let ghostStyle = Style::default().fg(Color::Rgb(100, 100, 120));
                let symbolStyle = Style::default()
                    .fg(Color::Rgb(200, 200, 210))
                    .bg(pillColor);
                let edgeStyle = Style::default().fg(pillColor);
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("\u{E0B6}", edgeStyle),
                    Span::styled("\u{2139}\u{FE0E}", symbolStyle),
                    Span::styled("\u{E0B4}", edgeStyle),
                    Span::styled(format!(" {text}"), ghostStyle),
                ]));
                cont.push(false);
            }
            PanelEntry::CompactionMarker { stage } => {
                let dimStyle = Style::default().fg(Color::DarkGray);
                // Render as: ─── ⚙︎ S2 compressed ───
                let label = format!(" \u{2699}\u{FE0E} {stage} compressed ");
                let dashCount = (width as usize).saturating_sub(label.len()) / 2;
                let dashes = "\u{2500}".repeat(dashCount.max(3));
                lines.push(Line::from(Span::styled(
                    format!("{dashes}{label}{dashes}"),
                    dimStyle,
                )));
                cont.push(false);
            }
            PanelEntry::Cancelled => {
                lines.push(Line::from(Span::styled(
                    "\u{2500} cancelled",
                    Style::default().fg(Color::DarkGray),
                )));
                cont.push(false);
            }
        }
    }
}

/// Extract a substring from a plain text string by display column range.
///
/// Format a byte count as a human-readable string (e.g. "1.2 MB").
fn formatBytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Accounts for multi-width characters. Returns the trimmed slice between
/// `colStart` and `colEnd` (exclusive) in display columns.
fn sliceByDisplayColumn(text: &str, colStart: u16, colEnd: u16) -> String {
    let mut result = String::new();
    let mut col: u16 = 0;

    for ch in text.chars() {
        let w = unicode_display_width(ch) as u16;
        if col + w > colEnd {
            break;
        }
        if col >= colStart {
            result.push(ch);
        }
        col += w;
    }

    result.trim_end().to_string()
}

/// Count the visual lines a reasoning text would produce when expanded.
fn countReasoningLines(text: &str, width: u16) -> u16 {
    let textWidth = (width as usize).saturating_sub(2);
    let style = Style::default();
    let mut count: u16 = 0;
    for logicalLine in text.lines() {
        let spanLine = Line::from(Span::styled(logicalLine.to_string(), style));
        count += wrapSpannedLine(spanLine, textWidth).len() as u16;
    }
    count
}

/// Count added/removed lines in a unified diff string.
fn diffStats(diff: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;
    for line in diff.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if line.starts_with('+') {
            added += 1;
        } else if line.starts_with('-') {
            removed += 1;
        }
    }
    (added, removed)
}

/// Prepend a styled symbol to the first line; indent continuations to match.
///
/// Pre-wraps each content line at `(width - prefixWidth)` so that wrapped
/// continuations align with the text start, not column 0.
fn prefixFirstLine(
    out: &mut Vec<Line<'static>>,
    cont: &mut Vec<bool>,
    contentLines: Vec<Line<'static>>,
    symbol: &str,
    symbolStyle: Style,
    width: u16,
) {
    let prefixWidth: usize = symbol.chars().map(unicode_display_width).sum();
    let indent = " ".repeat(prefixWidth);
    let textWidth = (width as usize).saturating_sub(prefixWidth);
    let mut isFirst = true;

    for line in contentLines {
        let wrapped = wrapSpannedLine(line, textWidth);
        for (idx, wLine) in wrapped.into_iter().enumerate() {
            cont.push(idx > 0);
            let prefix = if isFirst {
                Span::styled(symbol.to_string(), symbolStyle)
            } else {
                Span::raw(indent.clone())
            };
            isFirst = false;
            let mut spans = vec![prefix];
            spans.extend(wLine.spans);
            out.push(Line::from(spans));
        }
    }
}

/// Prepend a styled symbol to the first line of rendered blocks.
///
/// Text blocks are word-wrapped; code blocks are rendered with borders
/// and pushed without wrapping. Tracks code block metadata for scrolling and copy.
fn prefixRenderedBlocks(
    out: &mut Vec<Line<'static>>,
    cont: &mut Vec<bool>,
    blocks: Vec<RenderedBlock>,
    symbol: &str,
    symbolStyle: Style,
    width: u16,
    entryIndex: usize,
    codeScrollX: &HashMap<(usize, usize), u16>,
    codeExpanded: &HashSet<(usize, usize)>,
    codeBlockRanges: &mut Vec<CodeBlockRange>,
    copiedFlash: &Option<((usize, usize), Instant)>,
) {
    let prefixWidth: usize = symbol.chars().map(unicode_display_width).sum();
    let indent = " ".repeat(prefixWidth);
    let textWidth = (width as usize).saturating_sub(prefixWidth);
    let mut isFirst = true;
    let mut codeOrdinal: usize = 0;
    let mut isFirstBlock = true;

    for block in blocks {
        // Insert a blank line between blocks for paragraph spacing.
        if !isFirstBlock {
            cont.push(false);
            let prefix = Span::raw(indent.clone());
            out.push(Line::from(vec![prefix]));
        }
        isFirstBlock = false;

        match block {
            RenderedBlock::Text(textLines) => {
                for line in textLines {
                    let wrapped = wrapSpannedLine(line, textWidth);
                    for (idx, wLine) in wrapped.into_iter().enumerate() {
                        cont.push(idx > 0);
                        let prefix = if isFirst {
                            Span::styled(symbol.to_string(), symbolStyle)
                        } else {
                            Span::raw(indent.clone())
                        };
                        isFirst = false;
                        let mut spans = vec![prefix];
                        spans.extend(wLine.spans);
                        out.push(Line::from(spans));
                    }
                }
            }
            RenderedBlock::Code { lang, lines: codeLines, code } => {
                let blockId = (entryIndex, codeOrdinal);
                codeOrdinal += 1;
                let codeWidth = width.saturating_sub(prefixWidth as u16);
                let totalLines = codeLines.len();
                let collapsible = totalLines > MAX_CODE_BLOCK_LINES;
                let isExpanded = codeExpanded.contains(&blockId);

                // Decide visible lines, top indicator, and bottom label.
                let (visibleLines, topExtra, bottomLabel): (
                    &[Vec<Span<'static>>], Option<String>, Option<&str>,
                ) = if collapsible && !isExpanded {
                    let hidden = totalLines - MAX_CODE_BLOCK_LINES;
                    (
                        &codeLines[..MAX_CODE_BLOCK_LINES],
                        Some(format!("\u{25BE}{hidden}")),
                        None,
                    )
                } else if collapsible {
                    (&codeLines, Some("\u{25B4}".to_string()), Some("\u{25B4}"))
                } else {
                    (&codeLines, None, None)
                };

                let mcw = markdown::highlight::maxContentWidth(&codeLines);
                let innerW = (codeWidth as usize).saturating_sub(2);
                let scrollX = codeScrollX.get(&blockId).copied().unwrap_or(0);
                let showCopied = copiedFlash
                    .as_ref()
                    .is_some_and(|(bid, t)| *bid == blockId && t.elapsed().as_secs() < 2);
                let copyLabel = if showCopied { "copied" } else { "copy" };
                let copyLabelCol = codeWidth.saturating_sub(copyLabel.len() as u16 + 1);

                let startLine = out.len();
                let rendered = markdown::renderCodeBlock(
                    visibleLines, lang.as_deref(), codeWidth, scrollX, mcw, showCopied,
                    topExtra.as_deref(), bottomLabel,
                );
                for codeLine in rendered {
                    cont.push(false);
                    let prefix = if isFirst {
                        Span::styled(symbol.to_string(), symbolStyle)
                    } else {
                        Span::raw(indent.clone())
                    };
                    isFirst = false;
                    let mut spans = vec![prefix];
                    spans.extend(codeLine.spans);
                    out.push(Line::from(spans));
                }
                let endLine = out.len().saturating_sub(1);

                codeBlockRanges.push(CodeBlockRange {
                    startLine,
                    endLine,
                    blockId,
                    maxContentWidth: mcw,
                    innerWidth: innerW,
                    contentLines: codeLines,
                    rawCode: code,
                    copyLabelCol: copyLabelCol + prefixWidth as u16,
                    collapsible,
                    totalLines,
                });
            }
        }
    }
}

/// Flatten rendered blocks into lines for bounds calculation.
fn flattenRenderedBlocks(blocks: Vec<RenderedBlock>, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for block in blocks {
        match block {
            RenderedBlock::Text(textLines) => lines.extend(textLines),
            RenderedBlock::Code { lang, lines: codeLines, .. } => {
                let mcw = markdown::highlight::maxContentWidth(&codeLines);
                lines.extend(markdown::renderCodeBlock(
                    &codeLines, lang.as_deref(), width, 0, mcw, false, None, None,
                ));
            }
        }
    }
    lines
}

/// Interpolate a color toward dark based on fade progress (0.0 = dark, 1.0 = original).
fn fadeColor(color: Color, progress: f32) -> Color {
    let (r, g, b) = match color {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::White => (255, 255, 255),
        Color::Reset => (200, 200, 200),
        Color::Gray => (128, 128, 128),
        Color::DarkGray => (80, 80, 80),
        Color::Red => (255, 0, 0),
        Color::Green => (0, 255, 0),
        Color::Yellow => (255, 255, 0),
        Color::Blue => (0, 0, 255),
        Color::Magenta => (255, 0, 255),
        Color::Cyan => (0, 255, 255),
        _ => return color,
    };
    let dark = 25_f32;
    let blend = |c: u8| -> u8 {
        (dark + (c as f32 - dark) * progress).clamp(0.0, 255.0) as u8
    };
    Color::Rgb(blend(r), blend(g), blend(b))
}

/// Dim the last visible character in a slice of lines based on fade progress.
fn applyFadeToLastChar(lines: &mut [Line<'static>], progress: f32) {
    // Walk backward to find the last line with visible content.
    for line in lines.iter_mut().rev() {
        // Find the last non-empty span with non-whitespace content.
        let lastIdx = match line.spans.iter().rposition(|s| {
            !s.content.is_empty() && s.content.chars().any(|c| !c.is_whitespace())
        }) {
            Some(i) => i,
            None => continue,
        };

        let span = &line.spans[lastIdx];
        let content = span.content.to_string();
        let style = span.style;

        // Find the byte offset of the last char.
        if let Some((byteIdx, _)) = content.char_indices().last() {
            let prefix = content[..byteIdx].to_string();
            let lastCharStr = content[byteIdx..].to_string();

            let originalFg = style.fg.unwrap_or(Color::White);
            let fadedFg = fadeColor(originalFg, progress);
            let fadedStyle = Style { fg: Some(fadedFg), ..style };

            if prefix.is_empty() {
                // Entire span is one char — replace in place.
                line.spans[lastIdx] = Span::styled(lastCharStr, fadedStyle);
            } else {
                // Split: prefix keeps original style, last char gets faded style.
                line.spans[lastIdx] = Span::styled(prefix, style);
                line.spans.insert(lastIdx + 1, Span::styled(lastCharStr, fadedStyle));
            }
        }
        return;
    }
}

/// Wrap a multi-span Line into multiple lines fitting within maxWidth display columns.
///
/// Prefers breaking at space boundaries. Falls back to character-level
/// splitting when a word exceeds the available width.
/// Truncate a string to fit within `maxWidth` display columns, appending `…` if truncated.
fn truncateToWidth(s: &str, maxWidth: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    let mut width = 0;
    let mut end = 0;
    for (i, ch) in s.char_indices() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        // Reserve 1 column for the ellipsis.
        if width + cw > maxWidth.saturating_sub(1) {
            end = i;
            if width == 0 { end = s.len(); } // Don't produce empty string.
            return format!("{}\u{2026}", &s[..end]);
        }
        width += cw;
    }
    // Didn't need truncation.
    s.to_string()
}

fn wrapSpannedLine(line: Line<'static>, maxWidth: usize) -> Vec<Line<'static>> {
    use unicode_width::UnicodeWidthChar;

    if maxWidth == 0 {
        return vec![line];
    }

    // Flatten spans into (char, Style) pairs.
    let chars: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|span| span.content.chars().map(move |ch| (ch, span.style)))
        .collect();

    // NOTE: Use UnicodeWidthChar (crate) not unicode_display_width (custom) so
    // measurements match UnicodeWidthStr used by bordered line padding and ratatui.
    let totalWidth: usize = chars.iter()
        .map(|(ch, _)| UnicodeWidthChar::width(*ch).unwrap_or(0))
        .sum();
    if totalWidth <= maxWidth {
        return vec![line];
    }

    let mut result: Vec<Line<'static>> = Vec::new();
    let mut lineStart = 0;
    let mut currentWidth: usize = 0;
    let mut lastSpace: Option<usize> = None;

    for i in 0..chars.len() {
        if chars[i].0 == ' ' {
            lastSpace = Some(i);
        }

        let charW = UnicodeWidthChar::width(chars[i].0).unwrap_or(0);

        if currentWidth + charW > maxWidth && i > lineStart {
            // Break at the last space: exclude the trailing space from output
            // and skip it for the continuation, so lines never exceed maxWidth.
            let (sliceEnd, nextStart) = if let Some(sp) = lastSpace {
                if sp > lineStart { (sp, sp + 1) } else { (i, i) }
            } else {
                (i, i)
            };

            result.push(styledCharsToLine(&chars[lineStart..sliceEnd]));
            lineStart = nextStart;
            // Recount width from the new start through the current character.
            if lineStart <= i {
                currentWidth = chars[lineStart..=i]
                    .iter()
                    .map(|(ch, _)| UnicodeWidthChar::width(*ch).unwrap_or(0))
                    .sum();
            } else {
                currentWidth = 0;
            }
            lastSpace = None;
        } else {
            currentWidth += charW;
        }
    }

    if lineStart < chars.len() {
        result.push(styledCharsToLine(&chars[lineStart..]));
    }

    result
}

/// Reconstruct a Line from styled character pairs, merging adjacent same-style runs.
fn styledCharsToLine(chars: &[(char, Style)]) -> Line<'static> {
    if chars.is_empty() {
        return Line::from("");
    }

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut currentStr = String::new();
    let mut currentStyle = chars[0].1;

    for &(ch, style) in chars {
        if style == currentStyle {
            currentStr.push(ch);
        } else {
            if !currentStr.is_empty() {
                spans.push(Span::styled(
                    std::mem::take(&mut currentStr),
                    currentStyle,
                ));
            }
            currentStr.push(ch);
            currentStyle = style;
        }
    }

    if !currentStr.is_empty() {
        spans.push(Span::styled(currentStr, currentStyle));
    }

    Line::from(spans)
}

