use std::fmt;

use crate::message::ToolDef;

/// A single edit operation within a multiEdit batch.
#[derive(Debug)]
pub struct EditOp {
    pub oldString: String,
    pub newString: String,
    pub replaceAll: bool,
}

/// Scope of a shell command's effect on the environment (model-classified).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ShellImpact {
    /// Command only reads or inspects. No state change.
    Read,
    /// Creates or modifies a small number of files within the project.
    MinorMod,
    /// Installs packages, modifies configuration, or changes many files.
    MajorMod,
    /// Removes files, drops state, or performs operations difficult to reverse.
    Delete,
}

/// A parsed tool invocation from the LLM.
#[derive(Debug)]
pub enum ToolAction {
    Shell {
        command: String,
        explanation: String,
        impact: ShellImpact,
        timeout: Option<u64>,
        /// Target terminal by name. None resolves to the active terminal.
        /// For `runInBackground`, None creates a visible ephemeral terminal.
        terminal: Option<String>,
        /// Spawn non-blocking as a visible terminal run. When true, returns a
        /// terminal run id immediately and archives replay bytes for terminal history.
        runInBackground: bool,
    },
    ReadFile {
        path: String,
        offset: Option<usize>,
        limit: Option<usize>,
        anchor: Option<usize>,
    },
    WriteFile {
        path: String,
        content: String,
    },
    EditFile {
        path: String,
        oldString: String,
        newString: String,
        replaceAll: bool,
    },
    MultiEdit {
        path: String,
        edits: Vec<EditOp>,
    },
    CopyFile {
        src: String,
        dest: String,
        overwrite: bool,
    },
    MoveFile {
        src: String,
        dest: String,
        overwrite: bool,
    },
    DeleteFile {
        path: String,
        recursive: bool,
    },
    MakeDirs {
        path: String,
    },
    ShellHistory {
        terminal: Option<String>,
    },
    ReadOutput {
        index: usize,
        offset: Option<usize>,
        limit: Option<usize>,
        terminal: Option<String>,
    },
    SearchOutput {
        index: usize,
        pattern: String,
        context: usize,
        terminal: Option<String>,
    },
    ReadTerminal {
        lines: usize,
        terminal: Option<String>,
    },
    /// Spawn a new terminal. Name auto-generated when None.
    TerminalSpawn {
        name: Option<String>,
    },
    /// Set the active default target for shell-using tool calls.
    TerminalSwitch {
        name: String,
    },
    /// Kill a named terminal.
    TerminalKill {
        name: String,
    },
    /// Snapshot of all terminals.
    TerminalList,
    /// Snapshot of archived visible terminal runs.
    TerminalRunList,
    /// Interrupt a running archived terminal run by id.
    TerminalRunStop {
        runId: String,
    },
    /// Retrieve buffered output for a task.
    JobOutput {
        jobId: u64,
        sinceLine: Option<u64>,
        maxLines: Option<usize>,
    },
    /// Kill a running job.
    JobStop {
        jobId: u64,
    },
    /// Snapshot of all tasks.
    JobList,
    /// Register a line-streamed monitor attached to an existing terminal. Lines
    /// matching the regex `filter` emit `MonitorEvent`s, bump the
    /// monitor's counter, and wake the agent with a synthetic wake event.
    /// Floods auto-stop the listener.
    Monitor {
        description: String,
        terminal: Option<String>,
        filter: String,
    },
    /// Stop a monitor listener.
    MonitorStop {
        monitorId: u64,
    },
    /// Snapshot of all monitors.
    MonitorList,
    /// Arm a one-shot delay wake. Fires after `delaySeconds` with the
    /// model-supplied `prompt` as the wake payload.
    ScheduleWakeup {
        delaySeconds: u64,
        prompt: String,
    },
    /// Arm a cron-scheduled wake. 5-field cron in local time.
    CronCreate {
        spec: String,
        prompt: String,
        recurring: bool,
    },
    /// Snapshot all wake sources (delay, cron, file-watch).
    CronList,
    /// Disarm a wake source by id. Works for any wake kind, named
    /// `cronDelete` because cron is the most common case the model
    /// will want to cancel.
    CronDelete {
        wakeId: u64,
    },
    /// Arm a filesystem watch. Each fs event under `path` (created,
    /// modified, removed) fires a wake with the `prompt` payload.
    FileWatch {
        path: String,
        prompt: String,
    },
    Glob {
        pattern: String,
        path: Option<String>,
        metadata: bool,
    },
    Grep {
        pattern: String,
        path: Option<String>,
        include: Option<String>,
        fileType: Option<String>,
        outputMode: String,
        caseSensitive: Option<bool>,
        contextLines: Option<usize>,
        multiline: bool,
    },
    ListDir {
        path: String,
        depth: usize,
        offset: usize,
        limit: usize,
        metadata: bool,
    },
    StructSearch {
        pattern: String,
        language: String,
        path: Option<String>,
    },
    Diff {
        path: Option<String>,
        gitRef: Option<String>,
        pathA: Option<String>,
        pathB: Option<String>,
    },
    FuzzyFind {
        query: String,
        path: Option<String>,
    },
    FileOutline {
        path: String,
    },
    ViewSymbol {
        file: String,
        symbol: String,
    },
    RelatedFiles {
        path: String,
    },
    WebSearch {
        query: String,
        allowedDomains: Option<Vec<String>>,
        blockedDomains: Option<Vec<String>>,
        maxResults: Option<usize>,
    },
    WebFetch {
        url: String,
        prompt: Option<String>,
        subpages: Option<usize>,
    },
    WebSimilar {
        url: String,
        allowedDomains: Option<Vec<String>>,
        blockedDomains: Option<Vec<String>>,
        maxResults: Option<usize>,
    },
    HistoryFetch {
        blockId: String,
    },
    HistorySearch {
        query: String,
        mediaType: Option<String>,
    },
    Task {
        prompt: String,
        agent: Option<String>,
        /// When true, the child session is registered as a background
        /// task in the JobPlane and the call returns immediately with a
        /// task id. The parent polls `jobOutput` / `jobList` and
        /// retrieves the final content from the task's ring buffer.
        runInBackground: bool,
    },
    Diagnostics {
        path: String,
        severity: String,
    },
    Mcp {
        qualifiedName: String,
        args: String,
    },
    Unknown {
        name: String,
        args: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParseError {
    MalformedJson(String),
    MissingField {
        field: &'static str,
    },
    MissingFieldWithExpected {
        field: &'static str,
        expected: &'static str,
    },
    WrongType {
        field: &'static str,
        expected: &'static str,
    },
    WrongNestedType {
        context: &'static str,
        field: &'static str,
        expected: &'static str,
    },
    InvalidField {
        field: &'static str,
        expected: &'static str,
    },
    MissingNestedField {
        context: &'static str,
        field: &'static str,
    },
}

impl fmt::Display for ToolParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolParseError::MalformedJson(msg) => {
                write!(f, "Malformed JSON arguments: {msg}")
            }
            ToolParseError::MissingField { field } => {
                write!(f, "Missing required field '{field}'.")
            }
            ToolParseError::MissingFieldWithExpected { field, expected } => {
                write!(f, "Missing required field '{field}' ({expected}).")
            }
            ToolParseError::WrongType { field, expected } => {
                write!(f, "Field '{field}': expected {expected}.")
            }
            ToolParseError::WrongNestedType {
                context,
                field,
                expected,
            } => {
                write!(f, "{context} field '{field}': expected {expected}.")
            }
            ToolParseError::InvalidField { field, expected } => {
                write!(f, "Field '{field}': expected {expected}.")
            }
            ToolParseError::MissingNestedField { context, field } => {
                write!(f, "{context} missing '{field}'.")
            }
        }
    }
}

impl std::error::Error for ToolParseError {}

/// Which tools a subagent can access.
#[derive(Debug, Clone)]
pub enum ToolSet {
    /// All built-in tools except `task` (prevents nesting).
    All,
    /// Read-only tools only.
    ReadOnly,
}

/// Filter tool definitions by a ToolSet.
pub fn filterDefs(defs: &[ToolDef], set: &ToolSet) -> Vec<ToolDef> {
    // Terminal-management tools are excluded from subagent toolsets in
    // phase 1 — child sessions stay single-shell.
    const SUBAGENT_DENIED: &[&str] = &[
        "task",
        "terminalSpawn",
        "terminalSwitch",
        "terminalKill",
        "terminalList",
        "monitor",
        "monitorStop",
        "monitorList",
        "scheduleWakeup",
        "cronCreate",
        "cronList",
        "cronDelete",
        "fileWatch",
    ];
    match set {
        ToolSet::All => defs
            .iter()
            .filter(|d| !SUBAGENT_DENIED.contains(&d.function.name.as_str()))
            .cloned()
            .collect(),
        ToolSet::ReadOnly => {
            // Read-only toolset for explore subagents. Deliberately omits
            // `shell` — the model self-classifies impact and we don't trust
            // that classification for explore agents. Read-only inspection
            // of prior shell output is available via `shellHistory` +
            // `readOutput` + `searchOutput`.
            const ALLOWED: &[&str] = &[
                "readFile",
                "glob",
                "grep",
                "listDir",
                "structSearch",
                "diff",
                "fuzzyFind",
                "fileOutline",
                "viewSymbol",
                "relatedFiles",
                "shellHistory",
                "readOutput",
                "searchOutput",
                "readTerminal",
                "terminalList",
                "terminalRunList",
                "terminalRunStop",
            ];
            defs.iter()
                .filter(|d| ALLOWED.contains(&d.function.name.as_str()))
                .cloned()
                .collect()
        }
    }
}

/// Whether this action is a subagent task (handled by Session, not execute()).
pub fn needsTask(action: &ToolAction) -> bool {
    matches!(action, ToolAction::Task { .. })
}

/// Whether this action mutates the shell registry (handled by Session,
/// not `execute()`). Includes terminal management tools.
pub fn needsRegistry(action: &ToolAction) -> bool {
    matches!(
        action,
        ToolAction::TerminalSpawn { .. }
            | ToolAction::TerminalSwitch { .. }
            | ToolAction::TerminalKill { .. }
            | ToolAction::TerminalList
            | ToolAction::TerminalRunList
            | ToolAction::TerminalRunStop { .. },
    )
}

/// Whether this action touches async task state (handled by Session, not
/// `execute()`). `Shell { runInBackground: true, .. }` belongs here so it can
/// become a visible terminal-backed run instead of a blocking foreground call.
pub fn needsJobPlane(action: &ToolAction) -> bool {
    matches!(
        action,
        ToolAction::Shell {
            runInBackground: true,
            ..
        } | ToolAction::JobOutput { .. }
            | ToolAction::JobStop { .. }
            | ToolAction::JobList,
    )
}

/// True for actions that the MonitorPlane handles. Routed separately
/// from task handling because monitors are terminal-output subscriptions, not
/// command execution.
pub fn needsMonitor(action: &ToolAction) -> bool {
    matches!(
        action,
        ToolAction::Monitor { .. } | ToolAction::MonitorStop { .. } | ToolAction::MonitorList,
    )
}

/// True for actions that the WakeRegistry handles (schedule/cron/fs).
pub fn needsWakes(action: &ToolAction) -> bool {
    matches!(
        action,
        ToolAction::ScheduleWakeup { .. }
            | ToolAction::CronCreate { .. }
            | ToolAction::CronList
            | ToolAction::CronDelete { .. }
            | ToolAction::FileWatch { .. },
    )
}

/// Check if a tool action requires transcript access (handled by session, not here).
pub fn needsTranscript(action: &ToolAction) -> bool {
    matches!(
        action,
        ToolAction::HistoryFetch { .. } | ToolAction::HistorySearch { .. }
    )
}

/// Check if a tool action is an MCP tool (handled by session, not here).
pub fn needsLsp(action: &ToolAction) -> bool {
    matches!(action, ToolAction::Diagnostics { .. })
}

pub fn needsMcp(action: &ToolAction) -> bool {
    matches!(action, ToolAction::Mcp { .. })
}

/// Check if a tool action requires the web client (handled by session, not here).
pub fn needsWeb(action: &ToolAction) -> bool {
    matches!(
        action,
        ToolAction::WebSearch { .. } | ToolAction::WebFetch { .. } | ToolAction::WebSimilar { .. }
    )
}

impl ToolAction {
    /// Optional target terminal for shell-using actions.
    /// Returns `None` for all non-shell actions and for shell actions
    /// without an explicit `terminal` field (which resolves to active).
    pub fn terminal(&self) -> Option<&str> {
        match self {
            ToolAction::Shell { terminal, .. }
            | ToolAction::ShellHistory { terminal }
            | ToolAction::ReadOutput { terminal, .. }
            | ToolAction::SearchOutput { terminal, .. }
            | ToolAction::ReadTerminal { terminal, .. } => terminal.as_deref(),
            _ => None,
        }
    }
}
