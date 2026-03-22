//! Tool definitions and execution.
//!
//! Defines the tools available to the agent and handles execution.
//! Shell commands go through the construct-owned Shell session.
//! File operations run in-process. Search/diff tools use subprocesses
//! (rg, sg, git) to keep work off the shared terminal.
//!
//! # Public API
//! - [`builtinDefs`] — returns tool definitions for the LLM
//! - [`execute`] — execute a parsed tool action
//!
//! # Dependencies
//! `serde_json`, `regex`, `similar`, `tokio::process`

use crate::message::ToolDef;
use crate::shell::Shell;

// --- Limits ---

const MAX_READ_LINES: usize = 2000;
const MAX_READ_BYTES: usize = 100_000;
const MAX_LINE_LENGTH: usize = 2000;
const MAX_GLOB_RESULTS: usize = 100;
const MAX_GREP_FILES: usize = 100;
const MAX_GREP_CONTENT_LINES: usize = 200;
const MAX_LISTDIR_ENTRIES: usize = 200;
const MAX_STRUCT_MATCHES: usize = 50;
const MAX_FUZZY_RESULTS: usize = 20;
const MAX_OUTLINE_ENTRIES: usize = 100;
const MAX_RELATED_FILES: usize = 50;
const SUBPROCESS_TIMEOUT_SECS: u64 = 30;

/// Returns the built-in tool definitions to send to the LLM.
pub fn builtinDefs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "shell".into(),
                description: "Execute a shell command and return its output. \
                    Output is truncated at 2000 lines / 100KB. \
                    Use readOutput to access full output of past commands.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One-line description of what this command does and why you are running it."
                        },
                        "impact": {
                            "type": "string",
                            "enum": ["read", "minorMod", "majorMod", "delete"],
                            "description": "Scope of the command's effect on the environment. \
                                read: only reads or inspects, no state change. \
                                minorMod: creates or modifies a small number of files within the project. \
                                majorMod: installs packages, modifies configuration, or changes many files. \
                                delete: removes files, drops state, or performs operations difficult to reverse."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds. Default 30. The command is interrupted if it exceeds this."
                        }
                    },
                    "required": ["command", "explanation", "impact"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "readFile".into(),
                description: "Read the contents of a file. Returns numbered lines. \
                    For large files, use offset and limit to read specific ranges.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-indexed). Defaults to 1."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read. Defaults to 2000."
                        },
                        "anchor": {
                            "type": "integer",
                            "description": "Line number to anchor on. Expands outward based on \
                                indentation to return the enclosing code block. When set, \
                                offset and limit are ignored."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "writeFile".into(),
                description: "Write content to a file, creating it if needed. \
                    Overwrites the entire file. Prefer editFile for small changes. \
                    You must readFile the target before overwriting an existing file.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file."
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "editFile".into(),
                description: "Edit a file by replacing exact string matches. \
                    The old_string must match exactly one location in the file \
                    (unless replace_all is true). Include enough surrounding \
                    context in old_string to make the match unique. \
                    You must readFile the target before editing.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to edit."
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to find and replace."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement text."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences instead of requiring a unique match. Defaults to false."
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "multiEdit".into(),
                description: "Apply multiple edits to a single file in sequence. \
                    All edits are atomic \u{2014} if any edit fails, none are applied. \
                    Each edit works like editFile: old_string must match exactly one \
                    location (unless replace_all is true for that edit). \
                    You must readFile the target before using this tool.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to edit."
                        },
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_string": {
                                        "type": "string",
                                        "description": "The exact text to find and replace."
                                    },
                                    "new_string": {
                                        "type": "string",
                                        "description": "The replacement text."
                                    },
                                    "replace_all": {
                                        "type": "boolean",
                                        "description": "Replace all occurrences. Defaults to false."
                                    }
                                },
                                "required": ["old_string", "new_string"]
                            },
                            "description": "Ordered list of edits to apply sequentially."
                        }
                    },
                    "required": ["path", "edits"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "shellHistory".into(),
                description: "List recent shell commands with their index, exit code, \
                    and output size. Use readOutput to read a specific command's full output.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "readOutput".into(),
                description: "Read the output of a previous shell command by index. \
                    Use shellHistory to see available commands. \
                    Supports offset/limit like readFile for navigating large output.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Command index from shellHistory (0-indexed)."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-indexed). Defaults to 1."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read. Defaults to 2000."
                        }
                    },
                    "required": ["index"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "searchOutput".into(),
                description: "Search a previous command's output for a pattern (regex or substring). \
                    Returns matching lines with surrounding context. \
                    Use shellHistory to see available commands.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Command index from shellHistory (0-indexed)."
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern or substring to search for."
                        },
                        "context": {
                            "type": "integer",
                            "description": "Number of lines of context around each match. Defaults to 3."
                        }
                    },
                    "required": ["index", "pattern"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "readTerminal".into(),
                description: "Read recent terminal scrollback — everything visible in \
                    the shared terminal including user commands and their output. \
                    Use this to see what the user has been doing.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "lines": {
                            "type": "integer",
                            "description": "Number of recent lines to read. Defaults to 50."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "glob".into(),
                description: "Find files matching a glob pattern. Returns paths sorted \
                    by modification time (newest first). Capped at 100 results.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match (e.g. \"**/*.rs\", \"src/**/*.toml\")."
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in. Defaults to working directory."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "grep".into(),
                description: "Search file contents for a regex pattern. \
                    Three output modes: 'files' (file paths only), 'content' (matching \
                    lines with context), 'count' (match counts per file).".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for."
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory to search in. Defaults to working directory."
                        },
                        "include": {
                            "type": "string",
                            "description": "Glob filter for files to include (e.g. \"*.rs\", \"*.{ts,tsx}\")."
                        },
                        "type": {
                            "type": "string",
                            "description": "File type filter (e.g. \"rust\", \"py\", \"js\"). Uses rg's built-in type defs."
                        },
                        "output_mode": {
                            "type": "string",
                            "enum": ["files", "content", "count"],
                            "description": "Output format. 'files' = paths only (default), 'content' = matching lines, 'count' = match counts."
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Force case sensitivity. Omit for smart-case (case-sensitive if pattern has uppercase)."
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context around each match in 'content' mode. Defaults to 2."
                        },
                        "multiline": {
                            "type": "boolean",
                            "description": "Enable multiline matching (dot matches newlines). Defaults to false."
                        }
                    },
                    "required": ["pattern"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "listDir".into(),
                description: "List directory contents as an indented tree. Shows directories \
                    first (with trailing /), then files. Excludes .git, node_modules, \
                    target, __pycache__, .venv. Capped at 200 entries.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list."
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Max depth to recurse (1-5). Defaults to 2."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of entries to skip (for pagination). Defaults to 0."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum entries to return. Defaults to 200."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "structSearch".into(),
                description: "Search code using AST structural patterns. \
                    Matches syntax tree structure, not text. Use $IDENT for single nodes, \
                    $$$ARGS for multiple nodes. Example: '$EXPR.unwrap()' finds all .unwrap() \
                    calls.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "AST pattern to search for (e.g. \"$EXPR.unwrap()\", \"fn $NAME($$$ARGS) -> Result<$T, $E>\")."
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (e.g. \"rust\", \"python\", \"typescript\", \"javascript\")."
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory to search in. Defaults to working directory."
                        }
                    },
                    "required": ["pattern", "language"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "diff".into(),
                description: "Show differences between files or git revisions. Two modes: \
                    (1) Git mode: provide 'path' and optional 'ref' (defaults to HEAD) to \
                    see changes. (2) File mode: provide 'path_a' and 'path_b' to diff two \
                    files directly.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path for git diff mode."
                        },
                        "ref": {
                            "type": "string",
                            "description": "Git ref to diff against (commit, branch, tag). Defaults to HEAD."
                        },
                        "path_a": {
                            "type": "string",
                            "description": "First file path for file-vs-file diff."
                        },
                        "path_b": {
                            "type": "string",
                            "description": "Second file path for file-vs-file diff."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "fuzzyFind".into(),
                description: "Find files by fuzzy name matching. Scores filenames against your \
                    query using subsequence matching with bonuses for word boundaries and \
                    consecutive characters. Returns top 20 matches. Use when you know \
                    roughly what a file is called but not its exact path.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Fuzzy search query (e.g. \"conftoml\", \"mainrs\", \"authhandler\")."
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search in. Defaults to working directory."
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "fileOutline".into(),
                description: "Show the structural outline of a file — functions, classes, \
                    structs, traits, impls, and other top-level declarations. Returns \
                    declaration lines with line numbers.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to outline."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "viewSymbol".into(),
                description: "Look up a symbol by name in a file and return its full definition. \
                    Finds the declaration (function, struct, class, etc.) and returns the \
                    complete code block.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Path to the file containing the symbol."
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol to find (e.g. \"executeGrep\", \"ToolAction\", \"MyClass\")."
                        }
                    },
                    "required": ["file", "symbol"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "relatedFiles".into(),
                description: "Find files related to a given file by parsing its import/use/require \
                    statements and listing sibling files in the same directory. Helps discover \
                    the dependency graph around a file.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to analyze."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        // ── Web Tools (Exa API) ──
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "webSearch".into(),
                description: "Search the web. Returns titles, URLs, and content snippets. \
                    Use when you need current information beyond your training data.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query."
                        },
                        "allowed_domains": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Only include results from these domains."
                        },
                        "blocked_domains": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Exclude results from these domains."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results to return. Default 5, max 20."
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "webFetch".into(),
                description: "Fetch a URL and return its content as markdown. For large pages, \
                    provide a prompt to extract only relevant information via a sidecar model. \
                    Results cached for 15 minutes.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch. HTTP auto-upgraded to HTTPS."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "What to extract. When provided, a fast model \
                                extracts only the relevant parts from large pages. \
                                Omit for raw content."
                        },
                        "subpages": {
                            "type": "integer",
                            "description": "Number of linked pages to also crawl. \
                                Useful for doc sites where content spans multiple pages. \
                                Default 0."
                        }
                    },
                    "required": ["url"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "webSimilar".into(),
                description: "Find web pages semantically similar to a given URL. \
                    Uses embedding-based similarity \u{2014} good for finding related docs, \
                    alternative implementations, or similar projects.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to find similar pages for."
                        },
                        "allowed_domains": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Only include results from these domains."
                        },
                        "blocked_domains": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Exclude results from these domains."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results to return. Default 5, max 20."
                        }
                    },
                    "required": ["url"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "historyFetch".into(),
                description: "Retrieve the full original content of a specific exchange block \
                    from the transcript. Use this to access details that may have been \
                    compacted or truncated from the current context. Returns all turns \
                    (user message, assistant responses, tool calls, tool results) in the block.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "blockId": {
                            "type": "string",
                            "description": "Block ID to retrieve (e.g. \"b001\")."
                        }
                    },
                    "required": ["blockId"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "historySearch".into(),
                description: "Search across the full original transcript by text match. \
                    Returns matching blocks with snippets, block IDs, and topic labels. \
                    Use this to find specific information that may have been compacted \
                    away from the current context.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for across all transcript content."
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "task".into(),
                description: "Spawn a subtask agent to handle a focused piece of work. \
                    The agent runs with its own context and shell, then returns a result. \
                    Use 'explore' for read-only codebase research (cheap, fast model). \
                    Use 'general' for tasks that may modify files (full tools, same model). \
                    Prefer delegating bounded, well-defined work — don't use task for \
                    simple operations you can do directly.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Task description for the subtask agent. Be specific \
                                about what you want it to find, do, or produce."
                        },
                        "agent": {
                            "type": "string",
                            "enum": ["explore", "general"],
                            "description": "Agent type. 'explore' = read-only research with a cheap \
                                model. 'general' = full tools with the same model as the parent. \
                                Default: 'general'."
                        }
                    },
                    "required": ["prompt"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "diagnostics".into(),
                description: "Get LSP diagnostics (errors/warnings) for a file or directory. \
                    Use after making changes to verify correctness, or to check project \
                    health before declaring work complete. Requires a language server to be \
                    available for the file type.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to check for diagnostics."
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["error", "warning"],
                            "description": "Minimum severity to report. Default: 'error'."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
    ]
}

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
    Shell { command: String, explanation: String, impact: ShellImpact, timeout: Option<u64> },
    ReadFile { path: String, offset: Option<usize>, limit: Option<usize>, anchor: Option<usize> },
    WriteFile { path: String, content: String },
    EditFile { path: String, oldString: String, newString: String, replaceAll: bool },
    MultiEdit { path: String, edits: Vec<EditOp> },
    ShellHistory,
    ReadOutput { index: usize, offset: Option<usize>, limit: Option<usize> },
    SearchOutput { index: usize, pattern: String, context: usize },
    ReadTerminal { lines: usize },
    Glob { pattern: String, path: Option<String> },
    Grep {
        pattern: String, path: Option<String>, include: Option<String>,
        fileType: Option<String>, outputMode: String, caseSensitive: Option<bool>,
        contextLines: Option<usize>, multiline: bool,
    },
    ListDir { path: String, depth: usize, offset: usize, limit: usize },
    StructSearch { pattern: String, language: String, path: Option<String> },
    Diff {
        path: Option<String>, gitRef: Option<String>,
        pathA: Option<String>, pathB: Option<String>,
    },
    FuzzyFind { query: String, path: Option<String> },
    FileOutline { path: String },
    ViewSymbol { file: String, symbol: String },
    RelatedFiles { path: String },
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
    HistoryFetch { blockId: String },
    HistorySearch { query: String },
    Task { prompt: String, agent: Option<String> },
    Diagnostics { path: String, severity: String },
    Mcp { qualifiedName: String, args: String },
    Unknown { name: String, args: String },
}

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
    match set {
        ToolSet::All => defs
            .iter()
            .filter(|d| d.function.name != "task")
            .cloned()
            .collect(),
        ToolSet::ReadOnly => {
            const ALLOWED: &[&str] = &[
                "readFile", "glob", "grep", "listDir", "structSearch", "diff",
                "fuzzyFind", "fileOutline", "viewSymbol", "relatedFiles",
                "shellHistory", "readOutput", "searchOutput", "readTerminal",
                "shell",
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

/// Human-readable summary of what a tool action will do.
pub fn summarize(action: &ToolAction) -> String {
    match action {
        ToolAction::Shell { command, explanation, timeout, .. } => {
            let prefix = match timeout {
                Some(t) => format!("Run ({t}s)"),
                None => "Run".into(),
            };
            if explanation.is_empty() {
                format!("{prefix}: {command}")
            } else {
                format!("{prefix}: {command} \u{2014} {explanation}")
            }
        }
        ToolAction::ReadFile { path, offset, limit, anchor } => {
            if let Some(a) = anchor {
                format!("Read: {path} (block at line {a})")
            } else {
                match (offset, limit) {
                    (Some(o), Some(l)) => format!("Read: {path} (lines {o}..{})", o + l - 1),
                    (Some(o), None) => format!("Read: {path} (from line {o})"),
                    (None, Some(l)) => format!("Read: {path} (first {l} lines)"),
                    (None, None) => format!("Read: {path}"),
                }
            }
        }
        ToolAction::WriteFile { path, content } => {
            format!("Write {} bytes to {path}", content.len())
        }
        ToolAction::EditFile { path, oldString, replaceAll, .. } => {
            let preview = if oldString.len() > 40 {
                format!("{}\u{2026}", &oldString[..oldString.floor_char_boundary(40)])
            } else {
                oldString.clone()
            };
            if *replaceAll {
                format!("Edit {path}: replace all \"{preview}\"")
            } else {
                format!("Edit {path}: replace \"{preview}\"")
            }
        }
        ToolAction::MultiEdit { path, edits } => {
            format!("Multi-edit {path}: {} edits", edits.len())
        }
        ToolAction::ShellHistory => "List shell command history".into(),
        ToolAction::ReadOutput { index, offset, limit } => {
            match (offset, limit) {
                (Some(o), Some(l)) => format!("Read output #{index} (lines {o}..{})", o + l - 1),
                (Some(o), None) => format!("Read output #{index} (from line {o})"),
                _ => format!("Read output #{index}"),
            }
        }
        ToolAction::SearchOutput { index, pattern, .. } => {
            format!("Search output #{index} for \"{pattern}\"")
        }
        ToolAction::ReadTerminal { lines } => format!("Read last {lines} terminal lines"),
        ToolAction::Glob { pattern, path } => {
            let dir = path.as_deref().unwrap_or(".");
            format!("Find files: {pattern} in {dir}")
        }
        ToolAction::Grep { pattern, path, outputMode, .. } => {
            let dir = path.as_deref().unwrap_or(".");
            format!("Search ({outputMode}): \"{pattern}\" in {dir}")
        }
        ToolAction::ListDir { path, depth, offset, limit } => {
            if *offset > 0 {
                format!("List: {path} (depth {depth}, offset {offset}, limit {limit})")
            } else {
                format!("List: {path} (depth {depth})")
            }
        }
        ToolAction::StructSearch { pattern, language, .. } => {
            format!("AST search ({language}): \"{pattern}\"")
        }
        ToolAction::Diff { path, gitRef, pathA, pathB } => {
            if let (Some(a), Some(b)) = (pathA, pathB) {
                format!("Diff: {a} vs {b}")
            } else {
                let file = path.as_deref().unwrap_or(".");
                let reference = gitRef.as_deref().unwrap_or("HEAD");
                format!("Diff: {file} vs {reference}")
            }
        }
        ToolAction::FuzzyFind { query, path } => {
            let dir = path.as_deref().unwrap_or(".");
            format!("Fuzzy find: \"{query}\" in {dir}")
        }
        ToolAction::FileOutline { path } => format!("Outline: {path}"),
        ToolAction::ViewSymbol { file, symbol } => format!("View symbol: {symbol} in {file}"),
        ToolAction::RelatedFiles { path } => format!("Related files: {path}"),
        ToolAction::WebSearch { query, .. } => format!("Web search: \"{query}\""),
        ToolAction::WebFetch { url, prompt, .. } => {
            if prompt.is_some() {
                format!("Fetch+extract: {url}")
            } else {
                format!("Fetch: {url}")
            }
        }
        ToolAction::WebSimilar { url, .. } => format!("Find similar: {url}"),
        ToolAction::HistoryFetch { blockId } => format!("Fetch block: {blockId}"),
        ToolAction::HistorySearch { query } => format!("Search history: {query}"),
        ToolAction::Diagnostics { path, .. } => format!("Check diagnostics: {path}"),
        ToolAction::Task { prompt, agent } => {
            let agentName = agent.as_deref().unwrap_or("general");
            let preview = if prompt.len() > 60 {
                format!("{}\u{2026}", &prompt[..prompt.floor_char_boundary(60)])
            } else {
                prompt.clone()
            };
            format!("task [{agentName}]: {preview}")
        }
        ToolAction::Mcp { qualifiedName, .. } => {
            match crate::mcp::schema::splitQualifiedName(qualifiedName) {
                Some((server, tool)) => format!("MCP {server}/{tool}"),
                None => format!("MCP: {qualifiedName}"),
            }
        }
        ToolAction::Unknown { name, .. } => format!("Unknown tool: {name}"),
    }
}

/// Generate a unified-diff preview for file-mutating tools.
///
/// Returns a diff string for `editFile` and `writeFile` actions,
/// or `None` for everything else.
pub fn diffPreview(action: &ToolAction) -> Option<String> {
    match action {
        ToolAction::EditFile { path, oldString, newString, .. } => {
            let diff = similar::TextDiff::configure()
                .algorithm(similar::Algorithm::Patience)
                .diff_lines(oldString, newString);
            let unified = diff
                .unified_diff()
                .context_radius(3)
                .header(&format!("a/{path}"), &format!("b/{path}"))
                .to_string();
            if unified.trim().is_empty() {
                None
            } else {
                Some(unified)
            }
        }
        ToolAction::WriteFile { path, content } => {
            let old = std::fs::read_to_string(path).unwrap_or_default();
            if old.is_empty() {
                // New file — show all lines as additions.
                let lineCount = content.lines().count();
                let header = format!(
                    "--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{lineCount} @@"
                );
                let additions: String = content
                    .lines()
                    .map(|l| format!("+{l}\n"))
                    .collect();
                Some(format!("{header}\n{additions}"))
            } else {
                let diff = similar::TextDiff::configure()
                    .algorithm(similar::Algorithm::Patience)
                    .diff_lines(&old, content);
                let unified = diff
                    .unified_diff()
                    .context_radius(3)
                    .header(&format!("a/{path}"), &format!("b/{path}"))
                    .to_string();
                if unified.trim().is_empty() {
                    None
                } else {
                    Some(unified)
                }
            }
        }
        ToolAction::MultiEdit { path, edits } => {
            let original = std::fs::read_to_string(path).ok()?;
            let mut content = original.clone();
            for edit in edits {
                if edit.oldString.is_empty() || edit.oldString == edit.newString {
                    continue;
                }
                if edit.replaceAll {
                    content = content.replace(&edit.oldString, &edit.newString);
                } else {
                    content = content.replacen(&edit.oldString, &edit.newString, 1);
                }
            }
            let diff = similar::TextDiff::configure()
                .algorithm(similar::Algorithm::Patience)
                .diff_lines(&original, &content);
            let unified = diff
                .unified_diff()
                .context_radius(3)
                .header(&format!("a/{path}"), &format!("b/{path}"))
                .to_string();
            if unified.trim().is_empty() {
                None
            } else {
                Some(unified)
            }
        }
        _ => None,
    }
}

/// Compute the proposed file content after a file mutation, without writing.
///
/// Returns (path, proposed_content) for editFile/writeFile/multiEdit,
/// or None for all other actions. Used to send early LSP didChange
/// while the user is reviewing the approval prompt.
pub fn proposedContent(action: &ToolAction) -> Option<(String, String)> {
    match action {
        ToolAction::WriteFile { path, content } => {
            Some((path.clone(), content.clone()))
        }
        ToolAction::EditFile { path, oldString, newString, replaceAll } => {
            let original = std::fs::read_to_string(path).ok()?;
            let result = if *replaceAll {
                original.replace(oldString, newString)
            } else {
                original.replacen(oldString, newString, 1)
            };
            Some((path.clone(), result))
        }
        ToolAction::MultiEdit { path, edits } => {
            let mut content = std::fs::read_to_string(path).ok()?;
            for edit in edits {
                if edit.oldString.is_empty() || edit.oldString == edit.newString {
                    continue;
                }
                if edit.replaceAll {
                    content = content.replace(&edit.oldString, &edit.newString);
                } else {
                    content = content.replacen(&edit.oldString, &edit.newString, 1);
                }
            }
            Some((path.clone(), content))
        }
        _ => None,
    }
}

/// Check if a tool action requires transcript access (handled by session, not here).
pub fn needsTranscript(action: &ToolAction) -> bool {
    matches!(action, ToolAction::HistoryFetch { .. } | ToolAction::HistorySearch { .. })
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

/// Execute a tool action and return the output string.
pub async fn execute(action: &ToolAction, shell: &Shell) -> String {
    match action {
        ToolAction::Shell { command, timeout, .. } => {
            let dur = timeout.map(|s| std::time::Duration::from_secs(s));
            let raw = shell.execute(command, dur).await;
            // Apply same size guard as readFile. Full output is in shell history.
            let index = shell.historyLen().saturating_sub(1);
            truncateOutput(&raw, index)
        }
        ToolAction::ReadFile { path, offset, limit, anchor } => {
            executeReadFile(path, *offset, *limit, *anchor)
        }
        ToolAction::WriteFile { path, content } => executeWriteFile(path, content),
        ToolAction::EditFile { path, oldString, newString, replaceAll } => {
            executeEditFile(path, oldString, newString, *replaceAll)
        }
        ToolAction::MultiEdit { path, edits } => executeMultiEdit(path, edits),
        ToolAction::ShellHistory => executeShellHistory(shell),
        ToolAction::ReadOutput { index, offset, limit } => {
            executeReadOutput(shell, *index, *offset, *limit)
        }
        ToolAction::SearchOutput { index, pattern, context } => {
            executeSearchOutput(shell, *index, pattern, *context)
        }
        ToolAction::ReadTerminal { lines } => shell.readTerminal(*lines),
        ToolAction::Glob { pattern, path } => executeGlob(pattern, path.as_deref()).await,
        ToolAction::Grep {
            pattern, path, include, fileType, outputMode,
            caseSensitive, contextLines, multiline,
        } => {
            executeGrep(
                pattern, path.as_deref(), include.as_deref(), fileType.as_deref(),
                outputMode, *caseSensitive, *contextLines, *multiline,
            ).await
        }
        ToolAction::ListDir { path, depth, offset, limit } => {
            executeListDir(path, *depth, *offset, *limit)
        }
        ToolAction::StructSearch { pattern, language, path } => {
            executeStructSearch(pattern, language, path.as_deref()).await
        }
        ToolAction::Diff { path, gitRef, pathA, pathB } => {
            executeDiff(path.as_deref(), gitRef.as_deref(), pathA.as_deref(), pathB.as_deref()).await
        }
        ToolAction::FuzzyFind { query, path } => executeFuzzyFind(query, path.as_deref()).await,
        ToolAction::FileOutline { path } => executeFileOutline(path).await,
        ToolAction::ViewSymbol { file, symbol } => executeViewSymbol(file, symbol).await,
        ToolAction::RelatedFiles { path } => executeRelatedFiles(path),
        // Web tools are handled by session.rs (need ExaClient + cache).
        ToolAction::WebSearch { .. } | ToolAction::WebFetch { .. } | ToolAction::WebSimilar { .. } => {
            "Error: web tools must be executed through the session.".into()
        }
        // History tools are handled by session.rs (need transcript access).
        ToolAction::HistoryFetch { .. } | ToolAction::HistorySearch { .. } => {
            "Error: history tools must be executed through the session.".into()
        }
        // LSP diagnostics are handled by session.rs (need LspManager).
        ToolAction::Diagnostics { .. } => {
            "Error: diagnostics tool must be executed through the session.".into()
        }
        // MCP tools are handled by session.rs (need McpManager).
        ToolAction::Mcp { .. } => {
            "Error: MCP tools must be executed through the session.".into()
        }
        // Task tools are handled by session.rs (need to spawn child session).
        ToolAction::Task { .. } => {
            "Error: task tools must be executed through the session.".into()
        }
        ToolAction::Unknown { name, .. } => format!("Unknown tool: {name}"),
    }
}

/// Truncate shell output with a reference to readOutput for the rest.
fn truncateOutput(raw: &str, historyIndex: usize) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let totalLines = lines.len();

    if totalLines <= MAX_READ_LINES && raw.len() <= MAX_READ_BYTES {
        return raw.to_string();
    }

    let mut output = String::new();
    let mut byteCount = 0usize;
    let mut linesEmitted = 0usize;

    for line in &lines {
        if linesEmitted >= MAX_READ_LINES {
            break;
        }

        let displayLine = if line.len() > MAX_LINE_LENGTH {
            format!("{}...\n", &line[..MAX_LINE_LENGTH])
        } else {
            format!("{line}\n")
        };

        if byteCount + displayLine.len() > MAX_READ_BYTES {
            break;
        }

        byteCount += displayLine.len();
        output.push_str(&displayLine);
        linesEmitted += 1;
    }

    let remaining = totalLines - linesEmitted;
    output.push_str(&format!(
        "\n... truncated ({remaining} more lines, {totalLines} total). \
         Use readOutput(index: {historyIndex}) to access full output."
    ));

    output
}

fn executeReadFile(
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    anchor: Option<usize>,
) -> String {
    // Binary detection via first 512 bytes.
    match std::fs::File::open(path) {
        Ok(mut file) => {
            use std::io::Read;
            let mut probe = [0u8; 512];
            let probeLen = match file.read(&mut probe) {
                Ok(n) => n,
                Err(e) => return format!("Failed to read file: {e}"),
            };
            if isBinary(&probe[..probeLen]) {
                return format!("Binary file ({} bytes). Use shell tools to inspect.",
                    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0));
            }
        }
        Err(e) => return format!("Failed to read file: {e}"),
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read file: {e}"),
    };

    // Anchor mode: expand from a line based on indentation.
    if let Some(anchorLine) = anchor {
        return expandFromAnchor(&content, anchorLine);
    }

    formatNumberedLines(&content, offset, limit)
}

/// Format text as numbered lines with offset/limit and truncation.
/// Shared between readFile and readOutput.
fn formatNumberedLines(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> String {
    let startLine = offset.unwrap_or(1).max(1);
    let maxLines = limit.unwrap_or(MAX_READ_LINES).min(MAX_READ_LINES);
    let totalLines = content.lines().count();

    let mut output = String::new();
    let mut byteCount = 0usize;
    let mut linesEmitted = 0usize;
    let mut truncatedAt: Option<usize> = None;

    for (idx, line) in content.lines().enumerate() {
        let lineNum = idx + 1;
        if lineNum < startLine {
            continue;
        }
        if linesEmitted >= maxLines {
            truncatedAt = Some(lineNum);
            break;
        }

        // Cap individual line length.
        let displayLine = if line.len() > MAX_LINE_LENGTH {
            format!("{}\u{2026}", &line[..MAX_LINE_LENGTH])
        } else {
            line.to_string()
        };

        let formatted = format!("{lineNum:>6}\t{displayLine}\n");

        // Check byte budget.
        if byteCount + formatted.len() > MAX_READ_BYTES {
            truncatedAt = Some(lineNum);
            break;
        }

        byteCount += formatted.len();
        output.push_str(&formatted);
        linesEmitted += 1;
    }

    // Truncation notice.
    if let Some(cutLine) = truncatedAt {
        let remaining = totalLines - cutLine + 1;
        output.push_str(&format!(
            "\n... truncated at line {cutLine} ({remaining} more lines, {totalLines} total). \
             Use offset/limit to read more."
        ));
    }

    if output.is_empty() {
        if totalLines == 0 {
            "Empty (0 lines).".into()
        } else {
            format!("No content in range. Total: {totalLines} lines.")
        }
    } else {
        output
    }
}

fn executeWriteFile(path: &str, content: &str) -> String {
    // Create parent directories if needed.
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return format!("Failed to create directories: {e}");
            }
        }
    }
    match std::fs::write(path, content) {
        Ok(()) => format!("Wrote {} bytes to {path}", content.len()),
        Err(e) => format!("Failed to write file: {e}"),
    }
}

fn executeEditFile(path: &str, oldString: &str, newString: &str, replaceAll: bool) -> String {
    if oldString == newString {
        return "old_string and new_string are identical.".into();
    }
    if oldString.is_empty() {
        return "old_string cannot be empty.".into();
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read file: {e}"),
    };

    let matchCount = content.matches(oldString).count();

    if matchCount == 0 {
        return "No match found for old_string.".into();
    }

    if !replaceAll && matchCount > 1 {
        return format!(
            "Found {matchCount} matches for old_string. \
             Provide more context to make a unique match, or set replace_all to true."
        );
    }

    let newContent = if replaceAll {
        content.replace(oldString, newString)
    } else {
        content.replacen(oldString, newString, 1)
    };

    match std::fs::write(path, &newContent) {
        Ok(()) => {
            if replaceAll && matchCount > 1 {
                format!("Replaced {matchCount} occurrences in {path}.")
            } else {
                format!("Applied edit to {path}.")
            }
        }
        Err(e) => format!("Failed to write file: {e}"),
    }
}

fn executeMultiEdit(path: &str, edits: &[EditOp]) -> String {
    if edits.is_empty() {
        return "No edits provided.".into();
    }

    let original = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read file: {e}"),
    };

    let mut content = original;

    // Validate and apply each edit sequentially against the in-memory copy.
    for (i, edit) in edits.iter().enumerate() {
        if edit.oldString.is_empty() {
            return format!("Edit {}: old_string cannot be empty. No edits were applied.", i + 1);
        }
        if edit.oldString == edit.newString {
            return format!(
                "Edit {}: old_string and new_string are identical. No edits were applied.",
                i + 1
            );
        }

        let matchCount = content.matches(&edit.oldString).count();

        if matchCount == 0 {
            return format!(
                "Edit {}: no match found for old_string. No edits were applied.",
                i + 1
            );
        }

        if !edit.replaceAll && matchCount > 1 {
            return format!(
                "Edit {}: found {matchCount} matches for old_string. \
                 Provide more context or set replace_all. No edits were applied.",
                i + 1
            );
        }

        content = if edit.replaceAll {
            content.replace(&edit.oldString, &edit.newString)
        } else {
            content.replacen(&edit.oldString, &edit.newString, 1)
        };
    }

    // All edits validated — write once.
    match std::fs::write(path, &content) {
        Ok(()) => format!("Applied {} edits to {path}.", edits.len()),
        Err(e) => format!("Failed to write file: {e}"),
    }
}

fn executeShellHistory(shell: &Shell) -> String {
    let entries = shell.listHistory();
    if entries.is_empty() {
        return "No commands in history.".into();
    }

    let mut output = String::new();
    for (i, cmd, exitCode, lineCount) in &entries {
        let codeStr = match exitCode {
            Some(0) => String::new(),
            Some(c) => format!(" (exit {c})"),
            None => " (?)".into(),
        };
        // Truncate long commands for the listing.
        let cmdPreview = if cmd.len() > 80 {
            format!("{}\u{2026}", &cmd[..cmd.floor_char_boundary(80)])
        } else {
            cmd.clone()
        };
        output.push_str(&format!("[{i}] {cmdPreview}{codeStr}  ({lineCount} lines)\n"));
    }

    output
}

fn executeReadOutput(
    shell: &Shell,
    index: usize,
    offset: Option<usize>,
    limit: Option<usize>,
) -> String {
    match shell.getRecord(index) {
        Some(record) => {
            let header = format!(
                "Command [{}]: {}\n\n",
                index,
                if record.command.len() > 100 {
                    format!("{}\u{2026}", &record.command[..100])
                } else {
                    record.command
                }
            );
            let body = formatNumberedLines(&record.output, offset, limit);
            format!("{header}{body}")
        }
        None => format!("No command at index {index}. Use shellHistory to see available commands."),
    }
}

fn executeSearchOutput(shell: &Shell, index: usize, pattern: &str, context: usize) -> String {
    match shell.searchOutput(index, pattern, context) {
        Some(result) => result,
        None => format!("No command at index {index}. Use shellHistory to see available commands."),
    }
}

/// Detect binary content by checking for NUL bytes and known magic numbers.
fn isBinary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    // Magic number signatures for common binary formats.
    const MAGIC: &[&[u8]] = &[
        b"\x89PNG",            // PNG
        b"\xff\xd8\xff",      // JPEG
        b"GIF8",              // GIF
        b"BM",                // BMP
        b"PK\x03\x04",       // ZIP/DOCX/JAR
        b"\x7fELF",          // ELF
        b"\xfe\xed\xfa",     // Mach-O
        b"\xcf\xfa\xed\xfe", // Mach-O (reversed)
        b"%PDF",              // PDF
        b"\x1f\x8b",         // gzip
    ];
    for sig in MAGIC {
        if bytes.starts_with(sig) {
            return true;
        }
    }

    // NUL byte check (strong binary indicator in first 512 bytes).
    bytes.contains(&0x00)
}

// --- Subprocess helper ---

/// Run an external program, capture stdout+stderr, enforce timeout.
/// Returns Ok(stdout) on success or Err(message) on failure.
/// rg exit code 1 ("no matches") is treated as success with empty output.
async fn runSubprocess(
    program: &str,
    args: &[&str],
    notFoundMsg: &str,
) -> Result<String, String> {
    use tokio::process::Command;

    let result = Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let child = match result {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(notFoundMsg.to_string());
        }
        Err(e) => return Err(format!("Failed to start {program}: {e}")),
    };

    let timeout = tokio::time::Duration::from_secs(SUBPROCESS_TIMEOUT_SECS);
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if output.status.success() || output.status.code() == Some(1) {
                // rg returns 1 for "no matches" — treat as success.
                Ok(stdout)
            } else {
                let msg = if stderr.is_empty() { &stdout } else { &stderr };
                Err(format!("{program} failed (exit {}): {}", output.status, msg.trim()))
            }
        }
        Ok(Err(e)) => Err(format!("Failed to run {program}: {e}")),
        Err(_) => {
            // Process is still running but we lost ownership via wait_with_output.
            // The child is dropped here which sends SIGKILL on Unix.
            Err(format!("{program} timed out after {SUBPROCESS_TIMEOUT_SECS}s."))
        }
    }
}

// --- Search / structure / diff execute functions ---

async fn executeGlob(pattern: &str, path: Option<&str>) -> String {
    let mut args = vec![
        "--files", "--sort", "modified", "--hidden",
        "--glob", pattern, "--glob", "!.git/",
    ];
    if let Some(p) = path {
        args.push(p);
    }

    match runSubprocess("rg", &args, "ripgrep (rg) not found. Install: https://github.com/BurntSushi/ripgrep").await {
        Ok(stdout) => {
            if stdout.trim().is_empty() {
                return "No files found.".into();
            }
            let lines: Vec<&str> = stdout.lines().collect();
            let total = lines.len();
            let mut output = String::new();
            for line in lines.iter().take(MAX_GLOB_RESULTS) {
                output.push_str(line);
                output.push('\n');
            }
            if total > MAX_GLOB_RESULTS {
                output.push_str(&format!(
                    "\n... {total} files found, showing first {MAX_GLOB_RESULTS}."
                ));
            }
            output
        }
        Err(e) => e,
    }
}

/// Get symbol definitions for a file using ast-grep.
/// Returns sorted (lineNumber, symbolSignature) pairs.
fn getFileSymbols(path: &str) -> Vec<(usize, String)> {
    let lang = detectLanguage(path);
    let patterns = outlinePatterns(&lang);

    if patterns.is_empty() {
        return Vec::new();
    }

    let mut entries: Vec<(usize, String)> = Vec::new();

    for pattern in &patterns {
        let output = std::process::Command::new("sg")
            .args(["run", "-p", pattern, "-l", &lang, "--json=compact", path])
            .output();

        if let Ok(output) = output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                    let lineNum = obj["range"]["start"]["line"]
                        .as_u64()
                        .map(|l| l + 1)
                        .unwrap_or(0) as usize;
                    let text = obj["text"].as_str().unwrap_or("");
                    let firstLine = text.lines().next().unwrap_or("").trim().to_string();
                    let display = if firstLine.len() > 80 {
                        format!("{}...", &firstLine[..firstLine.floor_char_boundary(80)])
                    } else {
                        firstLine
                    };
                    if !display.is_empty() {
                        entries.push((lineNum, display));
                    }
                }
            }
        }
    }

    entries.sort_by_key(|(line, _)| *line);
    entries.dedup_by_key(|(line, _)| *line);
    entries
}

/// Find the enclosing symbol for a given line number.
/// Returns the symbol signature from the last definition before or at that line.
fn symbolAtLine<'a>(symbols: &'a [(usize, String)], line: usize) -> Option<&'a str> {
    // Binary search for the last symbol with startLine <= line.
    let idx = symbols.partition_point(|(l, _)| *l <= line);
    if idx == 0 {
        return None;
    }
    Some(&symbols[idx - 1].1)
}

/// Annotate rg content-mode output with enclosing symbol headers.
/// Inserts `── file > symbol ──` lines when matches cross symbol boundaries.
fn annotateGrepWithSymbols(rgOutput: &str) -> String {
    // Parse rg output to find unique files — match lines have format file:line:content.
    let matchLineRe = regex::Regex::new(r"^(.+?):(\d+):").unwrap();

    // Collect unique files.
    let mut fileSet = std::collections::HashSet::new();
    for line in rgOutput.lines() {
        if let Some(caps) = matchLineRe.captures(line) {
            fileSet.insert(caps[1].to_string());
        }
    }

    // Build symbol maps for each file (cap at 10 files to avoid excessive I/O).
    let mut symbolMaps: std::collections::HashMap<String, Vec<(usize, String)>> =
        std::collections::HashMap::new();
    for (count, file) in fileSet.iter().enumerate() {
        if count >= 10 {
            break;
        }
        let symbols = getFileSymbols(file);
        if !symbols.is_empty() {
            symbolMaps.insert(file.clone(), symbols);
        }
    }

    // If no symbols found for any file, return output unchanged.
    if symbolMaps.is_empty() {
        return rgOutput.to_string();
    }

    // Walk through output lines, inserting symbol headers when scope changes.
    let mut output = String::new();
    let mut lastSymbol: Option<String> = None;
    let mut lastFile: Option<String> = None;

    for line in rgOutput.lines() {
        if let Some(caps) = matchLineRe.captures(line) {
            let file = &caps[1];
            let lineNum: usize = caps[2].parse().unwrap_or(0);

            if let Some(symbols) = symbolMaps.get(file) {
                let currentSymbol = symbolAtLine(symbols, lineNum).map(String::from);
                let fileChanged = lastFile.as_deref() != Some(file);
                let symbolChanged = currentSymbol != lastSymbol;

                if fileChanged || symbolChanged {
                    if let Some(ref sym) = currentSymbol {
                        output.push_str(&format!("── {file} > {sym} ──\n"));
                    }
                    lastSymbol = currentSymbol;
                    lastFile = Some(file.to_string());
                }
            } else {
                lastFile = Some(file.to_string());
                lastSymbol = None;
            }
        }

        output.push_str(line);
        output.push('\n');
    }

    output
}

async fn executeGrep(
    pattern: &str,
    path: Option<&str>,
    include: Option<&str>,
    fileType: Option<&str>,
    outputMode: &str,
    caseSensitive: Option<bool>,
    contextLines: Option<usize>,
    multiline: bool,
) -> String {
    let mut argStrings: Vec<String> = Vec::new();

    // Output mode flags.
    match outputMode {
        "files" => argStrings.push("--files-with-matches".into()),
        "count" => argStrings.push("--count".into()),
        _ => {
            // Content mode.
            let ctx = contextLines.unwrap_or(2);
            argStrings.push(format!("--context={ctx}"));
            argStrings.push("--line-number".into());
        }
    }

    // Case sensitivity.
    match caseSensitive {
        Some(true) => argStrings.push("--case-sensitive".into()),
        Some(false) => argStrings.push("--ignore-case".into()),
        None => {} // Smart-case is rg default.
    }

    // Multiline.
    if multiline {
        argStrings.push("--multiline".into());
        argStrings.push("--multiline-dotall".into());
    }

    // Include glob filter.
    if let Some(g) = include {
        argStrings.push("--glob".into());
        argStrings.push(g.to_string());
    }

    // Type filter.
    if let Some(t) = fileType {
        argStrings.push("--type".into());
        argStrings.push(t.to_string());
    }

    // Always exclude .git.
    argStrings.push("--hidden".into());
    argStrings.push("--glob".into());
    argStrings.push("!.git/".into());

    // Pattern and path.
    argStrings.push(pattern.to_string());
    if let Some(p) = path {
        argStrings.push(p.to_string());
    }

    let args: Vec<&str> = argStrings.iter().map(|s| s.as_str()).collect();

    match runSubprocess("rg", &args, "ripgrep (rg) not found. Install: https://github.com/BurntSushi/ripgrep").await {
        Ok(stdout) => {
            if stdout.trim().is_empty() {
                return "No matches found.".into();
            }
            let lines: Vec<&str> = stdout.lines().collect();
            let cap = match outputMode {
                "content" => MAX_GREP_CONTENT_LINES,
                _ => MAX_GREP_FILES,
            };
            let total = lines.len();
            let mut truncated = String::new();
            for line in lines.iter().take(cap) {
                truncated.push_str(line);
                truncated.push('\n');
            }

            // Annotate content-mode output with enclosing symbol context.
            let mut output = if outputMode == "content" {
                annotateGrepWithSymbols(&truncated)
            } else {
                truncated
            };

            if total > cap {
                output.push_str(&format!(
                    "\n... {total} lines total, showing first {cap}."
                ));
            }
            output
        }
        Err(e) => e,
    }
}

fn executeListDir(path: &str, depth: usize, offset: usize, limit: usize) -> String {
    const EXCLUDED: &[&str] = &[".git", "node_modules", "target", "__pycache__", ".venv"];

    let rootPath = std::path::Path::new(path);
    if !rootPath.is_dir() {
        return format!("Not a directory: {path}");
    }

    // Collect all entries first (up to a hard cap), then paginate.
    let hardCap = MAX_LISTDIR_ENTRIES.max(offset + limit);
    let mut allEntries = Vec::new();
    let mut count = 0usize;
    let truncated = listDirRecurse(rootPath, 0, depth, "", &mut allEntries, &mut count, EXCLUDED, hardCap);
    let total = allEntries.len();

    if total == 0 {
        return format!("Empty directory: {path}");
    }

    // Apply pagination.
    let pageEntries: Vec<_> = allEntries.into_iter().skip(offset).take(limit).collect();

    if pageEntries.is_empty() {
        return format!("Offset {offset} is past the end ({total} entries total).");
    }

    let mut result = pageEntries.join("\n");
    result.push('\n');

    let shown = pageEntries.len();
    let remaining = total.saturating_sub(offset + shown);
    if remaining > 0 || truncated {
        result.push_str(&format!(
            "\nShowing {shown} of {total} entries (offset {offset})."
        ));
        if truncated {
            result.push_str(" Directory has more entries beyond the scan limit.");
        }
    }
    result
}

/// Recursive DFS for listDir. Returns true if truncated.
fn listDirRecurse(
    dir: &std::path::Path,
    currentDepth: usize,
    maxDepth: usize,
    indent: &str,
    output: &mut Vec<String>,
    count: &mut usize,
    excluded: &[&str],
    hardCap: usize,
) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };

    // Collect and sort: dirs first, then files, alphabetical within each group.
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let fileType = entry.file_type();
        let isDir = fileType.as_ref().map(|ft| ft.is_dir()).unwrap_or(false);
        let isSymlink = fileType.as_ref().map(|ft| ft.is_symlink()).unwrap_or(false);

        if isDir && excluded.contains(&name.as_str()) {
            continue;
        }

        if isDir {
            dirs.push((name, isSymlink));
        } else {
            files.push((name, isSymlink));
        }
    }
    dirs.sort_by(|a, b| a.0.cmp(&b.0));
    files.sort_by(|a, b| a.0.cmp(&b.0));

    // Emit dirs first.
    for (name, isSymlink) in &dirs {
        if *count >= hardCap {
            return true;
        }
        let suffix = if *isSymlink { "@ -> /" } else { "/" };
        output.push(format!("{indent}{name}{suffix}"));
        *count += 1;

        if currentDepth + 1 < maxDepth {
            let childIndent = format!("{indent}  ");
            let childPath = dir.join(name);
            if listDirRecurse(&childPath, currentDepth + 1, maxDepth, &childIndent, output, count, excluded, hardCap) {
                return true;
            }
        }
    }

    // Then files.
    for (name, isSymlink) in &files {
        if *count >= hardCap {
            return true;
        }
        let suffix = if *isSymlink { "@" } else { "" };
        output.push(format!("{indent}{name}{suffix}"));
        *count += 1;
    }

    false
}

async fn executeStructSearch(pattern: &str, language: &str, path: Option<&str>) -> String {
    let mut args = vec!["run", "-p", pattern, "-l", language, "--json=compact"];
    if let Some(p) = path {
        args.push(p);
    }

    let notFound = "ast-grep (sg) not available. Install: https://ast-grep.github.io \
                    Use grep for text-based search.";

    match runSubprocess("sg", &args, notFound).await {
        Ok(stdout) => {
            if stdout.trim().is_empty() {
                return "No matches found.".into();
            }
            formatStructSearchOutput(&stdout)
        }
        Err(e) => e,
    }
}

fn formatStructSearchOutput(jsonOutput: &str) -> String {
    let mut output = String::new();

    // ast-grep --json=compact returns a JSON array, not newline-delimited objects.
    let matches: Vec<serde_json::Value> = match serde_json::from_str(jsonOutput) {
        Ok(v) => v,
        Err(e) => return format!("Failed to parse ast-grep output: {e}"),
    };

    for (_matchCount, obj) in matches.iter().enumerate().take(MAX_STRUCT_MATCHES) {
        let file = obj["file"].as_str().unwrap_or("?");
        let startLine = obj["range"]["start"]["line"]
            .as_u64()
            .map(|l| l + 1)
            .unwrap_or(0);
        let text = obj["text"].as_str().unwrap_or("");

        output.push_str(&format!("{file}:{startLine}\n"));

        // Show up to 5 lines of matched text, indented.
        for (i, matchLine) in text.lines().enumerate() {
            if i >= 5 {
                output.push_str("    ...\n");
                break;
            }
            output.push_str(&format!("    {matchLine}\n"));
        }

        // Show meta-variable bindings if present.
        if let Some(metaVars) = obj["metaVariables"].as_object() {
            if !metaVars.is_empty() {
                for (name, val) in metaVars {
                    // NOTE: ast-grep meta-vars can be objects with "text" field.
                    let binding = if let Some(t) = val["single"]["text"].as_str() {
                        t.to_string()
                    } else if let Some(arr) = val["multi"].as_array() {
                        arr.iter()
                            .filter_map(|v| v["text"].as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    } else {
                        continue;
                    };
                    if !binding.is_empty() {
                        output.push_str(&format!("    {name} = {binding}\n"));
                    }
                }
            }
        }

        output.push('\n');
    }

    let totalMatches = matches.len();
    if totalMatches > MAX_STRUCT_MATCHES {
        output.push_str(&format!(
            "... {totalMatches} matches total, showing first {MAX_STRUCT_MATCHES}."
        ));
    } else {
        output.push_str(&format!("{totalMatches} match(es)."));
    }

    output
}

async fn executeDiff(
    path: Option<&str>,
    gitRef: Option<&str>,
    pathA: Option<&str>,
    pathB: Option<&str>,
) -> String {
    // File-vs-file mode.
    if let (Some(a), Some(b)) = (pathA, pathB) {
        return diffTwoFiles(a, b);
    }

    // Git diff mode.
    if let Some(p) = path {
        let reference = gitRef.unwrap_or("HEAD");
        return diffGitRef(p, reference).await;
    }

    // Bare git diff (no path, no pathA/pathB) — show unstaged changes.
    if pathA.is_none() && pathB.is_none() && path.is_none() {
        let reference = gitRef.unwrap_or("HEAD");
        let args = vec!["diff", reference];
        match runSubprocess("git", &args, "git not found.").await {
            Ok(stdout) => {
                if stdout.trim().is_empty() {
                    return format!("No differences against {reference}.");
                }
                return truncateDiffOutput(&stdout);
            }
            Err(e) => return e,
        }
    }

    "Provide 'path' + optional 'ref' for git diff, or 'path_a' + 'path_b' for file diff.".into()
}

fn diffTwoFiles(pathA: &str, pathB: &str) -> String {
    let contentA = match std::fs::read_to_string(pathA) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read {pathA}: {e}"),
    };
    let contentB = match std::fs::read_to_string(pathB) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read {pathB}: {e}"),
    };

    let diff = similar::TextDiff::configure()
        .algorithm(similar::Algorithm::Patience)
        .diff_lines(&contentA, &contentB);

    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(pathA, pathB)
        .to_string();

    if unified.trim().is_empty() {
        return "Files are identical.".into();
    }

    truncateDiffOutput(&unified)
}

async fn diffGitRef(path: &str, reference: &str) -> String {
    let args = vec!["diff", reference, "--", path];

    match runSubprocess("git", &args, "git not found.").await {
        Ok(stdout) => {
            if stdout.trim().is_empty() {
                return format!("No differences for {path} against {reference}.");
            }
            truncateDiffOutput(&stdout)
        }
        Err(e) => e,
    }
}

fn truncateDiffOutput(diff: &str) -> String {
    let lines: Vec<&str> = diff.lines().collect();
    if lines.len() <= MAX_READ_LINES && diff.len() <= MAX_READ_BYTES {
        return diff.to_string();
    }

    let mut output = String::new();
    let mut byteCount = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if i >= MAX_READ_LINES || byteCount + line.len() + 1 > MAX_READ_BYTES {
            let remaining = lines.len() - i;
            output.push_str(&format!("\n... truncated ({remaining} more lines)."));
            break;
        }
        output.push_str(line);
        output.push('\n');
        byteCount += line.len() + 1;
    }
    output
}

// --- Anchor mode for readFile ---

/// Expand from an anchor line based on indentation to return the enclosing block.
fn expandFromAnchor(content: &str, anchorLine: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let totalLines = lines.len();

    if anchorLine == 0 || anchorLine > totalLines {
        return format!("Anchor line {anchorLine} out of range (file has {totalLines} lines).");
    }

    let anchorIdx = anchorLine - 1;
    let anchorIndent = indentLevel(lines[anchorIdx]);

    // Walk backward to find the block start — a line with strictly less indentation.
    let mut startIdx = anchorIdx;
    for i in (0..anchorIdx).rev() {
        let line = lines[i];
        if line.trim().is_empty() {
            continue;
        }
        let indent = indentLevel(line);
        if indent < anchorIndent {
            // This line is the parent scope — include it as context.
            startIdx = i;
            break;
        }
        startIdx = i;
    }

    // Walk forward to find the block end.
    let mut endIdx = anchorIdx;
    for i in (anchorIdx + 1)..totalLines {
        let line = lines[i];
        if line.trim().is_empty() {
            continue;
        }
        let indent = indentLevel(line);
        if indent < anchorIndent {
            // Include the closing line (e.g. closing brace).
            endIdx = i;
            break;
        }
        endIdx = i;
    }

    // Cap the block to MAX_READ_LINES.
    let blockLen = endIdx - startIdx + 1;
    let effectiveEnd = if blockLen > MAX_READ_LINES {
        startIdx + MAX_READ_LINES - 1
    } else {
        endIdx
    };

    let mut output = String::new();
    for i in startIdx..=effectiveEnd {
        let lineNum = i + 1;
        let displayLine = if lines[i].len() > MAX_LINE_LENGTH {
            format!("{}...", &lines[i][..MAX_LINE_LENGTH])
        } else {
            lines[i].to_string()
        };
        output.push_str(&format!("{lineNum:>6}\t{displayLine}\n"));
    }

    if effectiveEnd < endIdx {
        let remaining = endIdx - effectiveEnd;
        output.push_str(&format!("\n... block truncated ({remaining} more lines)."));
    }

    output
}

/// Count leading whitespace as an indentation level (spaces + tabs*4).
fn indentLevel(line: &str) -> usize {
    let mut level = 0;
    for ch in line.chars() {
        match ch {
            ' ' => level += 1,
            '\t' => level += 4,
            _ => break,
        }
    }
    level
}

// --- Fuzzy find ---

async fn executeFuzzyFind(query: &str, path: Option<&str>) -> String {
    use nucleo_matcher::{Matcher, Config};
    use nucleo_matcher::pattern::{Pattern, CaseMatching, Normalization};

    let mut args = vec!["--files", "--hidden", "--glob", "!.git/"];
    if let Some(p) = path {
        args.push(p);
    }

    let stdout = match runSubprocess(
        "rg", &args, "ripgrep (rg) not found. Install: https://github.com/BurntSushi/ripgrep",
    ).await {
        Ok(s) => s,
        Err(e) => return e,
    };

    if stdout.trim().is_empty() {
        return "No files found.".into();
    }

    let files: Vec<&str> = stdout.lines().collect();
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let pattern = Pattern::parse(query, CaseMatching::Smart, Normalization::Smart);
    let matches = pattern.match_list(&files, &mut matcher);

    if matches.is_empty() {
        return format!("No files matched \"{query}\".");
    }

    // match_list returns sorted by score descending already.
    let mut output = String::new();
    for (path, score) in matches.iter().take(MAX_FUZZY_RESULTS) {
        output.push_str(&format!("{score:>4}  {path}\n"));
    }
    if matches.len() > MAX_FUZZY_RESULTS {
        output.push_str(&format!(
            "\n... {} more matches. Refine your query.", matches.len() - MAX_FUZZY_RESULTS
        ));
    }
    output
}

// --- File outline ---

async fn executeFileOutline(path: &str) -> String {
    let lang = detectLanguage(path);
    let patterns = outlinePatterns(&lang);

    if patterns.is_empty() {
        return format!("No outline patterns for language \"{lang}\". File: {path}");
    }

    let mut entries: Vec<(usize, String)> = Vec::new();

    for pattern in &patterns {
        let args = vec!["run", "-p", pattern, "-l", &lang, "--json=compact", path];
        match runSubprocess("sg", &args, "ast-grep (sg) is required for fileOutline. Install: https://ast-grep.github.io").await {
            Ok(stdout) => {
                for line in stdout.lines() {
                    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                        let lineNum = obj["range"]["start"]["line"]
                            .as_u64()
                            .map(|l| l + 1)
                            .unwrap_or(0) as usize;
                        let text = obj["text"].as_str().unwrap_or("");
                        let firstLine = text.lines().next().unwrap_or("").trim().to_string();
                        if !firstLine.is_empty() {
                            entries.push((lineNum, firstLine));
                        }
                    }
                }
            }
            Err(e) => return e,
        }
    }

    if entries.is_empty() {
        return format!("No symbols found in {path}.");
    }

    // Deduplicate by line number and sort.
    entries.sort_by_key(|(line, _)| *line);
    entries.dedup_by_key(|(line, _)| *line);

    let mut output = String::new();
    for (line, text) in entries.iter().take(MAX_OUTLINE_ENTRIES) {
        output.push_str(&format!("{line:>6}  {text}\n"));
    }
    if entries.len() > MAX_OUTLINE_ENTRIES {
        output.push_str(&format!(
            "\n... {} more symbols.", entries.len() - MAX_OUTLINE_ENTRIES
        ));
    }
    output
}

// --- View symbol ---

async fn executeViewSymbol(file: &str, symbol: &str) -> String {
    let lang = detectLanguage(file);

    // Support qualified paths like "ToolAction::Grep" or "Foo::bar::baz".
    let parts: Vec<&str> = symbol.split("::").collect();

    if parts.len() == 1 {
        // Simple symbol lookup.
        return viewSymbolSingle(file, symbol, &lang).await;
    }

    // Qualified path: find outermost symbol, then narrow into nested ones.
    let content = match std::fs::read_to_string(file) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read {file}: {e}"),
    };

    // Find the outermost symbol first.
    let outerName = parts[0];
    let outerBlock = match findSymbolRange(&content, outerName, &lang).await {
        Some(range) => range,
        None => return format!("Symbol \"{outerName}\" not found in {file}."),
    };

    // Walk inward through the chain.
    let currentText = outerBlock.text.clone();
    let currentStart = outerBlock.startLine;

    for &part in &parts[1..] {
        // Search within the current block text for the next symbol.
        let found = false;
        for (idx, line) in currentText.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains(part) && (looksLikeDeclaration(trimmed, part) || looksLikeVariant(trimmed, part)) {
                let anchorLine = currentStart + idx;
                let expanded = expandFromAnchor(&currentText, idx + 1);
                // Re-anchor the expanded block to the file line numbers.
                let mut output = String::new();
                for expandedLine in expanded.lines() {
                    // The expanded output has line numbers relative to currentText.
                    // Re-number relative to the file.
                    if let Some(tabPos) = expandedLine.find('\t') {
                        let numStr = expandedLine[..tabPos].trim();
                        if let Ok(relLine) = numStr.parse::<usize>() {
                            let absLine = currentStart + relLine - 1;
                            output.push_str(&format!("{absLine:>6}\t{}\n", &expandedLine[tabPos + 1..]));
                            continue;
                        }
                    }
                    output.push_str(expandedLine);
                    output.push('\n');
                }
                return format!("{file}:{anchorLine} ({symbol})\n\n{output}");
            }
        }
        if !found {
            // Can't narrow further — return what we have of the outer block.
            return format!("{file}:{currentStart} (found {outerName}, \"{part}\" not found within)\n\n{currentText}");
        }
    }

    format!("{file}:{currentStart}\n\n{currentText}")
}

/// Find a symbol's range within a content string by declaration matching.
struct SymbolRange {
    startLine: usize,
    text: String,
}

async fn findSymbolRange(content: &str, name: &str, _lang: &str) -> Option<SymbolRange> {
    let mut foundLine: Option<usize> = None;
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.contains(name) && (looksLikeDeclaration(trimmed, name) || looksLikeVariant(trimmed, name)) {
            foundLine = Some(idx + 1);
            break;
        }
    }

    let lineNum = foundLine?;
    let expanded = expandFromAnchor(content, lineNum);

    // Parse the expanded output to extract the text (strip line numbers).
    let mut text = String::new();
    for line in expanded.lines() {
        if let Some(tabPos) = line.find('\t') {
            text.push_str(&line[tabPos + 1..]);
            text.push('\n');
        }
    }

    Some(SymbolRange { startLine: lineNum, text })
}

/// Simple single-name symbol lookup (original behavior).
async fn viewSymbolSingle(file: &str, symbol: &str, lang: &str) -> String {
    let patterns = symbolPatterns(lang, symbol);

    // Try ast-grep first.
    for pattern in &patterns {
        let args = vec!["run", "-p", pattern, "-l", lang, "--json=compact", file];
        if let Ok(stdout) = runSubprocess("sg", &args, "").await {
            let matches: Vec<serde_json::Value> = match serde_json::from_str(&stdout) {
                Ok(v) => v,
                Err(_) => continue,
            };
            for obj in &matches {
                let text = obj["text"].as_str().unwrap_or("");
                let startLine = obj["range"]["start"]["line"]
                    .as_u64()
                    .map(|l| l + 1)
                    .unwrap_or(0);
                if !text.is_empty() {
                    return format!("{file}:{startLine}\n\n{text}");
                }
            }
        }
    }

    format!("Symbol \"{symbol}\" not found in {file} via ast-grep.")
}

/// Heuristic: does this line look like it declares the given symbol?
fn looksLikeDeclaration(line: &str, symbol: &str) -> bool {
    // Check if symbol appears after common declaration keywords.
    let declarationPrefixes = [
        "fn ", "pub fn ", "async fn ", "pub async fn ",
        "struct ", "pub struct ", "enum ", "pub enum ",
        "trait ", "pub trait ", "impl ", "type ", "pub type ",
        "const ", "pub const ", "static ", "pub static ",
        "mod ", "pub mod ",
        "def ", "async def ", "class ",
        "function ", "export function ", "export default function ",
        "export const ", "export let ", "export class ",
        "interface ", "export interface ", "export type ",
        "func ", "var ", "let ", "const ",
    ];

    for prefix in &declarationPrefixes {
        if let Some(rest) = line.strip_prefix(prefix) {
            if rest.starts_with(symbol) {
                return true;
            }
        }
    }

    false
}

/// Heuristic: does this line look like an enum variant or struct field with this name?
fn looksLikeVariant(line: &str, name: &str) -> bool {
    // Match patterns like "Grep {", "Grep(", "Grep," (enum variants).
    if let Some(pos) = line.find(name) {
        let afterName = &line[pos + name.len()..].trim_start();
        if afterName.starts_with('{') || afterName.starts_with('(')
            || afterName.starts_with(',') || afterName.starts_with(';')
            || afterName.is_empty()
        {
            // Make sure it's at a word boundary (not a substring of a longer name).
            if pos == 0 || !line.as_bytes()[pos - 1].is_ascii_alphanumeric() {
                return true;
            }
        }
    }
    false
}

// --- Related files ---

fn executeRelatedFiles(path: &str) -> String {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read {path}: {e}"),
    };

    let lang = detectLanguage(path);
    let imports = parseImports(&content, &lang);
    let filePath = std::path::Path::new(path);
    let fileDir = filePath.parent();

    // Resolve import paths to real files.
    let mut resolved: Vec<String> = Vec::new();
    for imp in &imports {
        // Try resolving relative to the file's directory.
        if let Some(dir) = fileDir {
            let candidate = dir.join(imp);
            if candidate.exists() {
                resolved.push(candidate.to_string_lossy().to_string());
                continue;
            }
            // Try with common extensions.
            for ext in &[".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go"] {
                let withExt = dir.join(format!("{imp}{ext}"));
                if withExt.exists() {
                    resolved.push(withExt.to_string_lossy().to_string());
                    break;
                }
            }
        }
    }

    // Sibling files in the same directory.
    let mut siblings: Vec<String> = Vec::new();
    if let Some(dir) = fileDir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let canonPath = filePath.canonicalize().ok();
            for entry in entries.flatten() {
                let entryFt = entry.file_type();
                if entryFt.map(|ft| ft.is_file()).unwrap_or(false) {
                    let entryCanon = entry.path().canonicalize().ok();
                    if entryCanon != canonPath {
                        siblings.push(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
    }
    siblings.sort();

    let mut output = String::new();

    if !imports.is_empty() {
        output.push_str("Imports/dependencies:\n");
        for imp in &imports {
            output.push_str(&format!("  {imp}\n"));
        }
    }

    if !resolved.is_empty() {
        output.push_str("\nResolved files:\n");
        for r in resolved.iter().take(MAX_RELATED_FILES) {
            output.push_str(&format!("  {r}\n"));
        }
    }

    if !siblings.is_empty() {
        output.push_str("\nSibling files:\n");
        for sib in siblings.iter().take(MAX_RELATED_FILES) {
            output.push_str(&format!("  {sib}\n"));
        }
    }

    if output.is_empty() {
        "No related files found.".into()
    } else {
        output
    }
}

/// Parse import statements from file content based on language.
fn parseImports(content: &str, lang: &str) -> Vec<String> {
    let mut imports = Vec::new();
    let patterns: Vec<&str> = match lang {
        "rust" => vec![
            r"^use\s+([\w:]+)",
            r"^mod\s+(\w+)\s*;",
        ],
        "python" => vec![
            r"^(?:from\s+([\w.]+)\s+)?import\s+([\w.]+)",
        ],
        "typescript" | "javascript" | "tsx" | "jsx" => vec![
            r#"(?:import|require)\s*\(?[^)]*['"]([^'"]+)['"]"#,
        ],
        "go" => vec![
            r#"^\s*"([^"]+)""#,
        ],
        _ => vec![],
    };

    for pat in patterns {
        if let Ok(re) = regex::Regex::new(pat) {
            for line in content.lines() {
                if let Some(caps) = re.captures(line.trim()) {
                    // Take the last non-empty capture group.
                    for i in (1..caps.len()).rev() {
                        if let Some(m) = caps.get(i) {
                            let val = m.as_str().to_string();
                            if !val.is_empty() && !imports.contains(&val) {
                                imports.push(val);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    imports
}

// --- Language detection and pattern helpers ---

/// Detect programming language from file extension.
fn detectLanguage(path: &str) -> String {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "ts" => "typescript",
        "tsx" => "tsx",
        "js" => "javascript",
        "jsx" => "jsx",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "rb" => "ruby",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "cs" => "csharp",
        "lua" => "lua",
        "zig" => "zig",
        _ => ext,
    }.into()
}

/// ast-grep patterns for file outline by language.
fn outlinePatterns(lang: &str) -> Vec<&'static str> {
    match lang {
        "rust" => vec![
            "fn $NAME($$$) $BODY",
            "pub fn $NAME($$$) $BODY",
            "async fn $NAME($$$) $BODY",
            "pub async fn $NAME($$$) $BODY",
            "struct $NAME $BODY",
            "pub struct $NAME $BODY",
            "enum $NAME $BODY",
            "pub enum $NAME $BODY",
            "trait $NAME $BODY",
            "pub trait $NAME $BODY",
            "impl $TYPE $BODY",
            "mod $NAME",
            "pub mod $NAME",
            "type $NAME = $TYPE;",
            "pub type $NAME = $TYPE;",
            "const $NAME: $TYPE = $EXPR;",
            "pub const $NAME: $TYPE = $EXPR;",
        ],
        "python" => vec![
            "def $NAME($$$): $BODY",
            "async def $NAME($$$): $BODY",
            "class $NAME: $BODY",
            "class $NAME($$$): $BODY",
        ],
        "typescript" | "javascript" | "tsx" | "jsx" => vec![
            "function $NAME($$$) $BODY",
            "export function $NAME($$$) $BODY",
            "export default function $NAME($$$) $BODY",
            "class $NAME $BODY",
            "export class $NAME $BODY",
            "interface $NAME $BODY",
            "export interface $NAME $BODY",
            "type $NAME = $TYPE",
            "export type $NAME = $TYPE",
        ],
        "go" => vec![
            "func $NAME($$$) $BODY",
            "func ($RECV) $NAME($$$) $BODY",
            "type $NAME struct $BODY",
            "type $NAME interface $BODY",
        ],
        _ => vec![],
    }
}

/// ast-grep patterns to find a specific symbol by name.
fn symbolPatterns(lang: &str, symbol: &str) -> Vec<String> {
    match lang {
        "rust" => vec![
            format!("fn {symbol}($$$) $BODY"),
            format!("pub fn {symbol}($$$) $BODY"),
            format!("async fn {symbol}($$$) $BODY"),
            format!("pub async fn {symbol}($$$) $BODY"),
            format!("struct {symbol} $BODY"),
            format!("pub struct {symbol} $BODY"),
            format!("enum {symbol} $BODY"),
            format!("pub enum {symbol} $BODY"),
            format!("trait {symbol} $BODY"),
            format!("pub trait {symbol} $BODY"),
            format!("impl {symbol} $BODY"),
            format!("mod {symbol}"),
            format!("pub mod {symbol}"),
            format!("type {symbol} = $TYPE;"),
            format!("pub type {symbol} = $TYPE;"),
            format!("const {symbol}: $TYPE = $EXPR;"),
            format!("pub const {symbol}: $TYPE = $EXPR;"),
        ],
        "python" => vec![
            format!("def {symbol}($$$): $BODY"),
            format!("async def {symbol}($$$): $BODY"),
            format!("class {symbol}: $BODY"),
            format!("class {symbol}($$$): $BODY"),
        ],
        "typescript" | "javascript" | "tsx" | "jsx" => vec![
            format!("function {symbol}($$$) $BODY"),
            format!("export function {symbol}($$$) $BODY"),
            format!("class {symbol} $BODY"),
            format!("export class {symbol} $BODY"),
            format!("interface {symbol} $BODY"),
            format!("export interface {symbol} $BODY"),
            format!("const {symbol} = $EXPR"),
            format!("export const {symbol} = $EXPR"),
        ],
        "go" => vec![
            format!("func {symbol}($$$) $BODY"),
            format!("type {symbol} struct $BODY"),
            format!("type {symbol} interface $BODY"),
        ],
        _ => vec![],
    }
}

/// Parse a tool call name + JSON arguments into a ToolAction.
///
/// Returns Err with a message listing missing/malformed required fields.
/// The error message is sent back to the model as the tool result so it can retry.
pub fn parse(name: &str, argsJson: &str) -> Result<ToolAction, String> {
    let args: serde_json::Value = serde_json::from_str(argsJson)
        .map_err(|e| format!("Malformed JSON arguments: {e}"))?;

    /// Extract a required string field, or collect its name into `missing`.
    macro_rules! reqStr {
        ($field:expr) => {
            match args[$field].as_str() {
                Some(s) if !s.is_empty() => s.to_string(),
                Some(_) => return Err(format!("Missing required field '{}'.", $field)),
                None => return Err(format!("Missing required field '{}'.", $field)),
            }
        };
    }

    /// Extract an optional string field. Returns Err if present but wrong type.
    macro_rules! optStr {
        ($field:expr) => {
            match &args[$field] {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Null => None,
                _ => return Err(format!("Field '{}': expected string.", $field)),
            }
        };
    }

    /// Extract an optional u64 field. Returns Err if present but wrong type.
    macro_rules! optU64 {
        ($field:expr) => {
            match &args[$field] {
                serde_json::Value::Number(n) => n.as_u64(),
                serde_json::Value::Null => None,
                _ => return Err(format!("Field '{}': expected integer.", $field)),
            }
        };
    }

    /// Extract an optional bool field. Returns Err if present but wrong type.
    macro_rules! optBool {
        ($field:expr) => {
            match &args[$field] {
                serde_json::Value::Bool(b) => Some(*b),
                serde_json::Value::Null => None,
                _ => return Err(format!("Field '{}': expected boolean.", $field)),
            }
        };
    }

    let action = match name {
        "shell" => {
            let impact: ShellImpact = args["impact"]
                .as_str()
                .and_then(|s| serde_json::from_value(serde_json::Value::String(s.into())).ok())
                .ok_or_else(|| "Missing required field 'impact' (one of: read, minorMod, majorMod, delete).".to_string())?;
            ToolAction::Shell {
                command: reqStr!("command"),
                explanation: reqStr!("explanation"),
                impact,
                timeout: optU64!("timeout"),
            }
        }
        "readFile" => ToolAction::ReadFile {
            path: reqStr!("path"),
            offset: optU64!("offset").map(|v| v as usize),
            limit: optU64!("limit").map(|v| v as usize),
            anchor: optU64!("anchor").map(|v| v as usize),
        },
        "writeFile" => ToolAction::WriteFile {
            path: reqStr!("path"),
            content: reqStr!("content"),
        },
        "editFile" => ToolAction::EditFile {
            path: reqStr!("path"),
            oldString: reqStr!("old_string"),
            newString: args["new_string"].as_str().unwrap_or("").into(),
            replaceAll: optBool!("replace_all").unwrap_or(false),
        },
        "multiEdit" => ToolAction::MultiEdit {
            path: reqStr!("path"),
            edits: args["edits"]
                .as_array()
                .ok_or_else(|| "Missing required field 'edits'.".to_string())?
                .iter()
                .map(|e| {
                    Ok(EditOp {
                        oldString: e["old_string"].as_str()
                            .ok_or_else(|| "Edit missing 'old_string'.".to_string())?.into(),
                        newString: e["new_string"].as_str().unwrap_or("").into(),
                        replaceAll: e["replace_all"].as_bool().unwrap_or(false),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        },
        "shellHistory" => ToolAction::ShellHistory,
        "readOutput" => ToolAction::ReadOutput {
            index: optU64!("index").unwrap_or(0) as usize,
            offset: optU64!("offset").map(|v| v as usize),
            limit: optU64!("limit").map(|v| v as usize),
        },
        "searchOutput" => ToolAction::SearchOutput {
            index: optU64!("index").unwrap_or(0) as usize,
            pattern: reqStr!("pattern"),
            context: optU64!("context").unwrap_or(3) as usize,
        },
        "readTerminal" => ToolAction::ReadTerminal {
            lines: optU64!("lines").unwrap_or(50) as usize,
        },
        "glob" => ToolAction::Glob {
            pattern: reqStr!("pattern"),
            path: optStr!("path"),
        },
        "grep" => ToolAction::Grep {
            pattern: reqStr!("pattern"),
            path: optStr!("path"),
            include: optStr!("include"),
            fileType: optStr!("type"),
            outputMode: args["output_mode"].as_str().unwrap_or("files").into(),
            caseSensitive: optBool!("case_sensitive"),
            contextLines: optU64!("context_lines").map(|v| v as usize),
            multiline: optBool!("multiline").unwrap_or(false),
        },
        "listDir" => ToolAction::ListDir {
            path: args["path"].as_str().unwrap_or(".").into(),
            depth: optU64!("depth").unwrap_or(2).min(5).max(1) as usize,
            offset: optU64!("offset").unwrap_or(0) as usize,
            limit: optU64!("limit").unwrap_or(500) as usize,
        },
        "structSearch" => ToolAction::StructSearch {
            pattern: reqStr!("pattern"),
            language: args["language"].as_str().unwrap_or("").into(),
            path: optStr!("path"),
        },
        "diff" => ToolAction::Diff {
            path: optStr!("path"),
            gitRef: optStr!("ref"),
            pathA: optStr!("path_a"),
            pathB: optStr!("path_b"),
        },
        "fuzzyFind" => ToolAction::FuzzyFind {
            query: reqStr!("query"),
            path: optStr!("path"),
        },
        "fileOutline" => ToolAction::FileOutline {
            path: reqStr!("path"),
        },
        "viewSymbol" => ToolAction::ViewSymbol {
            file: reqStr!("file"),
            symbol: reqStr!("symbol"),
        },
        "relatedFiles" => ToolAction::RelatedFiles {
            path: reqStr!("path"),
        },
        "webSearch" => ToolAction::WebSearch {
            query: reqStr!("query"),
            allowedDomains: args["allowed_domains"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()),
            blockedDomains: args["blocked_domains"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()),
            maxResults: optU64!("max_results").map(|v| v as usize),
        },
        "webFetch" => ToolAction::WebFetch {
            url: reqStr!("url"),
            prompt: optStr!("prompt"),
            subpages: optU64!("subpages").map(|v| v as usize),
        },
        "webSimilar" => ToolAction::WebSimilar {
            url: reqStr!("url"),
            allowedDomains: args["allowed_domains"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()),
            blockedDomains: args["blocked_domains"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()),
            maxResults: optU64!("max_results").map(|v| v as usize),
        },
        "historyFetch" => ToolAction::HistoryFetch {
            blockId: reqStr!("blockId"),
        },
        "historySearch" => ToolAction::HistorySearch {
            query: reqStr!("query"),
        },
        "task" => ToolAction::Task {
            prompt: reqStr!("prompt"),
            agent: optStr!("agent"),
        },
        "diagnostics" => ToolAction::Diagnostics {
            path: reqStr!("path"),
            severity: args["severity"].as_str().unwrap_or("error").into(),
        },
        _ if crate::mcp::schema::isMcpTool(name) => ToolAction::Mcp {
            qualifiedName: name.into(),
            args: argsJson.into(),
        },
        _ => ToolAction::Unknown {
            name: name.into(),
            args: argsJson.into(),
        },
    };

    Ok(action)
}
