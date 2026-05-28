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
                description: "Execute a shell command. By default BLOCKS the \
                    turn until the command completes or the timeout elapses; \
                    output is truncated at 2000 lines / 100KB (full output \
                    in shellHistory + readOutput). Runs in the agent's \
                    target terminal \u{2014} pass `terminal` to dispatch \
                    elsewhere.\n\n\
                    AUTO-BG: a foreground call that hits its timeout is \
                    AUTOMATICALLY converted to a background job. You will \
                    receive an AUTO_BG_CONVERTED result with the new job id \
                    and any partial output captured before conversion. The \
                    bg job is a FRESH run of the same command \u{2014} the \
                    foreground attempt was killed and not migrated. If the \
                    command is non-idempotent (writes files, sends network \
                    requests, mutates shared state), prefer setting \
                    `runInBackground: true` up front or a generous `timeout` \
                    so it never trips auto-bg.\n\n\
                    Set `runInBackground: true` for long builds, dev \
                    servers, log tails, or any command whose result you \
                    don't need before continuing. Background calls return \
                    a job id immediately; you'll be notified when the \
                    job completes \u{2014} do NOT poll `jobOutput` while \
                    waiting. Use foreground (default) when you need the \
                    result before you can proceed; background when you \
                    have genuinely independent work to do in parallel.\n\n\
                    LINE BUFFERING: long-running pipes (`cmd | grep ...`, \
                    `ssh host 'tail -F ...'`) block-buffer stdout until \
                    kilobytes accumulate, hiding output for minutes. Use \
                    `grep --line-buffered`, `awk 'BEGIN {{...}}'` with \
                    `fflush()`, `stdbuf -oL <cmd>`, or `ssh host 'stdbuf \
                    -oL tail -F /path'` to keep output flowing.\n\n\
                    Background jobs bypass the named terminal (output \
                    buffered in a 5000-line ring; retrieve via jobOutput, \
                    stop via jobStop)."
                    .into(),
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
                            "description": "Timeout in seconds. Default 30. When exceeded, the command is auto-converted to a background job (you receive the new job id; the fg attempt is killed and the bg job is a fresh re-run). Ignored when runInBackground is true."
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Name of the terminal to run in. Omit to use the agent's target terminal. Ignored when runInBackground is true."
                        },
                        "runInBackground": {
                            "type": "boolean",
                            "description": "Spawn non-blocking. Returns a task id immediately; the command keeps running while you work. You'll be notified when it completes — do not poll. Defaults to false."
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
                    For large files, use offset and limit to read specific ranges."
                    .into(),
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
                    You must readFile the target before overwriting an existing file."
                    .into(),
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
                    You must readFile the target before editing."
                    .into(),
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
                    You must readFile the target before using this tool."
                    .into(),
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
                name: "copyFile".into(),
                description: "Copy a file or directory tree from source to destination. \
                    Creates parent directories of dest as needed. Refuses to overwrite \
                    an existing destination unless overwrite=true."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "description": "Source path."},
                        "dest": {"type": "string", "description": "Destination path."},
                        "overwrite": {
                            "type": "boolean",
                            "description": "Allow overwriting an existing destination. Defaults to false."
                        }
                    },
                    "required": ["src", "dest"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "moveFile".into(),
                description: "Move or rename a file or directory. Creates parent \
                    directories of dest as needed. Refuses to overwrite an existing \
                    destination unless overwrite=true."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "description": "Source path."},
                        "dest": {"type": "string", "description": "Destination path."},
                        "overwrite": {
                            "type": "boolean",
                            "description": "Allow overwriting an existing destination. Defaults to false."
                        }
                    },
                    "required": ["src", "dest"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "deleteFile".into(),
                description: "Delete a file or empty directory. For directory trees, \
                    set recursive=true. This operation is destructive and not undoable."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to delete."},
                        "recursive": {
                            "type": "boolean",
                            "description": "Recursively delete directory contents. Defaults to false."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "makeDirs".into(),
                description: "Create a directory and any missing parents. \
                    Succeeds silently if the directory already exists."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to create."}
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "shellHistory".into(),
                description: "List recent shell commands for a terminal with their index, \
                    exit code, and output size. Use readOutput to read a specific \
                    command's full output. History is per-terminal — pass `terminal` \
                    to inspect a specific terminal, otherwise the agent's target terminal is used."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "terminal": {
                            "type": "string",
                            "description": "Name of the terminal to inspect. Omit to use the agent's target terminal."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "readOutput".into(),
                description: "Read the output of a previous shell command by index. \
                    Indices are per-terminal — pair this with `terminal` if you ran \
                    the command in a non-default terminal. Use shellHistory to see \
                    available commands. Supports offset/limit for navigating large output."
                    .into(),
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
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Name of the terminal whose history to read from. Omit to use the agent's target terminal."
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
                description:
                    "Search a previous command's output for a pattern (regex or substring). \
                    Returns matching lines with surrounding context. Indices are per-terminal — \
                    pair with `terminal` to search a specific terminal's history."
                        .into(),
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
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Name of the terminal whose history to search. Omit to use the agent's target terminal."
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
                description: "Read recent terminal scrollback for a terminal — everything \
                    visible there including user commands and their output. Pair with \
                    `terminal` to read a specific one."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "lines": {
                            "type": "integer",
                            "description": "Number of recent lines to read. Defaults to 50."
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Name of the terminal to read from. Omit to use the agent's target terminal."
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
                    by modification time (newest first). Capped at 100 results. \
                    Set metadata=true to append size and mtime per file."
                    .into(),
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
                        },
                        "metadata": {
                            "type": "boolean",
                            "description": "Append size and mtime to each path. Defaults to false."
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
                    Uses ripgrep with Rust regex syntax: alternation is `|` \
                    (not `\\|`), metachars `.+*?()[]{}|^$\\` need escaping. \
                    Three output modes: 'files' (file paths only), 'content' \
                    (matching lines with context), 'count' (match counts per file)."
                    .into(),
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
                    target, __pycache__, .venv. Capped at 200 entries. \
                    Set metadata=true to append size and mtime per file."
                    .into(),
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
                        },
                        "metadata": {
                            "type": "boolean",
                            "description": "Append size and mtime to each file entry. Defaults to false."
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
                    calls."
                    .into(),
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
                    files directly."
                    .into(),
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
                    roughly what a file is called but not its exact path."
                    .into(),
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
                    declaration lines with line numbers."
                    .into(),
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
                    complete code block."
                    .into(),
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
                description:
                    "Find files related to a given file by parsing its import/use/require \
                    statements and listing sibling files in the same directory. Helps discover \
                    the dependency graph around a file."
                        .into(),
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
                    Use when you need current information beyond your training data."
                    .into(),
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
                    Results cached for 15 minutes."
                    .into(),
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
                    alternative implementations, or similar projects."
                    .into(),
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
                    (user message, assistant responses, tool calls, tool results) in the block, \
                    including any images that were attached."
                    .into(),
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
                    away from the current context. Use mediaType to filter for turns \
                    with specific attachment types (e.g. 'image')."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for across all transcript content."
                        },
                        "mediaType": {
                            "type": "string",
                            "description": "Filter by attachment type (e.g. 'image'). When set, only returns turns that have attachments of this type."
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
                description: "Spawn a subtask agent to handle a focused piece \
                    of work. The agent runs with its own context and shell, \
                    then returns a result. Use 'explore' for read-only \
                    codebase research (cheap, fast model). Use 'general' for \
                    tasks that may modify files (full tools, same model). \
                    Prefer delegating bounded, well-defined work \u{2014} \
                    don't use task for simple operations you can do directly.\n\n\
                    **Foreground vs background**: Use foreground (default) \
                    when you need the agent's results before you can \
                    proceed \u{2014} e.g. research whose findings inform \
                    your next step. Use background when you have genuinely \
                    independent work to do in parallel (fan out 3 explores \
                    for unrelated topics, then continue working).\n\n\
                    When `runInBackground: true`, the call returns \
                    immediately with a task id and you'll be notified when \
                    the subagent completes \u{2014} do NOT poll \
                    `jobOutput` while waiting. Use `jobList` to see \
                    what's still running."
                    .into(),
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
                        },
                        "runInBackground": {
                            "type": "boolean",
                            "description": "Spawn non-blocking and return a task id immediately. You'll be notified on completion — do not poll. Default: false (blocking)."
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
                    available for the file type."
                    .into(),
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
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "terminalSpawn".into(),
                description: "Spawn a new terminal (PTY) and add it as a tab the user can see. \
                    Returns the resolved name. Useful when you need an isolated shell — \
                    e.g. to run a TUI tool that would interfere with the main shell, \
                    or to give the user a tab while you work in another. \
                    Foreground `shell` calls block the turn; use \
                    `shell(runInBackground: true)` for non-blocking commands."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the new terminal. Must be unique. \
                                Letters, digits, dashes, underscores only. \
                                If omitted, a name like 'term2' is auto-generated."
                        }
                    },
                    "required": []
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "terminalSwitch".into(),
                description: "Set the agent's target terminal — the default for subsequent \
                    shell, shellHistory, readOutput, searchOutput, and readTerminal calls \
                    when they omit the `terminal` field. Independent from the user's \
                    focused tab in the deck; switching the agent's target does NOT switch \
                    the user's view."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the terminal to make active."
                        }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "terminalKill".into(),
                description: "Kill a named terminal. Refused if it is the last live \
                    terminal in the session \u{2014} the session must have at least one \
                    PTY. `main` is killable when at least one other terminal exists; \
                    in that case future calls with `terminal: \"main\"` will return \
                    a 'closed' error, so check `terminalList` if you're unsure. Any \
                    in-flight commands targeting the killed terminal will also return \
                    a 'closed' error."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the terminal to kill."
                        }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "terminalList".into(),
                description: "List all live terminals with their names, age, and \
                    which one is currently active."
                    .into(),
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
                name: "jobOutput".into(),
                description: "Read buffered output from a background job \
                    spawned by `shell(runInBackground: true)`. Returns the \
                    latest output lines plus the job's current state \
                    (running, completed, killed, errored). \n\n\
                    You generally don't need to call this proactively \u{2014} \
                    you'll receive a completion notification with the final \
                    output. Use it when you want a mid-flight peek or to \
                    page through historical output via `sinceLine`. Omit \
                    `sinceLine` for the most recent tail; `maxLines` caps \
                    the response (default 200, max 500)."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "jobId": {
                            "type": "integer",
                            "description": "Job id returned by shell(runInBackground: true)."
                        },
                        "sinceLine": {
                            "type": "integer",
                            "description": "Line index to resume from. Each line of output has a \
                                stable 0-indexed line number; the response tells you the next \
                                line index to resume at."
                        },
                        "maxLines": {
                            "type": "integer",
                            "description": "Cap on lines returned. Default 200, max 500."
                        }
                    },
                    "required": ["jobId"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "jobStop".into(),
                description: "Kill a running background job by id. Sends SIGTERM to \
                    the job's process group (so the shell wrapper and any child \
                    processes it spawned), waits briefly, then SIGKILLs anything \
                    still alive. No-op if the job is already terminal."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "jobId": {
                            "type": "integer",
                            "description": "Job id from shell(runInBackground: true)."
                        }
                    },
                    "required": ["jobId"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "jobList".into(),
                description: "List all background jobs (running, completed, killed, errored) \
                    with their command, age, total lines emitted, and state. Use to \
                    rediscover job ids if you've lost track."
                    .into(),
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
                name: "monitor".into(),
                description: "Register a line-streamed watcher backed by a \
                    long-running bash command. Each matching line is a \
                    notification — the model is woken with the matched \
                    line as payload (once the wake plane lands; today \
                    matches are counted and visible via monitorList).\n\n\
                    Pick by how many notifications you need:\n\
                    \u{2022} One (\"tell me when the build finishes\") \u{2192} \
                    use `shell(runInBackground: true)` with a command that \
                    exits when the condition is true. Don't use Monitor.\n\
                    \u{2022} One per occurrence, indefinitely (\"every ERROR \
                    in the log\") \u{2192} Monitor with an unbounded command \
                    like `tail -F` or `inotifywait -m`.\n\
                    \u{2022} One per occurrence with a known end (\"each CI \
                    step result, stop at run end\") \u{2192} Monitor with a \
                    command that emits lines and then exits.\n\n\
                    LINE BUFFERING: pipe-buffered output delays notifications \
                    by kilobytes. Always pass through `grep --line-buffered`, \
                    `awk` with `fflush()`, or wrap with `stdbuf -oL <cmd>`. \
                    For remote ssh: `ssh host 'stdbuf -oL tail -F /path'`. \
                    Use `tail -F` (capital F) for log rotation; `-f` silently \
                    stops on rotation.\n\n\
                    COVERAGE: filter must match every terminal state, not \
                    just the happy path. Before arming, ask: if this process \
                    crashed right now, would my filter emit anything? If not, \
                    widen it. Use alternation to cover progress + failure \
                    signatures (`elapsed_steps=|Traceback|Error|Killed|OOM`).\n\n\
                    OUTPUT VOLUME: every match becomes a notification, so \
                    keep the filter selective \u{2014} but selective means \
                    \"lines you'd act on,\" not just success. Monitors \
                    sustaining >500 matches/sec for 5s auto-stop; tighten \
                    the filter and re-register."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Short label shown in every notification and in the tasks panel. Be specific: \"errors in deploy.log\", not \"watching logs\"."
                        },
                        "command": {
                            "type": "string",
                            "description": "Bash command to run (via `bash -c`). Should produce \
                                line-by-line output. Examples: `tail -F /var/log/app.log`, \
                                `inotifywait -m --format '%e %f' /watched`, \
                                `while sleep 30; do curl -s http://x/health || true; done`."
                        },
                        "filter": {
                            "type": "string",
                            "description": "REQUIRED regex applied to each output line. Lines \
                                that don't match are still kept in the backing job's ring \
                                buffer (visible via jobOutput) but do NOT count as events \
                                or trigger wakes. Use alternation to cover failure modes."
                        }
                    },
                    "required": ["description", "command", "filter"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "monitorStop".into(),
                description: "Stop a monitor by id. Also kills the backing bash task so \
                    the watched process exits cleanly. No-op on already-terminal \
                    monitors."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "monitorId": {
                            "type": "integer",
                            "description": "Monitor id returned by `monitor`."
                        }
                    },
                    "required": ["monitorId"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "monitorList".into(),
                description: "Snapshot of every monitor (running, stopped, auto-stopped) \
                    with command, filter, event count, last-event age, and state."
                    .into(),
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
                name: "scheduleWakeup".into(),
                description: "Arm a one-shot delay wake. After `delaySeconds`, \
                    you'll receive a `<wake source=\"delay#N\" kind=\"Delay\">` \
                    user-shaped message carrying `prompt` as its payload. \
                    Use for \"remind me to check X in 5 minutes\" without \
                    blocking the conversation. The minimum granularity is \
                    ~1s (scheduler tick). Cancellable via cronDelete(wakeId)."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "delaySeconds": {
                            "type": "integer",
                            "description": "Seconds from now. Minimum 1."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Text passed back as the wake payload \u{2014} the model sees it as the new \"user\" turn."
                        }
                    },
                    "required": ["delaySeconds", "prompt"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "cronCreate".into(),
                description: "Arm a cron-scheduled wake. `spec` is a standard \
                    5-field cron string in local time (`minute hour day-of-month \
                    month day-of-week`). On each fire you'll receive a \
                    `<wake source=\"cron#N\" kind=\"Cron\">` message carrying \
                    `prompt`.\n\n\
                    `recurring` defaults to true (loops). Set false for a \
                    single-shot fire (functionally similar to scheduleWakeup \
                    but with a calendar-aware specification).\n\n\
                    Cron evaluation uses wall-clock so machine sleep doesn't \
                    permanently shift the schedule. Cancel via \
                    `cronDelete(wakeId)`."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "string",
                            "description": "5-field cron in local time. Examples: `0 9 * * 1-5` (weekdays at 09:00), `*/15 * * * *` (every 15 min), `30 8 1 * *` (1st of each month at 08:30)."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Payload passed to the model on each fire."
                        },
                        "recurring": {
                            "type": "boolean",
                            "description": "Default true. False = single-shot fire then auto-disarm."
                        }
                    },
                    "required": ["spec", "prompt"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "cronList".into(),
                description: "Snapshot every active wake source (delay, cron, \
                    file-watch, plus passive monitor/task sources) with id, \
                    kind, summary, prompt, and fire count. Use to rediscover \
                    wake ids you've lost track of."
                    .into(),
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
                name: "cronDelete".into(),
                description: "Disarm any wake source by id. Works for delay, \
                    cron, and file-watch sources. Named `cronDelete` for \
                    convenience but accepts any wake id from cronList."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "wakeId": {
                            "type": "integer",
                            "description": "Wake source id (from cronList or the return value of scheduleWakeup/cronCreate/fileWatch)."
                        }
                    },
                    "required": ["wakeId"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "fileWatch".into(),
                description: "Watch a filesystem path for events. Each created, \
                    modified, or removed file under `path` (recursive) fires a \
                    `<wake source=\"fileWatch#N\" kind=\"FileWatch\">` message \
                    with the event kind + affected paths appended to your \
                    `prompt`.\n\n\
                    Suitable for \"resume when the config file is saved\" or \
                    \"react to new files in this directory.\" Returns a wake id \
                    — cancel via cronDelete.\n\n\
                    macOS uses FSEvents (coalesces rapid bursts into single \
                    notifications). Linux uses inotify. Watching a noisy \
                    directory (e.g. node_modules during a build) will fire \
                    many wakes \u{2014} narrow the path or use a Monitor with \
                    a filter instead."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to watch. Must exist. Directories are watched recursively."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Text prepended to the wake payload on each event."
                        }
                    },
                    "required": ["path", "prompt"]
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
    Shell {
        command: String,
        explanation: String,
        impact: ShellImpact,
        timeout: Option<u64>,
        /// Target terminal by name. None resolves to the active terminal.
        /// Ignored when `runInBackground` is true (background jobs bypass
        /// the shared PTY).
        terminal: Option<String>,
        /// Spawn non-blocking in the background. When true, returns a
        /// task id immediately and the command continues running under
        /// the JobPlane; retrieve output via `jobOutput`.
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
    /// Register a line-streamed monitor backed by a bash task. Lines
    /// matching the regex `filter` emit `MonitorEvent`s, bump the
    /// monitor's counter, and wake the agent with a synthetic wake event.
    /// Floods auto-stop the task.
    Monitor {
        description: String,
        command: String,
        filter: String,
    },
    /// Stop a monitor (and its backing bash task).
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
            | ToolAction::TerminalList,
    )
}

/// Whether this action touches the background-job plane (handled by
/// Session, not `execute()`). `Shell { runInBackground: true, .. }` also
/// belongs here — backgrounded shell calls route through JobPlane.
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
/// from the JobPlane handler because monitor lifecycle is decoupled
/// from the backing bash task's lifecycle (a stopped monitor can have
/// a still-completing task and vice-versa).
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

/// Human-readable summary of what a tool action will do.
pub fn summarize(action: &ToolAction) -> String {
    match action {
        ToolAction::Shell {
            command,
            explanation,
            timeout,
            runInBackground,
            ..
        } => {
            let prefix = if *runInBackground {
                "Spawn bg".to_string()
            } else {
                match timeout {
                    Some(t) => format!("Run ({t}s)"),
                    None => "Run".into(),
                }
            };
            if explanation.is_empty() {
                format!("{prefix}: {command}")
            } else {
                format!("{prefix}: {command} \u{2014} {explanation}")
            }
        }
        ToolAction::ReadFile {
            path,
            offset,
            limit,
            anchor,
        } => {
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
        ToolAction::EditFile {
            path,
            oldString,
            replaceAll,
            ..
        } => {
            let preview = if oldString.len() > 40 {
                format!(
                    "{}\u{2026}",
                    &oldString[..oldString.floor_char_boundary(40)]
                )
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
        ToolAction::CopyFile {
            src,
            dest,
            overwrite,
        } => {
            let suffix = if *overwrite { " (overwrite)" } else { "" };
            format!("Copy {src} \u{2192} {dest}{suffix}")
        }
        ToolAction::MoveFile {
            src,
            dest,
            overwrite,
        } => {
            let suffix = if *overwrite { " (overwrite)" } else { "" };
            format!("Move {src} \u{2192} {dest}{suffix}")
        }
        ToolAction::DeleteFile { path, recursive } => {
            if *recursive {
                format!("Delete {path} (recursive)")
            } else {
                format!("Delete {path}")
            }
        }
        ToolAction::MakeDirs { path } => format!("Create directory {path}"),
        ToolAction::ShellHistory { terminal } => match terminal {
            Some(t) => format!("List shell history [{t}]"),
            None => "List shell command history".into(),
        },
        ToolAction::ReadOutput {
            index,
            offset,
            limit,
            terminal,
        } => {
            let suffix = terminal
                .as_deref()
                .map(|t| format!(" [{t}]"))
                .unwrap_or_default();
            match (offset, limit) {
                (Some(o), Some(l)) => {
                    format!("Read output #{index} (lines {o}..{}){suffix}", o + l - 1)
                }
                (Some(o), None) => format!("Read output #{index} (from line {o}){suffix}"),
                _ => format!("Read output #{index}{suffix}"),
            }
        }
        ToolAction::SearchOutput {
            index,
            pattern,
            terminal,
            ..
        } => {
            let suffix = terminal
                .as_deref()
                .map(|t| format!(" [{t}]"))
                .unwrap_or_default();
            format!("Search output #{index} for \"{pattern}\"{suffix}")
        }
        ToolAction::ReadTerminal { lines, terminal } => {
            let suffix = terminal
                .as_deref()
                .map(|t| format!(" [{t}]"))
                .unwrap_or_default();
            format!("Read last {lines} terminal lines{suffix}")
        }
        ToolAction::TerminalSpawn { name } => match name {
            Some(n) => format!("Spawn terminal '{n}'"),
            None => "Spawn terminal".into(),
        },
        ToolAction::TerminalSwitch { name } => format!("Switch active terminal to '{name}'"),
        ToolAction::TerminalKill { name } => format!("Kill terminal '{name}'"),
        ToolAction::TerminalList => "List terminals".into(),
        ToolAction::JobOutput {
            jobId, sinceLine, ..
        } => match sinceLine {
            Some(n) => format!("jobOutput #{jobId} (since line {n})"),
            None => format!("jobOutput #{jobId}"),
        },
        ToolAction::JobStop { jobId } => format!("jobStop #{jobId}"),
        ToolAction::JobList => "jobList".into(),
        ToolAction::Monitor {
            description,
            command,
            filter,
        } => {
            let preview = if command.len() > 50 {
                format!("{}\u{2026}", &command[..command.floor_char_boundary(50)])
            } else {
                command.clone()
            };
            format!("monitor \"{description}\": {preview}  | /{filter}/")
        }
        ToolAction::MonitorStop { monitorId } => format!("monitorStop #{monitorId}"),
        ToolAction::MonitorList => "monitorList".into(),
        ToolAction::ScheduleWakeup {
            delaySeconds,
            prompt,
        } => {
            let preview = if prompt.len() > 40 {
                format!("{}\u{2026}", &prompt[..prompt.floor_char_boundary(40)])
            } else {
                prompt.clone()
            };
            format!("scheduleWakeup {delaySeconds}s: {preview}")
        }
        ToolAction::CronCreate {
            spec,
            prompt,
            recurring,
        } => {
            let preview = if prompt.len() > 40 {
                format!("{}\u{2026}", &prompt[..prompt.floor_char_boundary(40)])
            } else {
                prompt.clone()
            };
            let suffix = if *recurring { "" } else { " (once)" };
            format!("cronCreate `{spec}`{suffix}: {preview}")
        }
        ToolAction::CronList => "cronList".into(),
        ToolAction::CronDelete { wakeId } => format!("cronDelete #{wakeId}"),
        ToolAction::FileWatch { path, prompt } => {
            let preview = if prompt.len() > 40 {
                format!("{}\u{2026}", &prompt[..prompt.floor_char_boundary(40)])
            } else {
                prompt.clone()
            };
            format!("fileWatch {path}: {preview}")
        }
        ToolAction::Glob {
            pattern,
            path,
            metadata,
        } => {
            let dir = path.as_deref().unwrap_or(".");
            let suffix = if *metadata { " +meta" } else { "" };
            format!("Find files: {pattern} in {dir}{suffix}")
        }
        ToolAction::Grep {
            pattern,
            path,
            outputMode,
            ..
        } => {
            let dir = path.as_deref().unwrap_or(".");
            format!("Search ({outputMode}): \"{pattern}\" in {dir}")
        }
        ToolAction::ListDir {
            path,
            depth,
            offset,
            limit,
            metadata,
        } => {
            let suffix = if *metadata { " +meta" } else { "" };
            if *offset > 0 {
                format!("List: {path} (depth {depth}, offset {offset}, limit {limit}){suffix}")
            } else {
                format!("List: {path} (depth {depth}){suffix}")
            }
        }
        ToolAction::StructSearch {
            pattern, language, ..
        } => {
            format!("AST search ({language}): \"{pattern}\"")
        }
        ToolAction::Diff {
            path,
            gitRef,
            pathA,
            pathB,
        } => {
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
        ToolAction::HistorySearch { query, .. } => format!("Search history: {query}"),
        ToolAction::Diagnostics { path, .. } => format!("Check diagnostics: {path}"),
        ToolAction::Task {
            prompt,
            agent,
            runInBackground,
        } => {
            let agentName = agent.as_deref().unwrap_or("general");
            let preview = if prompt.len() > 60 {
                format!("{}\u{2026}", &prompt[..prompt.floor_char_boundary(60)])
            } else {
                prompt.clone()
            };
            let modeLabel = if *runInBackground { " (bg)" } else { "" };
            format!("task [{agentName}]{modeLabel}: {preview}")
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
        ToolAction::EditFile {
            path,
            oldString,
            newString,
            ..
        } => {
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
                let header = format!("--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{lineCount} @@");
                let additions: String = content.lines().map(|l| format!("+{l}\n")).collect();
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
        ToolAction::WriteFile { path, content } => Some((path.clone(), content.clone())),
        ToolAction::EditFile {
            path,
            oldString,
            newString,
            replaceAll,
        } => {
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

/// Execute a tool action and return the content (text or multimodal).
///
/// `terminalName` is the resolved display name of the shell (e.g. "main",
/// "build") so per-terminal tools can label their output. Caller passes
/// `action.terminal().unwrap_or(active_name)`.
pub async fn execute(
    action: &ToolAction,
    shell: &Shell,
    terminalName: &str,
) -> crate::message::Content {
    match action {
        ToolAction::Shell {
            command,
            timeout,
            runInBackground,
            ..
        } => {
            // Background shell calls are routed by the Session via the
            // JobPlane handler. If we got one here, it's a bug.
            if *runInBackground {
                return crate::message::Content::text(
                    "Error: background shell calls must be executed through the session.",
                );
            }
            let dur = timeout.map(std::time::Duration::from_secs);
            let raw = shell.execute(command, dur).await;
            // Apply same size guard as readFile. Full output is in shell history.
            let index = shell.historyLen().saturating_sub(1);
            crate::message::Content::text(truncateOutput(&raw, index, terminalName))
        }
        ToolAction::ReadFile {
            path,
            offset,
            limit,
            anchor,
        } => executeReadFile(path, *offset, *limit, *anchor),
        ToolAction::WriteFile { path, content } => executeWriteFile(path, content).into(),
        ToolAction::EditFile {
            path,
            oldString,
            newString,
            replaceAll,
        } => executeEditFile(path, oldString, newString, *replaceAll).into(),
        ToolAction::MultiEdit { path, edits } => executeMultiEdit(path, edits).into(),
        ToolAction::CopyFile {
            src,
            dest,
            overwrite,
        } => executeCopyFile(src, dest, *overwrite).into(),
        ToolAction::MoveFile {
            src,
            dest,
            overwrite,
        } => executeMoveFile(src, dest, *overwrite).into(),
        ToolAction::DeleteFile { path, recursive } => executeDeleteFile(path, *recursive).into(),
        ToolAction::MakeDirs { path } => executeMakeDirs(path).into(),
        ToolAction::ShellHistory { .. } => executeShellHistory(shell, terminalName).into(),
        ToolAction::ReadOutput {
            index,
            offset,
            limit,
            ..
        } => executeReadOutput(shell, *index, *offset, *limit, terminalName).into(),
        ToolAction::SearchOutput {
            index,
            pattern,
            context,
            ..
        } => executeSearchOutput(shell, *index, pattern, *context, terminalName).into(),
        ToolAction::ReadTerminal { lines, .. } => shell.readTerminal(*lines).into(),
        // Terminal management is handled by Session (needs ShellRegistry).
        ToolAction::TerminalSpawn { .. }
        | ToolAction::TerminalSwitch { .. }
        | ToolAction::TerminalKill { .. }
        | ToolAction::TerminalList => crate::message::Content::text(
            "Error: terminal tools must be executed through the session.",
        ),
        // Job plane and monitor tools are handled by Session (need
        // direct access to JobPlane / MonitorPlane and logTx).
        ToolAction::JobOutput { .. }
        | ToolAction::JobStop { .. }
        | ToolAction::JobList
        | ToolAction::Monitor { .. }
        | ToolAction::MonitorStop { .. }
        | ToolAction::MonitorList => crate::message::Content::text(
            "Error: job plane tools must be executed through the session.",
        ),
        // Wake registry tools are handled by Session (need direct
        // access to WakeRegistry).
        ToolAction::ScheduleWakeup { .. }
        | ToolAction::CronCreate { .. }
        | ToolAction::CronList
        | ToolAction::CronDelete { .. }
        | ToolAction::FileWatch { .. } => {
            crate::message::Content::text("Error: wake tools must be executed through the session.")
        }
        ToolAction::Glob {
            pattern,
            path,
            metadata,
        } => executeGlob(pattern, path.as_deref(), *metadata)
            .await
            .into(),
        ToolAction::Grep {
            pattern,
            path,
            include,
            fileType,
            outputMode,
            caseSensitive,
            contextLines,
            multiline,
        } => executeGrep(
            pattern,
            path.as_deref(),
            include.as_deref(),
            fileType.as_deref(),
            outputMode,
            *caseSensitive,
            *contextLines,
            *multiline,
        )
        .await
        .into(),
        ToolAction::ListDir {
            path,
            depth,
            offset,
            limit,
            metadata,
        } => executeListDir(path, *depth, *offset, *limit, *metadata).into(),
        ToolAction::StructSearch {
            pattern,
            language,
            path,
        } => executeStructSearch(pattern, language, path.as_deref())
            .await
            .into(),
        ToolAction::Diff {
            path,
            gitRef,
            pathA,
            pathB,
        } => executeDiff(
            path.as_deref(),
            gitRef.as_deref(),
            pathA.as_deref(),
            pathB.as_deref(),
        )
        .await
        .into(),
        ToolAction::FuzzyFind { query, path } => {
            executeFuzzyFind(query, path.as_deref()).await.into()
        }
        ToolAction::FileOutline { path } => executeFileOutline(path).await.into(),
        ToolAction::ViewSymbol { file, symbol } => executeViewSymbol(file, symbol).await.into(),
        ToolAction::RelatedFiles { path } => executeRelatedFiles(path).into(),
        // Web tools are handled by session.rs (need ExaClient + cache).
        ToolAction::WebSearch { .. }
        | ToolAction::WebFetch { .. }
        | ToolAction::WebSimilar { .. } => {
            crate::message::Content::text("Error: web tools must be executed through the session.")
        }
        // History tools are handled by session.rs (need transcript access).
        ToolAction::HistoryFetch { .. } | ToolAction::HistorySearch { .. } => {
            crate::message::Content::text(
                "Error: history tools must be executed through the session.",
            )
        }
        // LSP diagnostics are handled by session.rs (need LspManager).
        ToolAction::Diagnostics { .. } => crate::message::Content::text(
            "Error: diagnostics tool must be executed through the session.",
        ),
        // MCP tools are handled by session.rs (need McpManager).
        ToolAction::Mcp { .. } => {
            crate::message::Content::text("Error: MCP tools must be executed through the session.")
        }
        // Task tools are handled by session.rs (need to spawn child session).
        ToolAction::Task { .. } => {
            crate::message::Content::text("Error: task tools must be executed through the session.")
        }
        ToolAction::Unknown { name, .. } => {
            crate::message::Content::text(format!("Unknown tool: {name}"))
        }
    }
}

/// Truncate shell output into a head/middle/tail three-piece slice with
/// a reference to readOutput for the rest.
///
/// Tail-weighted (20/10/70) because for shell output the **tail** is
/// where the signal lives — exit codes, error summaries, final state.
/// Head gives setup context; a middle sample helps the model tell
/// whether something interesting sits in the elided range.
fn truncateOutput(raw: &str, historyIndex: usize, terminalName: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let totalLines = lines.len();

    if totalLines <= MAX_READ_LINES && raw.len() <= MAX_READ_BYTES {
        return raw.to_string();
    }

    const HEAD_RATIO: f64 = 0.20;
    const MIDDLE_RATIO: f64 = 0.10;
    // Tail takes the remainder (~0.70).

    let headLineBudget = (MAX_READ_LINES as f64 * HEAD_RATIO) as usize;
    let midLineBudget = (MAX_READ_LINES as f64 * MIDDLE_RATIO) as usize;
    let tailLineBudget = MAX_READ_LINES - headLineBudget - midLineBudget;

    let headByteBudget = (MAX_READ_BYTES as f64 * HEAD_RATIO) as usize;
    let midByteBudget = (MAX_READ_BYTES as f64 * MIDDLE_RATIO) as usize;
    let tailByteBudget = MAX_READ_BYTES - headByteBudget - midByteBudget;

    // Emit lines in `start..end` until either budget is hit. Each line
    // is clipped to MAX_LINE_LENGTH before being counted. Returns the
    // emitted text and the index of the first line NOT emitted.
    fn emitSlice(
        lines: &[&str],
        start: usize,
        end: usize,
        lineBudget: usize,
        byteBudget: usize,
    ) -> (String, usize) {
        let mut out = String::new();
        let mut emitted = 0usize;
        let mut bytes = 0usize;
        let mut idx = start;
        while idx < end && emitted < lineBudget {
            let line = lines[idx];
            let display = if line.len() > MAX_LINE_LENGTH {
                format!("{}...\n", &line[..MAX_LINE_LENGTH])
            } else {
                format!("{line}\n")
            };
            if bytes + display.len() > byteBudget {
                break;
            }
            bytes += display.len();
            out.push_str(&display);
            emitted += 1;
            idx += 1;
        }
        (out, idx)
    }

    // Head: first N lines.
    let (head, headEnd) = emitSlice(&lines, 0, totalLines, headLineBudget, headByteBudget);

    // Middle: window centered on the midpoint of the full output. Clamp
    // the start past headEnd so the sections never overlap.
    let midCenter = totalLines / 2;
    let midStart = midCenter.saturating_sub(midLineBudget / 2).max(headEnd);
    let (middle, midEnd) = emitSlice(&lines, midStart, totalLines, midLineBudget, midByteBudget);

    // Tail: last N lines. Clamp past midEnd so they can't overlap.
    let tailStart = totalLines.saturating_sub(tailLineBudget).max(midEnd);
    let (tail, _tailEnd) = emitSlice(
        &lines,
        tailStart,
        totalLines,
        tailLineBudget,
        tailByteBudget,
    );

    let headElided = midStart.saturating_sub(headEnd);
    let midElided = tailStart.saturating_sub(midEnd);

    let headMarker = if headElided > 0 {
        format!("\n... [{headElided} lines elided] ...\n\n")
    } else {
        String::new()
    };
    let midMarker = if midElided > 0 {
        format!("\n... [{midElided} lines elided] ...\n\n")
    } else {
        String::new()
    };

    let hint = format!(
        "\n[truncated \u{2014} {totalLines} total lines; \
         use readOutput(index: {historyIndex}, terminal: \"{terminalName}\") for full output]"
    );

    format!("{head}{headMarker}{middle}{midMarker}{tail}{hint}")
}

fn executeReadFile(
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    anchor: Option<usize>,
) -> crate::message::Content {
    use base64::Engine;

    // File type detection via first 512 bytes.
    match std::fs::File::open(path) {
        Ok(mut file) => {
            use std::io::Read;
            let mut probe = [0u8; 512];
            let probeLen = match file.read(&mut probe) {
                Ok(n) => n,
                Err(e) => {
                    return crate::message::Content::text(format!("Failed to read file: {e}"));
                }
            };
            match classifyFile(&probe[..probeLen]) {
                FileKind::Image(fmt) => {
                    let fileSize = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    if fileSize > MAX_IMAGE_BYTES {
                        return crate::message::Content::text(format!(
                            "Image file ({fileSize} bytes). Too large to send inline \u{2014} maximum is 4 MB."
                        ));
                    }
                    match std::fs::read(path) {
                        Ok(bytes) => {
                            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                            let dataUri = format!("data:{};base64,{b64}", fmt.mimeType());
                            return crate::message::Content::withImages(
                                &format!("[{path}]"),
                                vec![dataUri],
                            );
                        }
                        Err(e) => {
                            return crate::message::Content::text(format!(
                                "Failed to read file: {e}"
                            ));
                        }
                    }
                }
                FileKind::Binary => {
                    return crate::message::Content::text(format!(
                        "Binary file ({} bytes). Use shell tools to inspect.",
                        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
                    ));
                }
                FileKind::Text => {}
            }
        }
        Err(e) => return crate::message::Content::text(format!("Failed to read file: {e}")),
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return crate::message::Content::text(format!("Failed to read file: {e}")),
    };

    // Anchor mode: expand from a line based on indentation.
    if let Some(anchorLine) = anchor {
        return crate::message::Content::text(expandFromAnchor(&content, anchorLine));
    }

    crate::message::Content::text(formatNumberedLines(&content, offset, limit))
}

/// Format text as numbered lines with offset/limit and truncation.
/// Shared between readFile and readOutput.
fn formatNumberedLines(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
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
    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.exists()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("Failed to create directories: {e}");
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
            return format!(
                "Edit {}: old_string cannot be empty. No edits were applied.",
                i + 1
            );
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

fn executeCopyFile(src: &str, dest: &str, overwrite: bool) -> String {
    let srcPath = std::path::Path::new(src);
    let destPath = std::path::Path::new(dest);
    if !srcPath.exists() {
        return format!("Source does not exist: {src}");
    }
    if destPath.exists() && !overwrite {
        return format!("Destination already exists: {dest}. Set overwrite=true to replace.");
    }
    if let Some(parent) = destPath.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("Failed to create parent directories of {dest}: {e}");
    }
    if srcPath.is_dir() {
        match copyDirRecursive(srcPath, destPath) {
            Ok(()) => format!("Copied directory {src} \u{2192} {dest}."),
            Err(e) => format!("Failed to copy directory: {e}"),
        }
    } else {
        match std::fs::copy(srcPath, destPath) {
            Ok(bytes) => format!("Copied {src} \u{2192} {dest} ({bytes} bytes)."),
            Err(e) => format!("Failed to copy file: {e}"),
        }
    }
}

fn copyDirRecursive(src: &std::path::Path, dest: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if kind.is_dir() {
            copyDirRecursive(&from, &to)?;
        } else if kind.is_symlink() {
            // Reproduce symlinks by reading the target.
            let target = std::fs::read_link(&from)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &to)?;
            #[cfg(windows)]
            {
                if target.is_dir() {
                    std::os::windows::fs::symlink_dir(&target, &to)?;
                } else {
                    std::os::windows::fs::symlink_file(&target, &to)?;
                }
            }
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn executeMoveFile(src: &str, dest: &str, overwrite: bool) -> String {
    let srcPath = std::path::Path::new(src);
    let destPath = std::path::Path::new(dest);
    if !srcPath.exists() {
        return format!("Source does not exist: {src}");
    }
    if destPath.exists() && !overwrite {
        return format!("Destination already exists: {dest}. Set overwrite=true to replace.");
    }
    if let Some(parent) = destPath.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("Failed to create parent directories of {dest}: {e}");
    }
    // Try a rename first (cheap, atomic when on the same filesystem). Fall back
    // to copy+delete on EXDEV (cross-device link) or other rename failures.
    match std::fs::rename(srcPath, destPath) {
        Ok(()) => format!("Moved {src} \u{2192} {dest}."),
        Err(_) => {
            let copyResult = if srcPath.is_dir() {
                copyDirRecursive(srcPath, destPath)
            } else {
                std::fs::copy(srcPath, destPath).map(|_| ())
            };
            if let Err(e) = copyResult {
                return format!("Failed to move (copy phase): {e}");
            }
            let removeResult = if srcPath.is_dir() {
                std::fs::remove_dir_all(srcPath)
            } else {
                std::fs::remove_file(srcPath)
            };
            match removeResult {
                Ok(()) => format!("Moved {src} \u{2192} {dest} (cross-device, copy+delete)."),
                Err(e) => format!("Copied {src} \u{2192} {dest} but failed to remove source: {e}"),
            }
        }
    }
}

fn executeDeleteFile(path: &str, recursive: bool) -> String {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return format!("Path does not exist: {path}");
    }
    if p.is_dir() {
        if recursive {
            match std::fs::remove_dir_all(p) {
                Ok(()) => format!("Deleted directory tree {path}."),
                Err(e) => format!("Failed to delete directory: {e}"),
            }
        } else {
            match std::fs::remove_dir(p) {
                Ok(()) => format!("Deleted empty directory {path}."),
                Err(e) => format!(
                    "Failed to delete directory: {e}. Set recursive=true to delete contents."
                ),
            }
        }
    } else {
        match std::fs::remove_file(p) {
            Ok(()) => format!("Deleted {path}."),
            Err(e) => format!("Failed to delete file: {e}"),
        }
    }
}

fn executeMakeDirs(path: &str) -> String {
    match std::fs::create_dir_all(path) {
        Ok(()) => format!("Created directory {path}."),
        Err(e) => format!("Failed to create directory: {e}"),
    }
}

fn executeShellHistory(shell: &Shell, terminalName: &str) -> String {
    let entries = shell.listHistory();
    if entries.is_empty() {
        return format!("No commands in history for terminal '{terminalName}'.");
    }

    let mut output = format!("History for terminal '{terminalName}':\n");
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
        output.push_str(&format!(
            "[{i}] {cmdPreview}{codeStr}  ({lineCount} lines)\n"
        ));
    }

    output
}

fn executeReadOutput(
    shell: &Shell,
    index: usize,
    offset: Option<usize>,
    limit: Option<usize>,
    terminalName: &str,
) -> String {
    match shell.getRecord(index) {
        Some(record) => {
            let header = format!(
                "Terminal '{}', command [{}]: {}\n\n",
                terminalName,
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

fn executeSearchOutput(
    shell: &Shell,
    index: usize,
    pattern: &str,
    context: usize,
    _terminalName: &str,
) -> String {
    match shell.searchOutput(index, pattern, context) {
        Some(result) => result,
        None => format!("No command at index {index}. Use shellHistory to see available commands."),
    }
}

/// Classification of a file's content type based on magic bytes.
pub enum FileKind {
    Text,
    Image(ImageFormat),
    Binary,
}

/// Recognized image formats (by magic bytes).
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Bmp,
    Webp,
}

impl ImageFormat {
    pub fn mimeType(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Gif => "image/gif",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Webp => "image/webp",
        }
    }
}

/// Maximum image file size for inline base64 encoding (4 MB).
const MAX_IMAGE_BYTES: u64 = 4 * 1024 * 1024;

/// Classify file content by probing magic bytes.
fn classifyFile(bytes: &[u8]) -> FileKind {
    if bytes.is_empty() {
        return FileKind::Text;
    }

    // Image signatures — check before generic binary.
    if bytes.starts_with(b"\x89PNG") {
        return FileKind::Image(ImageFormat::Png);
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return FileKind::Image(ImageFormat::Jpeg);
    }
    if bytes.starts_with(b"GIF8") {
        return FileKind::Image(ImageFormat::Gif);
    }
    if bytes.starts_with(b"BM") {
        return FileKind::Image(ImageFormat::Bmp);
    }
    // WebP: starts with RIFF....WEBP.
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return FileKind::Image(ImageFormat::Webp);
    }

    // Non-image binary signatures.
    const BINARY_MAGIC: &[&[u8]] = &[
        b"PK\x03\x04",       // ZIP/DOCX/JAR
        b"\x7fELF",          // ELF
        b"\xfe\xed\xfa",     // Mach-O
        b"\xcf\xfa\xed\xfe", // Mach-O (reversed)
        b"%PDF",             // PDF
        b"\x1f\x8b",         // gzip
    ];
    for sig in BINARY_MAGIC {
        if bytes.starts_with(sig) {
            return FileKind::Binary;
        }
    }

    // NUL byte check (strong binary indicator in first 512 bytes).
    if bytes.contains(&0x00) {
        return FileKind::Binary;
    }

    FileKind::Text
}

// --- Subprocess helper ---

/// Run an external program, capture stdout+stderr, enforce timeout.
/// Returns Ok(stdout) on success or Err(message) on failure.
/// rg exit code 1 ("no matches") is treated as success with empty output.
async fn runSubprocess(program: &str, args: &[&str], notFoundMsg: &str) -> Result<String, String> {
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
                Err(format!(
                    "{program} failed (exit {}): {}",
                    output.status,
                    msg.trim()
                ))
            }
        }
        Ok(Err(e)) => Err(format!("Failed to run {program}: {e}")),
        Err(_) => {
            // Process is still running but we lost ownership via wait_with_output.
            // The child is dropped here which sends SIGKILL on Unix.
            Err(format!(
                "{program} timed out after {SUBPROCESS_TIMEOUT_SECS}s."
            ))
        }
    }
}

// --- Search / structure / diff execute functions ---

async fn executeGlob(pattern: &str, path: Option<&str>, metadata: bool) -> String {
    let mut args = vec![
        "--files", "--sort", "modified", "--hidden", "--glob", pattern, "--glob", "!.git/",
    ];
    if let Some(p) = path {
        args.push(p);
    }

    match runSubprocess(
        "rg",
        &args,
        "ripgrep (rg) not found. Install: https://github.com/BurntSushi/ripgrep",
    )
    .await
    {
        Ok(stdout) => {
            if stdout.trim().is_empty() {
                return "No files found.".into();
            }
            let lines: Vec<&str> = stdout.lines().collect();
            let total = lines.len();
            let mut output = String::new();
            for line in lines.iter().take(MAX_GLOB_RESULTS) {
                output.push_str(line);
                if metadata && let Some(meta) = formatMetadata(std::path::Path::new(line)) {
                    output.push_str("  ");
                    output.push_str(&meta);
                }
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
    let Some(rule) = outlineRule(&lang) else {
        return Vec::new();
    };

    let output = std::process::Command::new("sg")
        .args(["scan", "--inline-rules", &rule, "--json=stream", path])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut entries = parseSgEntries(&stdout);
    for (_, sig) in entries.iter_mut() {
        if sig.len() > 80 {
            *sig = format!("{}...", &sig[..sig.floor_char_boundary(80)]);
        }
    }
    entries.sort_by_key(|(line, _)| *line);
    entries.dedup_by_key(|(line, _)| *line);
    entries
}

/// Find the enclosing symbol for a given line number.
/// Returns the symbol signature from the last definition before or at that line.
fn symbolAtLine(symbols: &[(usize, String)], line: usize) -> Option<&str> {
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

#[allow(clippy::too_many_arguments)]
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
    // Pre-validate pattern syntax. ripgrep uses Rust's regex crate by default,
    // so this catches the common "I forgot to escape something" cases with a
    // clear error rather than a silent "No matches found".
    if let Err(e) = regex::Regex::new(pattern) {
        return format!(
            "Invalid regex pattern: {pattern:?}\n\nParser error: {e}\n\n\
             Hint: ripgrep uses Rust regex syntax. Escape regex metachars \
             (.+*?()[]{{}}|^$\\) with backslashes. Watch for stray quotes \
             from JSON escaping."
        );
    }

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

    match runSubprocess(
        "rg",
        &args,
        "ripgrep (rg) not found. Install: https://github.com/BurntSushi/ripgrep",
    )
    .await
    {
        Ok(stdout) => {
            if stdout.trim().is_empty() {
                let scope = path.unwrap_or(".");
                let mut msg = format!("No matches for pattern {pattern:?} in {scope}.");
                // Surface common foot-guns when a pattern looks suspect.
                if pattern.ends_with('"') || pattern.ends_with("\\\"") {
                    msg.push_str(
                        "\n\nNote: pattern ends with a quote. Likely a JSON \
                         escaping artifact rather than intended literal.",
                    );
                }
                if pattern.contains("\\|") && !pattern.contains("(?") {
                    msg.push_str(
                        "\n\nNote: `\\|` is a literal pipe in Rust regex. \
                         For alternation use `|` (or wrap in `(a|b)`).",
                    );
                }
                return msg;
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
                output.push_str(&format!("\n... {total} lines total, showing first {cap}."));
            }
            output
        }
        Err(e) => e,
    }
}

fn executeListDir(path: &str, depth: usize, offset: usize, limit: usize, metadata: bool) -> String {
    const EXCLUDED: &[&str] = &[".git", "node_modules", "target", "__pycache__", ".venv"];

    let rootPath = std::path::Path::new(path);
    if !rootPath.is_dir() {
        return format!("Not a directory: {path}");
    }

    // Collect all entries first (up to a hard cap), then paginate.
    let hardCap = MAX_LISTDIR_ENTRIES.max(offset + limit);
    let mut allEntries = Vec::new();
    let mut count = 0usize;
    let truncated = listDirRecurse(
        rootPath,
        0,
        depth,
        "",
        &mut allEntries,
        &mut count,
        EXCLUDED,
        hardCap,
        metadata,
    );
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
#[allow(clippy::too_many_arguments)]
fn listDirRecurse(
    dir: &std::path::Path,
    currentDepth: usize,
    maxDepth: usize,
    indent: &str,
    output: &mut Vec<String>,
    count: &mut usize,
    excluded: &[&str],
    hardCap: usize,
    metadata: bool,
) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };

    // Collect and sort: dirs first, then files, alphabetical within each group.
    let mut dirs = Vec::new();
    let mut files: Vec<(String, bool, std::path::PathBuf)> = Vec::new();
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
            files.push((name, isSymlink, entry.path()));
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
            if listDirRecurse(
                &childPath,
                currentDepth + 1,
                maxDepth,
                &childIndent,
                output,
                count,
                excluded,
                hardCap,
                metadata,
            ) {
                return true;
            }
        }
    }

    // Then files.
    for (name, isSymlink, path) in &files {
        if *count >= hardCap {
            return true;
        }
        let suffix = if *isSymlink { "@" } else { "" };
        let mut line = format!("{indent}{name}{suffix}");
        if metadata && let Some(meta) = formatMetadata(path) {
            line.push_str("  ");
            line.push_str(&meta);
        }
        output.push(line);
        *count += 1;
    }

    false
}

/// Render `<size>  <YYYY-MM-DD HH:MM>` for a file path. Returns None on error.
fn formatMetadata(path: &std::path::Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let size = formatSize(meta.len());
    let mtime = meta.modified().ok().map(formatMtime).unwrap_or_default();
    Some(format!("{size:>9}  {mtime}"))
}

fn formatSize(bytes: u64) -> String {
    const UNITS: &[(u64, &str)] = &[(1024 * 1024 * 1024, "G"), (1024 * 1024, "M"), (1024, "K")];
    for (threshold, suffix) in UNITS {
        if bytes >= *threshold {
            let value = bytes as f64 / *threshold as f64;
            return if value >= 10.0 {
                format!("{value:.0}{suffix}")
            } else {
                format!("{value:.1}{suffix}")
            };
        }
    }
    format!("{bytes}B")
}

fn formatMtime(time: std::time::SystemTime) -> String {
    let secs = time
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Local-tz aware breakdown via chrono would be cleaner, but we don't have
    // chrono. UTC is fine — the model just needs a stable ordering.
    let (y, mo, d, h, mi) = epochToYMDHM(secs);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}Z")
}

/// UTC epoch seconds → (year, month, day, hour, minute). Civil-time conversion
/// from Howard Hinnant's date algorithms (no chrono dependency).
fn epochToYMDHM(secs: u64) -> (u32, u32, u32, u32, u32) {
    let days = (secs / 86400) as i64;
    let timeOfDay = (secs % 86400) as u32;
    let h = timeOfDay / 3600;
    let mi = (timeOfDay % 3600) / 60;

    // Hinnant: civil_from_days
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let yy = (if mo <= 2 { y + 1 } else { y }) as u32;
    (yy, mo, d, h, mi)
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
        if let Some(metaVars) = obj["metaVariables"].as_object()
            && !metaVars.is_empty()
        {
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
    for (i, line) in lines.iter().enumerate().skip(anchorIdx + 1) {
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
    for (i, line) in lines
        .iter()
        .enumerate()
        .take(effectiveEnd + 1)
        .skip(startIdx)
    {
        let lineNum = i + 1;
        let displayLine = if line.len() > MAX_LINE_LENGTH {
            format!("{}...", &line[..MAX_LINE_LENGTH])
        } else {
            line.to_string()
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
    use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
    use nucleo_matcher::{Config, Matcher};

    let mut args = vec!["--files", "--hidden", "--glob", "!.git/"];
    if let Some(p) = path {
        args.push(p);
    }

    let stdout = match runSubprocess(
        "rg",
        &args,
        "ripgrep (rg) not found. Install: https://github.com/BurntSushi/ripgrep",
    )
    .await
    {
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
            "\n... {} more matches. Refine your query.",
            matches.len() - MAX_FUZZY_RESULTS
        ));
    }
    output
}

// --- File outline ---

async fn executeFileOutline(path: &str) -> String {
    let lang = detectLanguage(path);
    let Some(rule) = outlineRule(&lang) else {
        return format!("No outline support for language \"{lang}\". File: {path}");
    };

    let args = vec!["scan", "--inline-rules", &rule, "--json=stream", path];
    let stdout = match runSubprocess(
        "sg",
        &args,
        "ast-grep (sg) is required for fileOutline. Install: https://ast-grep.github.io",
    )
    .await
    {
        Ok(s) => s,
        Err(e) => return e,
    };

    let mut entries = parseSgEntries(&stdout);
    if entries.is_empty() {
        return format!("No symbols found in {path}.");
    }
    entries.sort_by_key(|(line, _)| *line);
    entries.dedup_by_key(|(line, _)| *line);

    let mut output = String::new();
    for (line, text) in entries.iter().take(MAX_OUTLINE_ENTRIES) {
        output.push_str(&format!("{line:>6}  {text}\n"));
    }
    if entries.len() > MAX_OUTLINE_ENTRIES {
        output.push_str(&format!(
            "\n... {} more symbols.",
            entries.len() - MAX_OUTLINE_ENTRIES
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
            if trimmed.contains(part)
                && (looksLikeDeclaration(trimmed, part) || looksLikeVariant(trimmed, part))
            {
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
                            output.push_str(&format!(
                                "{absLine:>6}\t{}\n",
                                &expandedLine[tabPos + 1..]
                            ));
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
            return format!(
                "{file}:{currentStart} (found {outerName}, \"{part}\" not found within)\n\n{currentText}"
            );
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
        if trimmed.contains(name)
            && (looksLikeDeclaration(trimmed, name) || looksLikeVariant(trimmed, name))
        {
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

    Some(SymbolRange {
        startLine: lineNum,
        text,
    })
}

/// Simple single-name symbol lookup (original behavior).
async fn viewSymbolSingle(file: &str, symbol: &str, lang: &str) -> String {
    let Some(rule) = symbolRule(lang, symbol) else {
        return format!("Symbol lookup not supported for language \"{lang}\".");
    };
    let args = vec!["scan", "--inline-rules", &rule, "--json=stream", file];
    let Ok(stdout) = runSubprocess("sg", &args, "").await else {
        return format!("Symbol \"{symbol}\" not found in {file} via ast-grep.");
    };

    for line in stdout.lines() {
        let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let text = obj["text"].as_str().unwrap_or("");
        let startLine = obj["range"]["start"]["line"]
            .as_u64()
            .map(|l| l + 1)
            .unwrap_or(0);
        if !text.is_empty() {
            return format!("{file}:{startLine}\n\n{text}");
        }
    }

    format!("Symbol \"{symbol}\" not found in {file} via ast-grep.")
}

/// Heuristic: does this line look like it declares the given symbol?
fn looksLikeDeclaration(line: &str, symbol: &str) -> bool {
    // Check if symbol appears after common declaration keywords.
    let declarationPrefixes = [
        "fn ",
        "pub fn ",
        "async fn ",
        "pub async fn ",
        "struct ",
        "pub struct ",
        "enum ",
        "pub enum ",
        "trait ",
        "pub trait ",
        "impl ",
        "type ",
        "pub type ",
        "const ",
        "pub const ",
        "static ",
        "pub static ",
        "mod ",
        "pub mod ",
        "def ",
        "async def ",
        "class ",
        "function ",
        "export function ",
        "export default function ",
        "export const ",
        "export let ",
        "export class ",
        "interface ",
        "export interface ",
        "export type ",
        "func ",
        "var ",
        "let ",
        "const ",
    ];

    for prefix in &declarationPrefixes {
        if let Some(rest) = line.strip_prefix(prefix)
            && rest.starts_with(symbol)
        {
            return true;
        }
    }

    false
}

/// Heuristic: does this line look like an enum variant or struct field with this name?
fn looksLikeVariant(line: &str, name: &str) -> bool {
    // Match patterns like "Grep {", "Grep(", "Grep," (enum variants).
    if let Some(pos) = line.find(name) {
        let afterName = &line[pos + name.len()..].trim_start();
        if afterName.starts_with('{')
            || afterName.starts_with('(')
            || afterName.starts_with(',')
            || afterName.starts_with(';')
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
    if let Some(dir) = fileDir
        && let Ok(entries) = std::fs::read_dir(dir)
    {
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
        "rust" => vec![r"^use\s+([\w:]+)", r"^mod\s+(\w+)\s*;"],
        "python" => vec![r"^(?:from\s+([\w.]+)\s+)?import\s+([\w.]+)"],
        "typescript" | "javascript" | "tsx" | "jsx" => {
            vec![r#"(?:import|require)\s*\(?[^)]*['"]([^'"]+)['"]"#]
        }
        "go" => vec![r#"^\s*"([^"]+)""#],
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
    }
    .into()
}

/// Tree-sitter node kinds that appear in a file outline for `lang`.
fn outlineKinds(lang: &str) -> Option<&'static [&'static str]> {
    match lang {
        "rust" => Some(&[
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
            "mod_item",
            "type_item",
            "const_item",
            "static_item",
            "macro_definition",
        ]),
        "python" => Some(&["function_definition", "class_definition"]),
        "typescript" | "tsx" => Some(&[
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "method_definition",
            "abstract_class_declaration",
        ]),
        "javascript" | "jsx" => Some(&[
            "function_declaration",
            "class_declaration",
            "method_definition",
        ]),
        "go" => Some(&[
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ]),
        _ => None,
    }
}

/// Tree-sitter kind of "function-like" nodes in this language. Items
/// nested inside such a node are excluded from outlines (locals,
/// nested consts inside fn bodies). Methods inside class/impl blocks
/// stay because their containing kind isn't this one.
fn fnKind(lang: &str) -> Option<&'static str> {
    match lang {
        "rust" => Some("function_item"),
        "python" => Some("function_definition"),
        "typescript" | "tsx" | "javascript" | "jsx" | "go" => Some("function_declaration"),
        _ => None,
    }
}

/// Build an ast-grep inline YAML rule matching outline items in `lang`.
fn outlineRule(lang: &str) -> Option<String> {
    let kinds = outlineKinds(lang)?;
    let mut yaml =
        format!("id: outline\nlanguage: {lang}\nseverity: info\nmessage: outline\nrule:\n  any:\n");
    for k in kinds {
        yaml.push_str(&format!("    - kind: {k}\n"));
    }
    if let Some(fk) = fnKind(lang) {
        yaml.push_str(&format!(
            "  not:\n    inside:\n      kind: {fk}\n      stopBy: end\n"
        ));
    }
    Some(yaml)
}

/// Build an ast-grep inline YAML rule matching a specific symbol by name.
/// Rust impl blocks (which lack a `name` field) are matched via their
/// `type` and `trait` fields.
fn symbolRule(lang: &str, symbol: &str) -> Option<String> {
    let kinds = outlineKinds(lang)?;
    let escaped = regex::escape(symbol);
    let mut yaml =
        format!("id: symbol\nlanguage: {lang}\nseverity: info\nmessage: symbol\nrule:\n  any:\n",);

    yaml.push_str("    - all:\n");
    yaml.push_str("        - any:\n");
    for k in kinds.iter().filter(|k| **k != "impl_item") {
        yaml.push_str(&format!("            - kind: {k}\n"));
    }
    yaml.push_str("        - has:\n");
    yaml.push_str("            field: name\n");
    yaml.push_str(&format!("            regex: \"^{escaped}$\"\n"));

    if lang == "rust" {
        yaml.push_str("    - all:\n");
        yaml.push_str("        - kind: impl_item\n");
        yaml.push_str("        - any:\n");
        yaml.push_str("            - has:\n");
        yaml.push_str("                field: type\n");
        yaml.push_str(&format!("                regex: \"^{escaped}$\"\n"));
        yaml.push_str("            - has:\n");
        yaml.push_str("                field: trait\n");
        yaml.push_str(&format!("                regex: \"^{escaped}$\"\n"));
    }
    Some(yaml)
}

/// Parse JSONL output from `sg scan --json=stream` into (line, firstLine)
/// pairs. Empty matches are skipped.
fn parseSgEntries(stdout: &str) -> Vec<(usize, String)> {
    let mut entries = Vec::new();
    for line in stdout.lines() {
        let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
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
    entries
}

/// Parse a tool call name + JSON arguments into a ToolAction.
///
/// Returns Err with a message listing missing/malformed required fields.
/// The error message is sent back to the model as the tool result so it can retry.
pub fn parse(name: &str, argsJson: &str) -> Result<ToolAction, String> {
    let args: serde_json::Value =
        serde_json::from_str(argsJson).map_err(|e| format!("Malformed JSON arguments: {e}"))?;

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
                .ok_or_else(|| {
                    "Missing required field 'impact' (one of: read, minorMod, majorMod, delete)."
                        .to_string()
                })?;
            ToolAction::Shell {
                command: reqStr!("command"),
                explanation: reqStr!("explanation"),
                impact,
                timeout: optU64!("timeout"),
                terminal: optStr!("terminal"),
                runInBackground: optBool!("runInBackground").unwrap_or(false),
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
                        oldString: e["old_string"]
                            .as_str()
                            .ok_or_else(|| "Edit missing 'old_string'.".to_string())?
                            .into(),
                        newString: e["new_string"].as_str().unwrap_or("").into(),
                        replaceAll: e["replace_all"].as_bool().unwrap_or(false),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        },
        "copyFile" => ToolAction::CopyFile {
            src: reqStr!("src"),
            dest: reqStr!("dest"),
            overwrite: optBool!("overwrite").unwrap_or(false),
        },
        "moveFile" => ToolAction::MoveFile {
            src: reqStr!("src"),
            dest: reqStr!("dest"),
            overwrite: optBool!("overwrite").unwrap_or(false),
        },
        "deleteFile" => ToolAction::DeleteFile {
            path: reqStr!("path"),
            recursive: optBool!("recursive").unwrap_or(false),
        },
        "makeDirs" => ToolAction::MakeDirs {
            path: reqStr!("path"),
        },
        "shellHistory" => ToolAction::ShellHistory {
            terminal: optStr!("terminal"),
        },
        "readOutput" => ToolAction::ReadOutput {
            index: optU64!("index").unwrap_or(0) as usize,
            offset: optU64!("offset").map(|v| v as usize),
            limit: optU64!("limit").map(|v| v as usize),
            terminal: optStr!("terminal"),
        },
        "searchOutput" => ToolAction::SearchOutput {
            index: optU64!("index").unwrap_or(0) as usize,
            pattern: reqStr!("pattern"),
            context: optU64!("context").unwrap_or(3) as usize,
            terminal: optStr!("terminal"),
        },
        "readTerminal" => ToolAction::ReadTerminal {
            lines: optU64!("lines").unwrap_or(50) as usize,
            terminal: optStr!("terminal"),
        },
        "terminalSpawn" => ToolAction::TerminalSpawn {
            name: optStr!("name"),
        },
        "terminalSwitch" => ToolAction::TerminalSwitch {
            name: reqStr!("name"),
        },
        "terminalKill" => ToolAction::TerminalKill {
            name: reqStr!("name"),
        },
        "terminalList" => ToolAction::TerminalList,
        "jobOutput" => ToolAction::JobOutput {
            jobId: optU64!("jobId").ok_or_else(|| "Missing required field 'jobId'.".to_string())?,
            sinceLine: optU64!("sinceLine"),
            maxLines: optU64!("maxLines").map(|v| v as usize),
        },
        "jobStop" => ToolAction::JobStop {
            jobId: optU64!("jobId").ok_or_else(|| "Missing required field 'jobId'.".to_string())?,
        },
        "jobList" => ToolAction::JobList,
        "monitor" => ToolAction::Monitor {
            description: reqStr!("description"),
            command: reqStr!("command"),
            filter: reqStr!("filter"),
        },
        "monitorStop" => ToolAction::MonitorStop {
            monitorId: optU64!("monitorId")
                .ok_or_else(|| "Missing required field 'monitorId'.".to_string())?,
        },
        "monitorList" => ToolAction::MonitorList,
        "scheduleWakeup" => ToolAction::ScheduleWakeup {
            delaySeconds: optU64!("delaySeconds")
                .ok_or_else(|| "Missing required field 'delaySeconds'.".to_string())?,
            prompt: reqStr!("prompt"),
        },
        "cronCreate" => ToolAction::CronCreate {
            spec: reqStr!("spec"),
            prompt: reqStr!("prompt"),
            recurring: optBool!("recurring").unwrap_or(true),
        },
        "cronList" => ToolAction::CronList,
        "cronDelete" => ToolAction::CronDelete {
            wakeId: optU64!("wakeId")
                .ok_or_else(|| "Missing required field 'wakeId'.".to_string())?,
        },
        "fileWatch" => ToolAction::FileWatch {
            path: reqStr!("path"),
            prompt: reqStr!("prompt"),
        },
        "glob" => ToolAction::Glob {
            pattern: reqStr!("pattern"),
            path: optStr!("path"),
            metadata: optBool!("metadata").unwrap_or(false),
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
            depth: optU64!("depth").unwrap_or(2).clamp(1, 5) as usize,
            offset: optU64!("offset").unwrap_or(0) as usize,
            limit: optU64!("limit").unwrap_or(500) as usize,
            metadata: optBool!("metadata").unwrap_or(false),
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
            allowedDomains: args["allowed_domains"].as_array().map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
            blockedDomains: args["blocked_domains"].as_array().map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
            maxResults: optU64!("max_results").map(|v| v as usize),
        },
        "webFetch" => ToolAction::WebFetch {
            url: reqStr!("url"),
            prompt: optStr!("prompt"),
            subpages: optU64!("subpages").map(|v| v as usize),
        },
        "webSimilar" => ToolAction::WebSimilar {
            url: reqStr!("url"),
            allowedDomains: args["allowed_domains"].as_array().map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
            blockedDomains: args["blocked_domains"].as_array().map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
            maxResults: optU64!("max_results").map(|v| v as usize),
        },
        "historyFetch" => ToolAction::HistoryFetch {
            blockId: reqStr!("blockId"),
        },
        "historySearch" => ToolAction::HistorySearch {
            query: reqStr!("query"),
            mediaType: args["mediaType"].as_str().map(String::from),
        },
        "task" => ToolAction::Task {
            prompt: reqStr!("prompt"),
            agent: optStr!("agent"),
            runInBackground: optBool!("runInBackground").unwrap_or(false),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifyPng() {
        let header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Png) => {}
            other => panic!("expected PNG, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyJpeg() {
        let header = b"\xff\xd8\xff\xe0\x00\x10JFIF";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Jpeg) => {}
            other => panic!("expected JPEG, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyGif() {
        let header = b"GIF89a\x01\x00\x01\x00";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Gif) => {}
            other => panic!("expected GIF, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyWebp() {
        let header = b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Webp) => {}
            other => panic!("expected WebP, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyElf() {
        let header = b"\x7fELF\x02\x01\x01\x00";
        match classifyFile(header) {
            FileKind::Binary => {}
            other => panic!("expected Binary, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyPlainText() {
        let header = b"fn main() {\n    println!(\"hello\");\n}";
        match classifyFile(header) {
            FileKind::Text => {}
            other => panic!("expected Text, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyNulAsBinary() {
        let header = b"some text\x00more text";
        match classifyFile(header) {
            FileKind::Binary => {}
            other => panic!("expected Binary, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyEmptyAsText() {
        match classifyFile(b"") {
            FileKind::Text => {}
            other => panic!("expected Text, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn readFileTextReturnsContent() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "line 1\nline 2\n").unwrap();
        let result = executeReadFile(tmp.path().to_str().unwrap(), None, None, None);
        let text = result.textContent();
        assert!(text.contains("line 1"));
        assert!(text.contains("line 2"));
        assert!(!result.hasImages());
    }

    #[test]
    fn readFilePngReturnsImageContent() {
        // Write a minimal 1x1 PNG.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let png: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
            0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
            0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, 0x33, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(tmp.path(), png).unwrap();
        let result = executeReadFile(tmp.path().to_str().unwrap(), None, None, None);
        assert!(result.hasImages());
        let uris = result.imageUris();
        assert_eq!(uris.len(), 1);
        assert!(uris[0].starts_with("data:image/png;base64,"));
    }

    #[test]
    fn readFileBinaryReturnsError() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let elf = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        std::fs::write(tmp.path(), elf).unwrap();
        let result = executeReadFile(tmp.path().to_str().unwrap(), None, None, None);
        assert!(!result.hasImages());
        assert!(result.textContent().contains("Binary file"));
    }
}
