use crate::message::ToolDef;

/// Returns the built-in tool definitions to send to the LLM.
pub(crate) fn builtinDefs() -> Vec<ToolDef> {
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
                    DETACH: a foreground call that hits its timeout keeps \
                    running in the same visible terminal as an archived \
                    terminal run; Ctrl+B does the same immediately. The \
                    command is not restarted. You receive a terminal run id \
                    and a wake when it completes.\n\n\
                    Set `runInBackground: true` for long builds, dev \
                    servers, log tails, or any command whose result you \
                    don't need before continuing. Background calls run in a \
                    visible terminal and return a terminal run id immediately; \
                    you'll be notified when the run completes \u{2014} do NOT poll while \
                    waiting. Use foreground (default) when you need the \
                    result before you can proceed; background when you \
                    have genuinely independent work to do in parallel.\n\n\
                    LINE BUFFERING: long-running pipes (`cmd | grep ...`, \
                    `ssh host 'tail -F ...'`) block-buffer stdout until \
                    kilobytes accumulate, hiding output for minutes. Use \
                    `grep --line-buffered`, `awk 'BEGIN {{...}}'` with \
                    `fflush()`, `stdbuf -oL <cmd>`, or `ssh host 'stdbuf \
                    -oL tail -F /path'` to keep output flowing.\n\n\
                    Async terminal runs are archived in terminal history. If \
                    `runInBackground` is true and `terminal` is omitted, \
                    Flatline creates a visible ephemeral terminal and closes \
                    it after the run is archived."
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
                            "description": "Timeout in seconds. Default 30. When exceeded, the command detaches into the same visible terminal as an archived terminal run; it is not restarted. Ignored when runInBackground is true."
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Name of the terminal to run in. Omit to use the agent's target terminal for foreground calls, or to create a visible ephemeral terminal for runInBackground."
                        },
                        "runInBackground": {
                            "type": "boolean",
                            "description": "Run non-blocking in a visible terminal. Returns a terminal run id immediately; the command keeps running while you work. You'll be notified when it completes — do not poll. Defaults to false."
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
                name: "terminalRunList".into(),
                description: "List archived visible terminal runs with run id, \
                    purpose, impact, terminal, status, and exit code. Use this \
                    to rediscover async run ids; terminal history is the \
                    user-facing view."
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
                name: "terminalRunStop".into(),
                description: "Cancel a running archived terminal run by id. \
                    This interrupts the visible terminal that owns the run; \
                    no-op if the run is already terminal."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "runId": {
                            "type": "string",
                            "description": "Run id returned by shell(runInBackground: true) or foreground timeout/Ctrl+B detach."
                        }
                    },
                    "required": ["runId"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "jobOutput".into(),
                description: "Read buffered output from a background task \
                    such as `task(runInBackground: true)`. Returns the \
                    latest output lines plus the task's current state \
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
                            "description": "Task/job id returned by task(runInBackground: true) or another JobPlane-backed tool."
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
                description: "Kill a running background task by id. No-op if \
                    the task is already terminal. Use terminalRunStop for \
                    visible async terminal runs."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "jobId": {
                            "type": "integer",
                            "description": "Task/job id."
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
                description: "List JobPlane-backed background tasks (running, \
                    completed, killed, errored) with age, total lines emitted, \
                    and state. Visible async terminal runs live in terminalRunList \
                    and terminal history."
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
                description: "Register a regex watcher on an existing terminal's \
                    output stream. A monitor does not start a command; run the \
                    command normally in a terminal first, then attach the \
                    monitor to that terminal. Each matching line wakes the \
                    model with the matched line as payload.\n\n\
                    Use `shell(runInBackground: true)` for one-shot async work \
                    that exits. Use `monitor` when an already-running terminal \
                    should keep notifying on matching output. Keep the regex \
                    selective but broad enough to include failure modes."
                    .into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Short label shown in every notification and in the tasks panel. Be specific: \"errors in deploy.log\", not \"watching logs\"."
                        },
                        "terminal": {
                            "type": "string",
                            "description": "Terminal name to watch. Omit to watch the agent's current target terminal."
                        },
                        "filter": {
                            "type": "string",
                            "description": "REQUIRED regex applied to each normalized terminal output line. Non-matching lines remain only in the terminal/replay history and do not trigger wakes."
                        }
                    },
                    "required": ["description", "filter"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "monitorStop".into(),
                description: "Stop a monitor by id. This detaches the regex watcher from \
                    its terminal but does not stop the terminal command itself."
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
                    with terminal, filter, event count, last-event age, and state."
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

pub(crate) fn builtinDefsWithPermissionEscalation() -> Vec<ToolDef> {
    let mut defs = builtinDefs();
    addPermissionEscalationFieldsToDefs(&mut defs);
    defs
}

pub(crate) fn addPermissionEscalationFieldsToDefs(defs: &mut [ToolDef]) {
    for def in defs {
        addPermissionEscalationFields(&mut def.function.parameters);
    }
}

fn addPermissionEscalationFields(parameters: &mut serde_json::Value) {
    let Some(obj) = parameters.as_object_mut() else {
        return;
    };
    let props = obj
        .entry("properties")
        .or_insert_with(|| serde_json::json!({}));
    let Some(propsObj) = props.as_object_mut() else {
        return;
    };

    propsObj.insert(
        "raiseToUser".into(),
        serde_json::json!({
            "type": "boolean",
            "description": "Permission retry only. Set true only after a previous auto-review denial explicitly allowed escalation for this exact same action."
        }),
    );
    propsObj.insert(
        "raiseReason".into(),
        serde_json::json!({
            "type": "string",
            "description": "Short reason to show the user when raiseToUser is true."
        }),
    );
}

pub(crate) fn stripPermissionEscalationArgs(value: &mut serde_json::Value) {
    if let Some(obj) = value.as_object_mut() {
        stripPermissionEscalationObject(obj);
    }
}

pub(crate) fn stripPermissionEscalationObject(
    obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    obj.remove("raiseToUser");
    obj.remove("raiseReason");
    obj.remove("raise_to_user");
    obj.remove("raise_reason");
}

#[cfg(test)]
mod tests {
    #[test]
    fn stripPermissionEscalationObjectRemovesFlatlineOnlyFields() {
        let mut obj = serde_json::json!({
            "query": "rust",
            "raiseToUser": true,
            "raiseReason": "blocking",
            "raise_to_user": true,
            "raise_reason": "blocking"
        })
        .as_object()
        .unwrap()
        .clone();

        super::stripPermissionEscalationObject(&mut obj);

        assert_eq!(obj.get("query").and_then(|v| v.as_str()), Some("rust"));
        assert!(obj.get("raiseToUser").is_none());
        assert!(obj.get("raiseReason").is_none());
        assert!(obj.get("raise_to_user").is_none());
        assert!(obj.get("raise_reason").is_none());
    }
}
