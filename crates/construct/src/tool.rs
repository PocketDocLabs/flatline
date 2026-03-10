//! Tool definitions and execution.
//!
//! Defines the tools available to the agent and handles execution.
//! Shell commands go through the construct-owned Shell session.
//! File operations run in-process.
//!
//! # Public API
//! - [`builtinDefs`] — returns tool definitions for the LLM
//! - [`execute`] — execute a parsed tool action
//!
//! # Dependencies
//! `serde_json`, `regex`

use crate::message::ToolDef;
use crate::shell::Shell;

// --- Limits ---

const MAX_READ_LINES: usize = 2000;
const MAX_READ_BYTES: usize = 100_000;
const MAX_LINE_LENGTH: usize = 2000;

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
                        }
                    },
                    "required": ["command"]
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
                    Overwrites the entire file. Prefer editFile for small changes.".into(),
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
                    context in old_string to make the match unique.".into(),
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
    ]
}

/// A parsed tool invocation from the LLM.
#[derive(Debug)]
pub enum ToolAction {
    Shell { command: String },
    ReadFile { path: String, offset: Option<usize>, limit: Option<usize> },
    WriteFile { path: String, content: String },
    EditFile { path: String, oldString: String, newString: String, replaceAll: bool },
    ShellHistory,
    ReadOutput { index: usize, offset: Option<usize>, limit: Option<usize> },
    SearchOutput { index: usize, pattern: String, context: usize },
    ReadTerminal { lines: usize },
    Unknown { name: String, args: String },
}

/// Human-readable summary of what a tool action will do.
pub fn summarize(action: &ToolAction) -> String {
    match action {
        ToolAction::Shell { command } => format!("Run: {command}"),
        ToolAction::ReadFile { path, offset, limit } => {
            match (offset, limit) {
                (Some(o), Some(l)) => format!("Read: {path} (lines {o}..{})", o + l - 1),
                (Some(o), None) => format!("Read: {path} (from line {o})"),
                (None, Some(l)) => format!("Read: {path} (first {l} lines)"),
                (None, None) => format!("Read: {path}"),
            }
        }
        ToolAction::WriteFile { path, content } => {
            format!("Write {} bytes to {path}", content.len())
        }
        ToolAction::EditFile { path, oldString, replaceAll, .. } => {
            let preview = if oldString.len() > 40 {
                format!("{}...", &oldString[..40])
            } else {
                oldString.clone()
            };
            if *replaceAll {
                format!("Edit {path}: replace all \"{preview}\"")
            } else {
                format!("Edit {path}: replace \"{preview}\"")
            }
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
        ToolAction::Unknown { name, .. } => format!("Unknown tool: {name}"),
    }
}

/// Execute a tool action and return the output string.
pub async fn execute(action: &ToolAction, shell: &Shell) -> String {
    match action {
        ToolAction::Shell { command } => {
            let raw = shell.execute(command).await;
            // Apply same size guard as readFile. Full output is in shell history.
            let index = shell.historyLen().saturating_sub(1);
            truncateOutput(&raw, index)
        }
        ToolAction::ReadFile { path, offset, limit } => executeReadFile(path, *offset, *limit),
        ToolAction::WriteFile { path, content } => executeWriteFile(path, content),
        ToolAction::EditFile { path, oldString, newString, replaceAll } => {
            executeEditFile(path, oldString, newString, *replaceAll)
        }
        ToolAction::ShellHistory => executeShellHistory(shell),
        ToolAction::ReadOutput { index, offset, limit } => {
            executeReadOutput(shell, *index, *offset, *limit)
        }
        ToolAction::SearchOutput { index, pattern, context } => {
            executeSearchOutput(shell, *index, pattern, *context)
        }
        ToolAction::ReadTerminal { lines } => shell.readTerminal(*lines),
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

fn executeReadFile(path: &str, offset: Option<usize>, limit: Option<usize>) -> String {
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
            format!("{}...", &line[..MAX_LINE_LENGTH])
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
            format!("{}...", &cmd[..80])
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
                    format!("{}...", &record.command[..100])
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

/// Parse a tool call name + JSON arguments into a ToolAction.
pub fn parse(name: &str, argsJson: &str) -> ToolAction {
    let args: serde_json::Value = serde_json::from_str(argsJson).unwrap_or_default();

    match name {
        "shell" => ToolAction::Shell {
            command: args["command"].as_str().unwrap_or("").into(),
        },
        "readFile" => ToolAction::ReadFile {
            path: args["path"].as_str().unwrap_or("").into(),
            offset: args["offset"].as_u64().map(|v| v as usize),
            limit: args["limit"].as_u64().map(|v| v as usize),
        },
        "writeFile" => ToolAction::WriteFile {
            path: args["path"].as_str().unwrap_or("").into(),
            content: args["content"].as_str().unwrap_or("").into(),
        },
        "editFile" => ToolAction::EditFile {
            path: args["path"].as_str().unwrap_or("").into(),
            oldString: args["old_string"].as_str().unwrap_or("").into(),
            newString: args["new_string"].as_str().unwrap_or("").into(),
            replaceAll: args["replace_all"].as_bool().unwrap_or(false),
        },
        "shellHistory" => ToolAction::ShellHistory,
        "readOutput" => ToolAction::ReadOutput {
            index: args["index"].as_u64().unwrap_or(0) as usize,
            offset: args["offset"].as_u64().map(|v| v as usize),
            limit: args["limit"].as_u64().map(|v| v as usize),
        },
        "searchOutput" => ToolAction::SearchOutput {
            index: args["index"].as_u64().unwrap_or(0) as usize,
            pattern: args["pattern"].as_str().unwrap_or("").into(),
            context: args["context"].as_u64().unwrap_or(3) as usize,
        },
        "readTerminal" => ToolAction::ReadTerminal {
            lines: args["lines"].as_u64().unwrap_or(50) as usize,
        },
        _ => ToolAction::Unknown {
            name: name.into(),
            args: argsJson.into(),
        },
    }
}
