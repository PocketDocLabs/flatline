use crate::shell::Shell;

use super::ToolAction;

mod fs;
mod search;
mod shell;
mod subprocess;

use fs::{
    executeCopyFile, executeDeleteFile, executeEditFile, executeMakeDirs, executeMoveFile,
    executeMultiEdit, executeWriteFile,
};
use search::{
    executeDiff, executeFileOutline, executeFuzzyFind, executeGlob, executeGrep, executeListDir,
    executeRelatedFiles, executeStructSearch, executeViewSymbol,
};
use shell::{executeReadOutput, executeSearchOutput, executeShellHistory};

pub(crate) use shell::truncateOutput;

#[cfg(test)]
pub(super) use fs::{FileKind, ImageFormat, classifyFile, executeReadFile};
#[cfg(test)]
pub(super) use subprocess::SubprocessError;

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

/// Execute a tool action and return the content (text or multimodal).
///
/// `terminalName` is the resolved display name of the shell (e.g. "main",
/// "build") so per-terminal tools can label their output. Caller passes
/// `action.terminal().unwrap_or(active_name)`.
pub(crate) async fn execute(
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
            // Background shell calls are routed by the Session so they can
            // become visible terminal-backed runs. If we got one here, it's
            // a bug.
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
        } => fs::executeReadFile(path, *offset, *limit, *anchor),
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
        | ToolAction::TerminalList
        | ToolAction::TerminalRunList
        | ToolAction::TerminalRunStop { .. } => crate::message::Content::text(
            "Error: terminal tools must be executed through the session.",
        ),
        // Task and monitor tools are handled by Session (need direct access
        // to JobPlane / MonitorPlane and logTx).
        ToolAction::JobOutput { .. }
        | ToolAction::JobStop { .. }
        | ToolAction::JobList
        | ToolAction::Monitor { .. }
        | ToolAction::MonitorStop { .. }
        | ToolAction::MonitorList => crate::message::Content::text(
            "Error: task/monitor tools must be executed through the session.",
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
