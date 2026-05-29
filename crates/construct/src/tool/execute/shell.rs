use crate::shell::Shell;

use super::{MAX_LINE_LENGTH, MAX_READ_BYTES, MAX_READ_LINES, formatNumberedLines};

/// Truncate shell output into a head/middle/tail three-piece slice with
/// a reference to readOutput for the rest.
///
/// Tail-weighted (20/10/70) because for shell output the **tail** is
/// where the signal lives — exit codes, error summaries, final state.
/// Head gives setup context; a middle sample helps the model tell
/// whether something interesting sits in the elided range.
pub(crate) fn truncateOutput(raw: &str, historyIndex: usize, terminalName: &str) -> String {
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

pub(super) fn executeShellHistory(shell: &Shell, terminalName: &str) -> String {
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

pub(super) fn executeReadOutput(
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

pub(super) fn executeSearchOutput(
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
