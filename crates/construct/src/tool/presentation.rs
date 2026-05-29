use super::ToolAction;

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
        ToolAction::TerminalRunList => "List terminal runs".into(),
        ToolAction::TerminalRunStop { runId } => format!("terminalRunStop {runId}"),
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
            terminal,
            filter,
        } => {
            let target = terminal.as_deref().unwrap_or("agent target");
            format!("monitor \"{description}\": terminal {target} | /{filter}/")
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
pub(crate) fn diffPreview(action: &ToolAction) -> Option<String> {
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
pub(crate) fn proposedContent(action: &ToolAction) -> Option<(String, String)> {
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
