use super::{EditOp, ShellImpact, ToolAction, ToolParseError};

type ToolParseResult<T> = std::result::Result<T, ToolParseError>;

fn reqString(args: &serde_json::Value, field: &'static str) -> ToolParseResult<String> {
    match &args[field] {
        serde_json::Value::String(s) if !s.is_empty() => Ok(s.clone()),
        serde_json::Value::String(_) | serde_json::Value::Null => {
            Err(ToolParseError::MissingField { field })
        }
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "string",
        }),
    }
}

fn reqStringAllowEmpty(args: &serde_json::Value, field: &'static str) -> ToolParseResult<String> {
    match &args[field] {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Null => Err(ToolParseError::MissingField { field }),
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "string",
        }),
    }
}

fn optString(args: &serde_json::Value, field: &'static str) -> ToolParseResult<Option<String>> {
    match &args[field] {
        serde_json::Value::String(s) => Ok(Some(s.clone())),
        serde_json::Value::Null => Ok(None),
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "string",
        }),
    }
}

fn optStringDefault(
    args: &serde_json::Value,
    field: &'static str,
    default: &'static str,
) -> ToolParseResult<String> {
    Ok(optString(args, field)?.unwrap_or_else(|| default.to_string()))
}

fn optStringArray(
    args: &serde_json::Value,
    field: &'static str,
) -> ToolParseResult<Option<Vec<String>>> {
    match &args[field] {
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| match item {
                serde_json::Value::String(s) => Ok(s.clone()),
                _ => Err(ToolParseError::WrongType {
                    field,
                    expected: "array of strings",
                }),
            })
            .collect::<ToolParseResult<Vec<_>>>()
            .map(Some),
        serde_json::Value::Null => Ok(None),
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "array of strings",
        }),
    }
}

fn optU64(args: &serde_json::Value, field: &'static str) -> ToolParseResult<Option<u64>> {
    match &args[field] {
        serde_json::Value::Number(n) => n.as_u64().map(Some).ok_or(ToolParseError::WrongType {
            field,
            expected: "integer",
        }),
        serde_json::Value::Null => Ok(None),
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "integer",
        }),
    }
}

fn reqU64(args: &serde_json::Value, field: &'static str) -> ToolParseResult<u64> {
    optU64(args, field)?.ok_or(ToolParseError::MissingField { field })
}

fn optBool(args: &serde_json::Value, field: &'static str) -> ToolParseResult<Option<bool>> {
    match &args[field] {
        serde_json::Value::Bool(b) => Ok(Some(*b)),
        serde_json::Value::Null => Ok(None),
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "boolean",
        }),
    }
}

fn reqArray<'a>(
    args: &'a serde_json::Value,
    field: &'static str,
) -> ToolParseResult<&'a Vec<serde_json::Value>> {
    match &args[field] {
        serde_json::Value::Array(items) => Ok(items),
        serde_json::Value::Null => Err(ToolParseError::MissingField { field }),
        _ => Err(ToolParseError::WrongType {
            field,
            expected: "array",
        }),
    }
}

fn reqNestedStringAllowEmpty(
    args: &serde_json::Value,
    context: &'static str,
    field: &'static str,
) -> ToolParseResult<String> {
    match &args[field] {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Null => Err(ToolParseError::MissingNestedField { context, field }),
        _ => Err(ToolParseError::WrongNestedType {
            context,
            field,
            expected: "string",
        }),
    }
}

fn optNestedBool(
    args: &serde_json::Value,
    context: &'static str,
    field: &'static str,
) -> ToolParseResult<Option<bool>> {
    match &args[field] {
        serde_json::Value::Bool(b) => Ok(Some(*b)),
        serde_json::Value::Null => Ok(None),
        _ => Err(ToolParseError::WrongNestedType {
            context,
            field,
            expected: "boolean",
        }),
    }
}

fn validateStringEnum(
    value: String,
    field: &'static str,
    expected: &'static str,
    allowed: &[&str],
) -> ToolParseResult<String> {
    if allowed.contains(&value.as_str()) {
        Ok(value)
    } else {
        Err(ToolParseError::InvalidField { field, expected })
    }
}

/// Parse a tool call name + JSON arguments into a ToolAction.
///
/// Returns Err with structured missing/malformed required-field details.
/// The error's Display text is sent back to the model as the tool result so it can retry.
pub fn parse(name: &str, argsJson: &str) -> std::result::Result<ToolAction, ToolParseError> {
    let args: serde_json::Value =
        serde_json::from_str(argsJson).map_err(|e| ToolParseError::MalformedJson(e.to_string()))?;

    let action = match name {
        "shell" => {
            let impactString =
                optString(&args, "impact")?.ok_or(ToolParseError::MissingFieldWithExpected {
                    field: "impact",
                    expected: "one of: read, minorMod, majorMod, delete",
                })?;
            let impact: ShellImpact =
                serde_json::from_value(serde_json::Value::String(impactString)).map_err(|_| {
                    ToolParseError::InvalidField {
                        field: "impact",
                        expected: "one of: read, minorMod, majorMod, delete",
                    }
                })?;
            ToolAction::Shell {
                command: reqString(&args, "command")?,
                explanation: reqString(&args, "explanation")?,
                impact,
                timeout: optU64(&args, "timeout")?,
                terminal: optString(&args, "terminal")?,
                runInBackground: optBool(&args, "runInBackground")?.unwrap_or(false),
            }
        }
        "readFile" => ToolAction::ReadFile {
            path: reqString(&args, "path")?,
            offset: optU64(&args, "offset")?.map(|v| v as usize),
            limit: optU64(&args, "limit")?.map(|v| v as usize),
            anchor: optU64(&args, "anchor")?.map(|v| v as usize),
        },
        "writeFile" => ToolAction::WriteFile {
            path: reqString(&args, "path")?,
            content: reqString(&args, "content")?,
        },
        "editFile" => ToolAction::EditFile {
            path: reqString(&args, "path")?,
            oldString: reqString(&args, "old_string")?,
            newString: reqStringAllowEmpty(&args, "new_string")?,
            replaceAll: optBool(&args, "replace_all")?.unwrap_or(false),
        },
        "multiEdit" => ToolAction::MultiEdit {
            path: reqString(&args, "path")?,
            edits: reqArray(&args, "edits")?
                .iter()
                .map(|e| {
                    Ok(EditOp {
                        oldString: reqNestedStringAllowEmpty(e, "Edit", "old_string")?,
                        newString: reqNestedStringAllowEmpty(e, "Edit", "new_string")?,
                        replaceAll: optNestedBool(e, "Edit", "replace_all")?.unwrap_or(false),
                    })
                })
                .collect::<ToolParseResult<Vec<_>>>()?,
        },
        "copyFile" => ToolAction::CopyFile {
            src: reqString(&args, "src")?,
            dest: reqString(&args, "dest")?,
            overwrite: optBool(&args, "overwrite")?.unwrap_or(false),
        },
        "moveFile" => ToolAction::MoveFile {
            src: reqString(&args, "src")?,
            dest: reqString(&args, "dest")?,
            overwrite: optBool(&args, "overwrite")?.unwrap_or(false),
        },
        "deleteFile" => ToolAction::DeleteFile {
            path: reqString(&args, "path")?,
            recursive: optBool(&args, "recursive")?.unwrap_or(false),
        },
        "makeDirs" => ToolAction::MakeDirs {
            path: reqString(&args, "path")?,
        },
        "shellHistory" => ToolAction::ShellHistory {
            terminal: optString(&args, "terminal")?,
        },
        "readOutput" => ToolAction::ReadOutput {
            index: optU64(&args, "index")?.unwrap_or(0) as usize,
            offset: optU64(&args, "offset")?.map(|v| v as usize),
            limit: optU64(&args, "limit")?.map(|v| v as usize),
            terminal: optString(&args, "terminal")?,
        },
        "searchOutput" => ToolAction::SearchOutput {
            index: optU64(&args, "index")?.unwrap_or(0) as usize,
            pattern: reqString(&args, "pattern")?,
            context: optU64(&args, "context")?.unwrap_or(3) as usize,
            terminal: optString(&args, "terminal")?,
        },
        "readTerminal" => ToolAction::ReadTerminal {
            lines: optU64(&args, "lines")?.unwrap_or(50) as usize,
            terminal: optString(&args, "terminal")?,
        },
        "terminalSpawn" => ToolAction::TerminalSpawn {
            name: optString(&args, "name")?,
        },
        "terminalSwitch" => ToolAction::TerminalSwitch {
            name: reqString(&args, "name")?,
        },
        "terminalKill" => ToolAction::TerminalKill {
            name: reqString(&args, "name")?,
        },
        "terminalList" => ToolAction::TerminalList,
        "terminalRunList" => ToolAction::TerminalRunList,
        "terminalRunStop" => ToolAction::TerminalRunStop {
            runId: reqString(&args, "runId")?,
        },
        "jobOutput" => ToolAction::JobOutput {
            jobId: reqU64(&args, "jobId")?,
            sinceLine: optU64(&args, "sinceLine")?,
            maxLines: optU64(&args, "maxLines")?.map(|v| v as usize),
        },
        "jobStop" => ToolAction::JobStop {
            jobId: reqU64(&args, "jobId")?,
        },
        "jobList" => ToolAction::JobList,
        "waitForSubagent" => ToolAction::WaitForSubagent {
            jobId: reqU64(&args, "jobId")?,
        },
        "monitor" => ToolAction::Monitor {
            description: reqString(&args, "description")?,
            terminal: optString(&args, "terminal")?,
            filter: reqString(&args, "filter")?,
        },
        "monitorStop" => ToolAction::MonitorStop {
            monitorId: reqU64(&args, "monitorId")?,
        },
        "monitorList" => ToolAction::MonitorList,
        "scheduleWakeup" => ToolAction::ScheduleWakeup {
            delaySeconds: reqU64(&args, "delaySeconds")?,
            prompt: reqString(&args, "prompt")?,
        },
        "cronCreate" => ToolAction::CronCreate {
            spec: reqString(&args, "spec")?,
            prompt: reqString(&args, "prompt")?,
            recurring: optBool(&args, "recurring")?.unwrap_or(true),
        },
        "cronList" => ToolAction::CronList,
        "cronDelete" => ToolAction::CronDelete {
            wakeId: reqU64(&args, "wakeId")?,
        },
        "fileWatch" => ToolAction::FileWatch {
            path: reqString(&args, "path")?,
            prompt: reqString(&args, "prompt")?,
        },
        "glob" => ToolAction::Glob {
            pattern: reqString(&args, "pattern")?,
            path: optString(&args, "path")?,
            metadata: optBool(&args, "metadata")?.unwrap_or(false),
        },
        "grep" => ToolAction::Grep {
            pattern: reqString(&args, "pattern")?,
            path: optString(&args, "path")?,
            include: optString(&args, "include")?,
            fileType: optString(&args, "type")?,
            outputMode: validateStringEnum(
                optStringDefault(&args, "output_mode", "files")?,
                "output_mode",
                "one of: files, content, count",
                &["files", "content", "count"],
            )?,
            caseSensitive: optBool(&args, "case_sensitive")?,
            contextLines: optU64(&args, "context_lines")?.map(|v| v as usize),
            multiline: optBool(&args, "multiline")?.unwrap_or(false),
        },
        "listDir" => ToolAction::ListDir {
            path: reqString(&args, "path")?,
            depth: optU64(&args, "depth")?.unwrap_or(2).clamp(1, 5) as usize,
            offset: optU64(&args, "offset")?.unwrap_or(0) as usize,
            limit: optU64(&args, "limit")?.unwrap_or(500) as usize,
            metadata: optBool(&args, "metadata")?.unwrap_or(false),
        },
        "structSearch" => ToolAction::StructSearch {
            pattern: reqString(&args, "pattern")?,
            language: reqString(&args, "language")?,
            path: optString(&args, "path")?,
        },
        "diff" => ToolAction::Diff {
            path: optString(&args, "path")?,
            gitRef: optString(&args, "ref")?,
            pathA: optString(&args, "path_a")?,
            pathB: optString(&args, "path_b")?,
        },
        "fuzzyFind" => ToolAction::FuzzyFind {
            query: reqString(&args, "query")?,
            path: optString(&args, "path")?,
        },
        "fileOutline" => ToolAction::FileOutline {
            path: reqString(&args, "path")?,
        },
        "viewSymbol" => ToolAction::ViewSymbol {
            file: reqString(&args, "file")?,
            symbol: reqString(&args, "symbol")?,
        },
        "relatedFiles" => ToolAction::RelatedFiles {
            path: reqString(&args, "path")?,
        },
        "webSearch" => ToolAction::WebSearch {
            query: reqString(&args, "query")?,
            allowedDomains: optStringArray(&args, "allowed_domains")?,
            blockedDomains: optStringArray(&args, "blocked_domains")?,
            maxResults: optU64(&args, "max_results")?.map(|v| v as usize),
        },
        "webFetch" => ToolAction::WebFetch {
            url: reqString(&args, "url")?,
            prompt: optString(&args, "prompt")?,
            subpages: optU64(&args, "subpages")?.map(|v| v as usize),
        },
        "webSimilar" => ToolAction::WebSimilar {
            url: reqString(&args, "url")?,
            allowedDomains: optStringArray(&args, "allowed_domains")?,
            blockedDomains: optStringArray(&args, "blocked_domains")?,
            maxResults: optU64(&args, "max_results")?.map(|v| v as usize),
        },
        "historyFetch" => ToolAction::HistoryFetch {
            blockId: reqString(&args, "blockId")?,
        },
        "historySearch" => ToolAction::HistorySearch {
            query: reqString(&args, "query")?,
            mediaType: optString(&args, "mediaType")?,
        },
        "task" => ToolAction::Task {
            prompt: reqString(&args, "prompt")?,
            agent: optString(&args, "agent")?,
            runInBackground: optBool(&args, "runInBackground")?.unwrap_or(false),
        },
        "diagnostics" => ToolAction::Diagnostics {
            path: reqString(&args, "path")?,
            severity: validateStringEnum(
                optStringDefault(&args, "severity", "error")?,
                "severity",
                "one of: error, warning",
                &["error", "warning"],
            )?,
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
