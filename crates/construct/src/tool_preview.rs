//! Streaming preview of tool-call arguments.
//!
//! As tool-call JSON arguments arrive delta-by-delta, [`previewForTool`]
//! pulls out a short human-readable string describing what the call will
//! do (target file, command, pattern, etc.). A field is only surfaced
//! after its closing quote has been observed, so the preview never flickers
//! on half-streamed values.
//!
//! # Public API
//! - [`previewForTool`] — dispatch by tool name, return a formatted preview
//! - [`findClosedStringField`] — extract one top-level string field
//!   (exposed for tests and custom tools)

/// Dispatch a tool's partial arguments to the right preview formatter.
/// Returns `None` until at least one required field has fully closed.
pub fn previewForTool(name: &str, args: &str) -> Option<String> {
    match name {
        "shell" => {
            let cmd = findClosedStringField(args, "command")?;
            Some(truncate(&cmd, PREVIEW_LIMIT))
        }
        "readFile" | "writeFile" | "editFile" | "multiEdit" | "listDir" | "deleteFile"
        | "makeDirs" | "fileOutline" | "relatedFiles" | "diagnostics" => {
            let path = findClosedStringField(args, "path")?;
            Some(truncate(&path, PREVIEW_LIMIT))
        }
        "copyFile" | "moveFile" => {
            let src = findClosedStringField(args, "src")?;
            match findClosedStringField(args, "dest") {
                Some(d) if !d.is_empty() => {
                    Some(truncate(&format!("{src} \u{2192} {d}"), PREVIEW_LIMIT))
                }
                _ => Some(truncate(&src, PREVIEW_LIMIT)),
            }
        }
        "glob" => {
            let pat = findClosedStringField(args, "pattern")?;
            match findClosedStringField(args, "path") {
                Some(p) if !p.is_empty() => Some(truncate(&format!("{pat} in {p}"), PREVIEW_LIMIT)),
                _ => Some(truncate(&pat, PREVIEW_LIMIT)),
            }
        }
        "grep" => {
            let pat = findClosedStringField(args, "pattern")?;
            match findClosedStringField(args, "path") {
                Some(p) if !p.is_empty() => {
                    Some(truncate(&format!("\"{pat}\" in {p}"), PREVIEW_LIMIT))
                }
                _ => Some(truncate(&format!("\"{pat}\""), PREVIEW_LIMIT)),
            }
        }
        "structSearch" => {
            let pat = findClosedStringField(args, "pattern")?;
            Some(truncate(&format!("\"{pat}\""), PREVIEW_LIMIT))
        }
        "fuzzyFind" => {
            let q = findClosedStringField(args, "query")?;
            Some(truncate(&q, PREVIEW_LIMIT))
        }
        "viewSymbol" => {
            let sym = findClosedStringField(args, "symbol")?;
            match findClosedStringField(args, "file") {
                Some(f) if !f.is_empty() => Some(truncate(&format!("{sym} in {f}"), PREVIEW_LIMIT)),
                _ => Some(truncate(&sym, PREVIEW_LIMIT)),
            }
        }
        "webFetch" | "webSimilar" => {
            let url = findClosedStringField(args, "url")?;
            Some(truncate(&url, PREVIEW_LIMIT))
        }
        "webSearch" => {
            let q = findClosedStringField(args, "query")?;
            Some(truncate(&q, PREVIEW_LIMIT))
        }
        "historyFetch" => findClosedStringField(args, "blockId"),
        "historySearch" => {
            let q = findClosedStringField(args, "query")?;
            Some(truncate(&q, PREVIEW_LIMIT))
        }
        "task" => {
            let p = findClosedStringField(args, "prompt")?;
            Some(truncate(&p, PREVIEW_LIMIT))
        }
        "searchOutput" => {
            let p = findClosedStringField(args, "pattern")?;
            Some(truncate(&format!("\"{p}\""), PREVIEW_LIMIT))
        }
        "diff" => {
            findClosedStringField(args, "path").or_else(|| findClosedStringField(args, "file1"))
        }
        _ => None,
    }
}

const PREVIEW_LIMIT: usize = 80;

/// Collapse runs of whitespace and truncate at a display-character limit.
fn truncate(s: &str, max: usize) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max).collect();
    format!("{cut}\u{2026}")
}

/// Scan a partial JSON object for a top-level string field's closed value.
///
/// Returns `Some(value)` only when the closing quote of the value has been
/// observed. Partial values return `None` so previews never flicker.
pub fn findClosedStringField(partial: &str, key: &str) -> Option<String> {
    let bytes = partial.as_bytes();
    let needle = format!("\"{key}\"");
    let needleBytes = needle.as_bytes();
    let mut i = 0;
    let mut depth: i32 = 0;
    let mut inString = false;
    let mut escaped = false;
    let mut expectingKey = false;
    while i < bytes.len() {
        let c = bytes[i];
        if inString {
            if escaped {
                escaped = false;
                i += 1;
                continue;
            }
            if c == b'\\' {
                escaped = true;
                i += 1;
                continue;
            }
            if c == b'"' {
                inString = false;
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }
        match c {
            b'{' => {
                depth += 1;
                expectingKey = depth == 1;
                i += 1;
            }
            b'[' => {
                depth += 1;
                i += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                i += 1;
            }
            b',' if depth == 1 => {
                expectingKey = true;
                i += 1;
            }
            b':' if depth == 1 => {
                expectingKey = false;
                i += 1;
            }
            b'"' => {
                if expectingKey && depth == 1 && bytes[i..].starts_with(needleBytes) {
                    return readColonStringValue(bytes, i + needleBytes.len());
                }
                inString = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    None
}

fn readColonStringValue(bytes: &[u8], start: usize) -> Option<String> {
    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;
    let mut out = String::new();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            if i + 1 >= bytes.len() {
                return None;
            }
            match bytes[i + 1] {
                b'"' => {
                    out.push('"');
                    i += 2;
                }
                b'\\' => {
                    out.push('\\');
                    i += 2;
                }
                b'/' => {
                    out.push('/');
                    i += 2;
                }
                b'n' => {
                    out.push('\n');
                    i += 2;
                }
                b't' => {
                    out.push('\t');
                    i += 2;
                }
                b'r' => {
                    out.push('\r');
                    i += 2;
                }
                b'b' => {
                    out.push('\x08');
                    i += 2;
                }
                b'f' => {
                    out.push('\x0C');
                    i += 2;
                }
                b'u' => {
                    if i + 6 > bytes.len() {
                        return None;
                    }
                    let hex = std::str::from_utf8(&bytes[i + 2..i + 6]).ok()?;
                    let code = u32::from_str_radix(hex, 16).ok()?;
                    if let Some(ch) = char::from_u32(code) {
                        out.push(ch);
                    }
                    i += 6;
                }
                _ => return None,
            }
        } else if c == b'"' {
            return Some(out);
        } else {
            let rest = std::str::from_utf8(&bytes[i..]).ok()?;
            let ch = rest.chars().next()?;
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn findsClosedFieldAtTopLevel() {
        let args = r#"{"path": "crates/deck/src/app.rs", "line": 42}"#;
        assert_eq!(
            findClosedStringField(args, "path").as_deref(),
            Some("crates/deck/src/app.rs"),
        );
    }

    #[test]
    fn returnsNoneWhileValueStillOpen() {
        let args = r#"{"path": "crates/deck/"#;
        assert_eq!(findClosedStringField(args, "path"), None);
    }

    #[test]
    fn returnsValueOnceClosingQuoteSeen() {
        let args = r#"{"path": "crates/deck/src/app.rs""#;
        assert_eq!(
            findClosedStringField(args, "path").as_deref(),
            Some("crates/deck/src/app.rs"),
        );
    }

    #[test]
    fn ignoresKeyNameAppearingAsValueElsewhere() {
        // "path" appears as a value of another key; only the real path key
        // (which is unclosed) should count. Expect None.
        let args = r#"{"label": "path", "path": "foo"#;
        assert_eq!(findClosedStringField(args, "path"), None);
    }

    #[test]
    fn ignoresNestedOccurrence() {
        // file_path is only nested; the top level doesn't have it.
        let args = r#"{"meta": {"file_path": "inner.rs"}}"#;
        assert_eq!(findClosedStringField(args, "file_path"), None);
    }

    #[test]
    fn handlesEscapedQuotesInValue() {
        let args = r#"{"command": "echo \"hi\" there"}"#;
        assert_eq!(
            findClosedStringField(args, "command").as_deref(),
            Some("echo \"hi\" there"),
        );
    }

    #[test]
    fn handlesUnicodeEscape() {
        let args = r#"{"label": "aéb"}"#;
        assert_eq!(findClosedStringField(args, "label").as_deref(), Some("aéb"),);
    }

    #[test]
    fn previewShellCommand() {
        let args =
            r#"{"command": "cargo check --all-targets", "explanation": "x", "impact": "read"}"#;
        assert_eq!(
            previewForTool("shell", args).as_deref(),
            Some("cargo check --all-targets"),
        );
    }

    #[test]
    fn previewEditFilePath() {
        let args = r#"{"path": "crates/deck/src/app.rs", "old_string": "foo"}"#;
        assert_eq!(
            previewForTool("editFile", args).as_deref(),
            Some("crates/deck/src/app.rs"),
        );
    }

    #[test]
    fn previewGrepPatternOnly() {
        let args = r#"{"pattern": "ToolCall"}"#;
        assert_eq!(
            previewForTool("grep", args).as_deref(),
            Some("\"ToolCall\""),
        );
    }

    #[test]
    fn previewGrepPatternAndPath() {
        let args = r#"{"pattern": "ToolCall", "path": "crates/"}"#;
        assert_eq!(
            previewForTool("grep", args).as_deref(),
            Some("\"ToolCall\" in crates/"),
        );
    }

    #[test]
    fn previewCollapsesWhitespaceAndTruncates() {
        let long = "a".repeat(200);
        let args = format!(r#"{{"command": "{long}"}}"#);
        let out = previewForTool("shell", &args).unwrap();
        assert!(out.ends_with('\u{2026}'));
        assert!(out.chars().count() <= 81);
    }

    #[test]
    fn previewUnknownToolReturnsNone() {
        assert_eq!(previewForTool("mysteryTool", r#"{"x": "y"}"#), None);
    }

    #[test]
    fn previewReturnsNoneUntilFieldCloses() {
        let args = r#"{"path": "crates/deck/src"#;
        assert_eq!(previewForTool("readFile", args), None);
    }
}
