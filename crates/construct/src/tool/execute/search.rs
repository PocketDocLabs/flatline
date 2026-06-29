use super::subprocess::runSubprocess;
use super::{
    MAX_FUZZY_RESULTS, MAX_GLOB_RESULTS, MAX_GREP_CONTENT_LINES, MAX_GREP_FILES,
    MAX_LISTDIR_ENTRIES, MAX_OUTLINE_ENTRIES, MAX_READ_BYTES, MAX_READ_LINES, MAX_RELATED_FILES,
    MAX_STRUCT_MATCHES, expandFromAnchor,
};

// --- Search / structure / diff execute functions ---

pub(super) async fn executeGlob(pattern: &str, path: Option<&str>, metadata: bool) -> String {
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
                tracing::debug!(%pattern, ?path, "glob: no matches");
                return "No files found.".into();
            }
            let lines: Vec<&str> = stdout.lines().collect();
            let total = lines.len();
            tracing::debug!(%pattern, total, "glob matched");
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
        Err(e) => {
            tracing::debug!(%pattern, error = %e, "glob subprocess failed");
            e.to_string()
        }
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
pub(super) async fn executeGrep(
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

    let fileType = fileType.map(str::trim).filter(|value| !value.is_empty());

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
                tracing::debug!(%pattern, %scope, "grep: no matches");
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
            tracing::debug!(%pattern, total, "grep matched");
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
        Err(e) => {
            tracing::debug!(%pattern, error = %e, "grep subprocess failed");
            e.to_string()
        }
    }
}

pub(super) fn executeListDir(
    path: &str,
    depth: usize,
    offset: usize,
    limit: usize,
    metadata: bool,
) -> String {
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

pub(super) async fn executeStructSearch(
    pattern: &str,
    language: &str,
    path: Option<&str>,
) -> String {
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
        Err(e) => e.to_string(),
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

pub(super) async fn executeDiff(
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
            Err(e) => return e.to_string(),
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
        Err(e) => e.to_string(),
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

// --- Fuzzy find ---

pub(super) async fn executeFuzzyFind(query: &str, path: Option<&str>) -> String {
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
        Err(e) => return e.to_string(),
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

pub(super) async fn executeFileOutline(path: &str) -> String {
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
        Err(e) => return e.to_string(),
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

pub(super) async fn executeViewSymbol(file: &str, symbol: &str) -> String {
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

pub(super) fn executeRelatedFiles(path: &str) -> String {
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
