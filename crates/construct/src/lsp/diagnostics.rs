//! Diagnostic aggregation, formatting, and severity filtering.
//!
//! Converts LSP Diagnostic objects into a compact string format suitable
//! for appending to tool output. Filters by severity and caps output to
//! avoid flooding the context.
//!
//! # Public API
//! - [`formatDiagnostics`] — format diagnostics for a single file
//! - [`formatMultiFileDiagnostics`] — format diagnostics across multiple files
//!
//! # Dependencies
//! `async-lsp` (for lsp-types re-export)

use async_lsp::lsp_types::{Diagnostic, DiagnosticSeverity};

const MAX_ERRORS_PER_FILE: usize = 20;
const MAX_FILES: usize = 5;
const MAX_TOTAL_DIAGNOSTICS: usize = 200;

/// Format diagnostics for a single file into the tool output format.
///
/// Args:
///     path: File path (relative or absolute).
///     diagnostics: LSP diagnostics for this file.
///     minSeverity: Minimum severity to include (1=Error, 2=Warning, 3=Info, 4=Hint).
///
/// Returns:
///     Formatted string, empty if no diagnostics match the filter.
pub fn formatDiagnostics(
    path: &str,
    diagnostics: &[Diagnostic],
    minSeverity: DiagnosticSeverity,
) -> String {
    let filtered: Vec<&Diagnostic> = diagnostics
        .iter()
        .filter(|d| {
            d.severity
                .map(|s| s <= minSeverity)
                .unwrap_or(true)
        })
        .collect();

    if filtered.is_empty() {
        return String::new();
    }

    let mut out = format!("<diagnostics file=\"{path}\">\n");
    for (i, d) in filtered.iter().enumerate() {
        if i >= MAX_ERRORS_PER_FILE {
            let remaining = filtered.len() - MAX_ERRORS_PER_FILE;
            out.push_str(&format!("... and {remaining} more\n"));
            break;
        }
        let line = d.range.start.line + 1;
        let col = d.range.start.character + 1;
        let label = severityLabel(d.severity);
        let msg = d.message.lines().next().unwrap_or("");
        out.push_str(&format!("{label} [{line}:{col}] {msg}\n"));
    }
    out.push_str("</diagnostics>");
    out
}

/// Format diagnostics across multiple files.
///
/// Args:
///     fileDiagnostics: Vec of (path, diagnostics) pairs.
///     minSeverity: Minimum severity to include.
///
/// Returns:
///     Formatted string with header, empty if no diagnostics.
pub fn formatMultiFileDiagnostics(
    fileDiagnostics: &[(&str, &[Diagnostic])],
    minSeverity: DiagnosticSeverity,
) -> String {
    let mut sections = Vec::new();
    let mut totalCount = 0;

    for (path, diagnostics) in fileDiagnostics {
        if sections.len() >= MAX_FILES {
            break;
        }
        if totalCount >= MAX_TOTAL_DIAGNOSTICS {
            break;
        }
        let section = formatDiagnostics(path, diagnostics, minSeverity);
        if !section.is_empty() {
            totalCount += diagnostics
                .iter()
                .filter(|d| d.severity.map(|s| s <= minSeverity).unwrap_or(true))
                .count();
            sections.push(section);
        }
    }

    if sections.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n\nLSP errors detected:\n");
    for section in sections {
        out.push_str(&section);
        out.push('\n');
    }
    out
}

/// Human-readable severity label.
fn severityLabel(severity: Option<DiagnosticSeverity>) -> &'static str {
    match severity {
        Some(DiagnosticSeverity::ERROR) => "ERROR",
        Some(DiagnosticSeverity::WARNING) => "WARNING",
        Some(DiagnosticSeverity::INFORMATION) => "INFO",
        Some(DiagnosticSeverity::HINT) => "HINT",
        _ => "ERROR",
    }
}
