//! Diff rendering with colored lines for file edits.
//!
//! Uses `similar` to compute line-level diffs and renders them
//! with green/red backgrounds for insertions/deletions. Context
//! lines are shown with a DIM modifier.
//!
//! # Public API
//! - [`renderDiff`] — render a unified diff as styled Lines
//!
//! # Dependencies
//! `similar`, `ratatui`

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

/// Render a unified diff between old and new text.
///
/// Args:
///     old: Original file content.
///     new: Modified file content.
///     lang: Optional language for syntax highlighting (reserved for V2).
///     width: Available rendering width.
///
/// Returns:
///     Vec<Line<'static>>: Styled diff lines with green/red backgrounds.
pub fn renderDiff(
    old: &str,
    new: &str,
    _lang: Option<&str>,
    _width: u16,
) -> Vec<Line<'static>> {
    let diff = TextDiff::from_lines(old, new);
    let mut lines = Vec::new();

    let deleteStyle = Style::default().fg(Color::Red).add_modifier(Modifier::DIM);
    let insertStyle = Style::default().fg(Color::Green);
    let contextStyle = Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM);
    let headerStyle = Style::default().fg(Color::DarkGray);

    for group in diff.grouped_ops(3) {
        // Hunk header.
        let first = group.first().unwrap();
        let last = group.last().unwrap();
        let oldRange = first.old_range();
        let newRange = last.new_range();
        let header = format!(
            "@@ -{},{} +{},{} @@",
            oldRange.start + 1,
            oldRange.end - oldRange.start,
            newRange.start + 1,
            newRange.end - newRange.start,
        );
        lines.push(Line::from(Span::styled(header, headerStyle)));

        for op in &group {
            for change in diff.iter_changes(op) {
                let (prefix, style) = match change.tag() {
                    ChangeTag::Delete => ("-", deleteStyle),
                    ChangeTag::Insert => ("+", insertStyle),
                    ChangeTag::Equal => (" ", contextStyle),
                };
                let text = change.value().trim_end_matches('\n');
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{text}"),
                    style,
                )));
            }
        }
    }

    lines
}
