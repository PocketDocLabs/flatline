//! Syntax highlighting for code blocks via syntect + two-face.
//!
//! Provides a global `SyntaxSet` (250+ languages via two-face)
//! and theme, with functions to highlight code and render bordered
//! code blocks with horizontal scrolling.
//!
//! # Public API
//! - [`highlightLines`] — highlight code into spans per line (no borders)
//! - [`renderCodeBlock`] — render highlighted lines with borders and scroll
//! - [`diffLines`] — render diff content with red/green coloring
//!
//! # Dependencies
//! `syntect`, `two-face`, `ratatui`

use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, Style as SyntectStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Iterate (grapheme, display_width) pairs for a string.
///
/// Uses grapheme clusters with str-level width so emoji sequences like
/// `⚠\u{FE0F}` render as one atomic 2-col unit, matching what ratatui
/// and the terminal actually draw.
fn graphemeWidths(s: &str) -> impl Iterator<Item = (&str, usize)> {
    s.graphemes(true).map(|g| (g, UnicodeWidthStr::width(g)))
}

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME: OnceLock<Theme> = OnceLock::new();

fn syntaxSet() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(two_face::syntax::extra_newlines)
}

fn theme() -> &'static Theme {
    THEME.get_or_init(|| {
        let ts = ThemeSet::load_defaults();
        ts.themes["base16-ocean.dark"].clone()
    })
}

/// Maximum input size for highlighting (512KB).
const MAX_HIGHLIGHT_BYTES: usize = 512 * 1024;
/// Maximum line count for highlighting.
const MAX_HIGHLIGHT_LINES: usize = 10_000;

/// Highlight code into spans per line (no borders, no truncation).
///
/// Args:
///     code: The source code text.
///     lang: Optional language identifier for syntax detection.
///
/// Returns:
///     Vec<Vec<Span<'static>>>: Highlighted spans per source line.
pub fn highlightLines(code: &str, lang: Option<&str>) -> Vec<Vec<Span<'static>>> {
    let codeLines: Vec<&str> = code.lines().collect();
    let tooLarge = code.len() > MAX_HIGHLIGHT_BYTES || codeLines.len() > MAX_HIGHLIGHT_LINES;

    if tooLarge {
        let style = Style::default().fg(Color::White);
        codeLines
            .iter()
            .map(|line| vec![Span::styled(line.to_string(), style)])
            .collect()
    } else {
        syntectLines(&codeLines, lang)
    }
}

/// Render highlighted code lines with borders and horizontal scroll.
///
/// Args:
///     contentLines: Highlighted spans per line (from highlightLines).
///     lang: Optional language label for the top border.
///     width: Available width for the bordered block.
///     scrollX: Horizontal scroll offset in display columns.
///     maxContentWidth: Widest line in display columns (for scroll indicators).
///     showCopied: True to show "copied" flash, false to show "copy".
///     topExtra: Optional label between lang and copy on top border (e.g. "47 more \u{25B8}").
///     bottomLabel: Optional label centered in bottom border (e.g. "collapse \u{25B4}").
///
/// Returns:
///     Vec<Line<'static>>: Bordered lines with scrolled content.
/// Truncate a label string to fit within `maxWidth` display columns.
/// Appends an ellipsis if truncation occurs.
fn truncateLabelToWidth(s: &str, maxWidth: usize) -> String {
    let fullWidth = UnicodeWidthStr::width(s);
    if fullWidth <= maxWidth {
        return s.to_string();
    }
    let mut width = 0;
    let mut end = 0;
    for (g, cw) in graphemeWidths(s) {
        // Reserve 1 column for the ellipsis.
        if width + cw > maxWidth.saturating_sub(1) {
            if width == 0 {
                return s.to_string();
            }
            return format!("{}\u{2026}", &s[..end]);
        }
        width += cw;
        end += g.len();
    }
    s.to_string()
}

pub fn renderCodeBlock(
    contentLines: &[Vec<Span<'static>>],
    lang: Option<&str>,
    width: u16,
    scrollX: u16,
    maxContentWidth: usize,
    showCopied: bool,
    topExtra: Option<&str>,
    bottomLabel: Option<&str>,
) -> Vec<Line<'static>> {
    let w = width as usize;
    let innerWidth = w.saturating_sub(2);
    let borderStyle = Style::default().fg(Color::DarkGray);
    let scroll = scrollX as usize;
    let hasOverflow = maxContentWidth > innerWidth;
    let hasLeftOverflow = scroll > 0;
    let hasRightOverflow = maxContentWidth > scroll + innerWidth;

    let mut lines = Vec::with_capacity(contentLines.len() + 3);

    // Top border with language label, optional extra, and copy button.
    let rawLabel = lang.unwrap_or("");
    let copyLabel = if showCopied { "copied" } else { "copy" };
    let copyStyle = if showCopied {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let copyLen = UnicodeWidthStr::width(copyLabel);

    // Truncate label so copy button and toggle always fit.
    // Reserve: 1 space after label + copy + 1 space before copy + 3 min rule chars.
    let maxLabelWidth = innerWidth.saturating_sub(copyLen + 5);
    let label = truncateLabelToWidth(rawLabel, maxLabelWidth);
    // Use display width, not byte length — multi-byte chars like ▴/▾ are 1 column.
    let labelLen = UnicodeWidthStr::width(label.as_str());
    let extraStyle = Style::default().fg(Color::Gray);

    let topSpans = if let Some(extra) = topExtra {
        let extraLen = UnicodeWidthStr::width(extra);
        // Layout: ╭{lang} ─ {extra} ─ copy╮
        let fixedWidth = labelLen + 1 + 1 + extraLen + 1 + 1 + copyLen;
        if innerWidth > fixedWidth {
            let ruleTotal = innerWidth - fixedWidth;
            let leftRule = ruleTotal / 2;
            let rightRule = ruleTotal - leftRule;
            vec![
                Span::styled("\u{256D}", borderStyle),
                Span::styled(format!("{label} "), borderStyle),
                Span::styled("\u{2500}".repeat(leftRule), borderStyle),
                Span::styled(format!(" {extra} "), extraStyle),
                Span::styled("\u{2500}".repeat(rightRule), borderStyle),
                Span::styled(format!(" {copyLabel}"), copyStyle),
                Span::styled("\u{256E}", borderStyle),
            ]
        } else {
            // Too narrow — just show extra, drop copy label.
            let ruleLen = innerWidth.saturating_sub(labelLen + 1 + extraLen + 1);
            vec![
                Span::styled("\u{256D}", borderStyle),
                Span::styled(format!("{label} "), borderStyle),
                Span::styled("\u{2500}".repeat(ruleLen), borderStyle),
                Span::styled(format!(" {extra}"), extraStyle),
                Span::styled("\u{256E}", borderStyle),
            ]
        }
    } else if innerWidth >= labelLen + copyLen + 4 {
        // Normal layout: ╭{lang} ─── copy╮
        let ruleLen = innerWidth.saturating_sub(labelLen + 1 + copyLen + 1);
        vec![
            Span::styled("\u{256D}", borderStyle),
            Span::styled(format!("{label} "), borderStyle),
            Span::styled("\u{2500}".repeat(ruleLen), borderStyle),
            Span::styled(format!(" {copyLabel}"), copyStyle),
            Span::styled("\u{256E}", borderStyle),
        ]
    } else {
        // Too narrow — simple border.
        vec![
            Span::styled("\u{256D}", borderStyle),
            Span::styled("\u{2500}".repeat(innerWidth), borderStyle),
            Span::styled("\u{256E}", borderStyle),
        ]
    };
    lines.push(Line::from(topSpans));

    // Code lines with side borders, scrolled and truncated.
    // Show arrows on at most 3 lines (top, middle, bottom).
    let arrowStyle = Style::default().fg(Color::Gray);
    let lineCount = contentLines.len();
    let arrowRow = lineCount / 2;

    for (i, spans) in contentLines.iter().enumerate() {
        let scrolled = scrollAndTruncateSpans(spans, scroll, innerWidth);
        let showArrow = i == arrowRow;

        // Left border.
        let leftChar = if hasLeftOverflow && showArrow { "\u{25C2}" } else { "\u{2502}" };
        let leftStyle = if hasLeftOverflow && showArrow { arrowStyle } else { borderStyle };

        // Right border.
        let rightChar = if hasRightOverflow && showArrow { "\u{25B8}" } else { "\u{2502}" };
        let rightStyle = if hasRightOverflow && showArrow { arrowStyle } else { borderStyle };

        // Build the line as a single fixed-width string so Wrap cannot break it.
        // Collect styled segments, then pad to innerWidth and append the right border.
        let mut buf = String::with_capacity(innerWidth + 2);
        let mut bufWidth: usize = 0;
        buf.push_str(leftChar);

        // Flatten content spans into the buffer, tracking display width.
        struct StyledRange {
            start: usize,
            end: usize,
            style: Style,
        }
        let mut ranges: Vec<StyledRange> = Vec::new();

        // Left border range.
        let leftEnd = leftChar.len();
        ranges.push(StyledRange { start: 0, end: leftEnd, style: leftStyle });

        for span in &scrolled {
            let start = buf.len();
            for (g, cw) in graphemeWidths(&span.content) {
                // Expand tabs to spaces (terminals interpret raw \t as tab stops).
                if g == "\t" {
                    let tabW = 4usize.saturating_sub(bufWidth % 4);
                    for _ in 0..tabW {
                        if bufWidth >= innerWidth { break; }
                        buf.push(' ');
                        bufWidth += 1;
                    }
                    continue;
                }
                // Skip control-only clusters (e.g. \0, \r).
                if cw == 0 && g.chars().all(|c| c.is_control()) {
                    continue;
                }
                if bufWidth + cw > innerWidth {
                    break;
                }
                buf.push_str(g);
                bufWidth += cw;
            }
            let end = buf.len();
            if end > start {
                ranges.push(StyledRange { start, end, style: span.style });
            }
        }

        // Pad to innerWidth.
        let padStart = buf.len();
        while bufWidth < innerWidth {
            buf.push(' ');
            bufWidth += 1;
        }

        // Right border.
        buf.push_str(rightChar);

        // Padding + right border share the right style.
        if buf.len() > padStart {
            ranges.push(StyledRange { start: padStart, end: buf.len(), style: rightStyle });
        }

        // Convert to spans.
        let lineSpans: Vec<Span<'static>> = ranges
            .into_iter()
            .map(|r| Span::styled(buf[r.start..r.end].to_string(), r.style))
            .collect();

        lines.push(Line::from(lineSpans));
    }

    // Bottom border with optional label or scrollbar.
    if let Some(label) = bottomLabel {
        // Centered label: ╰─── label ───╯
        let lblLen = UnicodeWidthStr::width(label);
        let lblStyle = Style::default().fg(Color::Gray);
        // Inner content: leftRule + " " + label + " " + rightRule = innerWidth.
        let ruleSpace = innerWidth.saturating_sub(lblLen + 2);
        let leftRule = ruleSpace / 2;
        let rightRule = ruleSpace.saturating_sub(leftRule);
        let bottomSpans = vec![
            Span::styled("\u{2570}", borderStyle),
            Span::styled("\u{2500}".repeat(leftRule), borderStyle),
            Span::styled(format!(" {label} "), lblStyle),
            Span::styled("\u{2500}".repeat(rightRule), borderStyle),
            Span::styled("\u{256F}", borderStyle),
        ];
        lines.push(Line::from(bottomSpans));
    } else if hasOverflow {
        let trackLen = innerWidth;
        let scrollMax = maxContentWidth.saturating_sub(innerWidth);
        let thumbLen = (trackLen * trackLen / maxContentWidth).max(1);
        let thumbPos = if scrollMax > 0 {
            (scroll * (trackLen.saturating_sub(thumbLen))) / scrollMax
        } else {
            0
        };
        let thumbEnd = (thumbPos + thumbLen).min(trackLen);

        let mut bottomSpans: Vec<Span<'static>> = vec![
            Span::styled("\u{2570}", borderStyle),
        ];
        // Pre-thumb track.
        let preTrack: String = (0..thumbPos).map(|_| '\u{2500}').collect();
        let postTrack: String = (thumbEnd..trackLen).map(|_| '\u{2500}').collect();
        let thumbStr: String = (thumbPos..thumbEnd).map(|_| '\u{2501}').collect();

        if !preTrack.is_empty() {
            bottomSpans.push(Span::styled(preTrack, borderStyle));
        }
        if !thumbStr.is_empty() {
            bottomSpans.push(Span::styled(thumbStr, Style::default().fg(Color::Gray)));
        }
        if !postTrack.is_empty() {
            bottomSpans.push(Span::styled(postTrack, borderStyle));
        }
        bottomSpans.push(Span::styled("\u{256F}", borderStyle));
        lines.push(Line::from(bottomSpans));
    } else {
        let bottomRule = format!(
            "\u{2570}{}\u{256F}",
            "\u{2500}".repeat(innerWidth)
        );
        lines.push(Line::from(Span::styled(bottomRule, borderStyle)));
    }

    lines
}

// Muted background tints for diff lines (dark theme).
const DIFF_ADD_BG: Color = Color::Rgb(33, 58, 43);
const DIFF_DEL_BG: Color = Color::Rgb(74, 34, 29);

/// Diff line classification.
#[derive(Clone, Copy, PartialEq)]
enum DiffLineKind {
    Insert,
    Delete,
    Context,
}

/// A parsed hunk from a unified diff.
struct DiffHunk {
    oldStart: usize,
    newStart: usize,
    lines: Vec<(DiffLineKind, String)>,
}

/// Parse a unified diff string into file path and hunks.
fn parseDiff(code: &str) -> (Option<String>, Vec<DiffHunk>) {
    let mut filePath: Option<String> = None;
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut currentHunk: Option<DiffHunk> = None;

    for line in code.lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            filePath = Some(path.to_string());
        } else if line.starts_with("+++ ") || line.starts_with("--- ") {
            // Skip diff headers (including "+++ /dev/null").
        } else if line.starts_with("@@ ") {
            // Flush previous hunk.
            if let Some(h) = currentHunk.take() {
                hunks.push(h);
            }
            // Parse "@@ -old_start,count +new_start,count @@".
            let (oldStart, newStart) = parseHunkHeader(line);
            currentHunk = Some(DiffHunk {
                oldStart,
                newStart,
                lines: Vec::new(),
            });
        } else if let Some(ref mut hunk) = currentHunk {
            if let Some(content) = line.strip_prefix('+') {
                hunk.lines.push((DiffLineKind::Insert, content.to_string()));
            } else if let Some(content) = line.strip_prefix('-') {
                hunk.lines.push((DiffLineKind::Delete, content.to_string()));
            } else if let Some(content) = line.strip_prefix(' ') {
                hunk.lines.push((DiffLineKind::Context, content.to_string()));
            } else if line.starts_with('\\') {
                // Skip "\ No newline at end of file" markers.
            } else {
                hunk.lines.push((DiffLineKind::Context, line.to_string()));
            }
        }
    }
    if let Some(h) = currentHunk {
        hunks.push(h);
    }
    (filePath, hunks)
}

/// Extract old_start and new_start from a `@@ -A,B +C,D @@` header.
fn parseHunkHeader(header: &str) -> (usize, usize) {
    // Strip "@@ " prefix and " @@..." suffix.
    let inner = header
        .strip_prefix("@@ ")
        .and_then(|s| s.split(" @@").next())
        .unwrap_or("");
    let mut oldStart = 1usize;
    let mut newStart = 1usize;
    for part in inner.split_whitespace() {
        if let Some(range) = part.strip_prefix('-') {
            oldStart = range
                .split(',')
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(1);
        } else if let Some(range) = part.strip_prefix('+') {
            newStart = range
                .split(',')
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(1);
        }
    }
    (oldStart, newStart)
}

/// Detect language token from a file path for syntax highlighting.
///
/// Returns an owned string because the extension is borrowed from the input path.
fn detectLang(path: &str) -> Option<String> {
    let ext = std::path::Path::new(path).extension()?.to_str()?;
    // Verify syntect recognizes this extension.
    syntaxSet().find_syntax_by_extension(ext)?;
    Some(ext.to_string())
}

/// Render a unified diff with line numbers, background tints, and syntax highlighting.
///
/// Returns spans-per-line (no borders) for use in `RenderedBlock::Code`.
pub fn diffLines(code: &str) -> Vec<Vec<Span<'static>>> {
    let (filePath, hunks) = parseDiff(code);

    if hunks.is_empty() {
        // Fallback: render as plain text.
        return code
            .lines()
            .map(|l| vec![Span::styled(l.to_string(), Style::default().fg(Color::DarkGray))])
            .collect();
    }

    let langOwned = filePath.as_deref().and_then(detectLang);
    let lang = langOwned.as_deref();

    // Find the maximum line number across all hunks for gutter width.
    let maxLineNum = {
        let mut max = 0usize;
        for hunk in &hunks {
            let mut oldLn = hunk.oldStart;
            let mut newLn = hunk.newStart;
            for (kind, _) in &hunk.lines {
                match kind {
                    DiffLineKind::Insert => { max = max.max(newLn); newLn += 1; }
                    DiffLineKind::Delete => { max = max.max(oldLn); oldLn += 1; }
                    DiffLineKind::Context => {
                        max = max.max(oldLn).max(newLn);
                        oldLn += 1;
                        newLn += 1;
                    }
                }
            }
            max = max.max(oldLn).max(newLn);
        }
        max
    };
    let gutterWidth = if maxLineNum == 0 { 1 } else { maxLineNum.to_string().len() };

    let gutterStyle = Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM);
    let spacerStyle = Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM);

    let mut output: Vec<Vec<Span<'static>>> = Vec::new();

    for (hunkIdx, hunk) in hunks.iter().enumerate() {
        // Hunk separator.
        if hunkIdx > 0 {
            let spacer = format!("{:>width$} ", "", width = gutterWidth);
            output.push(vec![
                Span::styled(spacer, spacerStyle),
                Span::styled("\u{22EE}", spacerStyle),
            ]);
        }

        // Syntax-highlight the hunk content as a single block.
        let hunkCode: String = hunk.lines
            .iter()
            .map(|(_, content)| format!("{content}\n"))
            .collect();
        let hunkCodeLines: Vec<&str> = hunkCode.lines().collect();
        let syntaxSpans = syntectLines(&hunkCodeLines, lang);

        let mut oldLn = hunk.oldStart;
        let mut newLn = hunk.newStart;

        for (lineIdx, (kind, content)) in hunk.lines.iter().enumerate() {
            let lineNum = match kind {
                DiffLineKind::Insert => newLn,
                DiffLineKind::Delete => oldLn,
                DiffLineKind::Context => newLn,
            };

            // Gutter span.
            let gutterText = format!("{lineNum:>gutterWidth$} ");
            let lineGutterStyle = match kind {
                DiffLineKind::Insert => gutterStyle.bg(DIFF_ADD_BG),
                DiffLineKind::Delete => gutterStyle.bg(DIFF_DEL_BG),
                DiffLineKind::Context => gutterStyle,
            };

            // Sign span.
            let (signChar, signStyle) = match kind {
                DiffLineKind::Insert => (
                    "+",
                    Style::default().fg(Color::Green).bg(DIFF_ADD_BG),
                ),
                DiffLineKind::Delete => (
                    "-",
                    Style::default().fg(Color::Red).bg(DIFF_DEL_BG),
                ),
                DiffLineKind::Context => (
                    " ",
                    Style::default(),
                ),
            };

            // Content spans — syntax-highlighted with diff background overlay.
            let contentSpans: Vec<Span<'static>> = if let Some(synLine) = syntaxSpans.get(lineIdx) {
                synLine
                    .iter()
                    .map(|sp| {
                        let style = match kind {
                            DiffLineKind::Insert => sp.style.bg(DIFF_ADD_BG),
                            DiffLineKind::Delete => sp.style
                                .bg(DIFF_DEL_BG)
                                .add_modifier(Modifier::DIM),
                            DiffLineKind::Context => sp.style,
                        };
                        Span::styled(sp.content.to_string(), style)
                    })
                    .collect()
            } else {
                let style = match kind {
                    DiffLineKind::Insert => Style::default().fg(Color::Green).bg(DIFF_ADD_BG),
                    DiffLineKind::Delete => Style::default().fg(Color::Red).bg(DIFF_DEL_BG)
                        .add_modifier(Modifier::DIM),
                    DiffLineKind::Context => Style::default().fg(Color::DarkGray),
                };
                vec![Span::styled(content.to_string(), style)]
            };

            let mut spans = Vec::with_capacity(contentSpans.len() + 2);
            spans.push(Span::styled(gutterText, lineGutterStyle));
            spans.push(Span::styled(signChar.to_string(), signStyle));
            spans.extend(contentSpans);

            output.push(spans);

            // Advance line counters.
            match kind {
                DiffLineKind::Insert => newLn += 1,
                DiffLineKind::Delete => oldLn += 1,
                DiffLineKind::Context => { oldLn += 1; newLn += 1; }
            }
        }
    }

    output
}

/// Compute the maximum content width across all lines (in display columns).
pub fn maxContentWidth(contentLines: &[Vec<Span<'static>>]) -> usize {
    contentLines
        .iter()
        .map(|spans| spans.iter().map(|s| s.width()).sum())
        .max()
        .unwrap_or(0)
}

/// Highlight code lines using syntect.
fn syntectLines(codeLines: &[&str], lang: Option<&str>) -> Vec<Vec<Span<'static>>> {
    let ss = syntaxSet();
    let syntax = lang
        .and_then(|l| ss.find_syntax_by_token(l))
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut h = HighlightLines::new(syntax, theme());

    codeLines
        .iter()
        .map(|line| {
            match h.highlight_line(line, ss) {
                Ok(ranges) => ranges
                    .into_iter()
                    .map(|(style, text)| Span::styled(text.to_string(), syntectToRatatui(style)))
                    .collect(),
                Err(_) => vec![Span::raw(line.to_string())],
            }
        })
        .collect()
}

/// Skip `skipCols` display columns, then take up to `maxCols` display columns.
fn scrollAndTruncateSpans(
    spans: &[Span<'static>],
    skipCols: usize,
    maxCols: usize,
) -> Vec<Span<'static>> {
    let mut result = Vec::new();
    let mut skipped = 0;
    let mut taken = 0;

    for span in spans {
        let mut text = String::new();
        for (g, w) in graphemeWidths(&span.content) {
            // Still skipping.
            if skipped < skipCols {
                skipped += w;
                continue;
            }

            // Taking.
            if taken + w > maxCols {
                break;
            }
            text.push_str(g);
            taken += w;
        }
        if !text.is_empty() {
            result.push(Span::styled(text, span.style));
        }
        if taken >= maxCols {
            break;
        }
    }

    result
}

/// Convert a syntect Style to a ratatui Style.
fn syntectToRatatui(s: SyntectStyle) -> Style {
    let mut style = Style::default().fg(Color::Rgb(
        s.foreground.r,
        s.foreground.g,
        s.foreground.b,
    ));
    if s.font_style.contains(FontStyle::BOLD) {
        style = style.add_modifier(Modifier::BOLD);
    }
    if s.font_style.contains(FontStyle::ITALIC) {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if s.font_style.contains(FontStyle::UNDERLINE) {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineText(l: &Line<'_>) -> String {
        l.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn emojiSequenceWidthMatchesRender() {
        let code = "\u{2514}\u{2500} \u{26A0}\u{FE0F} Cannot automatically restore";
        let lines = highlightLines(code, None);
        let mcw = maxContentWidth(&lines);
        assert_eq!(mcw, UnicodeWidthStr::width(code));

        for width in [20u16, 30u16, 40u16, 50u16, 60u16] {
            let rendered = renderCodeBlock(&lines, None, width, 0, mcw, false, None, None);
            for (i, ln) in rendered.iter().enumerate() {
                let text = lineText(ln);
                let w = UnicodeWidthStr::width(text.as_str());
                assert_eq!(
                    w, width as usize,
                    "width={} line{}: rendered width {} mismatch for text={:?}",
                    width, i, w, text
                );
            }
        }
    }

    #[test]
    fn emojiInScrolledContent() {
        let code = "\u{251C}\u{2500}\u{25CB} ! bash \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500} 3 files \u{26A0}\u{FE0F}    2:45 PM";
        let lines = highlightLines(code, None);
        let mcw = maxContentWidth(&lines);
        for width in [30u16, 40u16, 50u16, 60u16, 70u16] {
            for scroll in [0u16, 5u16, 20u16] {
                let rendered = renderCodeBlock(&lines, None, width, scroll, mcw, false, None, None);
                for (i, ln) in rendered.iter().enumerate() {
                    let text = lineText(ln);
                    let w = UnicodeWidthStr::width(text.as_str());
                    assert_eq!(
                        w, width as usize,
                        "w={} scroll={} line{}: rendered width {} mismatch for text={:?}",
                        width, scroll, i, w, text
                    );
                }
            }
        }
    }
}
