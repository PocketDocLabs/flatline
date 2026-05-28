//! Box-drawing table renderer for markdown tables.
//!
//! Measures column widths, distributes available space, and renders
//! tables using Unicode box-drawing characters. Cells wrap instead
//! of truncating. When columns would shrink below a readable threshold,
//! switches to a vertical record layout (label │ value per field).
//!
//! # Public API
//! - [`renderTable`] — render a table as box-drawn Lines
//!
//! # Dependencies
//! `ratatui`

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::block::{Alignment, StyledSegment};
use crate::text_area::unicode_display_width;

/// Minimum column width in characters (including padding).
const MIN_COL_WIDTH: usize = 3;

/// Default minimum inner width before switching to vertical layout.
const VERTICAL_THRESHOLD: usize = 6;

/// Render a markdown table as box-drawn Lines.
///
/// Uses a grid layout with cell wrapping when columns fit. Falls back
/// to a vertical record layout when any column would be too narrow.
///
/// Args:
///     alignments: Column alignment specifications.
///     header: Header row cell contents.
///     rows: Data row cell contents.
///     width: Available rendering width.
///
/// Returns:
///     Vec<Line<'static>>: Rendered table lines.
pub fn renderTable(
    alignments: &[Alignment],
    header: &[Vec<StyledSegment>],
    rows: &[Vec<Vec<StyledSegment>>],
    width: u16,
) -> Vec<Line<'static>> {
    let w = width as usize;
    let colCount = header
        .len()
        .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));

    if colCount == 0 {
        return vec![];
    }

    // Measure natural column widths.
    let mut colWidths: Vec<usize> = vec![0; colCount];
    for (i, cell) in header.iter().enumerate() {
        colWidths[i] = colWidths[i].max(cellWidth(cell));
    }
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < colCount {
                colWidths[i] = colWidths[i].max(cellWidth(cell));
            }
        }
    }

    // Add 1-char padding per side, ensure minimum widths.
    for cw in &mut colWidths {
        *cw = (*cw + 2).max(MIN_COL_WIDTH);
    }

    // Distribute widths to fit available space.
    // Account for borders: 1 left + 1 right + (colCount - 1) separators.
    let overhead = 1 + 1 + colCount.saturating_sub(1);
    let availableContent = w.saturating_sub(overhead);
    let totalNatural: usize = colWidths.iter().sum();

    // Snapshot natural widths before shrinking.
    let naturalWidths = colWidths.clone();

    if totalNatural > availableContent && availableContent > 0 {
        // Shrink widest columns first, protecting narrow ones.
        // Iteratively brings the widest tier down to the next tier until
        // enough space is recovered, distributing any remainder evenly.
        let mut excess = totalNatural - availableContent;
        while excess > 0 {
            let maxW = match colWidths.iter().max() {
                Some(&w) if w > MIN_COL_WIDTH => w,
                _ => break,
            };
            // Next width tier below the current widest.
            let nextW = colWidths
                .iter()
                .filter(|&&w| w < maxW)
                .max()
                .copied()
                .unwrap_or(MIN_COL_WIDTH);
            // Indices of columns at the widest tier.
            let atMax: Vec<usize> = colWidths
                .iter()
                .enumerate()
                .filter(|(_, w)| **w == maxW)
                .map(|(i, _)| i)
                .collect();
            let shrinkPerCol = maxW - nextW;
            let totalShrinkable = shrinkPerCol * atMax.len();

            if totalShrinkable <= excess {
                // Bring all widest columns down to the next tier.
                for &i in &atMax {
                    colWidths[i] = nextW;
                }
                excess -= totalShrinkable;
            } else {
                // Distribute remaining excess evenly among widest columns.
                let perCol = excess / atMax.len();
                let mut remainder = excess % atMax.len();
                for &i in &atMax {
                    let shrink = perCol
                        + if remainder > 0 {
                            remainder -= 1;
                            1
                        } else {
                            0
                        };
                    colWidths[i] -= shrink;
                }
                excess = 0;
            }
        }
    }

    // Only switch to vertical when shrinking forced a column below threshold.
    let longest = longestWordWidth(header, rows);
    let minUsable = VERTICAL_THRESHOLD.max(longest);
    let tooNarrow = colWidths
        .iter()
        .zip(naturalWidths.iter())
        .any(|(&actual, &natural)| actual < natural && actual.saturating_sub(2) < minUsable);

    let borderStyle = Style::default().fg(Color::DarkGray);

    if tooNarrow {
        return renderVerticalRecord(header, rows, w, borderStyle);
    }

    let mut lines = Vec::new();

    // Top border.
    lines.push(horizontalRule(
        &colWidths,
        "\u{250C}",
        "\u{252C}",
        "\u{2510}",
        borderStyle,
    ));

    // Header row (multi-line with wrapping).
    if !header.is_empty() {
        let headerStyle = Style::default().add_modifier(Modifier::BOLD);
        lines.extend(dataRows(
            header,
            &colWidths,
            alignments,
            borderStyle,
            Some(headerStyle),
        ));
    }

    // Header separator.
    lines.push(horizontalRule(
        &colWidths,
        "\u{251C}",
        "\u{253C}",
        "\u{2524}",
        borderStyle,
    ));

    // Data rows with separators between them.
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            lines.push(horizontalRule(
                &colWidths,
                "\u{251C}",
                "\u{253C}",
                "\u{2524}",
                borderStyle,
            ));
        }
        lines.extend(dataRows(row, &colWidths, alignments, borderStyle, None));
    }

    // Bottom border.
    lines.push(horizontalRule(
        &colWidths,
        "\u{2514}",
        "\u{2534}",
        "\u{2518}",
        borderStyle,
    ));

    lines
}

// ── Width measurement ─────────────────────────────────────────────

/// Measure the display width of a table cell's content.
fn cellWidth(cell: &[StyledSegment]) -> usize {
    cell.iter().map(|seg| strDisplayWidth(&seg.text)).sum()
}

/// Display width of a string, accounting for multi-byte and wide characters.
fn strDisplayWidth(s: &str) -> usize {
    s.chars().map(unicode_display_width).sum()
}

/// Find the longest whitespace-delimited word across all cells.
///
/// Returns 0 if no words exist.
fn longestWordWidth(header: &[Vec<StyledSegment>], rows: &[Vec<Vec<StyledSegment>>]) -> usize {
    let allCells = header.iter().chain(rows.iter().flat_map(|r| r.iter()));
    let mut longest: usize = 0;
    for cell in allCells {
        for seg in cell {
            for word in seg.text.split_whitespace() {
                longest = longest.max(strDisplayWidth(word));
            }
        }
    }
    longest
}

// ── Segment wrapping ──────────────────────────────────────────────

/// Merge adjacent same-style (char, Style) pairs into StyledSegments.
fn styledCharsToSegments(chars: &[(char, Style)]) -> Vec<StyledSegment> {
    if chars.is_empty() {
        return vec![];
    }
    let mut segments = Vec::new();
    let mut currentText = String::new();
    let mut currentStyle = chars[0].1;

    for &(ch, style) in chars {
        if style == currentStyle {
            currentText.push(ch);
        } else {
            if !currentText.is_empty() {
                segments.push(StyledSegment {
                    text: std::mem::take(&mut currentText),
                    style: currentStyle,
                });
            }
            currentStyle = style;
            currentText.push(ch);
        }
    }
    if !currentText.is_empty() {
        segments.push(StyledSegment {
            text: currentText,
            style: currentStyle,
        });
    }
    segments
}

/// Word-wrap styled segments to fit within a maximum display width.
///
/// Breaks at whitespace boundaries when possible, falls back to
/// character-level breaks when a word exceeds the line width.
///
/// Args:
///     segments: Styled text to wrap.
///     maxWidth: Maximum display width per line.
///
/// Returns:
///     Vec<Vec<StyledSegment>>: Wrapped lines of segments.
fn wrapSegments(segments: &[StyledSegment], maxWidth: usize) -> Vec<Vec<StyledSegment>> {
    if maxWidth == 0 {
        return vec![segments.to_vec()];
    }

    // Flatten to (char, Style) pairs.
    let chars: Vec<(char, Style)> = segments
        .iter()
        .flat_map(|seg| seg.text.chars().map(move |ch| (ch, seg.style)))
        .collect();

    if chars.is_empty() {
        return vec![vec![]];
    }

    // Check if it fits on one line.
    let totalWidth: usize = chars.iter().map(|(ch, _)| unicode_display_width(*ch)).sum();
    if totalWidth <= maxWidth {
        return vec![segments.to_vec()];
    }

    let mut result: Vec<Vec<StyledSegment>> = Vec::new();
    let mut lineStart: usize = 0;
    let mut currentWidth: usize = 0;
    let mut lastSpace: Option<usize> = None;

    for (i, &(ch, _)) in chars.iter().enumerate() {
        let charWidth = unicode_display_width(ch);

        if ch == ' ' {
            lastSpace = Some(i);
        }

        if currentWidth + charWidth > maxWidth && i > lineStart {
            // Break at last space if available, otherwise at current position.
            let breakAt = if let Some(sp) = lastSpace {
                if sp > lineStart { sp } else { i }
            } else {
                i
            };

            result.push(styledCharsToSegments(&chars[lineStart..breakAt]));

            // Skip the space at the break point if we broke at a space.
            if breakAt < chars.len() && chars[breakAt].0 == ' ' {
                lineStart = breakAt + 1;
            } else {
                lineStart = breakAt;
            }

            // Recompute width from new line start to current position.
            currentWidth = 0;
            for j in lineStart..=i {
                if j < chars.len() {
                    currentWidth += unicode_display_width(chars[j].0);
                }
            }
            lastSpace = None;
        } else {
            currentWidth += charWidth;
        }
    }

    // Remaining characters.
    if lineStart < chars.len() {
        result.push(styledCharsToSegments(&chars[lineStart..]));
    }

    if result.is_empty() {
        result.push(vec![]);
    }

    result
}

// ── Grid layout rendering ─────────────────────────────────────────

/// Render a horizontal rule with box-drawing characters.
fn horizontalRule(
    colWidths: &[usize],
    left: &str,
    mid: &str,
    right: &str,
    style: Style,
) -> Line<'static> {
    let mut s = left.to_string();
    for (i, &w) in colWidths.iter().enumerate() {
        s.push_str(&"\u{2500}".repeat(w));
        if i < colWidths.len() - 1 {
            s.push_str(mid);
        }
    }
    s.push_str(right);
    Line::from(Span::styled(s, style))
}

/// Render a data row with cell wrapping, producing multiple visual lines.
///
/// Each cell's content is wrapped to fit the column width. The row
/// height equals the tallest cell. Shorter cells are padded with blanks.
fn dataRows(
    cells: &[Vec<StyledSegment>],
    colWidths: &[usize],
    alignments: &[Alignment],
    borderStyle: Style,
    extraStyle: Option<Style>,
) -> Vec<Line<'static>> {
    // Wrap each cell's content to its column's inner width.
    let wrappedCells: Vec<Vec<Vec<StyledSegment>>> = colWidths
        .iter()
        .enumerate()
        .map(|(i, &cw)| {
            let innerWidth = cw.saturating_sub(2);
            if let Some(segs) = cells.get(i) {
                wrapSegments(segs, innerWidth)
            } else {
                vec![vec![]]
            }
        })
        .collect();

    // Row height = tallest cell.
    let rowHeight = wrappedCells.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut lines = Vec::new();
    for lineIdx in 0..rowHeight {
        let mut spans: Vec<Span<'static>> = vec![Span::styled("\u{2502}", borderStyle)];

        for (colIdx, colWidth) in colWidths.iter().enumerate() {
            let alignment = alignments.get(colIdx).copied().unwrap_or(Alignment::None);

            let cellLine = wrappedCells.get(colIdx).and_then(|wc| wc.get(lineIdx));

            let cellSpans = if let Some(segs) = cellLine {
                renderCellContent(segs, *colWidth, alignment, extraStyle)
            } else {
                // Blank padding for shorter cells.
                vec![Span::raw(" ".repeat(*colWidth))]
            };

            spans.extend(cellSpans);
            spans.push(Span::styled("\u{2502}", borderStyle));
        }

        lines.push(Line::from(spans));
    }

    lines
}

/// Render a cell's content, padded and aligned to the column width.
///
/// Content is guaranteed to fit within the column (wrapping happens
/// upstream). This function only handles padding and alignment.
///
/// Column width includes 1-char padding on each side.
fn renderCellContent(
    segments: &[StyledSegment],
    colWidth: usize,
    alignment: Alignment,
    extraStyle: Option<Style>,
) -> Vec<Span<'static>> {
    let innerWidth = colWidth.saturating_sub(2);
    let contentWidth: usize = segments.iter().map(|seg| strDisplayWidth(&seg.text)).sum();

    let slack = innerWidth.saturating_sub(contentWidth);
    let (leftPad, rightPad) = match alignment {
        Alignment::Right => (slack, 0),
        Alignment::Center => (slack / 2, slack - slack / 2),
        Alignment::Left | Alignment::None => (0, slack),
    };

    let mut spans = Vec::new();
    // Leading pad: 1 fixed + alignment padding.
    spans.push(Span::raw(" ".repeat(1 + leftPad)));
    for seg in segments {
        let style = if let Some(extra) = extraStyle {
            seg.style.patch(extra)
        } else {
            seg.style
        };
        spans.push(Span::styled(seg.text.clone(), style));
    }
    // Trailing pad: alignment padding + 1 fixed.
    spans.push(Span::raw(" ".repeat(rightPad + 1)));
    spans
}

// ── Vertical record layout ───────────────────────────────────────

/// Render a horizontal rule with an embedded label.
///
/// Produces: `── {label} ──────...` filling to the given width.
fn horizontalRuleWithLabel(width: usize, label: &str, style: Style) -> Line<'static> {
    let labelDisplay = strDisplayWidth(label);
    // Format: "── label ──..."
    let prefixLen = 2; // "── " before label, but we use "─ " (2 chars).
    let suffixStart = prefixLen + 1 + labelDisplay + 1; // "─ " + label + " " + rest.
    let remaining = width.saturating_sub(suffixStart);

    let mut s = String::with_capacity(width);
    s.push_str("\u{2500} ");
    s.push_str(label);
    s.push(' ');
    s.push_str(&"\u{2500}".repeat(remaining));

    Line::from(Span::styled(s, style))
}

/// Render a table as vertical records when columns are too narrow for a grid.
///
/// Each data row becomes a labeled block of field: value pairs.
/// No side borders — just horizontal rules and a `│` separator.
fn renderVerticalRecord(
    header: &[Vec<StyledSegment>],
    rows: &[Vec<Vec<StyledSegment>>],
    width: usize,
    borderStyle: Style,
) -> Vec<Line<'static>> {
    let colCount = header.len();
    if colCount == 0 {
        return vec![];
    }

    // Compute label column width: max header width, capped at width/3.
    let maxLabelWidth = header.iter().map(|h| cellWidth(h)).max().unwrap_or(0);
    let labelColWidth = (maxLabelWidth + 1).min(width / 3).max(1);

    // Separator takes 3 chars: " │ ".
    let separatorWidth = 3;
    let valueColWidth = width.saturating_sub(labelColWidth + separatorWidth);

    if valueColWidth == 0 {
        return vec![];
    }

    let labelStyle = Style::default().add_modifier(Modifier::BOLD);
    let mut lines = Vec::new();

    for (rowIdx, row) in rows.iter().enumerate() {
        // Row header rule.
        let label = format!("{}", rowIdx + 1);
        lines.push(horizontalRuleWithLabel(width, &label, borderStyle));

        for colIdx in 0..colCount {
            // Render label (right-aligned, truncated if needed).
            let headerText = headerPlainText(header.get(colIdx));
            let labelText = rightAlignTruncate(&headerText, labelColWidth);

            // Wrap value content.
            let valueSegments = row.get(colIdx).map(|s| s.as_slice()).unwrap_or(&[]);
            let wrappedValue = wrapSegments(valueSegments, valueColWidth);

            for (lineIdx, valueLine) in wrappedValue.iter().enumerate() {
                let mut spans: Vec<Span<'static>> = Vec::new();

                if lineIdx == 0 {
                    // First line: label │ value.
                    spans.push(Span::styled(labelText.clone(), labelStyle));
                } else {
                    // Continuation: blank │ value.
                    spans.push(Span::raw(" ".repeat(labelColWidth)));
                }

                spans.push(Span::styled(" \u{2502} ", borderStyle));

                for seg in valueLine {
                    spans.push(Span::styled(seg.text.clone(), seg.style));
                }

                lines.push(Line::from(spans));
            }
        }
    }

    lines
}

/// Extract plain text from a header cell's styled segments.
fn headerPlainText(cell: Option<&Vec<StyledSegment>>) -> String {
    match cell {
        Some(segs) => segs.iter().map(|s| s.text.as_str()).collect(),
        None => String::new(),
    }
}

/// Right-align a string within a given width, truncating from the left if needed.
fn rightAlignTruncate(text: &str, width: usize) -> String {
    let textWidth = strDisplayWidth(text);
    if textWidth <= width {
        let padding = width - textWidth;
        format!("{}{}", " ".repeat(padding), text)
    } else {
        // Truncate from the right, keeping what fits.
        let mut result = String::new();
        let mut remaining = width;
        for ch in text.chars() {
            let w = unicode_display_width(ch);
            if w > remaining {
                break;
            }
            result.push(ch);
            remaining -= w;
        }
        result
    }
}
