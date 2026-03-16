//! Markdown block types and per-block rendering.
//!
//! Each `MdBlock` variant holds the semantic content of a parsed
//! markdown element and renders to `Vec<Line<'static>>`.
//!
//! # Public API
//! - [`MdBlock`] — enum of all supported block types
//! - [`StyledSegment`] — inline text with ratatui styling
//!
//! # Dependencies
//! `ratatui`

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::highlight;
use super::table;
use super::RenderedBlock;

/// Column alignment for table cells.
#[derive(Debug, Clone, Copy)]
pub enum Alignment {
    None,
    Left,
    Center,
    Right,
}

/// Inline text segment with styling applied.
#[derive(Debug, Clone)]
pub struct StyledSegment {
    pub text: String,
    pub style: Style,
}

/// A parsed markdown block ready for rendering.
#[derive(Debug, Clone)]
pub enum MdBlock {
    Paragraph {
        spans: Vec<StyledSegment>,
    },
    Heading {
        level: u8,
        spans: Vec<StyledSegment>,
    },
    CodeBlock {
        lang: Option<String>,
        code: String,
    },
    Table {
        alignments: Vec<Alignment>,
        header: Vec<Vec<StyledSegment>>,
        rows: Vec<Vec<Vec<StyledSegment>>>,
    },
    List {
        ordered: bool,
        startIndex: Option<u64>,
        items: Vec<Vec<MdBlock>>,
    },
    Blockquote {
        blocks: Vec<MdBlock>,
    },
    Rule,
}

impl MdBlock {
    /// Render this block into rendered blocks.
    ///
    /// Code blocks produce `RenderedBlock::Code` with untruncated
    /// highlighted content. Everything else produces `RenderedBlock::Text`.
    ///
    /// Args:
    ///     width: Available rendering width in columns.
    ///
    /// Returns:
    ///     Vec<RenderedBlock>: One or more rendered blocks.
    pub fn renderBlocks(&self, width: u16) -> Vec<RenderedBlock> {
        match self {
            MdBlock::CodeBlock { lang, code } => {
                let isDiff = lang.as_deref().is_some_and(|l| l == "diff" || l == "patch");
                let isMermaid = lang.as_deref().is_some_and(|l| l == "mermaid");
                if isDiff {
                    // Diff blocks render with line numbers, backgrounds, and syntax highlighting.
                    vec![RenderedBlock::Code {
                        lang: Some("diff".to_string()),
                        lines: highlight::diffLines(code),
                        code: code.clone(),
                    }]
                } else if isMermaid {
                    // Inner width = total width minus 2 border columns.
                    let innerWidth = width.saturating_sub(2) as usize;
                    if let Some(diagramLines) = super::mermaid::tryRenderMermaid(code, innerWidth) {
                        vec![RenderedBlock::Code {
                            lang: Some("mermaid".to_string()),
                            lines: diagramLines,
                            code: code.clone(),
                        }]
                    } else {
                        // Parse failed — fall back to plain code block.
                        vec![RenderedBlock::Code {
                            lang: lang.clone(),
                            lines: highlight::highlightLines(code, lang.as_deref()),
                            code: code.clone(),
                        }]
                    }
                } else {
                    vec![RenderedBlock::Code {
                        lang: lang.clone(),
                        lines: highlight::highlightLines(code, lang.as_deref()),
                        code: code.clone(),
                    }]
                }
            }
            other => vec![RenderedBlock::Text(other.renderLines(width))],
        }
    }

    /// Render this block into styled ratatui Lines.
    fn renderLines(&self, width: u16) -> Vec<Line<'static>> {
        match self {
            MdBlock::Paragraph { spans } => renderParagraph(spans),
            MdBlock::Heading { level, spans } => renderHeading(*level, spans),
            MdBlock::CodeBlock { lang, code } => {
                // Fallback — code blocks normally go through renderBlocks.
                highlight::highlightLines(code, lang.as_deref())
                    .into_iter()
                    .map(Line::from)
                    .collect()
            }
            MdBlock::Table {
                alignments,
                header,
                rows,
            } => table::renderTable(alignments, header, rows, width),
            MdBlock::List {
                ordered,
                startIndex,
                items,
            } => renderList(*ordered, *startIndex, items, width),
            MdBlock::Blockquote { blocks } => renderBlockquote(blocks, width),
            MdBlock::Rule => renderRule(width),
        }
    }
}

/// Flatten styled segments into a single Line.
fn renderParagraph(spans: &[StyledSegment]) -> Vec<Line<'static>> {
    if spans.is_empty() {
        return vec![];
    }

    // Split on hard breaks (newlines embedded in text).
    let mut lines = Vec::new();
    let mut currentSpans: Vec<Span<'static>> = Vec::new();

    for seg in spans {
        if seg.text.contains('\n') {
            let parts: Vec<&str> = seg.text.split('\n').collect();
            for (i, part) in parts.iter().enumerate() {
                if !part.is_empty() {
                    currentSpans.push(Span::styled(part.to_string(), seg.style));
                }
                if i < parts.len() - 1 {
                    lines.push(Line::from(std::mem::take(&mut currentSpans)));
                }
            }
        } else {
            currentSpans.push(Span::styled(seg.text.clone(), seg.style));
        }
    }

    if !currentSpans.is_empty() {
        lines.push(Line::from(currentSpans));
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }

    lines
}

/// Render a heading with level-appropriate styling.
fn renderHeading(level: u8, spans: &[StyledSegment]) -> Vec<Line<'static>> {
    let style = match level {
        1 => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        2 => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        3 => Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
        _ => Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD | Modifier::DIM),
    };

    let text: String = spans.iter().map(|s| s.text.as_str()).collect();
    vec![Line::from(Span::styled(text, style))]
}

/// Render a list with marker prefixes and nested indentation.
fn renderList(
    ordered: bool,
    startIndex: Option<u64>,
    items: &[Vec<MdBlock>],
    width: u16,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut index = startIndex.unwrap_or(1);

    for item in items {
        let marker = if ordered {
            format!("{index}. ")
        } else {
            "- ".to_string()
        };
        let indent = " ".repeat(marker.len());
        let innerWidth = width.saturating_sub(marker.len() as u16);

        let mut itemLines = Vec::new();
        for block in item {
            itemLines.extend(block.renderLines(innerWidth));
        }

        for (i, line) in itemLines.into_iter().enumerate() {
            let prefix = if i == 0 {
                Span::raw(marker.clone())
            } else {
                Span::raw(indent.clone())
            };
            let mut spans = vec![prefix];
            spans.extend(line.spans);
            lines.push(Line::from(spans));
        }

        index += 1;
    }

    lines
}

/// Render a blockquote with vertical bar prefix.
fn renderBlockquote(blocks: &[MdBlock], width: u16) -> Vec<Line<'static>> {
    let innerWidth = width.saturating_sub(2);
    let barStyle = Style::default().fg(Color::DarkGray);
    let mut lines = Vec::new();

    for block in blocks {
        for line in block.renderLines(innerWidth) {
            let mut spans = vec![Span::styled("\u{2502} ", barStyle)];
            spans.extend(line.spans);
            lines.push(Line::from(spans));
        }
    }

    lines
}

/// Render a horizontal rule.
fn renderRule(width: u16) -> Vec<Line<'static>> {
    let rule: String = "\u{2500}".repeat(width as usize);
    vec![Line::from(Span::styled(
        rule,
        Style::default().fg(Color::DarkGray),
    ))]
}
