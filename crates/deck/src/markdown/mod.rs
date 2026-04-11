//! Block-based markdown rendering with streaming support.
//!
//! Parses markdown via pulldown-cmark into structured blocks,
//! then renders each block type with appropriate styling:
//! syntax-highlighted code, box-drawn tables, styled headings, etc.
//!
//! Before parsing, scans for unclosed delimiters and appends closers
//! so partial streaming content renders correctly.
//!
//! # Public API
//! - [`render`] — convert markdown text to rendered blocks
//! - [`RenderedBlock`] — output block: text (wrappable) or code (scrollable)
//! - [`renderCodeBlock`] — render a code block with borders and horizontal scroll
//!
//! # Dependencies
//! `pulldown-cmark`, `syntect`, `two-face`, `ratatui`

mod block;
pub mod highlight;
mod mermaid;
mod parse;
mod table;

use ratatui::text::{Line, Span};

pub use highlight::renderCodeBlock;

/// A rendered markdown block — either wrappable text or a scrollable code block.
pub enum RenderedBlock {
    /// Regular text lines (paragraphs, headings, lists, tables, etc.).
    ///
    /// These should be word-wrapped by the caller.
    Text(Vec<Line<'static>>),

    /// Code block with highlighted content, rendered separately.
    ///
    /// Content is stored as untruncated spans per line so the caller
    /// can render with horizontal scroll and borders.
    Code {
        lang: Option<String>,
        /// Highlighted spans per source line (no borders, no truncation).
        lines: Vec<Vec<Span<'static>>>,
        /// Original source text for clipboard copy.
        code: String,
    },
}

/// Render markdown text into structured blocks.
///
/// Caps any unclosed delimiters for streaming, parses into
/// structured blocks, then renders each with appropriate styling.
///
/// Args:
///     text: Raw markdown source (may be incomplete/streaming).
///     width: Available rendering width in columns.
///
/// Returns:
///     Vec<RenderedBlock>: Blocks ready for rendering.
pub fn render(text: &str, width: u16) -> Vec<RenderedBlock> {
    let capped = capUnclosed(text);
    let blocks = parse::parse(&capped);
    let mut rendered = Vec::new();
    for mdBlock in &blocks {
        rendered.extend(mdBlock.renderBlocks(width));
    }
    rendered
}

/// Append closing delimiters for any unclosed markdown markers.
///
/// Scans the text tracking delimiter state and appends closers
/// at the end so pulldown-cmark produces the intended formatting
/// rather than treating unclosed markers as literal text.
fn capUnclosed(text: &str) -> String {
    let mut result = text.to_string();
    let mut suffix = String::new();

    // Code fences: count ``` occurrences on their own line.
    let fenceCount = text
        .lines()
        .filter(|l| l.trim_start().starts_with("```"))
        .count();
    if fenceCount % 2 != 0 {
        suffix.push_str("\n```");
    }

    // For inline markers, track open/close state outside code fences.
    let mut inFence = false;
    let mut inCode = false;
    let mut inBold = false;
    let mut inItalic = false;

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            inFence = !inFence;
            continue;
        }
        if inFence {
            continue;
        }

        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            match chars[i] {
                '\\' => {
                    i += 2;
                    continue;
                }
                '`' => {
                    inCode = !inCode;
                }
                '*' if !inCode => {
                    // Check for ** (bold) vs * (italic).
                    if i + 1 < chars.len() && chars[i + 1] == '*' {
                        inBold = !inBold;
                        i += 2;
                        continue;
                    } else {
                        inItalic = !inItalic;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    if inCode {
        suffix.push('`');
    }
    if inBold {
        suffix.push_str("**");
    }
    if inItalic {
        suffix.push('*');
    }

    result.push_str(&suffix);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capsUnmatchedBold() {
        let capped = capUnclosed("hello **world");
        assert!(capped.ends_with("**"));
    }

    #[test]
    fn capsUnmatchedCodeFence() {
        let capped = capUnclosed("text\n```rust\nfn main() {}");
        assert!(capped.ends_with("\n```"));
    }

    #[test]
    fn leavesCompleteMarkdownAlone() {
        let input = "hello **world** and *stuff*";
        let capped = capUnclosed(input);
        assert_eq!(capped, input);
    }

    #[test]
    fn capsUnmatchedInlineCode() {
        let capped = capUnclosed("use `foo");
        assert!(capped.ends_with("`"));
    }
}
