//! Markdown rendering with streaming support.
//!
//! Wraps `tui-markdown` (pulldown-cmark) with predictive delimiter
//! capping so partial streaming content renders correctly.
//!
//! Before parsing, scans for unclosed delimiters (`**`, `*`, `` ` ``,
//! `` ``` ``) and appends closers. This means mid-stream bold text
//! shows as bold rather than raw `**`, unclosed code fences render
//! as code blocks, etc.
//!
//! # Public API
//! - [`render`] — convert markdown text to ratatui `Text`

use std::borrow::Cow;

use ratatui::text::{Line, Span, Text};

/// Render markdown text into styled ratatui `Text`.
///
/// Caps any unclosed delimiters before parsing so partial
/// (streaming) content renders correctly.
pub fn render(text: &str) -> Text<'static> {
    let capped = capUnclosed(text);
    let parsed = tui_markdown::from_str(&capped);
    // Convert borrowed spans to owned so the result outlives `capped`.
    let ownedLines: Vec<Line<'static>> = parsed
        .lines
        .into_iter()
        .map(|line| {
            let ownedSpans: Vec<Span<'static>> = line
                .spans
                .into_iter()
                .map(|span| Span {
                    content: Cow::Owned(span.content.into_owned()),
                    style: span.style,
                })
                .collect();
            Line {
                spans: ownedSpans,
                style: line.style,
                ..Default::default()
            }
        })
        .collect();
    Text::from(ownedLines)
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
