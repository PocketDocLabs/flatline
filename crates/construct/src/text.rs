//! Text sanitization utilities for LLM output.
//!
//! Handles Unicode variation selector cleanup so emoji-only characters
//! don't get broken text-presentation selectors that cause rendering
//! artifacts in terminals.
//!
//! # Public API
//! - [`sanitizeVariationSelectors`] — strip U+FE0E from emoji-only codepoints
//! - [`recoverScratchpadClose`] — retroactively split a malformed `</scratchpad>`

/// Strip U+FE0E (text variation selector) when it follows a character
/// that has `Emoji_Presentation=Yes` — meaning it has no text glyph
/// and the selector just causes rendering artifacts.
///
/// Characters with text glyphs (like `\u{2713}` check mark or `\u{2699}`
/// gear) keep their variation selectors intact.
pub fn sanitizeVariationSelectors(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev: Option<char> = None;

    for ch in input.chars() {
        if ch == '\u{FE0E}' {
            if let Some(p) = prev {
                if isEmojiPresentation(p) {
                    // Drop the selector — no text glyph exists.
                    continue;
                }
            }
        }
        out.push(ch);
        prev = Some(ch);
    }
    out
}

/// Returns true if a character defaults to emoji presentation and
/// typically lacks a text-style glyph in monospace fonts.
///
/// Source: Unicode 15.1 emoji-data.txt `Emoji_Presentation=Yes`.
/// BMP entries are listed explicitly; SMP emoji blocks (U+1F000+)
/// use a range catch-all since virtually all are emoji-only.
fn isEmojiPresentation(ch: char) -> bool {
    // SMP catch-all: everything in the emoji supplement planes.
    if ch as u32 >= 0x1F000 {
        return true;
    }
    // BMP characters with Emoji_Presentation=Yes.
    matches!(
        ch,
        '\u{231A}'..='\u{231B}'
            | '\u{23E9}'..='\u{23F3}'
            | '\u{23F8}'..='\u{23FA}'
            | '\u{25FD}'..='\u{25FE}'
            | '\u{2614}'..='\u{2615}'
            | '\u{2648}'..='\u{2653}'
            | '\u{267F}'
            | '\u{2693}'
            | '\u{26A1}'
            | '\u{26AA}'..='\u{26AB}'
            | '\u{26BD}'..='\u{26BE}'
            | '\u{26C4}'..='\u{26C5}'
            | '\u{26CE}'
            | '\u{26D4}'
            | '\u{26EA}'
            | '\u{26F2}'..='\u{26F3}'
            | '\u{26F5}'
            | '\u{26FA}'
            | '\u{26FD}'
            | '\u{2702}'
            | '\u{2705}'
            | '\u{2708}'..='\u{270D}'
            | '\u{270F}'
            | '\u{2712}'
            | '\u{2714}'
            | '\u{2716}'
            | '\u{271D}'
            | '\u{2721}'
            | '\u{2728}'
            | '\u{2733}'..='\u{2734}'
            | '\u{2744}'
            | '\u{2747}'
            | '\u{274C}'
            | '\u{274E}'
            | '\u{2753}'..='\u{2755}'
            | '\u{2757}'
            | '\u{2763}'..='\u{2764}'
            | '\u{2795}'..='\u{2797}'
            | '\u{27A1}'
            | '\u{27B0}'
            | '\u{27BF}'
            | '\u{2934}'..='\u{2935}'
            | '\u{2B05}'..='\u{2B07}'
            | '\u{2B1B}'..='\u{2B1C}'
            | '\u{2B50}'
            | '\u{2B55}'
            | '\u{3030}'
            | '\u{303D}'
            | '\u{3297}'
            | '\u{3299}'
    )
}

/// Result of a successful retroactive scratchpad-close recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScratchpadRecovery {
    /// The reasoning buffer with the malformed close and trailing content removed.
    pub reasoning: String,
    /// The content that was originally misclassified as reasoning.
    pub content: String,
    /// The literal matched tag (e.g. `</scratch>`, `</scratchpa>`, `</scratchpad`).
    pub matchedTag: String,
}

/// Find a malformed `</scratchpad>` close near the end of `reasoning` and
/// split the trailing visible content out of the reasoning buffer.
///
/// Used when the streaming extractor never matched a proper `</scratchpad>`
/// — typically because the model dropped a letter (`</scratch>`,
/// `</scratchpa>`), forgot the `>` (`</scratchpad`), or wrote the bare
/// prefix (`</scratch`). In all those cases the entire turn collapses
/// into reasoning and the visible answer is lost to the user.
///
/// Heuristic: take the LAST `</scratch[A-Za-z]*\s*>?` in the buffer. Earlier
/// occurrences may be the model legitimately quoting the tag while thinking,
/// so we leave them alone. Returns `None` if no candidate exists or if the
/// candidate has nothing after it (a trailing typo with no recoverable
/// content isn't worth touching).
pub fn recoverScratchpadClose(reasoning: &str) -> Option<ScratchpadRecovery> {
    const NEEDLE: &str = "</scratch";

    let start = reasoning.rfind(NEEDLE)?;
    let bytes = reasoning.as_bytes();

    // Consume the alphabetic suffix after `</scratch` (e.g. `pad`, `pads`).
    let mut end = start + NEEDLE.len();
    while end < bytes.len() && bytes[end].is_ascii_alphabetic() {
        end += 1;
    }

    // Optionally consume `\s*>`. If neither shows up, the tag is bare —
    // keep `end` at the last alphabetic char so we don't eat real content.
    let mut probe = end;
    while probe < bytes.len() && (bytes[probe] == b' ' || bytes[probe] == b'\t') {
        probe += 1;
    }
    if probe < bytes.len() && bytes[probe] == b'>' {
        end = probe + 1;
    }

    let matched = &reasoning[start..end];
    let before = &reasoning[..start];
    let after = reasoning[end..].trim_start();
    if after.is_empty() {
        return None;
    }

    Some(ScratchpadRecovery {
        reasoning: before.to_string(),
        content: after.to_string(),
        matchedTag: matched.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keepTextVariationOnTextChars() {
        // Check mark has a text glyph — keep the selector.
        let input = "\u{2713}\u{FE0E}";
        assert_eq!(sanitizeVariationSelectors(input), input);
    }

    #[test]
    fn stripTextVariationOnEmojiChars() {
        // Thumbs up has no text glyph — strip the selector.
        let input = "\u{1F44D}\u{FE0E}";
        let expected = "\u{1F44D}";
        assert_eq!(sanitizeVariationSelectors(input), expected);
    }

    #[test]
    fn mixedContent() {
        // Gear (text) keeps VS, fire (emoji) loses it.
        let input = "\u{2699}\u{FE0E} and \u{1F525}\u{FE0E}";
        let expected = "\u{2699}\u{FE0E} and \u{1F525}";
        assert_eq!(sanitizeVariationSelectors(input), expected);
    }

    #[test]
    fn noSelectorsPassThrough() {
        let input = "plain text with no selectors";
        assert_eq!(sanitizeVariationSelectors(input), input);
    }

    #[test]
    fn bmpEmojiStripped() {
        // Umbrella with rain (U+2614) is Emoji_Presentation=Yes.
        let input = "\u{2614}\u{FE0E}";
        let expected = "\u{2614}";
        assert_eq!(sanitizeVariationSelectors(input), expected);
    }

    #[test]
    fn recoverDroppedPad() {
        let input = "thinking out loud</scratch>final answer";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(r.reasoning, "thinking out loud");
        assert_eq!(r.content, "final answer");
        assert_eq!(r.matchedTag, "</scratch>");
    }

    #[test]
    fn recoverPartialPad() {
        let input = "thoughts</scratchpa>visible";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(r.reasoning, "thoughts");
        assert_eq!(r.content, "visible");
        assert_eq!(r.matchedTag, "</scratchpa>");
    }

    #[test]
    fn recoverMissingAngleBracket() {
        let input = "thoughts</scratchpad\nvisible answer";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(r.reasoning, "thoughts");
        assert_eq!(r.content, "visible answer");
        assert_eq!(r.matchedTag, "</scratchpad");
    }

    #[test]
    fn recoverBarePrefix() {
        let input = "thinking</scratch the answer is 42";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(r.reasoning, "thinking");
        assert_eq!(r.content, "the answer is 42");
        assert_eq!(r.matchedTag, "</scratch");
    }

    #[test]
    fn recoverProperTag() {
        // Defensive: if a proper tag somehow leaks through, recovery still works.
        let input = "thoughts</scratchpad>\n\nanswer";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(r.reasoning, "thoughts");
        assert_eq!(r.content, "answer");
        assert_eq!(r.matchedTag, "</scratchpad>");
    }

    #[test]
    fn recoverPicksLastOccurrence() {
        // Earlier `</scratch` mention is the model quoting itself; only the
        // last one should be treated as the close.
        let input = "I should remember to write </scratchpad> at the end</scratch>final";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(
            r.reasoning,
            "I should remember to write </scratchpad> at the end"
        );
        assert_eq!(r.content, "final");
        assert_eq!(r.matchedTag, "</scratch>");
    }

    #[test]
    fn recoverNothingAfterTag() {
        // Trailing malformed tag with no content — nothing to recover.
        let input = "just thoughts</scratch";
        assert!(recoverScratchpadClose(input).is_none());
    }

    #[test]
    fn recoverNoMatch() {
        let input = "no scratch tag here at all";
        assert!(recoverScratchpadClose(input).is_none());
    }

    #[test]
    fn recoverEmptyInput() {
        assert!(recoverScratchpadClose("").is_none());
    }

    #[test]
    fn recoverStripsLeadingWhitespace() {
        let input = "thoughts</scratch>\n\n  final answer";
        let r = recoverScratchpadClose(input).unwrap();
        assert_eq!(r.content, "final answer");
    }
}
