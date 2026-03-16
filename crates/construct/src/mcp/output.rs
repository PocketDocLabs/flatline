//! MCP output limits and token estimation.
//!
//! Caps tool output to prevent context flooding from verbose MCP servers.
//! Uses a rough chars/4 heuristic for token estimation.
//!
//! # Public API
//! - [`limitOutput`] — truncate output exceeding token limit
//! - [`estimateTokens`] — rough token count from text
//!
//! # Dependencies
//! None.

/// Token count at which a warning is appropriate.
pub const OUTPUT_WARNING_TOKENS: usize = 10_000;

/// Default maximum output tokens per MCP tool call.
pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = 25_000;

/// Rough token estimate: chars / 4.
///
/// This is a standard heuristic across LLM tooling. Not precise,
/// but consistent and fast enough for budget decisions.
pub fn estimateTokens(text: &str) -> usize {
    text.len() / 4
}

/// Truncate MCP tool output if it exceeds the configured limit.
///
/// Returns the original string if within budget, or a truncated version
/// with a suffix indicating how much was cut.
pub fn limitOutput(output: &str, maxTokens: usize) -> String {
    let tokens = estimateTokens(output);
    if tokens <= maxTokens {
        return output.to_string();
    }

    // Truncate at the char boundary closest to maxTokens * 4.
    let maxChars = maxTokens * 4;
    let truncated = if output.is_char_boundary(maxChars) {
        &output[..maxChars]
    } else {
        // Walk back to a valid char boundary.
        let mut end = maxChars;
        while end > 0 && !output.is_char_boundary(end) {
            end -= 1;
        }
        &output[..end]
    };

    format!(
        "{truncated}\n\n... [truncated at ~{maxTokens} tokens, \
         {tokens} total estimated]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimateTokensBasic() {
        assert_eq!(estimateTokens("abcd"), 1);
        assert_eq!(estimateTokens("abcdefgh"), 2);
        assert_eq!(estimateTokens(""), 0);
    }

    #[test]
    fn limitOutputPassthrough() {
        let short = "hello world";
        assert_eq!(limitOutput(short, 100), short);
    }

    #[test]
    fn limitOutputTruncates() {
        let long = "a".repeat(200_000); // ~50K tokens
        let result = limitOutput(&long, 1000);
        assert!(result.len() < long.len());
        assert!(result.contains("[truncated"));
    }

    #[test]
    fn limitOutputHandlesUtf8() {
        // Multibyte chars: ensure we don't split mid-character.
        let text = "\u{1f600}".repeat(20_000); // emoji, 4 bytes each
        let result = limitOutput(&text, 100);
        // Should be valid UTF-8.
        assert!(result.is_char_boundary(result.len()));
    }
}
