use crate::config::Config;
use crate::message::{Content, Message};

/// A user input message with optional image attachments.
#[derive(Debug, Clone)]
pub struct UserInput {
    pub text: String,
    pub attachments: Vec<Attachment>,
}

/// A binary attachment to a user message.
///
/// For clipboard-pasted images, `data` contains raw RGBA pixels and
/// `rgbaDimensions` holds (width, height). PNG encoding is deferred to
/// submit time to avoid blocking the TUI event loop.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub mimeType: String,
    pub data: Vec<u8>,
    pub label: String,
    /// Raw RGBA dimensions — set when data is raw pixels, None when already encoded.
    pub rgbaDimensions: Option<(u32, u32)>,
}

impl From<String> for UserInput {
    fn from(text: String) -> Self {
        UserInput {
            text,
            attachments: Vec::new(),
        }
    }
}

/// A scaffolding instruction prepended to the latest User message at API
/// call time. Riders never appear in `self.history`, the transcript, or
/// snapshots — they're ephemeral nudges to reinforce system-prompt behavior.
///
/// All active riders render into a single `<CRITICAL_INSTRUCTIONS>` block
/// placed above the user's text in the API request copy only.
#[derive(Debug, Clone)]
pub struct Rider {
    /// Tag name used inside the `<CRITICAL_INSTRUCTIONS>` wrapper
    /// (e.g. `"THINKING"`, `"MODE"`). Kept as a `&'static str` so each
    /// rider's identity is fixed at compile time.
    pub id: &'static str,
    /// Body text. Must not contain the outer wrapper tags.
    pub content: String,
}

/// Build the active rider list for a session from its config.
pub(super) fn buildRiders(config: &Config) -> Vec<Rider> {
    let mut riders = Vec::new();
    if config.heavy.promptThinking {
        riders.push(Rider {
            id: "THINKING",
            content: crate::prompt::THINKING_RIDER_BODY.to_string(),
        });
    }
    riders
}

/// Render active riders into the prefix that goes before the user's text.
/// Returns an empty string when no riders are active.
fn renderRiderPrefix(riders: &[Rider]) -> String {
    if riders.is_empty() {
        return String::new();
    }
    let mut out = String::from("<CRITICAL_INSTRUCTIONS>\n");
    for r in riders {
        out.push_str(&format!(
            "<{id}>\n{body}\n</{id}>\n",
            id = r.id,
            body = r.content
        ));
    }
    out.push_str("</CRITICAL_INSTRUCTIONS>\n\n");
    out
}

/// Build the messages array sent to the API from the clean `history`.
///
/// Two transforms happen here — neither touches `self.history`:
/// 1. The latest User message gets a `<CRITICAL_INSTRUCTIONS>` prefix when
///    any riders are active.
/// 2. When `promptThinking` is on, assistant messages' `reasoning` field
///    is baked into their `content` as `<scratchpad>...</scratchpad>` so
///    the model sees the pattern it's being asked to produce. Models in
///    this mode don't get a separate `reasoning` JSON key.
pub(super) fn buildRequestMessages(
    history: &[Message],
    riders: &[Rider],
    promptThinking: bool,
) -> Vec<Message> {
    let prefix = renderRiderPrefix(riders);
    let lastUserIdx = history
        .iter()
        .rposition(|m| matches!(m, Message::User { .. }));

    history
        .iter()
        .enumerate()
        .map(|(i, msg)| match msg {
            Message::User { content } if Some(i) == lastUserIdx && !prefix.is_empty() => {
                Message::User {
                    content: prependToContent(content, &prefix),
                }
            }
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
            } if promptThinking => {
                let merged = match (reasoning.as_ref(), content.as_ref()) {
                    (Some(r), Some(c)) => Some(format!("<scratchpad>\n{r}\n</scratchpad>\n{c}")),
                    (Some(r), None) => Some(format!("<scratchpad>\n{r}\n</scratchpad>")),
                    (None, c) => c.cloned(),
                };
                Message::Assistant {
                    content: merged,
                    tool_calls: tool_calls.clone(),
                    reasoning: None,
                }
            }
            other => other.clone(),
        })
        .collect()
}

/// Prepend `prefix` to a `Content`'s text portion. Preserves multimodal
/// structure — riders attach to the first text block, image blocks
/// keep their position.
fn prependToContent(content: &Content, prefix: &str) -> Content {
    use crate::message::ContentBlock;
    match content {
        Content::Text(s) => Content::Text(format!("{prefix}{s}")),
        Content::Blocks(blocks) => {
            let mut out = Vec::with_capacity(blocks.len() + 1);
            let mut attached = false;
            for b in blocks {
                match b {
                    ContentBlock::Text { text } if !attached => {
                        out.push(ContentBlock::Text {
                            text: format!("{prefix}{text}"),
                        });
                        attached = true;
                    }
                    other => out.push(other.clone()),
                }
            }
            if !attached {
                out.insert(
                    0,
                    ContentBlock::Text {
                        text: prefix.to_string(),
                    },
                );
            }
            Content::Blocks(out)
        }
    }
}
