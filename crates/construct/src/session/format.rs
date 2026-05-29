//! Formatting and stream-assembly helpers for [`crate::session`].
//!
//! These helpers are intentionally session-private: they keep `session.rs`
//! focused on orchestration without creating a broader API surface.

use super::Attachment;
use crate::message::{FunctionCall, Message, ToolCall};

/// Best-effort path normalization for read-before-write comparison.
pub(super) fn normalizePath(path: &str) -> String {
    std::path::Path::new(path)
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string())
}

/// Extract a short snippet around the first occurrence of a query in text.
pub(super) fn extractSnippet(text: &str, queryLower: &str) -> String {
    let textLower = text.to_lowercase();
    let pos = match textLower.find(queryLower) {
        Some(p) => p,
        None => return String::new(),
    };

    let contextChars = 80;
    let start = pos.saturating_sub(contextChars);
    let end = (pos + queryLower.len() + contextChars).min(text.len());

    let start = text.floor_char_boundary(start);
    let end = text.ceil_char_boundary(end);

    text[start..end].replace('\n', " ")
}

/// Build an `Assistant` message for history.
pub(super) fn buildAssistantMessage(
    content: Option<String>,
    toolCalls: Option<Vec<ToolCall>>,
    reasoning: Option<String>,
) -> Message {
    Message::Assistant {
        content,
        tool_calls: toolCalls,
        reasoning,
    }
}

/// Outcome of a single API call.
pub(super) enum TurnResult {
    Done {
        promptTokens: Option<usize>,
    },
    ToolCalls {
        calls: Vec<ToolCall>,
        content: Option<String>,
        reasoning: Option<String>,
        promptTokens: Option<usize>,
    },
    Cancelled,
    /// A transient API error that can be retried (e.g. 500, 502, timeout).
    TransientError(String),
}

/// Check whether an API error message looks transient (worth retrying).
pub(super) fn isTransientError(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("internal server error")
        || lower.contains("bad gateway")
        || lower.contains("service unavailable")
        || lower.contains("gateway timeout")
        || lower.contains("stream stalled")
        || lower.contains("overloaded")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("temporarily unavailable")
        || lower.contains("server error")
        || lower.contains("timeout")
        || lower.contains("connection reset")
        || lower.contains("stream read error")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
}

/// Format a coalesced `WakeBatch` as a single user-shaped envelope. The
/// model sees one user message containing N nested `<wake>` elements,
/// one per fire, in arrival order. Single-fire batches still go through
/// this path so the on-the-wire shape is uniform.
pub(super) fn formatWakeBatch(batch: &crate::wakes::WakeBatch) -> String {
    use std::fmt::Write;
    let count = batch.fires.len();
    let mut buf = String::with_capacity(64 + count * 96);
    let _ = write!(buf, "<wakes count=\"{count}\">");
    for fire in &batch.fires {
        let firedAtSecs = fire.firedAt.elapsed().as_secs();
        let kindStr = fire.kind.asStr();
        let source = escapeWakeXml(&fire.source);
        let payload = escapeWakeXml(&fire.payload);
        let _ = write!(
            buf,
            "\n<wake source=\"{}\" kind=\"{kindStr}\" ageSecs=\"{firedAtSecs}\">\n{}\n</wake>",
            source, payload,
        );
    }
    buf.push_str("\n</wakes>");
    buf
}

fn escapeWakeXml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// One-line summary for `LogEvent::WakeBatchInjected` — drives the deck
/// notice chip without exposing the full envelope text.
pub(super) fn wakeBatchSummary(batch: &crate::wakes::WakeBatch) -> String {
    let count = batch.fires.len();
    let first = batch.fires.first();
    match (count, first) {
        (1, Some(f)) => format!("{} \u{00B7} {}", f.source, snippet(&f.payload, 80)),
        (n, Some(f)) => format!("{n} wakes (first: {})", f.source),
        _ => "wake".to_string(),
    }
}

fn snippet(s: &str, n: usize) -> String {
    let first = s.lines().next().unwrap_or("");
    if first.len() <= n {
        first.to_string()
    } else {
        let mut cut = n;
        while cut > 0 && !first.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}\u{2026}", &first[..cut])
    }
}

/// Returns the `Content` for `Message::User` and optional `TurnAttachment` list for transcript.
pub(super) fn buildUserContent(
    text: &str,
    attachments: &[Attachment],
) -> (
    crate::message::Content,
    Option<Vec<crate::transcript::TurnAttachment>>,
) {
    use base64::Engine;

    let encoded: Vec<(String, Vec<u8>)> = attachments
        .iter()
        .map(|att| {
            if let Some((w, h)) = att.rgbaDimensions {
                let png = encodeRgbaToPng(&att.data, w, h);
                ("image/png".to_string(), png)
            } else {
                (att.mimeType.clone(), att.data.clone())
            }
        })
        .collect();

    let content = if encoded.is_empty() {
        crate::message::Content::text(text)
    } else {
        let imageUris: Vec<String> = encoded
            .iter()
            .map(|(mime, data)| {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                format!("data:{mime};base64,{b64}")
            })
            .collect();
        crate::message::Content::withImages(text, imageUris)
    };

    let turnAttachments = if encoded.is_empty() {
        None
    } else {
        Some(
            encoded
                .iter()
                .map(|(mime, data)| crate::transcript::TurnAttachment {
                    mimeType: mime.clone(),
                    data: base64::engine::general_purpose::STANDARD.encode(data),
                })
                .collect(),
        )
    };

    (content, turnAttachments)
}

/// Encode raw RGBA pixel data to PNG bytes.
/// Resizes images larger than 2048px on either side.
fn encodeRgbaToPng(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .expect("RGBA buffer size mismatch");
    let dynamic = image::DynamicImage::ImageRgba8(img);

    let finalImg = if width > 2048 || height > 2048 {
        dynamic.resize(2048, 2048, image::imageops::FilterType::Triangle)
    } else {
        dynamic
    };

    let mut buf = Vec::new();
    finalImg
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("PNG encoding failed");
    buf
}

/// Accumulates streaming tool call deltas into complete tool calls.
pub(super) struct ToolCallAccumulator {
    pending: Vec<PendingCall>,
}

struct PendingCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    pub(super) fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Whether any tool call deltas have been accumulated.
    pub(super) fn hasContent(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Peek at the current (name, arguments) for an index. Used to refresh
    /// tool-call previews after each arg delta.
    pub(super) fn pendingCall(&self, index: usize) -> Option<(&str, &str)> {
        self.pending
            .get(index)
            .map(|p| (p.name.as_str(), p.arguments.as_str()))
    }

    /// Returns `(newName, totalArgBytes)`:
    /// - `newName` is `Some(name)` only on the delta that first sets the name
    ///   for this index (used to emit ToolCallPending exactly once).
    /// - `totalArgBytes` is the running total of accumulated argument bytes
    ///   for this index, or `None` if this delta didn't carry any args.
    pub(super) fn accumulate(
        &mut self,
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) -> (Option<String>, Option<usize>) {
        while self.pending.len() <= index {
            self.pending.push(PendingCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
        }

        let entry = &mut self.pending[index];
        if let Some(id) = id {
            entry.id = id;
        }
        let mut firstName = None;
        if let Some(name) = name {
            if !name.is_empty() && entry.name.is_empty() {
                firstName = Some(name.clone());
            }
            entry.name = name;
        }
        let bytes = if let Some(args) = arguments {
            entry.arguments.push_str(&args);
            Some(entry.arguments.len())
        } else {
            None
        };
        (firstName, bytes)
    }

    pub(super) fn finish(self) -> Vec<ToolCall> {
        self.pending
            .into_iter()
            .filter(|p| !p.name.is_empty())
            .map(|p| ToolCall {
                id: p.id,
                callType: "function".into(),
                function: FunctionCall {
                    name: p.name,
                    arguments: p.arguments,
                },
            })
            .collect()
    }
}

/// Format a JobOutputSnapshot for the agent. Includes the command,
/// state, line range, and the buffered lines themselves with a hint
/// about how to page if `totalLines` exceeds what we returned.
pub(super) fn formatJobOutput(
    taskId: u64,
    snap: &crate::jobs::JobOutputSnapshot,
    sinceLine: Option<u64>,
) -> String {
    use crate::jobs::JobState;
    let stateLabel = match &snap.state {
        JobState::Running => "running".to_string(),
        JobState::Completed { exitCode } => format!("completed exit {exitCode}"),
        JobState::Killed => "killed".into(),
        JobState::Errored(msg) => format!("errored: {msg}"),
    };
    let returned = snap.lines.len() as u64;
    let nextLine = snap.firstLine + returned;
    let header = format!(
        "Task #{} \u{2014} {}\nState: {} \u{00B7} {} total lines \u{00B7} \
         showing lines {}..{}\n\n",
        taskId,
        snap.command,
        stateLabel,
        snap.totalLines,
        snap.firstLine,
        snap.firstLine + returned.saturating_sub(1),
    );
    let mut body = String::new();
    let askedFor = sinceLine.unwrap_or(0);
    if snap.firstLine > askedFor && askedFor < snap.earliestBuffered {
        let lost = snap.earliestBuffered - askedFor;
        body.push_str(&format!(
            "[earlier {lost} lines fell off the ring buffer; oldest buffered is line {}]\n",
            snap.earliestBuffered,
        ));
    } else if sinceLine.is_none() && snap.firstLine > snap.earliestBuffered {
        let recoverable = snap.firstLine - snap.earliestBuffered;
        body.push_str(&format!(
            "[{recoverable} earlier lines still buffered \u{2014} call \
             jobOutput(jobId: {taskId}, sinceLine: {}) to read from the start]\n",
            snap.earliestBuffered,
        ));
    }
    for line in &snap.lines {
        body.push_str(line);
        body.push('\n');
    }
    if nextLine < snap.totalLines {
        body.push_str(&format!(
            "\n[{} more lines \u{2014} call jobOutput(jobId: {}, sinceLine: {}) to continue]",
            snap.totalLines - nextLine,
            taskId,
            nextLine,
        ));
    } else if matches!(snap.state, JobState::Running) {
        body.push_str(&format!(
            "\n[task is still running \u{2014} next sinceLine: {}]",
            nextLine,
        ));
    }
    format!("{header}{body}")
}

/// Format a TaskList snapshot.
pub(super) fn formatJobList(tasks: &[crate::jobs::JobInfo]) -> String {
    use crate::jobs::JobState;
    if tasks.is_empty() {
        return "No background jobs.".into();
    }
    let mut out = String::from("Background tasks:\n");
    for info in tasks {
        let stateLabel = match &info.state {
            JobState::Running => "running".to_string(),
            JobState::Completed { exitCode } => format!("completed exit {exitCode}"),
            JobState::Killed => "killed".into(),
            JobState::Errored(msg) => format!("errored: {msg}"),
        };
        let age = info.spawnedAt.elapsed().as_secs();
        let cmdPreview = if info.command.len() > 80 {
            format!(
                "{}\u{2026}",
                &info.command[..info.command.floor_char_boundary(80)]
            )
        } else {
            info.command.clone()
        };
        out.push_str(&format!(
            "  #{} {} \u{2014} {} \u{00B7} {}s \u{00B7} {} lines\n",
            info.id, cmdPreview, stateLabel, age, info.totalLines,
        ));
    }
    out
}

pub(super) fn formatMonitorList(monitors: &[crate::monitors::MonitorInfo]) -> String {
    use crate::monitors::MonitorState;
    if monitors.is_empty() {
        return "No monitors.".into();
    }
    let mut out = String::from("Monitors:\n");
    for info in monitors {
        let stateLabel = match &info.state {
            MonitorState::Running => "running".to_string(),
            MonitorState::Stopped => "stopped".into(),
            MonitorState::AutoStopped(reason) => format!("auto-stopped ({reason})"),
        };
        let lastEvent = match info.lastEventAt {
            Some(t) => format!("{}s ago", t.elapsed().as_secs()),
            None => "never".into(),
        };
        out.push_str(&format!(
            "  #{} \"{}\" terminal {} | /{}/ \u{2014} {} \u{00B7} {} events \u{00B7} last {}\n",
            info.id,
            info.description,
            info.terminal,
            info.filter,
            stateLabel,
            info.eventCount,
            lastEvent,
        ));
    }
    out
}

pub(super) fn formatWakeList(sources: &[crate::wakes::WakeSourceInfo]) -> String {
    if sources.is_empty() {
        return "No wake sources.".into();
    }
    let mut out = String::from("Wake sources:\n");
    for info in sources {
        let promptPreview = info
            .prompt
            .as_deref()
            .filter(|p| !p.is_empty())
            .map(|p| {
                if p.len() > 40 {
                    format!(" \u{2014} {}\u{2026}", &p[..p.floor_char_boundary(40)])
                } else {
                    format!(" \u{2014} {p}")
                }
            })
            .unwrap_or_default();
        let age = info.createdAt.elapsed().as_secs();
        out.push_str(&format!(
            "  #{} [{}] {} \u{00B7} {} fires \u{00B7} {age}s ago{promptPreview}\n",
            info.id,
            info.kind.asStr(),
            info.summary,
            info.firesSoFar,
        ));
    }
    out
}
