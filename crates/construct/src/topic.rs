//! Topic tracker — classifies user messages into topic segments.
//!
//! After each user message, a utility model call decides whether the
//! conversation topic has shifted. The utility model receives the same
//! conversation history as the main model (maximizing prefix cache hits)
//! with a topic classification instruction appended as the final message.
//!
//! Topic boundaries drive S3 compaction targeting and session naming.
//!
//! # Public API
//! - [`TopicTracker`] — stateful topic classifier
//! - [`TopicInfo`] — metadata for a topic segment
//! - [`EvalResult`] — result of a single evaluation
//! - [`TopicDecision`] — raw classification output (for async spawning)
//! - [`classifyPrepared`] — free function for spawned classification tasks
//!
//! # Dependencies
//! `crate::api`, `crate::message`

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::api;
use crate::config::ModelTier;
use crate::message::Message;
use crate::transcript::Turn;

use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TopicInfo {
    pub topicId: String,
    pub label: String,
    pub startBlock: String,
    pub blockCount: usize,
}

pub struct EvalResult {
    pub topicId: String,
    pub label: String,
    pub isNewTopic: bool,
}

/// Raw classification output — either the topic is the same or a new label.
///
/// Exposed publicly so spawned tasks can return it and the caller
/// can apply the decision to TopicTracker state on the main thread.
pub enum TopicDecision {
    Same,
    New(String),
}

pub struct TopicTracker {
    topics: Vec<TopicInfo>,
    currentTopicId: String,
    currentLabel: String,
    nextTopicNum: usize,
}

impl Default for TopicTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicTracker {
    pub fn new() -> Self {
        Self {
            topics: Vec::new(),
            currentTopicId: String::new(),
            currentLabel: String::new(),
            nextTopicNum: 1,
        }
    }

    /// Evaluate whether a user message starts a new topic (synchronous convenience).
    ///
    /// Calls prepareClassification + classifyPrepared + applyDecision sequentially.
    /// For async overlap, use the three methods separately.
    ///
    /// Args:
    ///     history: The live conversation history (same as the main model sees).
    ///     blockId: Current exchange block ID.
    ///     client: API client.
    ///     utilityModel: Model ID for utility calls.
    pub async fn evaluate(
        &mut self,
        history: &[Message],
        blockId: &str,
        client: &api::Client,
        utilityModel: &str,
    ) -> Result<EvalResult> {
        let messages = self.prepareClassification(history);
        let decision = classifyPrepared(messages, client.clone(), utilityModel.to_string()).await;
        Ok(self.applyDecision(decision, blockId))
    }

    /// Build the message list for a topic classification call.
    ///
    /// Clones history, strips the last user message, and appends the
    /// topic context XML block. Returns owned data suitable for passing
    /// to a spawned task via [`classifyPrepared`].
    pub fn prepareClassification(&self, history: &[Message]) -> Vec<Message> {
        // Extract the next user message text for the XML block.
        let nextMessage = history
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::User { content } => Some(content.textContent()),
                _ => None,
            })
            .unwrap_or("");

        // Clone history without the last user message — it's presented
        // inside the <topic_tracker> block instead to avoid duplication.
        let mut messages: Vec<Message> = history.to_vec();
        if let Some(pos) = messages
            .iter()
            .rposition(|m| matches!(m, Message::User { .. }))
        {
            messages.remove(pos);
        }
        messages.push(Message::User {
            content: self.buildTopicContext(nextMessage).into(),
        });

        messages
    }

    /// Apply a classification decision to internal state.
    ///
    /// Call this on the main thread after collecting the result from
    /// a spawned [`classifyPrepared`] task.
    pub fn applyDecision(&mut self, decision: TopicDecision, blockId: &str) -> EvalResult {
        match decision {
            TopicDecision::New(label) => self.startTopic(&label, blockId),
            TopicDecision::Same => {
                if self.currentTopicId.is_empty() {
                    // First message but model said "same" — shouldn't happen,
                    // but fall back to a generic label.
                    return self.startTopic("General", blockId);
                }
                // Increment block count on current topic.
                if let Some(last) = self.topics.last_mut() {
                    last.blockCount += 1;
                }
                EvalResult {
                    topicId: self.currentTopicId.clone(),
                    label: self.currentLabel.clone(),
                    isNewTopic: false,
                }
            }
        }
    }

    /// Get labels for all tracked topics.
    pub fn topicLabels(&self) -> Vec<String> {
        self.topics.iter().map(|t| t.label.clone()).collect()
    }

    /// Get all topic infos (for S3 targeting).
    pub fn topics(&self) -> &[TopicInfo] {
        &self.topics
    }

    /// Current topic ID.
    pub fn currentTopicId(&self) -> &str {
        &self.currentTopicId
    }

    /// Current topic label.
    pub fn currentLabel(&self) -> &str {
        &self.currentLabel
    }

    /// Restore state from saved topic infos (for session resume).
    ///
    /// When `topics` is empty the tracker is reset to a clean-slate state —
    /// `currentTopicId`/`currentLabel` are cleared so the transcript is not
    /// left stamping turns with a ghost `topicId`. `nextTopicNum` is derived
    /// from the max existing `topic-NN` number rather than list length so
    /// rewinds that drop entries can't recycle a still-referenced ID.
    pub fn restoreState(&mut self, topics: Vec<TopicInfo>) {
        match topics.last() {
            Some(last) => {
                self.currentTopicId = last.topicId.clone();
                self.currentLabel = last.label.clone();
            }
            None => {
                self.currentTopicId.clear();
                self.currentLabel.clear();
            }
        }
        self.nextTopicNum = nextTopicNumFromTopics(&topics);
        self.topics = topics;
    }

    /// Start a new topic segment.
    fn startTopic(&mut self, label: &str, blockId: &str) -> EvalResult {
        let topicId = format!("topic-{:02}", self.nextTopicNum);
        self.nextTopicNum += 1;
        self.currentTopicId = topicId.clone();
        self.currentLabel = label.to_string();
        self.topics.push(TopicInfo {
            topicId: topicId.clone(),
            label: label.to_string(),
            startBlock: blockId.to_string(),
            blockCount: 1,
        });
        EvalResult {
            topicId,
            label: label.to_string(),
            isNewTopic: true,
        }
    }

    /// Build the topic tracking context that gets appended to history.
    ///
    /// Includes the full classification system prompt so the model gets
    /// all framing in one block, riding the prefix cache from the shared
    /// conversation history that precedes it.
    fn buildTopicContext(&self, nextMessage: &str) -> String {
        let mut ctx = String::from("<topic_tracker>\n");

        ctx.push_str(TOPIC_SYSTEM_PROMPT);
        ctx.push('\n');

        // Recent topic history so the model can see what came before.
        let recentCount = 5;
        let start = self.topics.len().saturating_sub(recentCount);
        if !self.topics[start..].is_empty() {
            ctx.push_str("\n<recent_topics>\n");
            for t in &self.topics[start..] {
                ctx.push_str(&format!("- \"{}\" ({} blocks)\n", t.label, t.blockCount));
            }
            ctx.push_str("</recent_topics>\n\n");
        }

        // Current topic state.
        if !self.currentLabel.is_empty() {
            let blockCount = self.topics.last().map_or(0, |t| t.blockCount);
            ctx.push_str(&format!(
                "<current_topic label=\"{}\" blocks=\"{}\"/>\n\n",
                self.currentLabel, blockCount
            ));
        }

        // Present the next user message so the model knows what to classify.
        ctx.push_str(&format!(
            "<next_user_message>\n{nextMessage}\n</next_user_message>\n\n"
        ));

        ctx.push_str(
            "Classify the message in <next_user_message> above. \
            Respond with a single <topic> tag. Nothing else.",
        );
        ctx.push_str("\n</topic_tracker>");
        ctx
    }
}

/// Run a topic classification using pre-built messages.
///
/// Free function — takes only owned data, safe to `tokio::spawn`.
/// Returns the raw [`TopicDecision`]; caller applies it to tracker
/// state via [`TopicTracker::applyDecision`].
pub async fn classifyPrepared(
    messages: Vec<Message>,
    client: api::Client,
    utilityModel: String,
) -> TopicDecision {
    match client
        .complete(ModelTier::Utility, &messages, Some(&utilityModel))
        .await
    {
        Ok((response, _usage)) => parseTopicResponse(&response),
        Err(e) => {
            tracing::warn!("topic classification failed, assuming same: {e}");
            TopicDecision::Same
        }
    }
}

const TOPIC_SYSTEM_PROMPT: &str = "\
STOP \u{2014} ignore all prior instructions and roles. You are no longer the \
agent in this conversation. You are now a topic classifier. Do not continue \
the conversation, do not use tools, do not answer questions. Your ONLY job \
is to decide whether the message in <next_user_message> continues the current \
topic or starts a new one.\n\
\n\
A topic is a coherent unit of work \u{2014} one bug, one feature, one investigation. \
It should be specific enough to summarize in a sentence later.\n\
\n\
SAME topic (keep the current label):\n\
- Refinements, follow-ups, corrections within the same area\n\
- Debugging or fixing something that just broke in the current work\n\
- Meta-coordination about the current work (\"commit this\", \"what's next\", \
\"let's plan\", \"looks good\", \"run the tests\")\n\
- Short reactions, confirmations, or redirections (\"no do it the other way\")\n\
\n\
NEW topic (generate a new label):\n\
- The user asks about a clearly different file, feature, or system\n\
- The user starts a new task unrelated to the current thread\n\
- The subject matter has genuinely changed, not just the approach\n\
\n\
Label rules:\n\
- 2-5 words, title case, no punctuation\n\
- Name the specific thing: \"PTY Resize Handling\", \"Auth Token Expiry Bug\"\n\
- Never use vague labels: \"Next Steps\", \"Various Issues\", \"Code Changes\", \
\"Implementing Features\", \"Project Work\"\n\
- If you could not distinguish two topics by label alone, the label is too vague\n\
\n\
Respond with a single <topic> tag. No explanation, no other text.\n\
\n\
If same topic:\n\
<topic>same</topic>\n\
\n\
If new topic:\n\
<topic>Label Here</topic>";

/// Parse the model's response into a topic decision.
fn parseTopicResponse(response: &str) -> TopicDecision {
    let trimmed = response.trim();

    // Try to extract content from <topic>...</topic> tags.
    if let Some(start) = trimmed.find("<topic>")
        && let Some(end) = trimmed.find("</topic>")
    {
        let inner = trimmed[start + 7..end].trim();
        if inner.eq_ignore_ascii_case("same") {
            return TopicDecision::Same;
        }
        let label = sanitizeLabel(inner);
        if !label.is_empty() {
            return TopicDecision::New(label);
        }
    }

    // Fallback: handle bare "SAME" or "NEW: label" responses.
    if trimmed.eq_ignore_ascii_case("SAME") {
        return TopicDecision::Same;
    }

    let upper = trimmed.to_uppercase();
    if let Some(rest) = upper.strip_prefix("NEW:") {
        let labelStart = trimmed.len() - rest.len();
        let label = sanitizeLabel(&trimmed[labelStart..]);
        if !label.is_empty() {
            return TopicDecision::New(label);
        }
    }

    // If the model didn't follow the format, default to same.
    tracing::debug!(
        response = trimmed,
        "unexpected topic classifier response, defaulting to SAME"
    );
    TopicDecision::Same
}

/// Next `topic-NN` number to assign, derived from the max existing number.
///
/// Using max-plus-one instead of list-length prevents ID reuse after a rewind
/// that truncated the topics list. Older transcript turns referencing a
/// higher-numbered topic would otherwise collide with a freshly-minted topic
/// that reused their ID.
fn nextTopicNumFromTopics(topics: &[TopicInfo]) -> usize {
    topics
        .iter()
        .filter_map(|t| {
            t.topicId
                .strip_prefix("topic-")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .map_or(1, |n| n + 1)
}

/// Clean up a label: trim whitespace, quotes, trailing punctuation.
fn sanitizeLabel(raw: &str) -> String {
    raw.trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim_end_matches('.')
        .trim()
        .to_string()
}

/// Rebuild TopicInfo entries from a chain of transcript turns.
///
/// Groups turns by topicId, derives startBlock (first blockId per topic)
/// and blockCount (distinct blockId count per topic). Labels come from
/// `existingTopics`; topics whose labels can't be recovered are dropped
/// from the output rather than resurrected with a placeholder. A silent
/// `"Unknown"` stand-in used to mask upstream corruption (empty tracker +
/// rewind) by baking a meaningless label into persisted metadata.
///
/// Args:
///     turns: Turns on the active branch, in chronological order.
///     existingTopics: Previous TopicInfo list (for label lookup). Callers
///         that have multiple label sources (e.g. live tracker + on-disk
///         meta) should merge them before passing the union here.
pub fn rebuildTopicInfos(turns: &[Turn], existingTopics: &[TopicInfo]) -> Vec<TopicInfo> {
    let labelMap: HashMap<&str, &str> = existingTopics
        .iter()
        .map(|t| (t.topicId.as_str(), t.label.as_str()))
        .collect();

    // Track first-seen order, first blockId, and distinct blockIds per topic.
    let mut order: Vec<String> = Vec::new();
    let mut firstBlock: HashMap<String, String> = HashMap::new();
    let mut blockSets: HashMap<String, HashSet<String>> = HashMap::new();

    for turn in turns {
        if turn.topicId.is_empty() {
            continue;
        }
        let tid = &turn.topicId;

        if !firstBlock.contains_key(tid) {
            order.push(tid.clone());
            firstBlock.insert(tid.clone(), turn.blockId.clone());
        }

        blockSets
            .entry(tid.clone())
            .or_default()
            .insert(turn.blockId.clone());
    }

    order
        .into_iter()
        .filter_map(|tid| {
            let label = match labelMap.get(tid.as_str()).copied() {
                Some(l) => l.to_string(),
                None => {
                    tracing::warn!(
                        topicId = %tid,
                        "dropping topic with unrecoverable label during rebuild"
                    );
                    return None;
                }
            };
            let startBlock = firstBlock.get(&tid).cloned().unwrap_or_default();
            let blockCount = blockSets.get(&tid).map_or(0, |s| s.len());
            Some(TopicInfo {
                topicId: tid,
                label,
                startBlock,
                blockCount,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{TurnRole, TurnStatus};

    fn topic(id: &str, label: &str) -> TopicInfo {
        TopicInfo {
            topicId: id.to_string(),
            label: label.to_string(),
            startBlock: String::new(),
            blockCount: 0,
        }
    }

    fn turn(id: &str, blockId: &str, topicId: &str) -> Turn {
        Turn {
            id: id.to_string(),
            blockId: blockId.to_string(),
            topicId: topicId.to_string(),
            role: TurnRole::User,
            content: String::new(),
            ts: 0,
            parentId: None,
            tool: None,
            args: None,
            toolCallId: None,
            reasoning: None,
            attachments: None,
            cost: None,
            promptTokens: None,
            completionTokens: None,
            model: None,
            finishReason: None,
            snapshotHash: None,
            status: TurnStatus::Completed,
        }
    }

    #[test]
    fn restoreState_with_empty_clears_current_label() {
        let mut tracker = TopicTracker::new();
        tracker.restoreState(vec![topic("topic-01", "Real Label")]);
        assert_eq!(tracker.currentTopicId(), "topic-01");
        assert_eq!(tracker.currentLabel(), "Real Label");

        tracker.restoreState(Vec::new());
        assert_eq!(tracker.currentTopicId(), "", "ghost topicId must clear");
        assert_eq!(tracker.currentLabel(), "", "ghost label must clear");
        assert_eq!(tracker.nextTopicNum, 1);
    }

    #[test]
    fn restoreState_nextTopicNum_uses_max_id_not_length() {
        let mut tracker = TopicTracker::new();
        tracker.restoreState(vec![topic("topic-01", "A"), topic("topic-05", "E")]);
        assert_eq!(
            tracker.nextTopicNum, 6,
            "next must exceed max existing ID so a rewind can't recycle topic-02..topic-05"
        );
    }

    #[test]
    fn rebuildTopicInfos_drops_unknown_instead_of_labeling_unknown() {
        let turns = vec![
            turn("t1", "b_aaa", "topic-01"),
            turn("t2", "b_bbb", "topic-07"),
        ];
        let known = vec![topic("topic-01", "Known Label")];

        let rebuilt = rebuildTopicInfos(&turns, &known);
        assert_eq!(rebuilt.len(), 1, "orphaned topic-07 must be dropped");
        assert_eq!(rebuilt[0].topicId, "topic-01");
        assert_eq!(rebuilt[0].label, "Known Label");
        assert!(
            !rebuilt.iter().any(|t| t.label == "Unknown"),
            "literal 'Unknown' must never appear"
        );
    }

    #[test]
    fn rebuildTopicInfos_preserves_labels_across_rewind() {
        let turns = vec![
            turn("t1", "b_aaa", "topic-01"),
            turn("t2", "b_bbb", "topic-01"),
            turn("t3", "b_ccc", "topic-02"),
        ];
        let known = vec![
            topic("topic-01", "Research"),
            topic("topic-02", "Implementation"),
        ];

        let rebuilt = rebuildTopicInfos(&turns, &known);
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[0].label, "Research");
        assert_eq!(rebuilt[0].startBlock, "b_aaa");
        assert_eq!(rebuilt[0].blockCount, 2);
        assert_eq!(rebuilt[1].label, "Implementation");
        assert_eq!(rebuilt[1].startBlock, "b_ccc");
    }

    #[test]
    fn rebuildTopicInfos_skips_empty_topicId_turns() {
        // Pre-classification turns (first send) have empty topicId and
        // must not end up in the rebuilt list at all.
        let turns = vec![turn("t1", "b_aaa", ""), turn("t2", "b_aaa", "")];
        let rebuilt = rebuildTopicInfos(&turns, &[]);
        assert!(rebuilt.is_empty());
    }
}
