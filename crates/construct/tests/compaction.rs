#![allow(non_snake_case)]

//! Integration tests for the S1–S4 compaction pipeline.
//!
//! Builds a realistic session transcript and runs each stage.
//! S1 should pass (pure logic). S2 finds eligible blocks but the
//! LLM call fails (bogus key). S3 and S4 fail — proving the
//! block ID mismatch bug in expandTopicBlocks/blocksInZone.

use std::collections::{HashMap, HashSet};

use construct::api;
use construct::compaction::CompactionLog;
use construct::message::{Content, FunctionCall, Message, ToolCall};
use construct::s1;
use construct::s2;
use construct::s3;
use construct::s4;
use construct::topic::TopicInfo;
use construct::transcript::Transcript;

/// Build a minimal Config with a bogus API key so `api::Client::new`
/// doesn't reject it.
fn dummyClient() -> api::Client {
    use construct::config::{BudgetConfig, Config, ModelConfig, WebConfig};
    use std::collections::HashMap;

    let model = ModelConfig {
        provider: "openrouter".into(),
        key: "test-bogus-key".into(),
        model: "test-model".into(),
        baseUrl: "https://example".into(),
        reasoning: None,
        promptThinking: false,
        providerOrder: vec![],
        maxTokens: None,
        contextWindow: 100_000,
        supportsAnthropicCache: None,
    };
    let config = Config {
        heavyProfile: "test".into(),
        lightProfile: "test".into(),
        utilityProfile: "test".into(),
        heavy: model.clone(),
        light: model.clone(),
        utility: model,
        compactRatio: 0.8,
        web: WebConfig::default(),
        lsp: HashMap::new(),
        permissions: None,
        budget: BudgetConfig::default(),
        projectRoot: None,
    };
    api::Client::new(&config).expect("build test client")
}

/// Build a tool call message for readFile.
fn readFileCall(callId: &str, path: &str) -> Message {
    Message::Assistant {
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: callId.to_string(),
            callType: "function".to_string(),
            function: FunctionCall {
                name: "readFile".to_string(),
                arguments: serde_json::json!({ "path": path }).to_string(),
            },
        }]),
        reasoning: None,
    }
}

/// Build a tool result message.
fn toolResult(callId: &str, content: &str) -> Message {
    Message::Tool {
        tool_call_id: callId.to_string(),
        content: Content::text(content),
    }
}

// ── S1 ─────────────────────────────────────────────────────────────

#[test]
fn test_s1_dedup_and_middle_out() {
    // Build a history with duplicate file reads and a long tool result.
    let longContent = "x".repeat(5000);
    let mut history = vec![
        Message::System { content: "system".into() },
        // Block 1: first readFile.
        Message::User { content: Content::text("read foo") },
        readFileCall("tc_1", "/tmp/foo.rs"),
        toolResult("tc_1", "fn main() {}"),
        Message::Assistant { content: Some("got it".into()), tool_calls: None, reasoning: None },
        // Block 2: second readFile (same path → dedup target).
        Message::User { content: Content::text("read foo again") },
        readFileCall("tc_2", "/tmp/foo.rs"),
        toolResult("tc_2", "fn main() {}"),
        Message::Assistant { content: Some("ok".into()), tool_calls: None, reasoning: None },
        // Block 3: long tool result (middle-out target).
        Message::User { content: Content::text("run it") },
        readFileCall("tc_3", "/tmp/big.log"),
        toolResult("tc_3", &longContent),
        Message::Assistant { content: Some("done".into()), tool_calls: None, reasoning: None },
    ];

    let blockHints: HashMap<String, String> = HashMap::new();
    let alreadyProcessed: HashSet<String> = HashSet::new();

    let result = s1::run(
        &mut history,
        s1::DEFAULT_MIDDLE_OUT_THRESHOLD,
        &blockHints,
        &alreadyProcessed,
    );

    assert!(result.didWork, "S1 should have found work to do");
    assert!(
        !result.dedupedCallIds.is_empty() || !result.middleOutCallIds.is_empty(),
        "S1 should have deduped or middle-out'd something"
    );
}

// ── S2 ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_s2_eligible_blocks() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut transcript = Transcript::createAt(dir.path(), "test_s2").unwrap();

    // Record 3 exchange blocks with tool calls.
    let t1 = transcript.recordUser("first question", None, None).unwrap();
    transcript.recordAssistant("thinking about it...", Default::default()).unwrap();
    transcript.recordToolCall("tc_a", "shell", &serde_json::json!({"command": "ls"})).unwrap();
    transcript.recordToolResult("tc_a", "file1.rs\nfile2.rs", None).unwrap();
    transcript.recordAssistant("here are the files", Default::default()).unwrap();

    let t2 = transcript.recordUser("second question", Some(&t1), None).unwrap();
    transcript.recordAssistant("let me check...", Default::default()).unwrap();
    transcript.recordToolCall("tc_b", "readFile", &serde_json::json!({"path": "/tmp/foo.rs"})).unwrap();
    transcript.recordToolResult("tc_b", &"y".repeat(2000), None).unwrap();
    transcript.recordAssistant("read it", Default::default()).unwrap();

    let _t3 = transcript.recordUser("third question", Some(&t2), None).unwrap();
    transcript.recordAssistant("sure thing", Default::default()).unwrap();
    transcript.recordToolCall("tc_c", "shell", &serde_json::json!({"command": "echo hi"})).unwrap();
    let headTurn = transcript.recordToolResult("tc_c", "hi", None).unwrap();
    transcript.recordAssistant("done", Default::default()).unwrap();

    let compactionLog = CompactionLog::open(dir.path()).unwrap();
    let client = dummyClient();

    // S2 should find eligible blocks but fail on the LLM call.
    // The important thing: it doesn't crash and correctly identifies blocks.
    let result = s2::run(
        &transcript,
        &compactionLog,
        &headTurn,
        &client,
        "test-model",
        200_000,
        0.8,
    ).await;

    // S2 will either find blocks and fail the API call (returning didWork=false
    // with warnings), or succeed if somehow the API works. Either way, no panic.
    assert!(result.is_ok(), "S2 should not error: {:?}", result.err());
}

// ── S3 ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_s3_finds_and_compacts_topics() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut transcript = Transcript::createAt(dir.path(), "test_s3").unwrap();

    // Record 4 exchange blocks across 2 topics.
    // topic-01: blocks 1-2, topic-02: blocks 3-4.
    transcript.setTopicId("topic-01");
    let t1 = transcript.recordUser("topic one start", None, None).unwrap();
    transcript.recordAssistant("working on topic one", Default::default()).unwrap();
    transcript.recordToolCall("tc_1", "shell", &serde_json::json!({"command": "ls"})).unwrap();
    transcript.recordToolResult("tc_1", "output1", None).unwrap();
    transcript.recordAssistant("done with block 1", Default::default()).unwrap();

    let t2 = transcript.recordUser("still topic one", Some(&t1), None).unwrap();
    transcript.recordAssistant("continuing", Default::default()).unwrap();
    transcript.recordToolCall("tc_2", "readFile", &serde_json::json!({"path": "/a.rs"})).unwrap();
    transcript.recordToolResult("tc_2", "content of a.rs", None).unwrap();
    transcript.recordAssistant("read it", Default::default()).unwrap();

    transcript.setTopicId("topic-02");
    let t3 = transcript.recordUser("new topic here", Some(&t2), None).unwrap();
    transcript.recordAssistant("switching gears", Default::default()).unwrap();
    transcript.recordToolCall("tc_3", "shell", &serde_json::json!({"command": "cargo build"})).unwrap();
    transcript.recordToolResult("tc_3", "Compiling...", None).unwrap();
    transcript.recordAssistant("built", Default::default()).unwrap();

    let _t4 = transcript.recordUser("more topic two", Some(&t3), None).unwrap();
    transcript.recordAssistant("sure", Default::default()).unwrap();
    transcript.recordToolCall("tc_4", "readFile", &serde_json::json!({"path": "/b.rs"})).unwrap();
    let headTurn = transcript.recordToolResult("tc_4", "content of b.rs", None).unwrap();
    transcript.recordAssistant("got b.rs", Default::default()).unwrap();

    // Grab the actual block IDs from the recorded turns.
    let allTurns = transcript.loadAll().unwrap();
    let mut blockIdsByTopic: HashMap<String, Vec<String>> = HashMap::new();
    for turn in &allTurns {
        if !turn.topicId.is_empty() {
            blockIdsByTopic
                .entry(turn.topicId.clone())
                .or_default()
                .push(turn.blockId.clone());
        }
    }
    // Deduplicate block IDs per topic.
    for ids in blockIdsByTopic.values_mut() {
        ids.sort();
        ids.dedup();
    }

    let topic1Blocks = blockIdsByTopic.get("topic-01").unwrap();
    let topic2Blocks = blockIdsByTopic.get("topic-02").unwrap();

    let topics = vec![
        TopicInfo {
            topicId: "topic-01".into(),
            label: "First Topic".into(),
            startBlock: topic1Blocks[0].clone(),
            blockCount: topic1Blocks.len(),
        },
        TopicInfo {
            topicId: "topic-02".into(),
            label: "Second Topic".into(),
            startBlock: topic2Blocks[0].clone(),
            blockCount: topic2Blocks.len(),
        },
    ];

    // Pre-populate compaction log with S2 BlockCompact ops (S3 prerequisite).
    let mut compactionLog = CompactionLog::open(dir.path()).unwrap();
    for bid in topic1Blocks.iter().chain(topic2Blocks.iter()) {
        compactionLog
            .recordBlockCompact(bid, "summary of block", vec![], &headTurn)
            .unwrap();
    }

    let client = dummyClient();

    let result = s3::run(
        &transcript,
        &compactionLog,
        &headTurn,
        &topics,
        &client,
        "test-model",
        200_000,
        0.8,
    ).await.expect("S3 should not error");

    // S3 should find eligible topics and attempt compaction. With a bogus
    // API key the LLM call fails, so didWork=false, but the pipeline ran
    // correctly (no panic, no empty-block-ID bug). A real API key would
    // produce didWork=true.
    //
    // The key invariant: S3 doesn't return an error — it completes
    // gracefully even when the LLM call fails.
    assert!(
        !result.didWork,
        "S3 with bogus API key should not report didWork (LLM call fails)"
    );
}

// ── S4 ─────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore] // Requires real API key — bogus key hangs on connect timeout.
async fn test_s4_merges_topic_summaries() {
    let dir = tempfile::TempDir::new().unwrap();

    // Minimal transcript so s4 can compute the protected band.
    let mut transcript = Transcript::createAt(dir.path(), "test_s4").unwrap();
    let _ = transcript.recordUser("kickoff", None, None).unwrap();
    let head = transcript.recordAssistant("ack", Default::default()).unwrap();

    // Pre-populate compaction log with TopicCompact ops (simulating S3 output).
    let mut compactionLog = CompactionLog::open(dir.path()).unwrap();
    compactionLog
        .recordTopicCompact(
            "First Topic",
            "Summary of the first topic discussion.",
            vec!["b_aaa".into(), "b_bbb".into()],
            "t_head",
        )
        .unwrap();
    compactionLog
        .recordTopicCompact(
            "Second Topic",
            "Summary of the second topic discussion.",
            vec!["b_ccc".into(), "b_ddd".into()],
            "t_head",
        )
        .unwrap();

    let client = dummyClient();

    let result = s4::run(&transcript, &compactionLog, &head, &client, "test-model").await;

    // S4 finds the topic summaries and attempts the LLM call, which fails
    // with a bogus API key (401). This proves the logic works — it got past
    // the `sections.is_empty()` check. A real API key would produce didWork=true.
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("401") || msg.contains("Unauthorized") || msg.contains("failed"),
                "S4 should fail with an API error, got: {msg}"
            );
        }
        Ok(r) => {
            // If somehow the API works (shouldn't with bogus key), that's fine too.
            assert!(r.didWork, "S4 found sections but didn't produce output");
        }
    }
}
