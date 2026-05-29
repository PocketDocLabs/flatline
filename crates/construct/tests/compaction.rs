#![allow(non_snake_case)]

//! Integration tests for compaction stage no-op boundaries.
//!
//! Stage eligibility details live in the unit tests beside each stage.
//! These integration tests deliberately avoid model calls so the default
//! test suite stays hermetic and fast.

use std::collections::{HashMap, HashSet};

use construct::api;
use construct::compaction::CompactionLog;
use construct::message::{Content, FunctionCall, Message, ToolCall};
use construct::s1;
use construct::s2;
use construct::s3;
use construct::s4;
use construct::transcript::Transcript;

/// Build a minimal Config with a bogus API key so `api::Client::new`
/// doesn't reject it.
fn dummyClient() -> api::Client {
    use construct::config::{BudgetConfig, Config, ModelConfig, WebConfig};
    use std::collections::BTreeMap;
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
        maxContextWindow: Some(100_000),
        supportsAnthropicCache: None,
    };
    let config = Config {
        heavyProfile: "test".into(),
        lightProfile: "test".into(),
        utilityProfile: "test".into(),
        heavy: model.clone(),
        light: model.clone(),
        utility: model.clone(),
        profiles: BTreeMap::from([("test".into(), model)]),
        compactRatio: 0.8,
        web: WebConfig::default(),
        lsp: HashMap::new(),
        permissions: None,
        budget: BudgetConfig::default(),
        projectRoot: None,
        launchDir: std::path::PathBuf::from("."),
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
        Message::System {
            content: "system".into(),
        },
        // Block 1: first readFile.
        Message::User {
            content: Content::text("read foo"),
        },
        readFileCall("tc_1", "/tmp/foo.rs"),
        toolResult("tc_1", "fn main() {}"),
        Message::Assistant {
            content: Some("got it".into()),
            tool_calls: None,
            reasoning: None,
        },
        // Block 2: second readFile (same path → dedup target).
        Message::User {
            content: Content::text("read foo again"),
        },
        readFileCall("tc_2", "/tmp/foo.rs"),
        toolResult("tc_2", "fn main() {}"),
        Message::Assistant {
            content: Some("ok".into()),
            tool_calls: None,
            reasoning: None,
        },
        // Block 3: long tool result (middle-out target).
        Message::User {
            content: Content::text("run it"),
        },
        readFileCall("tc_3", "/tmp/big.log"),
        toolResult("tc_3", &longContent),
        Message::Assistant {
            content: Some("done".into()),
            tool_calls: None,
            reasoning: None,
        },
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

#[tokio::test]
async fn test_s2_no_agent_blocks_is_noop_without_model_call() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut transcript = Transcript::createAt(dir.path(), "test_s2").unwrap();
    let headTurn = transcript.recordUser("user-only turn", None, None).unwrap();

    let compactionLog = CompactionLog::open(dir.path()).unwrap();
    let client = dummyClient();

    let result = s2::run(
        &transcript,
        &compactionLog,
        &headTurn,
        &client,
        "test-model",
        200_000,
        0.8,
    )
    .await
    .expect("S2 no-op should not error");

    assert!(!result.didWork);
    assert!(result.compacted.is_empty());
}

#[tokio::test]
async fn test_s3_no_topics_is_noop_without_model_call() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut transcript = Transcript::createAt(dir.path(), "test_s3").unwrap();
    let headTurn = transcript.recordUser("topicless turn", None, None).unwrap();

    let compactionLog = CompactionLog::open(dir.path()).unwrap();
    let client = dummyClient();

    let result = s3::run(
        &transcript,
        &compactionLog,
        &headTurn,
        &[],
        &client,
        "test-model",
        200_000,
        0.8,
    )
    .await
    .expect("S3 no-op should not error");

    assert!(!result.didWork);
    assert!(result.compacted.is_empty());
}

#[tokio::test]
async fn test_s4_no_inputs_is_noop_without_model_call() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut transcript = Transcript::createAt(dir.path(), "test_s4").unwrap();
    let head = transcript.recordUser("kickoff", None, None).unwrap();

    let compactionLog = CompactionLog::open(dir.path()).unwrap();
    let client = dummyClient();

    let result = s4::run(&transcript, &compactionLog, &head, &client, "test-model")
        .await
        .expect("S4 no-op should not error");

    assert!(!result.didWork);
    assert!(result.sourceBlockIds.is_empty());
}
