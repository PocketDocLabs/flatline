#![allow(non_snake_case)]

//! End-to-end integration tests for `flatline export`.
//!
//! Builds synthetic `sessions/{id}/` directories in tempdirs, points
//! `FLATLINE_SESSIONS_DIR` at them, and invokes the `flatline` binary
//! directly so the CLI surface is exercised as a user would use it.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `flatline` binary built for this test.
fn flatlineBin() -> PathBuf {
    let exe = env!("CARGO_BIN_EXE_flatline");
    PathBuf::from(exe)
}

/// Build a minimal fixture session on disk.
///
/// One user message, one assistant response, one snapshot. Stable prefix,
/// so export should produce exactly one example.
fn writeLinearFixture(dir: &Path, sessionId: &str) {
    let session = dir.join(sessionId);
    fs::create_dir_all(session.join("snapshots/blobs/sp")).unwrap();
    fs::create_dir_all(session.join("snapshots/blobs/tl")).unwrap();
    fs::create_dir_all(session.join("snapshots/blobs/ms")).unwrap();

    // meta.json
    let meta = serde_json::json!({
        "sessionId": sessionId,
        "projectDir": "",
        "createdAt": 0,
        "updatedAt": 0,
        "name": null,
        "topicLabels": [],
        "headTurn": "t_asst_1",
        "forks": [],
        "totalCost": 0.0,
    });
    fs::write(
        session.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    // Blobs (contents match the hashes referenced in the snapshot).
    let sysText = "You are a helpful assistant.";
    let sysHash = sha1(sysText.as_bytes());
    fs::write(
        session
            .join("snapshots/blobs/sp")
            .join(format!("{sysHash}.txt")),
        sysText,
    )
    .unwrap();

    let tools = serde_json::json!([]);
    let toolsText = serde_json::to_string(&tools).unwrap();
    let toolsHash = sha1(toolsText.as_bytes());
    fs::write(
        session
            .join("snapshots/blobs/tl")
            .join(format!("{toolsHash}.json")),
        &toolsText,
    )
    .unwrap();

    let userMsg = serde_json::json!({ "role": "user", "content": "hello" });
    let userText = serde_json::to_string(&userMsg).unwrap();
    let userHash = sha1(userText.as_bytes());
    fs::write(
        session
            .join("snapshots/blobs/ms")
            .join(format!("{userHash}.json")),
        &userText,
    )
    .unwrap();

    // Snapshot index with one entry. The snapshot hash itself isn't validated
    // by export, so we fabricate a stable one.
    let snap = serde_json::json!({
        "v": 1,
        "model": "test/model",
        "provider": "openrouter",
        "baseUrl": "https://x",
        "toolsCount": 0,
        "systemPromptHash": sysHash,
        "toolsHash": toolsHash,
        "messages": [userHash],
        "ts": 0,
    });
    let snapHash = "snap000000000000000000000000000000000001";
    let entry = serde_json::json!({ "hash": snapHash, "snapshot": snap });
    fs::write(
        session.join("snapshots/index.jsonl"),
        format!("{}\n", serde_json::to_string(&entry).unwrap()),
    )
    .unwrap();

    // Transcript — user + assistant.
    let userTurn = serde_json::json!({
        "id": "t_user_1",
        "blockId": "b_1",
        "topicId": "",
        "role": "user",
        "content": "hello",
        "ts": 0,
    });
    let asstTurn = serde_json::json!({
        "id": "t_asst_1",
        "blockId": "b_1",
        "topicId": "",
        "role": "assistant",
        "content": "hi there",
        "ts": 0,
        "parentId": "t_user_1",
        "snapshotHash": snapHash,
    });
    fs::write(
        session.join("transcript.jsonl"),
        format!(
            "{}\n{}\n",
            serde_json::to_string(&userTurn).unwrap(),
            serde_json::to_string(&asstTurn).unwrap()
        ),
    )
    .unwrap();

    // Empty compaction log.
    fs::write(session.join("compaction.jsonl"), "").unwrap();
}

fn sha1(bytes: &[u8]) -> String {
    let d = sha1_smol::Sha1::from(bytes).digest();
    d.bytes().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn exportLinearSessionEmitsOneExample() {
    let dir = tempfile::TempDir::new().unwrap();
    let sessionId = "ses_linear_test";
    writeLinearFixture(dir.path(), sessionId);

    let out = dir.path().join("out.json");
    let status = Command::new(flatlineBin())
        .env("FLATLINE_SESSIONS_DIR", dir.path())
        .args(["export", sessionId, "-o"])
        .arg(&out)
        .status()
        .expect("spawn flatline export");

    assert!(status.success(), "export failed: {:?}", status);
    let rendered = fs::read_to_string(&out).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    let arr = parsed.as_array().expect("top-level must be an array");
    assert_eq!(arr.len(), 1, "expected exactly one example");

    let ex = &arr[0];
    let msgs = ex["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["content"], "hi there");
    assert_eq!(ex["_flatline"]["sessionId"], sessionId);
    assert_eq!(ex["_flatline"]["segmentIndex"], 0);
}

#[test]
fn exportAllMergesSessions() {
    let dir = tempfile::TempDir::new().unwrap();
    writeLinearFixture(dir.path(), "ses_a");
    writeLinearFixture(dir.path(), "ses_b");
    // A third session without snapshots — should be silently skipped.
    let noSnaps = dir.path().join("ses_legacy");
    fs::create_dir_all(&noSnaps).unwrap();
    fs::write(noSnaps.join("transcript.jsonl"), "").unwrap();

    let out = dir.path().join("all.json");
    let status = Command::new(flatlineBin())
        .env("FLATLINE_SESSIONS_DIR", dir.path())
        .args(["export", "--all", "-o"])
        .arg(&out)
        .status()
        .expect("spawn flatline export --all");

    assert!(status.success(), "export --all failed: {:?}", status);
    let rendered = fs::read_to_string(&out).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    let arr = parsed.as_array().expect("top-level must be an array");
    assert_eq!(arr.len(), 2, "expected one example per snapshotted session");

    // Both session IDs should appear in provenance.
    let ids: Vec<&str> = arr
        .iter()
        .map(|ex| ex["_flatline"]["sessionId"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"ses_a"));
    assert!(ids.contains(&"ses_b"));
}

#[test]
fn exportRejectsSessionWithoutSnapshots() {
    let dir = tempfile::TempDir::new().unwrap();
    let sessionId = "ses_no_snaps";
    let session = dir.path().join(sessionId);
    fs::create_dir_all(&session).unwrap();
    fs::write(session.join("transcript.jsonl"), "").unwrap();
    fs::write(
        session.join("meta.json"),
        r#"{"sessionId":"ses_no_snaps","projectDir":"","createdAt":0,"updatedAt":0,"name":null,"topicLabels":[],"forks":[],"totalCost":0.0}"#,
    )
    .unwrap();

    let status = Command::new(flatlineBin())
        .env("FLATLINE_SESSIONS_DIR", dir.path())
        .args(["export", sessionId])
        .status()
        .expect("spawn flatline export");

    // Exit code 2 — session predates snapshot feature.
    assert_eq!(status.code(), Some(2));
}
