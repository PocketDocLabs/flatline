use std::fmt;

use super::SUBPROCESS_TIMEOUT_SECS;

// --- Subprocess helper ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tool) enum SubprocessError {
    NotFound {
        message: String,
    },
    Spawn {
        program: String,
        error: String,
    },
    Failed {
        program: String,
        status: String,
        output: String,
    },
    Run {
        program: String,
        error: String,
    },
    Timeout {
        program: String,
        seconds: u64,
    },
}

impl fmt::Display for SubprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubprocessError::NotFound { message } => f.write_str(message),
            SubprocessError::Spawn { program, error } => {
                write!(f, "Failed to start {program}: {error}")
            }
            SubprocessError::Failed {
                program,
                status,
                output,
            } => write!(f, "{program} failed (exit {status}): {}", output.trim()),
            SubprocessError::Run { program, error } => {
                write!(f, "Failed to run {program}: {error}")
            }
            SubprocessError::Timeout { program, seconds } => {
                write!(f, "{program} timed out after {seconds}s.")
            }
        }
    }
}

impl std::error::Error for SubprocessError {}

/// Run an external program, capture stdout+stderr, enforce timeout.
/// Returns Ok(stdout) on success or Err(message) on failure.
/// rg exit code 1 ("no matches") is treated as success with empty output.
pub(super) async fn runSubprocess(
    program: &str,
    args: &[&str],
    notFoundMsg: &str,
) -> std::result::Result<String, SubprocessError> {
    use tokio::process::Command;

    tracing::debug!(%program, args = ?args, "running subprocess");

    let result = Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let child = match result {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!(%program, "subprocess not found");
            return Err(SubprocessError::NotFound {
                message: notFoundMsg.to_string(),
            });
        }
        Err(e) => {
            tracing::warn!(%program, error = %e, "subprocess spawn failed");
            return Err(SubprocessError::Spawn {
                program: program.to_string(),
                error: e.to_string(),
            });
        }
    };

    let timeout = tokio::time::Duration::from_secs(SUBPROCESS_TIMEOUT_SECS);
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if output.status.success() || output.status.code() == Some(1) {
                // rg returns 1 for "no matches" — treat as success.
                Ok(stdout)
            } else {
                let msg = if stderr.is_empty() { &stdout } else { &stderr };
                tracing::warn!(%program, status = %output.status, error = %msg.trim(), "subprocess failed");
                Err(SubprocessError::Failed {
                    program: program.to_string(),
                    status: output.status.to_string(),
                    output: msg.to_string(),
                })
            }
        }
        Ok(Err(e)) => {
            tracing::warn!(%program, error = %e, "subprocess wait failed");
            Err(SubprocessError::Run {
                program: program.to_string(),
                error: e.to_string(),
            })
        }
        Err(_) => {
            // Process is still running but we lost ownership via wait_with_output.
            // The child is dropped here which sends SIGKILL on Unix.
            tracing::warn!(%program, secs = SUBPROCESS_TIMEOUT_SECS, "subprocess timed out");
            Err(SubprocessError::Timeout {
                program: program.to_string(),
                seconds: SUBPROCESS_TIMEOUT_SECS,
            })
        }
    }
}
