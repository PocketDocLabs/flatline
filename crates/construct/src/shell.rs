//! Stateful shell session backed by a PTY.
//!
//! Owns the shell process and provides command execution with
//! output capture. The harness (TUI or headless) receives raw
//! PTY output for display and can forward user keystrokes.
//!
//! Maintains a ring buffer of recent command results so the agent
//! can recall and search past output without re-running commands.
//!
//! # Public API
//! - [`Shell`] — command execution handle
//! - [`ShellIo`] — I/O channels for the harness
//! - [`CommandRecord`] — stored command result
//! - [`spawnShell`] — create a shell session
//!
//! # Dependencies
//! `portable-pty`, `tokio`

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use portable_pty::{CommandBuilder, PtyPair, PtySize, native_pty_system};
use tokio::sync::{mpsc, oneshot};
use tokio::task;

const MAX_HISTORY: usize = 50;
const MAX_SCROLLBACK_BYTES: usize = 512_000;

/// Default shell command timeout (seconds). Commands are interrupted after this.
pub const SHELL_DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Grace period between each escalation phase (Ctrl+C → Ctrl+\ → force-extract).
const SHELL_ESCALATION_SECS: u64 = 3;

/// A stored command result.
#[derive(Debug, Clone)]
pub struct CommandRecord {
    pub command: String,
    pub output: String,
    pub exitCode: Option<i32>,
    pub lineCount: usize,
}

/// Command execution handle — send commands, get output.
///
/// Held by the Session. Commands run in the stateful shell,
/// so `cd`, env vars, etc. persist across calls.
#[derive(Clone)]
pub struct Shell {
    cmdTx: mpsc::Sender<ShellRequest>,
    inputTx: mpsc::Sender<Vec<u8>>,
    history: Arc<Mutex<Vec<CommandRecord>>>,
    scrollback: Arc<Mutex<VecDeque<u8>>>,
}

impl Shell {
    /// Send Ctrl+C to the shell to interrupt any running command.
    pub fn interrupt(&self) {
        let _ = self.inputTx.try_send(vec![0x03]);
    }

    /// Execute a command in the shell and return captured output.
    ///
    /// If `timeout` is Some, the command is interrupted after that duration.
    /// If None, uses the default timeout (120s).
    pub async fn execute(&self, command: &str, timeout: Option<Duration>) -> String {
        let (tx, rx) = oneshot::channel();
        let req = ShellRequest {
            command: command.into(),
            timeout: Some(timeout.unwrap_or(Duration::from_secs(SHELL_DEFAULT_TIMEOUT_SECS))),
            respondTo: tx,
        };
        if self.cmdTx.send(req).await.is_err() {
            return "Shell closed.".into();
        }
        rx.await.unwrap_or_else(|_| "Shell disconnected.".into())
    }

    /// Number of stored command records.
    pub fn historyLen(&self) -> usize {
        self.history.lock().unwrap().len()
    }

    /// Get a command record by index (0-indexed, oldest first).
    pub fn getRecord(&self, index: usize) -> Option<CommandRecord> {
        self.history.lock().unwrap().get(index).cloned()
    }

    /// List all command records (index, command, exit code, line count).
    pub fn listHistory(&self) -> Vec<(usize, String, Option<i32>, usize)> {
        self.history
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.command.clone(), r.exitCode, r.lineCount))
            .collect()
    }

    /// Search a command's output for a pattern, returning matching lines with context.
    pub fn searchOutput(
        &self,
        index: usize,
        pattern: &str,
        contextLines: usize,
    ) -> Option<String> {
        let history = self.history.lock().unwrap();
        let record = history.get(index)?;

        let regex = regex::Regex::new(pattern).ok();
        let lines: Vec<&str> = record.output.lines().collect();
        let totalLines = lines.len();

        // Find matching line indices.
        let mut matchIndices: Vec<usize> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let matched = if let Some(ref re) = regex {
                re.is_match(line)
            } else {
                line.contains(pattern)
            };
            if matched {
                matchIndices.push(i);
            }
        }

        if matchIndices.is_empty() {
            return Some(format!("No matches for \"{pattern}\" in command {index} output."));
        }

        // Build context windows around matches, merging overlaps.
        let mut included = vec![false; totalLines];
        for &mi in &matchIndices {
            let start = mi.saturating_sub(contextLines);
            let end = (mi + contextLines + 1).min(totalLines);
            for j in start..end {
                included[j] = true;
            }
        }

        let mut output = String::new();
        let mut inBlock = false;
        for (i, line) in lines.iter().enumerate() {
            if included[i] {
                if !inBlock && !output.is_empty() {
                    output.push_str("  ...\n");
                }
                inBlock = true;
                let marker = if matchIndices.contains(&i) { ">" } else { " " };
                output.push_str(&format!("{marker}{:>5}\t{line}\n", i + 1));
            } else {
                inBlock = false;
            }
        }

        output.push_str(&format!(
            "\n{} matches in {} lines.",
            matchIndices.len(),
            totalLines
        ));

        Some(output)
    }

    /// Read the last N lines from the terminal scrollback.
    /// This includes everything visible in the terminal — user commands,
    /// agent commands, and all output.
    pub fn readTerminal(&self, lines: usize) -> String {
        let scrollback = self.scrollback.lock().unwrap();
        let raw = Vec::from(scrollback.clone());
        let text = stripAnsi(&String::from_utf8_lossy(&raw));

        let allLines: Vec<&str> = text.lines().collect();
        let start = allLines.len().saturating_sub(lines);
        allLines[start..].join("\n")
    }
}

/// I/O channels for the harness to display output and forward input.
pub struct ShellIo {
    /// PTY output for rendering in the terminal widget.
    pub outputRx: mpsc::Receiver<Vec<u8>>,
    /// User keystrokes to forward to the shell.
    pub inputTx: mpsc::Sender<Vec<u8>>,
    /// Resize events (cols, rows).
    pub resizeTx: mpsc::Sender<(u16, u16)>,
    /// Trigger the killchain for any active captured command.
    pub killTx: mpsc::Sender<()>,
}

/// Create a shell session.
///
/// Spawns a PTY with the user's default shell and a background
/// task that manages I/O, command execution, and output capture.
///
/// Args:
///     cols: Initial terminal width.
///     rows: Initial terminal height.
///
/// Returns:
///     (Shell, ShellIo): Command handle for the session, I/O handle for the harness.
pub fn spawnShell(cols: u16, rows: u16) -> Result<(Shell, ShellIo)> {
    let ptySystem = native_pty_system();
    let PtyPair { master, slave } = ptySystem
        .openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .context("Failed to open PTY")?;

    let shellBin = std::env::var("SHELL").unwrap_or_else(|_| "sh".into());
    let mut cmd = CommandBuilder::new(&shellBin);
    if let Ok(cwd) = std::env::current_dir() {
        cmd.cwd(cwd);
    }
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    cmd.env("GIT_PAGER", "cat");
    cmd.env("PAGER", "cat");

    // Inject OSC 133 shell integration for command boundary tracking.
    injectShellIntegration(&shellBin, &mut cmd)?;

    slave
        .spawn_command(cmd)
        .context("Failed to spawn shell")?;

    // Drop the slave side — master owns the connection now.
    drop(slave);

    let mut writer = master.take_writer().context("Failed to take PTY writer")?;
    let mut reader = master
        .try_clone_reader()
        .context("Failed to clone PTY reader")?;

    // PTY reader thread → ptyOutputTx (blocking I/O in a dedicated thread).
    let (ptyOutputTx, mut ptyOutputRx) = mpsc::channel::<Vec<u8>>(256);
    task::spawn_blocking(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if ptyOutputTx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Channels exposed to the harness.
    let (outputTx, outputRx) = mpsc::channel::<Vec<u8>>(256);
    let (inputTx, mut inputRx) = mpsc::channel::<Vec<u8>>(64);
    let (resizeTx, mut resizeRx) = mpsc::channel::<(u16, u16)>(4);
    let (cmdTx, mut cmdRx) = mpsc::channel::<ShellRequest>(4);
    let (killTx, mut killRx) = mpsc::channel::<()>(4);

    let history: Arc<Mutex<Vec<CommandRecord>>> = Arc::new(Mutex::new(Vec::new()));
    let historyRef = Arc::clone(&history);
    let scrollback: Arc<Mutex<VecDeque<u8>>> = Arc::new(Mutex::new(VecDeque::new()));
    let scrollbackRef = Arc::clone(&scrollback);

    // Background task — central hub for all PTY I/O.
    tokio::spawn(async move {
        let mut captureState: Option<CaptureState> = None;
        // Pending bytes to write to the PTY, drained in chunks so
        // large commands (heredocs) don't deadlock with output echo.
        let mut pendingWrite: VecDeque<u8> = VecDeque::new();

        // Append bytes to the scrollback ring buffer.
        let appendScrollback = |bytes: &[u8]| {
            let mut sb = scrollbackRef.lock().unwrap();
            sb.extend(bytes);
            // Trim from front if over capacity, on a line boundary.
            while sb.len() > MAX_SCROLLBACK_BYTES {
                match sb.iter().position(|&b| b == b'\n') {
                    Some(pos) => { sb.drain(..=pos); }
                    None => { sb.drain(..1024); }
                }
            }
        };

        /// Finalize a capture: extract output, store in history, resolve oneshot.
        fn finalizeCapture(
            cap: CaptureState,
            historyRef: &Arc<Mutex<Vec<CommandRecord>>>,
            timedOut: bool,
        ) -> (String, Vec<u8>) {
            let mut cap = cap;
            let remaining = cap.filter.flush();
            let result = extractResult(&cap.buffer, &cap.uuid);

            let mut hist = historyRef.lock().unwrap();
            hist.push(CommandRecord {
                command: cap.command.clone(),
                output: result.output.clone(),
                exitCode: result.exitCode,
                lineCount: result.output.lines().count(),
            });
            if hist.len() > MAX_HISTORY {
                hist.remove(0);
            }

            let mut response = if let Some(code) = result.exitCode {
                if code != 0 {
                    format!("{}\n(exit code: {code})", result.output)
                } else {
                    result.output
                }
            } else {
                result.output
            };

            if timedOut {
                response.push_str("\n(command timed out and was interrupted)");
            }

            let _ = cap.respondTo.send(response);
            (cap.uuid, remaining)
        }

        loop {
            // Compute deadline future from capture state.
            let deadline = captureState.as_ref().and_then(|c| c.deadline);

            tokio::select! {
                // PTY output — forward to harness, capture if active.
                Some(bytes) = ptyOutputRx.recv() => {
                    if let Some(ref mut cap) = captureState {
                        // Raw bytes go to capture buffer (for extractOutput).
                        cap.buffer.extend_from_slice(&bytes);

                        // Filtered bytes go to harness (no sentinels in display).
                        let filtered = cap.filter.process(&bytes);
                        if !filtered.is_empty() {
                            appendScrollback(&filtered);
                            let _ = outputTx.try_send(filtered);
                        }
                    } else {
                        appendScrollback(&bytes);
                        let _ = outputTx.try_send(bytes);
                    }

                    // Check for end marker. The echo can't contain the exact
                    // pattern (it has printf %s, not the uuid), so no \n prefix needed.
                    let done = if let Some(ref cap) = captureState {
                        let bufStr = String::from_utf8_lossy(&cap.buffer);
                        bufStr.contains(&format!("__FLATLINE_END_{}_", cap.uuid))
                    } else {
                        false
                    };

                    if done {
                        let cap = captureState.take().unwrap();
                        let (_, remaining) = finalizeCapture(cap, &historyRef, false);
                        if !remaining.is_empty() {
                            appendScrollback(&remaining);
                            let _ = outputTx.try_send(remaining);
                        }
                    }
                }

                // Timeout deadline — escalating kill chain.
                _ = async {
                    match deadline {
                        Some(d) => tokio::time::sleep_until(d).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    if let Some(ref mut cap) = captureState {
                        let nextDeadline = tokio::time::Instant::now()
                            + Duration::from_secs(SHELL_ESCALATION_SECS);

                        match cap.killPhase {
                            KillPhase::Running => {
                                // Phase 1: Ctrl+C (SIGINT).
                                tracing::info!(command = %cap.command, "shell command timed out, sending SIGINT");
                                let _ = writer.write_all(&[0x03]);
                                let _ = writer.flush();
                                cap.killPhase = KillPhase::SentInterrupt;
                                cap.deadline = Some(nextDeadline);
                            }
                            KillPhase::SentInterrupt => {
                                // Phase 2: Ctrl+\ (SIGQUIT) — kills most SIGINT-resistant processes.
                                tracing::warn!(command = %cap.command, "SIGINT ignored, sending SIGQUIT");
                                let _ = writer.write_all(&[0x1C]);
                                let _ = writer.flush();
                                cap.killPhase = KillPhase::SentQuit;
                                cap.deadline = Some(nextDeadline);
                            }
                            KillPhase::SentQuit => {
                                // Phase 3: give up — force-extract whatever we have.
                                tracing::error!(command = %cap.command, "kill chain exhausted, force-extracting output");
                                let cap = captureState.take().unwrap();
                                let (_, remaining) = finalizeCapture(cap, &historyRef, true);
                                if !remaining.is_empty() {
                                    appendScrollback(&remaining);
                                    let _ = outputTx.try_send(remaining);
                                }
                                // Poke the shell with a newline to try to recover the prompt.
                                let _ = writer.write_all(b"\n");
                                let _ = writer.flush();
                            }
                        }
                    }
                }

                // User keystrokes from harness.
                Some(data) = inputRx.recv() => {
                    let _ = writer.write_all(&data);
                    let _ = writer.flush();
                }

                // Chunked command write — drain pendingWrite in small
                // pieces so output processing can interleave, preventing
                // PTY buffer deadlocks on large commands.
                _ = std::future::ready(()), if !pendingWrite.is_empty() => {
                    const CHUNK: usize = 256;
                    let n = CHUNK.min(pendingWrite.len());
                    let chunk: Vec<u8> = pendingWrite.drain(..n).collect();
                    let _ = writer.write_all(&chunk);
                    let _ = writer.flush();
                }

                // Command execution request from session.
                Some(req) = cmdRx.recv() => {
                    let uuid = generateUuid();

                    // Echo the command in the terminal so the user sees what ran.
                    // Sent directly on outputTx, bypassing the PTY/filter.
                    // Dim cyan to distinguish agent commands from user input.
                    let cmdEcho = format!("\x1b[2;36m{}\x1b[0m\r\n", req.command);
                    let echoBytes = cmdEcho.into_bytes();
                    appendScrollback(&echoBytes);
                    let _ = outputTx.try_send(echoBytes);

                    // NOTE: Markers use printf with %s so the shell's command
                    // echo never contains the exact uuid pattern. This prevents
                    // PTY line-wrapping from creating false marker matches.
                    // The DisplayFilter mutes all output until the actual start
                    // marker appears, hiding the echoed command text.
                    //
                    // The exit-code capture goes on a NEW LINE after the command.
                    // A semicolon join would break heredocs — the shell needs the
                    // delimiter (e.g. EOF) on a line by itself.
                    let cmd = req.command.trim_end();
                    let wrapped = format!(
                        "printf '__FLATLINE_START_%s__\\n' '{uuid}'; printf '\\e]133;C\\a'; {cmd}\n__flatline_ec=$?; printf '\\e]133;D;%s\\a' \"$__flatline_ec\"; printf '\\n__FLATLINE_END_%s_%s__\\n' '{uuid}' \"$__flatline_ec\"\n",
                    );
                    // Queue for chunked write — large commands (heredocs)
                    // would deadlock if written in one blocking call.
                    pendingWrite.extend(wrapped.as_bytes());

                    let deadline = req.timeout.map(|d| tokio::time::Instant::now() + d);
                    captureState = Some(CaptureState {
                        uuid: uuid.clone(),
                        command: req.command,
                        buffer: Vec::new(),
                        filter: DisplayFilter::new(&uuid),
                        respondTo: req.respondTo,
                        deadline,
                        killPhase: KillPhase::Running,
                    });
                }

                // Resize from harness.
                Some((cols, rows)) = resizeRx.recv() => {
                    let _ = master.resize(PtySize {
                        rows,
                        cols,
                        pixel_width: 0,
                        pixel_height: 0,
                    });
                }

                // User-triggered kill — start/advance the killchain immediately.
                Some(()) = killRx.recv() => {
                    if let Some(ref mut cap) = captureState {
                        cap.deadline = Some(tokio::time::Instant::now());
                    } else {
                        // No active capture — just forward Ctrl+C.
                        let _ = writer.write_all(&[0x03]);
                        let _ = writer.flush();
                    }
                }

                // All channels closed — shut down.
                else => break,
            }
        }
    });

    let shell = Shell {
        cmdTx,
        inputTx: inputTx.clone(),
        history,
        scrollback,
    };
    let io = ShellIo {
        outputRx,
        inputTx,
        resizeTx,
        killTx,
    };

    Ok((shell, io))
}

// --- Internal types ---

struct ShellRequest {
    command: String,
    timeout: Option<Duration>,
    respondTo: oneshot::Sender<String>,
}

/// Escalation phase for timed-out commands.
enum KillPhase {
    /// No timeout yet — command is still running normally.
    Running,
    /// Ctrl+C (SIGINT) sent, waiting for process to exit.
    SentInterrupt,
    /// Ctrl+\ (SIGQUIT) sent — last resort before force-extract.
    SentQuit,
}

struct CaptureState {
    uuid: String,
    command: String,
    buffer: Vec<u8>,
    filter: DisplayFilter,
    respondTo: oneshot::Sender<String>,
    /// Absolute deadline for the next escalation phase.
    deadline: Option<tokio::time::Instant>,
    /// Current kill escalation phase.
    killPhase: KillPhase,
}

/// Extracted result from the capture buffer.
struct ExtractedResult {
    output: String,
    exitCode: Option<i32>,
}

/// Line-based filter that suppresses sentinel markers and the shell's
/// command echo from display output.
///
/// Buffers partial lines until a newline arrives. All output is muted
/// until the actual start marker appears (suppressing the PTY echo of
/// the wrapped command). After the start marker, sentinel lines are
/// dropped and everything else passes through to the harness.
struct DisplayFilter {
    lineBuf: Vec<u8>,
    startPattern: String,
    endPattern: String,
    /// Whether the actual start marker output has been seen.
    /// Before this, all lines are suppressed (they're command echo).
    seenStart: bool,
}

impl DisplayFilter {
    fn new(uuid: &str) -> Self {
        Self {
            lineBuf: Vec::new(),
            startPattern: format!("__FLATLINE_START_{uuid}"),
            endPattern: format!("__FLATLINE_END_{uuid}"),
            seenStart: false,
        }
    }

    /// Process a chunk of bytes, returning only command output lines.
    fn process(&mut self, bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        for &b in bytes {
            self.lineBuf.push(b);
            if b == b'\n' {
                let line = String::from_utf8_lossy(&self.lineBuf);
                if !self.seenStart {
                    // Muted phase: suppress everything until start marker.
                    if line.contains(&self.startPattern) {
                        self.seenStart = true;
                    }
                } else if !line.contains(&self.startPattern)
                    && !line.contains(&self.endPattern)
                    && !line.contains("__flatline_ec")
                {
                    output.extend_from_slice(&self.lineBuf);
                }
                self.lineBuf.clear();
            }
        }
        output
    }

    /// Flush any remaining partial line (when capture ends).
    fn flush(&mut self) -> Vec<u8> {
        let line = String::from_utf8_lossy(&self.lineBuf);
        if !self.seenStart || line.contains("__FLATLINE_") {
            self.lineBuf.clear();
            Vec::new()
        } else {
            std::mem::take(&mut self.lineBuf)
        }
    }
}

// --- Output extraction ---

/// Extract command output and exit code from the capture buffer.
///
/// The buffer contains raw PTY output with command echoes, ANSI codes,
/// and sentinel markers. The wrapped command uses printf with %s for the
/// uuid, so the echo never contains the exact marker strings — only the
/// actual output does. This makes rfind unambiguous.
fn extractResult(buffer: &[u8], uuid: &str) -> ExtractedResult {
    let text = String::from_utf8_lossy(buffer);
    let startMarker = format!("__FLATLINE_START_{uuid}__");
    let endPrefix = format!("__FLATLINE_END_{uuid}_");

    let afterStart = match text.rfind(&startMarker) {
        Some(pos) => {
            let rest = &text[pos + startMarker.len()..];
            match rest.find('\n') {
                Some(nl) => &rest[nl + 1..],
                None => rest,
            }
        }
        None => {
            return ExtractedResult {
                output: stripAnsi(&text),
                exitCode: None,
            };
        }
    };

    let output = match afterStart.find(&endPrefix) {
        Some(pos) => afterStart[..pos]
            .trim_end_matches('\n')
            .trim_end_matches('\r'),
        None => afterStart,
    };

    // Strip ANSI escapes and filter out echoed sentinel machinery.
    let cleaned: String = stripAnsi(output)
        .lines()
        .filter(|l| !l.contains("__flatline_ec"))
        .collect::<Vec<_>>()
        .join("\n");

    // Extract exit code from end marker.
    let exitCode = text.rfind(&endPrefix).and_then(|endPos| {
        let afterPrefix = &text[endPos + endPrefix.len()..];
        afterPrefix.split("__").next()?.parse::<i32>().ok()
    });

    ExtractedResult {
        output: cleaned,
        exitCode,
    }
}

/// Strip ANSI escape sequences from text.
fn stripAnsi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() {
            match bytes[i + 1] {
                // CSI sequence: ESC [ ... (letter or ~).
                b'[' => {
                    i += 2;
                    while i < bytes.len() && !bytes[i].is_ascii_alphabetic() && bytes[i] != b'~' {
                        i += 1;
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                // OSC sequence: ESC ] ... (BEL or ST).
                b']' => {
                    i += 2;
                    while i < bytes.len() {
                        if bytes[i] == 0x07 {
                            i += 1;
                            break;
                        }
                        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                            i += 2;
                            break;
                        }
                        i += 1;
                    }
                }
                // Other escape: ESC + single char.
                _ => {
                    i += 2;
                }
            }
        } else if bytes[i] == b'\r' {
            // Skip carriage returns (PTY uses \r\n).
            i += 1;
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }

    result
}

/// Inject OSC 133 shell integration so the shell emits command boundary markers.
///
/// For zsh: overrides ZDOTDIR with a temp directory containing init files
/// that source the originals and add precmd/preexec hooks.
/// For bash: uses --rcfile pointing to a wrapper that sources .bashrc first.
fn injectShellIntegration(shell: &str, cmd: &mut CommandBuilder) -> Result<()> {
    let integrationDir =
        std::env::temp_dir().join(format!("flatline-si-{}", std::process::id()));
    std::fs::create_dir_all(&integrationDir)?;

    if shell.ends_with("zsh") {
        let originalZdotdir = std::env::var("ZDOTDIR")
            .unwrap_or_else(|_| std::env::var("HOME").unwrap_or_default());

        // Forward all zsh init files to originals.
        for file in &[".zshenv", ".zprofile", ".zlogin"] {
            let content = format!(
                "[[ -f \"{originalZdotdir}/{file}\" ]] && source \"{originalZdotdir}/{file}\"\n"
            );
            std::fs::write(integrationDir.join(file), content)?;
        }

        // .zshrc: source original, restore ZDOTDIR, add hooks.
        let zshrc = format!(
            r#"[[ -f "{originalZdotdir}/.zshrc" ]] && source "{originalZdotdir}/.zshrc"
ZDOTDIR="{originalZdotdir}"
flatline_precmd() {{ printf '\e]133;D;%s\a\e]133;A\a' "$?" }}
flatline_preexec() {{ printf '\e]133;C\a' }}
precmd_functions+=(flatline_precmd)
preexec_functions+=(flatline_preexec)
"#
        );
        std::fs::write(integrationDir.join(".zshrc"), zshrc)?;
        cmd.env("ZDOTDIR", integrationDir.to_str().unwrap_or_default());
    } else if shell.ends_with("bash") {
        let bashrc = r#"[[ -f "$HOME/.bashrc" ]] && source "$HOME/.bashrc"
flatline_prompt_command() { printf '\033]133;D;%s\a\033]133;A\a' "$?"; }
PROMPT_COMMAND="flatline_prompt_command${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
trap 'printf "\033]133;C\a"' DEBUG
"#;
        let bashrcPath = integrationDir.join("flatline_bashrc");
        std::fs::write(&bashrcPath, bashrc)?;
        cmd.arg("--rcfile");
        cmd.arg(bashrcPath.to_str().unwrap_or_default());
    }

    Ok(())
}

/// Generate a short pseudo-random ID for sentinel markers.
fn generateUuid() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let pid = std::process::id();
    format!("{pid:x}{nanos:x}")
}
