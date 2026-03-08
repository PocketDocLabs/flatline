//! Stateful shell session backed by a PTY.
//!
//! Owns the shell process and provides command execution with
//! output capture. The harness (TUI or headless) receives raw
//! PTY output for display and can forward user keystrokes.
//!
//! # Public API
//! - [`Shell`] — command execution handle
//! - [`ShellIo`] — I/O channels for the harness
//! - [`spawnShell`] — create a shell session
//!
//! # Dependencies
//! `portable-pty`, `tokio`

use std::io::{Read, Write};

use anyhow::{Context, Result};
use portable_pty::{CommandBuilder, PtyPair, PtySize, native_pty_system};
use tokio::sync::{mpsc, oneshot};
use tokio::task;

/// Command execution handle — send commands, get output.
///
/// Held by the Session. Commands run in the stateful shell,
/// so `cd`, env vars, etc. persist across calls.
pub struct Shell {
    cmdTx: mpsc::Sender<ShellRequest>,
}

impl Shell {
    /// Execute a command in the shell and return captured output.
    pub async fn execute(&self, command: &str) -> String {
        let (tx, rx) = oneshot::channel();
        let req = ShellRequest {
            command: command.into(),
            respondTo: tx,
        };
        if self.cmdTx.send(req).await.is_err() {
            return "Shell closed.".into();
        }
        rx.await.unwrap_or_else(|_| "Shell disconnected.".into())
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

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "sh".into());
    let mut cmd = CommandBuilder::new(&shell);
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");

    // Inject OSC 133 shell integration for command boundary tracking.
    injectShellIntegration(&shell, &mut cmd)?;

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

    // Background task — central hub for all PTY I/O.
    tokio::spawn(async move {
        let mut captureState: Option<CaptureState> = None;

        loop {
            tokio::select! {
                // PTY output — forward to harness, capture if active.
                Some(bytes) = ptyOutputRx.recv() => {
                    if let Some(ref mut cap) = captureState {
                        // Raw bytes go to capture buffer (for extractOutput).
                        cap.buffer.extend_from_slice(&bytes);

                        // Filtered bytes go to harness (no sentinels in display).
                        let filtered = cap.filter.process(&bytes);
                        if !filtered.is_empty() {
                            let _ = outputTx.send(filtered).await;
                        }
                    } else {
                        let _ = outputTx.send(bytes).await;
                    }

                    // Check for end marker (separate borrow scope).
                    let done = if let Some(ref cap) = captureState {
                        let bufStr = String::from_utf8_lossy(&cap.buffer);
                        bufStr.contains(&format!("__FLATLINE_END_{}__", cap.uuid))
                            || bufStr.contains(&format!("__FLATLINE_END_{}_", cap.uuid))
                    } else {
                        false
                    };

                    if done {
                        let mut cap = captureState.take().unwrap();
                        let remaining = cap.filter.flush();
                        if !remaining.is_empty() {
                            let _ = outputTx.send(remaining).await;
                        }
                        let output = extractOutput(&cap.buffer, &cap.uuid);
                        let _ = cap.respondTo.send(output);
                    }
                }

                // User keystrokes from harness.
                Some(data) = inputRx.recv() => {
                    let _ = writer.write_all(&data);
                    let _ = writer.flush();
                }

                // Command execution request from session.
                Some(req) = cmdRx.recv() => {
                    let uuid = generateUuid();
                    // Emit OSC 133 C/D markers so the terminal widget can track
                    // command output regions for entity selection (quad-click).
                    let wrapped = format!(
                        "echo __FLATLINE_START_{uuid}__; printf '\\e]133;C\\a'; {}; __flatline_ec=$?; printf '\\e]133;D;%s\\a' \"$__flatline_ec\"; echo __FLATLINE_END_{uuid}_${{__flatline_ec}}__\n",
                        req.command,
                    );
                    let _ = writer.write_all(wrapped.as_bytes());
                    let _ = writer.flush();
                    captureState = Some(CaptureState {
                        uuid: uuid.clone(),
                        buffer: Vec::new(),
                        filter: DisplayFilter::new(&uuid),
                        respondTo: req.respondTo,
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

                // All channels closed — shut down.
                else => break,
            }
        }
    });

    let shell = Shell { cmdTx };
    let io = ShellIo {
        outputRx,
        inputTx,
        resizeTx,
    };

    Ok((shell, io))
}

// --- Internal types ---

struct ShellRequest {
    command: String,
    respondTo: oneshot::Sender<String>,
}

struct CaptureState {
    uuid: String,
    buffer: Vec<u8>,
    filter: DisplayFilter,
    respondTo: oneshot::Sender<String>,
}

/// Line-based filter that suppresses sentinel markers from display output.
///
/// Buffers partial lines until a newline arrives, then checks if the
/// complete line contains a sentinel pattern. Matching lines are dropped;
/// everything else passes through to the harness.
struct DisplayFilter {
    lineBuf: Vec<u8>,
    startPattern: String,
    endPattern: String,
}

impl DisplayFilter {
    fn new(uuid: &str) -> Self {
        Self {
            lineBuf: Vec::new(),
            startPattern: format!("__FLATLINE_START_{uuid}"),
            endPattern: format!("__FLATLINE_END_{uuid}"),
        }
    }

    /// Process a chunk of bytes, returning only non-sentinel lines.
    fn process(&mut self, bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        for &b in bytes {
            self.lineBuf.push(b);
            if b == b'\n' {
                let line = String::from_utf8_lossy(&self.lineBuf);
                if !line.contains(&self.startPattern) && !line.contains(&self.endPattern) {
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
        if line.contains("__FLATLINE_") {
            self.lineBuf.clear();
            Vec::new()
        } else {
            std::mem::take(&mut self.lineBuf)
        }
    }
}

// --- Output extraction ---

/// Extract command output from the capture buffer using sentinel markers.
///
/// The buffer contains raw PTY output with command echoes, ANSI codes,
/// and sentinel markers. We use rfind for the start marker to skip past
/// the shell's input echo (which also contains the marker text).
fn extractOutput(buffer: &[u8], uuid: &str) -> String {
    let text = String::from_utf8_lossy(buffer);
    let startMarker = format!("__FLATLINE_START_{uuid}__");
    let endPrefix = format!("__FLATLINE_END_{uuid}_");

    // Use rfind to get the LAST start marker — the actual echo output,
    // not the shell's input echo of the wrapped command.
    let afterStart = match text.rfind(&startMarker) {
        Some(pos) => {
            let rest = &text[pos + startMarker.len()..];
            // Skip past the newline after the marker.
            match rest.find('\n') {
                Some(nl) => &rest[nl + 1..],
                None => rest,
            }
        }
        None => return stripAnsi(&text),
    };

    // Everything between start and end markers is command output.
    let output = match afterStart.find(&endPrefix) {
        Some(pos) => afterStart[..pos]
            .trim_end_matches('\n')
            .trim_end_matches('\r'),
        None => afterStart,
    };

    let cleaned = stripAnsi(output);

    // Extract exit code from end marker.
    if let Some(endPos) = text.rfind(&endPrefix) {
        let afterPrefix = &text[endPos + endPrefix.len()..];
        if let Some(codeStr) = afterPrefix.split("__").next() {
            if let Ok(code) = codeStr.parse::<i32>() {
                if code != 0 {
                    return format!("{cleaned}\n(exit code: {code})");
                }
            }
        }
    }

    cleaned
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
