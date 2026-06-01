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

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use alacritty_terminal::{
    Term,
    event::VoidListener,
    grid::Dimensions as VtDimensions,
    index::{Column, Line},
    term::Config as VtConfig,
    vte::ansi::Processor,
};
use anyhow::{Context, Result, bail};
use portable_pty::{CommandBuilder, PtyPair, PtySize, native_pty_system};
use tokio::sync::{mpsc, oneshot};
use tokio::task;

const MAX_HISTORY: usize = 50;
const FLATLINE_SHELL_ENV: &str = "FLATLINE_SHELL";

pub type ShellLineCallback = Arc<dyn Fn(&str) + Send + Sync + 'static>;

// VT render grid for turning captured PTY byte streams into the final
// visible text the user would see at the prompt. Also used by the live
// shell-side VT that backs `readTerminal`.
const VT_RENDER_COLS: usize = 200;
const VT_RENDER_ROWS: usize = 50;
const VT_SCROLLBACK_LINES: usize = 5000;

/// Default shell command timeout (seconds). Commands are interrupted after this.
pub const SHELL_DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Grace period between each escalation phase (Ctrl+C → Ctrl+\ → force-extract).
const SHELL_ESCALATION_SECS: u64 = 3;

/// Headless VT dimensions for rendering captured output.
struct VtSize {
    cols: usize,
    rows: usize,
}

impl VtDimensions for VtSize {
    fn total_lines(&self) -> usize {
        self.rows
    }
    fn screen_lines(&self) -> usize {
        self.rows
    }
    fn columns(&self) -> usize {
        self.cols
    }
}

/// Dump a VT grid (scrollback + visible region) as one row per line,
/// trailing whitespace trimmed per row, trailing blank rows dropped.
fn dumpVtGrid(term: &Term<VoidListener>) -> Vec<String> {
    let grid = term.grid();
    let numCols = grid.columns();
    let historySize = grid.total_lines().saturating_sub(grid.screen_lines());
    let firstLine = -(historySize as i32);
    let lastLine = grid.screen_lines() as i32 - 1;

    let mut rows: Vec<String> = Vec::with_capacity((lastLine - firstLine + 1) as usize);
    for gridLine in firstLine..=lastLine {
        let line = &grid[Line(gridLine)];
        let mut row = String::with_capacity(numCols);
        for col in 0..numCols {
            row.push(line[Column(col)].c);
        }
        while row.ends_with(' ') {
            row.pop();
        }
        rows.push(row);
    }

    while rows.last().map(|r| r.is_empty()).unwrap_or(false) {
        rows.pop();
    }

    rows
}

/// Construct a fresh headless VT with the standard render dimensions.
fn newRenderVt() -> (Term<VoidListener>, Processor) {
    let size = VtSize {
        cols: VT_RENDER_COLS,
        rows: VT_RENDER_ROWS,
    };
    let config = VtConfig {
        scrolling_history: VT_SCROLLBACK_LINES,
        ..VtConfig::default()
    };
    let term = Term::new(config, &size, VoidListener);
    let processor: Processor = Processor::new();
    (term, processor)
}

/// Feed captured command-output bytes through a fresh headless VT
/// emulator and return the final visible text. This is what turns
/// rich/textual/progress-bar frame spam into the single final state
/// the user would actually see at the prompt.
fn renderCommandOutput(bytes: &[u8]) -> String {
    let (mut term, mut processor) = newRenderVt();
    processor.advance(&mut term, bytes);
    dumpVtGrid(&term).join("\n")
}

/// Live VT emulator that tracks everything the shell has written to
/// its PTY. Used by `readTerminal` so the model sees what the user
/// actually sees on screen — including alt-screen apps, cursor-
/// positioned dashboards, and anything else that relies on terminal
/// semantics instead of plain text.
struct ShellVt {
    term: Term<VoidListener>,
    processor: Processor,
}

impl ShellVt {
    fn new() -> Self {
        let (term, processor) = newRenderVt();
        Self { term, processor }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.processor.advance(&mut self.term, bytes);
    }

    fn resize(&mut self, cols: u16, rows: u16) {
        let size = VtSize {
            cols: cols as usize,
            rows: rows as usize,
        };
        self.term.resize(size);
    }

    /// Dump the last `lines` rows of the rendered grid. Returns the
    /// final visible state of the terminal, joined with newlines.
    fn dumpRecentLines(&self, lines: usize) -> String {
        let rows = dumpVtGrid(&self.term);
        let start = rows.len().saturating_sub(lines);
        rows[start..].join("\n")
    }
}

/// A stored command result.
#[derive(Debug, Clone)]
pub struct CommandRecord {
    pub command: String,
    pub output: String,
    pub exitCode: Option<i32>,
    pub lineCount: usize,
}

/// Detailed result of an agent-issued terminal command.
#[derive(Debug, Clone)]
pub struct CommandExecution {
    pub command: String,
    pub output: String,
    pub exitCode: Option<i32>,
    pub lineCount: usize,
    pub replayBytes: Vec<u8>,
    pub timedOut: bool,
}

impl CommandExecution {
    pub fn responseText(&self) -> String {
        let mut response = if let Some(code) = self.exitCode {
            if code != 0 {
                format!("{}\n(exit code: {code})", self.output)
            } else {
                self.output.clone()
            }
        } else {
            self.output.clone()
        };
        if self.timedOut && self.exitCode.is_none() {
            response.push_str("\n(command timed out and was interrupted)");
        }
        response
    }
}

/// Command execution handle — send commands, get output.
///
/// Held by the Session. Commands run in the stateful shell,
/// so `cd`, env vars, etc. persist across calls.
#[derive(Clone)]
pub struct Shell {
    cmdTx: mpsc::Sender<ShellRequest>,
    inputTx: mpsc::Sender<Vec<u8>>,
    /// Tear-down signal. Sending here closes the PTY and exits the
    /// driver task; the harness's `ShellIo::outputRx` then sees its
    /// sender drop, signaling "shell ended". Used by `ShellRegistry::kill`.
    shutdownTx: mpsc::Sender<()>,
    history: Arc<Mutex<Vec<CommandRecord>>>,
    vt: Arc<Mutex<ShellVt>>,
    busy: Arc<AtomicBool>,
    lineListeners: Arc<Mutex<HashMap<u64, ShellLineCallback>>>,
}

impl Shell {
    /// Send Ctrl+C to the shell to interrupt any running command.
    pub fn interrupt(&self) {
        let _ = self.inputTx.try_send(vec![0x03]);
    }

    /// Tear down the shell: close the PTY, exit the driver task. The
    /// underlying process receives SIGHUP when its controlling terminal
    /// goes away. Idempotent — multiple calls are safe.
    pub fn shutdown(&self) {
        let _ = self.shutdownTx.try_send(());
    }

    /// Execute a command in the shell and return captured output.
    ///
    /// If `timeout` is Some, the command is interrupted after that duration.
    /// If None, uses the default timeout (120s).
    pub async fn execute(&self, command: &str, timeout: Option<Duration>) -> String {
        self.executeDetailed(command, timeout).await.responseText()
    }

    /// Execute a command and return structured execution metadata. This
    /// powers terminal-run history while `execute` preserves the old text API.
    pub async fn executeDetailed(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> CommandExecution {
        self.executeDetailedWithDriverTimeout(
            command,
            Some(timeout.unwrap_or(Duration::from_secs(SHELL_DEFAULT_TIMEOUT_SECS))),
        )
        .await
    }

    /// Execute a command with no shell-driver timeout. The caller is
    /// responsible for deciding whether to keep waiting or detach the run.
    /// This powers timeout/Ctrl+B backgrounding without killing and
    /// restarting the foreground command in a separate process.
    pub async fn executeDetailedNoTimeout(&self, command: &str) -> CommandExecution {
        self.executeDetailedWithDriverTimeout(command, None).await
    }

    async fn executeDetailedWithDriverTimeout(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> CommandExecution {
        if self
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return CommandExecution {
                command: command.into(),
                output: "Terminal is busy running another command. Wait for it to finish or choose another terminal.".into(),
                exitCode: None,
                lineCount: 1,
                replayBytes: Vec::new(),
                timedOut: false,
            };
        }
        let (tx, rx) = oneshot::channel();
        let req = ShellRequest {
            command: command.into(),
            timeout,
            respondTo: tx,
        };
        if self.cmdTx.send(req).await.is_err() {
            self.busy.store(false, Ordering::SeqCst);
            return CommandExecution {
                command: command.into(),
                output: "Shell closed.".into(),
                exitCode: None,
                lineCount: 1,
                replayBytes: Vec::new(),
                timedOut: false,
            };
        }
        match rx.await {
            Ok(output) => output,
            Err(_) => {
                self.busy.store(false, Ordering::SeqCst);
                CommandExecution {
                    command: command.into(),
                    output: "Shell disconnected.".into(),
                    exitCode: None,
                    lineCount: 1,
                    replayBytes: Vec::new(),
                    timedOut: false,
                }
            }
        }
    }

    /// True while an agent-command capture is running in this terminal.
    pub fn isBusy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Attach a normalized-line listener to this terminal's output stream.
    /// Listeners see lines after Flatline sentinel filtering and ANSI stripping.
    pub fn addLineListener(&self, id: u64, cb: ShellLineCallback) {
        self.lineListeners.lock().unwrap().insert(id, cb);
    }

    /// Remove a line listener. No-op when the id is unknown.
    pub fn removeLineListener(&self, id: u64) {
        self.lineListeners.lock().unwrap().remove(&id);
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
    pub fn searchOutput(&self, index: usize, pattern: &str, contextLines: usize) -> Option<String> {
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
            return Some(format!(
                "No matches for \"{pattern}\" in command {index} output."
            ));
        }

        // Build context windows around matches, merging overlaps.
        let mut included = vec![false; totalLines];
        for &mi in &matchIndices {
            let start = mi.saturating_sub(contextLines);
            let end = (mi + contextLines + 1).min(totalLines);
            for flag in included.iter_mut().take(end).skip(start) {
                *flag = true;
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
    ///
    /// Dumps the live VT grid — scrollback above the visible region
    /// plus the current screen — as plain text. For alt-screen apps
    /// (btm, htop, textual) this returns the current dashboard state
    /// with its layout intact, not the raw positioning stream.
    pub fn readTerminal(&self, lines: usize) -> String {
        self.vt.lock().unwrap().dumpRecentLines(lines)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellKind {
    Bash,
    Zsh,
}

fn resolveShellBin() -> Result<String> {
    if let Some(shell) = envVar(FLATLINE_SHELL_ENV).or_else(|| envVar("SHELL")) {
        return Ok(shell);
    }

    if let Some(shell) = findOnPath(&["bash.exe", "bash", "zsh.exe", "zsh"]) {
        return Ok(shell);
    }

    #[cfg(windows)]
    bail!(
        "could not find a supported shell. Install Git for Windows or MSYS2, or set {FLATLINE_SHELL_ENV} to a bash/zsh executable before starting Flatline."
    );

    #[cfg(not(windows))]
    bail!(
        "could not find a supported shell. Install bash or zsh, or set {FLATLINE_SHELL_ENV} to a bash/zsh executable before starting Flatline."
    );
}

fn envVar(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn findOnPath(candidates: &[&str]) -> Option<String> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        for candidate in candidates {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path.to_string_lossy().into_owned());
            }
        }
    }
    None
}

fn supportedShellKind(shell: &str) -> Option<ShellKind> {
    match shellProgramName(shell).as_str() {
        "bash" => Some(ShellKind::Bash),
        "zsh" => Some(ShellKind::Zsh),
        _ => None,
    }
}

fn shellProgramName(shell: &str) -> String {
    let file = shell.rsplit(['/', '\\']).next().unwrap_or(shell);
    let file = file.to_ascii_lowercase();
    file.strip_suffix(".exe").unwrap_or(&file).to_string()
}

fn pathForShellArg(path: &Path) -> String {
    let path = path.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        path.replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        path
    }
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

    let shellBin = resolveShellBin()?;
    let shellKind = supportedShellKind(&shellBin).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported shell '{}'. Flatline's shared terminal currently needs bash or zsh for command tracking. Set {FLATLINE_SHELL_ENV} to a bash/zsh executable, such as Git Bash on Windows.",
            shellBin
        )
    })?;
    let mut cmd = CommandBuilder::new(&shellBin);
    if let Ok(cwd) = std::env::current_dir() {
        cmd.cwd(cwd);
    }
    cmd.env("SHELL", &shellBin);
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    cmd.env("GIT_PAGER", "cat");
    cmd.env("PAGER", "cat");

    // Inject OSC 133 shell integration for command boundary tracking.
    injectShellIntegration(shellKind, &mut cmd)?;

    slave.spawn_command(cmd).context("Failed to spawn shell")?;

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
    let (shutdownTx, mut shutdownRx) = mpsc::channel::<()>(1);

    let history: Arc<Mutex<Vec<CommandRecord>>> = Arc::new(Mutex::new(Vec::new()));
    let historyRef = Arc::clone(&history);
    let vt: Arc<Mutex<ShellVt>> = Arc::new(Mutex::new(ShellVt::new()));
    let vtRef = Arc::clone(&vt);
    let busy: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let busyRef = Arc::clone(&busy);
    let lineListeners: Arc<Mutex<HashMap<u64, ShellLineCallback>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let lineListenersRef = Arc::clone(&lineListeners);
    // Size the live VT to the initial PTY dimensions so cursor
    // positioning from the shell lines up with what we track.
    vtRef.lock().unwrap().resize(cols, rows);

    // Background task — central hub for all PTY I/O.
    tokio::spawn(async move {
        let mut captureState: Option<CaptureState> = None;
        // Pending bytes to write to the PTY, drained in chunks so
        // large commands (heredocs) don't deadlock with output echo.
        let mut pendingWrite: VecDeque<u8> = VecDeque::new();
        let mut listenerLineBuf: Vec<u8> = Vec::new();

        // Feed bytes into the live VT that backs `readTerminal`.
        let feedVt = |bytes: &[u8]| {
            vtRef.lock().unwrap().feed(bytes);
        };

        let feedLineListeners = |bytes: &[u8], lineBuf: &mut Vec<u8>| {
            for &b in bytes {
                lineBuf.push(b);
                if b == b'\n' {
                    let raw = String::from_utf8_lossy(lineBuf);
                    let cleaned = stripAnsi(&raw).trim_end_matches(['\r', '\n']).to_string();
                    lineBuf.clear();
                    if cleaned.trim().is_empty() {
                        continue;
                    }
                    let listeners: Vec<ShellLineCallback> =
                        lineListenersRef.lock().unwrap().values().cloned().collect();
                    for cb in listeners {
                        cb(&cleaned);
                    }
                }
            }
        };

        /// Finalize a capture: extract output, store in history, resolve oneshot.
        fn finalizeCapture(
            cap: CaptureState,
            historyRef: &Arc<Mutex<Vec<CommandRecord>>>,
            busyRef: &Arc<AtomicBool>,
            timedOut: bool,
        ) -> (String, Vec<u8>) {
            let mut cap = cap;
            let remaining = cap.filter.flush();
            let result = extractResult(&cap.buffer, &cap.uuid);
            cap.displayBuffer.extend_from_slice(&remaining);

            let mut hist = historyRef.lock().unwrap();
            let lineCount = result.output.lines().count();
            hist.push(CommandRecord {
                command: cap.command.clone(),
                output: result.output.clone(),
                exitCode: result.exitCode,
                lineCount,
            });
            if hist.len() > MAX_HISTORY {
                hist.remove(0);
            }

            let exec = CommandExecution {
                command: cap.command,
                output: result.output,
                exitCode: result.exitCode,
                lineCount,
                replayBytes: cap.displayBuffer,
                timedOut,
            };
            let _ = cap.respondTo.send(exec);
            busyRef.store(false, Ordering::SeqCst);
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
                            cap.displayBuffer.extend_from_slice(&filtered);
                            feedLineListeners(&filtered, &mut listenerLineBuf);
                            feedVt(&filtered);
                            let _ = outputTx.try_send(filtered);
                        }
                    } else {
                        feedLineListeners(&bytes, &mut listenerLineBuf);
                        feedVt(&bytes);
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
                        let (_, remaining) = finalizeCapture(cap, &historyRef, &busyRef, false);
                        if !remaining.is_empty() {
                            feedVt(&remaining);
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
                        // Race recovery: a chunk containing the END
                        // marker may have arrived in the same tick as
                        // the deadline. If we SIGINT now we'll clear
                        // the shell's pending line 2 (the END printf)
                        // at the prompt — destroying our own signal.
                        // Re-check the buffer before advancing kill
                        // phase.
                        let endPattern = format!("__FLATLINE_END_{}_", cap.uuid);
                        if String::from_utf8_lossy(&cap.buffer).contains(&endPattern) {
                            let cap = captureState.take().unwrap();
                            let (_, remaining) = finalizeCapture(cap, &historyRef, &busyRef, false);
                            if !remaining.is_empty() {
                                feedVt(&remaining);
                                let _ = outputTx.try_send(remaining);
                            }
                            continue;
                        }

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
                                let (_, remaining) = finalizeCapture(cap, &historyRef, &busyRef, true);
                                if !remaining.is_empty() {
                                    feedVt(&remaining);
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
                // pieces with a delay so the output arm can interleave
                // and drain PTY echo, preventing buffer deadlocks.
                _ = tokio::time::sleep(Duration::from_millis(5)), if !pendingWrite.is_empty() => {
                    const CHUNK: usize = 128;
                    let n = CHUNK.min(pendingWrite.len());
                    let chunk: Vec<u8> = pendingWrite.drain(..n).collect();
                    let _ = writer.write_all(&chunk);
                    let _ = writer.flush();
                }

                // Command execution request from session.
                Some(req) = cmdRx.recv() => {
                    if captureState.is_some() {
                        let _ = req.respondTo.send(
                            CommandExecution {
                                command: req.command,
                                output: "Terminal is busy running another command. Wait for it to finish or choose another terminal.".into(),
                                exitCode: None,
                                lineCount: 1,
                                replayBytes: Vec::new(),
                                timedOut: false,
                            },
                        );
                        continue;
                    }
                    let uuid = generateUuid();

                    // Echo the command in the terminal so the user sees what ran.
                    // Sent directly on outputTx, bypassing the PTY/filter.
                    // Dim cyan to distinguish agent commands from user input.
                    let cmdEcho = format!("\x1b[2;36m{}\x1b[0m\n", req.command);
                    let echoBytes = toTermBytes(&cmdEcho);
                    feedVt(&echoBytes);
                    let _ = outputTx.try_send(echoBytes);

                    // SINGLE-LINE WRAPPER: set the per-command uuid
                    // variable, emit START, then run the user's command.
                    // The END marker is emitted by the shell's natural
                    // `precmd` hook (injected via shell integration)
                    // immediately after the command finishes — no
                    // separate follow-up line that would race with the
                    // shell's prompt redraw and hang until timeout.
                    //
                    // Heredoc-safe: setting `_flatline_uuid` is on the
                    // same logical line as the user command via `;`,
                    // not on a separate line that would attach to a
                    // heredoc delimiter.
                    let cmd = req.command.trim_end();
                    let wrapped = format!(
                        "_flatline_uuid='{uuid}'; printf '__FLATLINE_START_%s__\\n' '{uuid}'; {cmd}\n",
                    );
                    // Queue for chunked write — large commands (heredocs)
                    // would deadlock if written in one blocking call.
                    pendingWrite.extend(wrapped.as_bytes());

                    let deadline = req.timeout.map(|d| tokio::time::Instant::now() + d);
                    captureState = Some(CaptureState {
                        uuid: uuid.clone(),
                        command: req.command,
                        buffer: Vec::new(),
                        displayBuffer: Vec::new(),
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
                    vtRef.lock().unwrap().resize(cols, rows);
                }

                // User-triggered kill — start/advance the killchain immediately.
                Some(()) = killRx.recv() => {
                    // Abort any in-progress command write.
                    pendingWrite.clear();
                    if let Some(ref mut cap) = captureState {
                        cap.deadline = Some(tokio::time::Instant::now());
                    } else {
                        // No active capture — just forward Ctrl+C.
                        let _ = writer.write_all(&[0x03]);
                        let _ = writer.flush();
                    }
                }

                // Tear-down — close the PTY and exit the loop. The PTY
                // reader thread sees EOF on its next read and exits,
                // which drops outputTx so the harness knows the shell
                // is gone.
                Some(()) = shutdownRx.recv() => {
                    busyRef.store(false, Ordering::SeqCst);
                    break;
                }

                // All channels closed — shut down.
                else => break,
            }
        }
        // Master and writer drop here when the task exits; the OS
        // sends SIGHUP to the shell's process group.
    });

    let shell = Shell {
        cmdTx,
        inputTx: inputTx.clone(),
        shutdownTx,
        history,
        vt,
        busy,
        lineListeners,
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
    respondTo: oneshot::Sender<CommandExecution>,
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
    displayBuffer: Vec<u8>,
    filter: DisplayFilter,
    respondTo: oneshot::Sender<CommandExecution>,
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
///
/// The sliced byte range between START and END is fed through a headless
/// VT emulator so control sequences (`\r` overwrites, cursor moves,
/// alt-screen switches) are **applied** rather than stripped. This turns
/// rich/textual/progress-bar frame spam into the single final rendered
/// state the user would see at the prompt.
fn extractResult(buffer: &[u8], uuid: &str) -> ExtractedResult {
    let startMarker = format!("__FLATLINE_START_{uuid}__");
    let endPrefix = format!("__FLATLINE_END_{uuid}_");

    // Find markers by byte position in the raw buffer. rfind on a slice
    // gives us the byte offset directly — no UTF-8 boundary worries
    // because the marker bytes are pure ASCII.
    let startPos = rfindBytes(buffer, startMarker.as_bytes());
    let endPos = rfindBytes(buffer, endPrefix.as_bytes());

    // Exit code is parsed from the tail of the END marker
    // (`__FLATLINE_END_{uuid}_{code}__`), same as before.
    let exitCode = endPos.and_then(|pos| {
        let afterPrefix = &buffer[pos + endPrefix.len()..];
        let text = String::from_utf8_lossy(afterPrefix);
        text.split("__").next()?.parse::<i32>().ok()
    });

    let Some(startPos) = startPos else {
        // START marker never appeared — command likely failed before
        // the printf ran (e.g., zsh parse error on history expansion).
        // Fall back to ANSI-strip + line-filter since there's no
        // well-defined command-output slice to feed the VT.
        let text = String::from_utf8_lossy(buffer);
        let cleaned: String = stripAnsi(&text)
            .lines()
            .filter(|l| !l.contains("__FLATLINE_") && !l.contains("__flatline_ec"))
            .collect::<Vec<_>>()
            .join("\n");
        return ExtractedResult {
            output: cleaned.trim().to_string(),
            exitCode,
        };
    };

    // Command output begins after the START marker's newline.
    let afterStart = startPos + startMarker.len();
    let contentStart = match buffer[afterStart..].iter().position(|&b| b == b'\n') {
        Some(nl) => afterStart + nl + 1,
        None => afterStart,
    };

    // Command output ends at the END marker, if present.
    let contentEnd = endPos
        .filter(|&p| p >= contentStart)
        .unwrap_or(buffer.len());

    // Feed command output bytes through the VT emulator.
    let rendered = renderCommandOutput(&buffer[contentStart..contentEnd]);

    // Residual sentinel lines can still land in the grid if the END
    // marker partially wrote before we clipped — filter them out.
    let cleaned: String = rendered
        .lines()
        .filter(|l| !l.contains("__FLATLINE_") && !l.contains("__flatline_ec"))
        .collect::<Vec<_>>()
        .join("\n");

    ExtractedResult {
        output: cleaned,
        exitCode,
    }
}

/// Convert a synthesized display string into terminal bytes.
///
/// The PTY byte stream follows terminal-protocol line discipline (CRLF).
/// Bytes coming FROM the shell already do. Bytes we build ourselves from
/// a Rust `&str` (LF convention) must be translated before they join the
/// stream — otherwise bare `\n` moves the cursor down without returning
/// to column 0, producing a staircase in multi-line content.
///
/// Use this at every boundary where synthesized text enters `outputTx`
/// or `feedVt`: agent-command echoes, injected banners, status lines.
/// Idempotent on existing CRLF.
fn toTermBytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 8);
    let mut prev = 0u8;
    for &b in s.as_bytes() {
        if b == b'\n' && prev != b'\r' {
            out.push(b'\r');
        }
        out.push(b);
        prev = b;
    }
    out
}

/// Byte-level `rfind` for a needle in a haystack. Returns the byte
/// offset of the last occurrence.
fn rfindBytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    let limit = haystack.len() - needle.len();
    (0..=limit)
        .rev()
        .find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Strip ANSI escape sequences from text.
///
/// Iterates by Unicode scalar (not bytes) so multi-byte UTF-8
/// characters like box-drawing glyphs survive intact. ANSI control
/// sequences are all ASCII, so char-level scanning handles them
/// correctly with no byte gymnastics.
fn stripAnsi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.next() {
                // CSI sequence: ESC [ ... (letter or ~).
                Some('[') => {
                    for c in chars.by_ref() {
                        if c.is_ascii_alphabetic() || c == '~' {
                            break;
                        }
                    }
                }
                // OSC sequence: ESC ] ... (BEL or ST).
                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\x07' {
                            break;
                        }
                        if c == '\x1b' {
                            // ST terminator (ESC \) — consume the backslash.
                            chars.next();
                            break;
                        }
                    }
                }
                // Other escape: ESC + single char already consumed.
                _ => {}
            }
        } else if c == '\r' {
            // Skip carriage returns (PTY uses \r\n).
        } else {
            result.push(c);
        }
    }

    result
}

/// Inject OSC 133 shell integration so the shell emits command boundary markers.
///
/// For zsh: overrides ZDOTDIR with a temp directory containing init files
/// that source the originals and add precmd/preexec hooks.
/// For bash: uses --rcfile pointing to a wrapper that sources .bashrc first.
fn injectShellIntegration(shellKind: ShellKind, cmd: &mut CommandBuilder) -> Result<()> {
    let integrationDir = std::env::temp_dir().join(format!("flatline-si-{}", std::process::id()));
    std::fs::create_dir_all(&integrationDir)?;

    match shellKind {
        ShellKind::Zsh => {
            let originalZdotdir = std::env::var("ZDOTDIR")
                .unwrap_or_else(|_| std::env::var("HOME").unwrap_or_default());

            // Forward all zsh init files to originals.
            for file in &[".zshenv", ".zprofile", ".zlogin"] {
                let content = format!(
                    "[[ -f \"{originalZdotdir}/{file}\" ]] && source \"{originalZdotdir}/{file}\"\n"
                );
                std::fs::write(integrationDir.join(file), content)?;
            }

            // .zshrc: source original, restore ZDOTDIR, add hooks. The
            // precmd hook does double duty: emits the standard OSC 133
            // boundary markers, AND when the caller has set
            // `_flatline_uuid`, emits a per-command END marker carrying
            // that uuid and the just-finished command's exit code. That
            // lets the wrapper be a single line (no need for a follow-up
            // "; printf END" statement that the shell would have to read
            // as a second line — which always raced with the prompt and
            // hung until timeout).
            let zshrc = format!(
                r#"[[ -f "{originalZdotdir}/.zshrc" ]] && source "{originalZdotdir}/.zshrc"
ZDOTDIR="{originalZdotdir}"
# Agent shell has no interactive history — disable ! expansion
# so globs like [!.]* don't trigger "event not found".
set +H
# zsh's prompt-spacer paints PROMPT_EOL_MARK (usually inverse "%") and
# can add a protective newline when a hook writes control-only bytes
# before the prompt. Flatline's OSC 133 markers are terminal metadata,
# not user output, so disable the spacer in managed shells.
unsetopt PROMPT_SP
PROMPT_EOL_MARK=''
_flatline_seen_prompt=0
flatline_precmd() {{
    local _ec=$?
    if [[ "$_flatline_seen_prompt" != 1 ]]; then
        _flatline_seen_prompt=1
        return
    fi
    printf '\e]133;D;%s\a\e]133;A\a' "$_ec"
    if [[ -n "$_flatline_uuid" ]]; then
        printf '\n__FLATLINE_END_%s_%s__\n' "$_flatline_uuid" "$_ec"
        _flatline_uuid=""
    fi
}}
flatline_preexec() {{ printf '\e]133;C\a' }}
precmd_functions+=(flatline_precmd)
preexec_functions+=(flatline_preexec)
"#
            );
            std::fs::write(integrationDir.join(".zshrc"), zshrc)?;
            cmd.env("ZDOTDIR", pathForShellArg(&integrationDir));
        }
        ShellKind::Bash => {
            let bashrc = r#"[[ -f "$HOME/.bashrc" ]] && source "$HOME/.bashrc"
# Agent shell has no interactive history — disable ! expansion
# so globs like [!.]* don't trigger "event not found".
set +H
flatline_prompt_command() {
    local _ec=$?
    printf '\033]133;D;%s\a\033]133;A\a' "$_ec"
    if [[ -n "$_flatline_uuid" ]]; then
        printf '\n__FLATLINE_END_%s_%s__\n' "$_flatline_uuid" "$_ec"
        _flatline_uuid=""
    fi
}
PROMPT_COMMAND="flatline_prompt_command${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
trap 'printf "\033]133;C\a"' DEBUG
"#;
            let bashrcPath = integrationDir.join("flatline_bashrc");
            std::fs::write(&bashrcPath, bashrc)?;
            cmd.arg("--rcfile");
            cmd.arg(pathForShellArg(&bashrcPath));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shellProgramNameHandlesWindowsAndUnixPaths() {
        assert_eq!(shellProgramName("/bin/bash"), "bash");
        assert_eq!(shellProgramName("/opt/homebrew/bin/zsh"), "zsh");
        assert_eq!(
            shellProgramName(r"C:\Program Files\Git\bin\bash.exe"),
            "bash"
        );
        assert_eq!(shellProgramName(r"C:\Tools\BASH.EXE"), "bash");
        assert_eq!(shellProgramName("pwsh.exe"), "pwsh");
    }

    #[test]
    fn supportedShellKindAcceptsBashAndZshExecutables() {
        assert_eq!(
            supportedShellKind(r"C:\Program Files\Git\bin\bash.exe"),
            Some(ShellKind::Bash)
        );
        assert_eq!(supportedShellKind("/bin/zsh"), Some(ShellKind::Zsh));
        assert_eq!(supportedShellKind("powershell.exe"), None);
    }
}
