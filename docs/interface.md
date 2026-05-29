# Interface

Flatline's interactive mode is a terminal UI with a shared shell on the left
and an agent conversation panel on the right.

## Layout

- Terminal pane: a real PTY that both you and the agent can use
- Agent pane: conversation, markdown, reasoning display, tool progress, and
  permission prompts
- Status bar: model/context/cost/task status and clickable panels

The agent sees the same terminal session you see. Shell commands run visibly in
the shared terminal unless a tool routes them through the background job plane.

## Global Hotkeys

- `Tab`: switch focus between terminal and agent panel
- `Esc`: cancel a running agent turn or close an overlay
- `Ctrl+T`: spawn a new terminal tab when the terminal has focus
- `Ctrl+1` through `Ctrl+9`: jump to terminal tab N
- `Ctrl+B`: send a long-running foreground shell command to the background
- `Ctrl+Q` twice: quit Flatline
- `Ctrl+L`: force a terminal redraw
- `Ctrl+O`: toggle layout controls
- `Ctrl+H`: toggle hotkey help
- `Up` from the agent input: focus the status bar when status chips exist

Mouse support includes terminal tab clicks, status-chip clicks, scrolling, and
selection in the agent panel.

## Agent Input

When the agent panel has focus:

- `Enter`: send the message
- `Shift+Enter`: insert a newline
- `Up` and `Down`: navigate input history when the cursor is at the edge
- `PageUp` and `PageDown`: scroll the agent panel
- `Ctrl+A`: move to start of input
- `Ctrl+E`: move to end of input
- `Ctrl+K`: delete to end of input
- `Ctrl+U`: delete to start of input
- `Ctrl+Y`: yank deleted text
- `Ctrl+T`: toggle the latest reasoning block
- `Ctrl+D`: attach EOF-style input marker where supported by the input state

## Slash Commands

Slash commands are handled by the TUI before text is sent to the agent.

- `/help`: show available commands
- `/context` or `/ctx`: show context usage and compaction state
- `/undo`: restore the project to before the last file-modifying tool
- `/rewind`: rewind conversation, optionally to a specific turn id
- `/resume`: list or resume a previous session
- `/clear`, `/cls`, or `/new`: clear display and start a fresh session
- `/forks`: list saved forks or switch to one
- `/mcp`: show MCP server status and tool counts
- `/lsp`: show LSP server status and install hints
- `/permissions` or `/perms`: view and manage permission rules
- `/model` or `/models`: view, create, rename, delete, tune context/thinking, and switch live model profiles; discover provider models with context and reasoning effort hints; choose the config file to save into
- `/cost`: show session and rolling cost breakdown
- `/tasks` or `/jobs`: show background jobs, monitors, and wake schedules
- `/layout`: open layout controls

Tab completion is available for command names and aliases.

## Permission Prompts

When a tool call needs approval, Flatline shows an inline permission prompt.

Common actions:

- `y`: allow once
- `n`: deny once
- `A`: always allow using the selected pattern
- `D`: always deny using the selected pattern
- `Shift+Up` and `Shift+Down`: choose a narrower or broader rule pattern
- `v`: toggle expanded details or diff preview where available

Persisted permission rules are written to `.flatline/config.toml` when a project
root is available.

## Panels

Status-bar chips and slash commands open panels:

- MCP panel: connected servers, status, and tools
- LSP panel: known servers, install hints, and server state
- Permissions panel: effective permission rules with delete/toggle/save
- Tasks panel: background jobs, monitors, schedules, and wake sources
- Layout panel: interactive layout controls

Most panels use `Up`/`Down` or `j`/`k` to navigate and `Esc` or `q` to close.
