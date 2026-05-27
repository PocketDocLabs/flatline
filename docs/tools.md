# Tools

Flatline exposes built-in tools to the agent. Built-in tools are preferred over
shell commands for file reads, edits, search, and code navigation because they
return structured output, apply size guards, and participate in permissions.

## Tool Limits

Important limits:

- `readFile`: 2000 lines and 100 KB per call
- Shell output: 2000 lines and 100 KB in the immediate result
- Full shell output: retained in shell history and readable with `readOutput`
- `glob`: 100 results
- `grep`: 100 matching files or 200 content lines
- `listDir`: 200 entries
- `structSearch`: 50 matches
- `fuzzyFind`: 20 results
- `fileOutline`: 100 entries

## Shell and Terminal Tools

- `shell`: run a command in the shared terminal
- `shellHistory`: list recent commands with indexes and output sizes
- `readOutput`: read full output from a previous command
- `searchOutput`: regex-search a previous command's output
- `readTerminal`: read recent terminal scrollback
- `terminalSpawn`: create a new named terminal
- `terminalSwitch`: change the agent's default terminal target
- `terminalKill`: kill a named terminal
- `terminalList`: list live terminals

Foreground `shell` calls block the turn until completion or timeout. If a
foreground shell call times out, Flatline kills the foreground attempt and
restarts the same command as a background job. For non-idempotent commands, use
`runInBackground: true` up front or pass a generous timeout.

## File Mutation Tools

- `writeFile`: create or overwrite a whole file
- `editFile`: exact string replacement
- `multiEdit`: multiple exact replacements in one file, applied atomically
- `copyFile`: copy a file or directory tree
- `moveFile`: move or rename a file or directory tree
- `deleteFile`: delete a file or directory
- `makeDirs`: create a directory and missing parents

Existing files must be read with `readFile` before `writeFile`, `editFile`, or
`multiEdit` can modify them. This read-before-write guard helps prevent stale or
blind edits.

File mutation tools can show diff previews in permission prompts.

## File Discovery and Search

- `readFile`: read file content with optional offset, limit, or anchor mode
- `glob`: find files by glob pattern using ripgrep
- `grep`: search file contents using ripgrep and Rust regex syntax
- `listDir`: list a directory as an indented tree
- `fuzzyFind`: fuzzy filename search

Use these tools before shell equivalents such as `cat`, `find`, or `grep -rn`
when the goal is codebase inspection.

## Code Navigation

- `structSearch`: AST structural search with ast-grep
- `fileOutline`: show symbols in a file
- `viewSymbol`: jump to a symbol definition in a file
- `relatedFiles`: inspect imports and sibling files
- `diff`: compare files or a file against a git ref
- `diagnostics`: get LSP diagnostics for a file or directory

`structSearch`, `fileOutline`, and `viewSymbol` require `sg` from ast-grep.
`diagnostics` requires an appropriate language server.

## Web Tools

- `webSearch`: search the web through Exa
- `webFetch`: fetch a URL as markdown
- `webSimilar`: find pages similar to a URL

Set `web.searchKey` or `EXA_API_KEY` before using web tools. Large fetched
pages can be summarized by the utility model when the tool call includes a
prompt.

## Transcript Tools

- `historyFetch`: retrieve a full original exchange block from the transcript
- `historySearch`: search the full original transcript

These help recover details after compaction or truncation.

## Background Work and Wakes

- `jobOutput`: read buffered output from a background job
- `jobStop`: stop a background job
- `jobList`: list background jobs
- `monitor`: watch a long-running command and match lines by regex
- `monitorStop`: stop a monitor
- `monitorList`: list monitors
- `scheduleWakeup`: schedule a one-shot delayed wake
- `cronCreate`: schedule a cron wake
- `cronList`: list wake sources
- `cronDelete`: disarm a wake source
- `fileWatch`: wake on filesystem changes under a path

Use background shell jobs for work that will complete once. Use monitors for
repeated events, such as log lines matching `ERROR`.

## Subagents

- `task`: spawn a focused subtask agent

Agent types:

- `explore`: read-only research using the utility tier
- `general`: full tool access using the heavy tier

Use `runInBackground: true` when the subagent can work while the parent agent
continues on independent work.

## MCP Tools

External MCP tools are exposed with qualified names. If a connected MCP toolset
is too large for the context budget, Flatline exposes `mcpToolSearch` so the
agent can discover relevant MCP tools before calling them.

See [MCP](mcp.md).

