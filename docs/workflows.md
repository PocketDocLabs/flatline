# Workflows

These examples show common Flatline tasks from start to finish.

## Approve a Tool Safely

When Flatline asks for permission, read the summary first:

1. Check the tool name and target path or command.
2. Open the diff preview with `v` when the tool changes files.
3. Use `y` for a one-time approval when you are unsure.
4. Use `A` only when the selected pattern is narrow enough to trust again.
5. Use `Shift+Up` and `Shift+Down` to choose narrower or broader patterns.
6. Use `n` or `D` when a command is too broad, destructive, or surprising.

Good always-allow rules are scoped:

```toml
[[permissions.rules]]
tool = "shell"
pattern = "git status*"
allow = true
```

Avoid broad rules like `tool = "*"` unless you are deliberately running in a
trusted, high-autonomy mode.

## Run a Headless Code Inspection

Headless mode cannot ask you for permission in the middle of a run. For
read-only inspection, expose and allow only inspection tools:

```sh
flatline exec \
  --tools readFile,grep,listDir,fileOutline,viewSymbol,diagnostics \
  --allowed-tools readFile,grep,listDir,fileOutline,viewSymbol,diagnostics \
  "Inspect this project and report likely config-loading issues."
```

Use JSON output when another script will consume the result:

```sh
flatline exec --output json --allowed-tools readFile,grep "summarize config files"
```

## Add an MCP Server

Create or edit project `.mcp.json`:

```json
{
  "mcpServers": {
    "docs": {
      "command": "npx",
      "args": ["-y", "@example/docs-mcp"],
      "enabled": true,
      "startupTimeout": 20,
      "toolTimeout": 120
    }
  }
}
```

Restart Flatline or start a new session so the MCP manager reloads config. In
the TUI, run:

```text
/mcp
```

If there are many MCP tools, ask the agent to search them first. Flatline will
expose `mcpToolSearch` when the full MCP tool list is too large for the context
budget.

## Debug Missing LSP Diagnostics

Start with the LSP panel:

```text
/lsp
```

If the server is missing, install the hinted binary. For Rust:

```sh
rustup component add rust-analyzer
```

For project-specific overrides, add `.flatline/lsp.toml`:

```toml
[lsp.rust-analyzer]
enabled = true
startup_timeout = 30
diagnostics_timeout = 3
```

Then ask the agent to call `diagnostics` on a file or directory.

## Investigate a Long-Running Command

Ask the agent to use a background shell call for builds, servers, or watchers:

```text
Run the test suite in the background and keep working while it runs.
```

Inspect jobs with:

```text
/tasks
```

Use `jobOutput` for a mid-run peek or `jobStop` to stop a job. Use `monitor`
when you need repeated notifications for matching log lines, such as every
`ERROR` in a server log.
