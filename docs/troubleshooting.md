# Troubleshooting

This page covers common Flatline setup and runtime issues.

## API Key Not Set

If Flatline reports that the API key is missing, set the key for the active
provider:

```sh
export OPENROUTER_API_KEY=...
```

Other supported provider keys:

```sh
export FIREWORKS_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Check the active profile names in `~/.config/flatline/config.toml`:

```toml
heavyProfile = "opus"
lightProfile = "sonnet"
utilityProfile = "kimi"
```

## Headless Tool Calls Are Denied

In headless mode there is no TUI user to answer permission prompts. Use
`--allowed-tools` for tasks that need tools:

```sh
flatline exec --allowed-tools readFile,grep,listDir "inspect the project"
```

Use `--tools` to restrict the exposed tool list:

```sh
flatline exec --tools readFile,grep "search for config loading"
```

## Search Tools Fail

`glob`, `grep`, and `fuzzyFind` need ripgrep:

```sh
rg --version
```

Install ripgrep with your package manager if missing.

## Structural Tools Fail

`structSearch`, `fileOutline`, and `viewSymbol` need ast-grep:

```sh
sg --version
```

Install ast-grep if missing.

## Web Tools Are Not Configured

`webSearch`, `webFetch`, and `webSimilar` require Exa:

```sh
export EXA_API_KEY=...
```

Or configure:

```toml
[web]
searchKey = "..."
```

## LSP Diagnostics Are Empty

The `diagnostics` tool needs a matching language server binary. Check `/lsp` in
the TUI for server status and install hints.

Built-in server ids include:

- `rust-analyzer`
- `ty`
- `biome`
- `gopls`
- `clangd`
- `bash-language-server`
- `yaml-language-server`
- `typescript-language-server`
- `jdtls`
- `csharp-ls`

Project overrides live in `.flatline/lsp.toml`.

## MCP Server Does Not Appear

Check:

- JSON syntax in `~/.config/flatline/mcp.json` or project `.mcp.json`
- server name contains only valid MCP identifier characters
- command is available on `PATH`
- required environment variables are set
- `enabled` is not false and `disabled` is not true

Use `/mcp` to inspect status after startup.

## Terminal Rendering Looks Wrong

Press `Ctrl+L` to force a redraw.

If the layout feels wrong, use `Ctrl+O` or `/layout` to open layout controls.

## A Shell Command Ran Too Long

Foreground shell calls may be auto-converted to background jobs after timeout.
The background job is a fresh run of the same command, not the original process
migrated into the background.

Use `runInBackground: true` for long builds, servers, or watchers. In the TUI,
`Ctrl+B` hands a foreground shell call to the background job plane.

Use `/tasks` or `/jobs` to inspect jobs, monitors, and schedules.

## Permission Rules Are Surprising

Rules are first-match-wins. A broad rule above a narrow rule can hide the narrow
rule.

Use `/permissions` to inspect, toggle, delete, and save project or local rules.
