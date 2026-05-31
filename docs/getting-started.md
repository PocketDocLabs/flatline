# Getting Started

This guide gets Flatline running from a source checkout.

## Requirements

Flatline is a Rust workspace. You need a recent Rust toolchain with Cargo.

Optional but strongly recommended:

- `rg` from ripgrep for `glob`, `grep`, and `fuzzyFind`
- `sg` from ast-grep for `structSearch`, `fileOutline`, and `viewSymbol`

Optional integrations:

- Exa API key for web tools
- Language server binaries for LSP diagnostics
- MCP servers configured in `.mcp.json`

## Run the TUI

From the repository root:

```sh
cargo run -p deck
```

The `deck` crate builds the `flatline` binary. Running without a subcommand
starts the interactive terminal UI.

To install the binary into your Cargo bin directory:

```sh
cargo install --path crates/deck --locked
```

After that, run:

```sh
flatline
```

## Run Headless

Use `exec` to run a prompt without the TUI:

```sh
cargo run -p deck -- exec "summarize this project"
```

Read the prompt from stdin:

```sh
printf 'list the important config files\n' | cargo run -p deck -- exec
```

Useful headless flags:

- `--output text`: stream plain text output
- `--output json`: print final JSON
- `--output events`: stream JSON events
- `--max-turns N`: cap the number of agent turns
- `--heavy-profile NAME`: choose the heavy model profile
- `--light-profile NAME`: choose the light model profile
- `--utility-profile NAME`: choose the utility model profile
- `--model MODEL`: override the heavy model id
- `--allowed-tools readFile,grep`: allow specific tools in headless mode
- `--tools readFile,grep`: expose only specific tools
- `--max-budget-usd N`: stop when the session cost reaches a limit

In non-streaming headless mode, Flatline cannot ask a person to approve tool
calls. Use `--allowed-tools` when the task needs tools.

## First Config

Flatline creates `~/.config/flatline/config.toml` on first load if it does not
exist. The generated default uses OpenRouter profiles:

- heavy: Claude Opus through OpenRouter
- light: Claude Sonnet through OpenRouter
- utility: Kimi K2.6 through OpenRouter

Set an API key:

```sh
export OPENROUTER_API_KEY=...
```

Flatline also supports Fireworks and DeepSeek profile defaults:

```sh
export FIREWORKS_API_KEY=...
export DEEPSEEK_API_KEY=...
```

OpenAI API profiles use:

```sh
export OPENAI_API_KEY=...
```

ChatGPT/Codex OAuth profiles do not use an API key. Sign in with:

```sh
cargo run -p deck -- auth login openai-codex
```

Inside the TUI, use `/model` to inspect configured profiles and save a
heavy/light/utility profile choice. Press `[` or `]` in the panel to choose
whether the selection is saved to user, project, project-local, launch-dir, or
launch-local config. Saved model/profile changes apply to the live session's
next model turn. Press `e` to open the profile config view for model discovery,
profile create/rename/delete, usable context budget, thinking mode, and
provider-native reasoning settings.

See [Configuration](configuration.md) for profile examples and config layering.
For a full starter config, see [examples/config.toml](examples/config.toml).

For practical examples, see [Workflows](workflows.md).

## Project Context

Flatline injects context from:

- `~/.config/flatline/AGENTS.md` for user-level guidance
- the first `AGENTS.md` found while walking upward from the working directory

Use these files for persistent instructions such as build commands, test
commands, project conventions, and known gotchas.

## Verify the Build

Run tests:

```sh
cargo test
```

Before sharing a release, also run:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

If a tool appears unavailable, check [Troubleshooting](troubleshooting.md).
