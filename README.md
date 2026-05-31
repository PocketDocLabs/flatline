# Flatline

Flatline is an agentic terminal where you and the AI share the same live shell.
The agent can inspect files, run tools, and use the terminal while you watch,
interrupt, approve, or take over from the same workspace.

The project has two main crates:

- `crates/deck`: the terminal UI and `flatline` binary
- `crates/construct`: the headless agent core, tools, config, permissions,
  sessions, MCP, LSP, and web integrations

## Quick Start

From a source checkout:

```sh
cargo run -p deck
```

Run a headless prompt:

```sh
cargo run -p deck -- exec "summarize this project"
```

Flatline creates a user config at `~/.config/flatline/config.toml` if one does
not already exist. The default config uses OpenRouter profiles, so set:

```sh
export OPENROUTER_API_KEY=...
```

OpenAI API profiles use `OPENAI_API_KEY`. ChatGPT/Codex OAuth profiles use a
device-code sign-in:

```sh
cargo run -p deck -- auth login openai-codex
```

See [Getting Started](docs/getting-started.md) for first-run setup and common
commands.

An example config is available at [docs/examples/config.toml](docs/examples/config.toml).

## Install From Source

Flatline is currently distributed from source. Install the `flatline` binary
with Cargo:

```sh
cargo install --path crates/deck --locked
```

Then run:

```sh
flatline
```

Running without a subcommand starts the interactive terminal UI. Use
`flatline exec` for headless/non-interactive runs.

## Documentation

- [Getting Started](docs/getting-started.md): install, run, first session
- [Interface](docs/interface.md): TUI layout, hotkeys, slash commands, panels
- [Configuration](docs/configuration.md): profiles, config layers, env vars,
  permissions, LSP, web, budget
- [Tools](docs/tools.md): built-in tool reference and usage patterns
- [Workflows](docs/workflows.md): examples for approvals, headless inspection,
  MCP setup, and LSP debugging
- [MCP](docs/mcp.md): using MCP servers and exposing Flatline as an MCP server
- [Troubleshooting](docs/troubleshooting.md): common setup and runtime problems

Roadmap notes will move into GitHub issues and release notes as they stabilize.

## Development

Run the test suite:

```sh
cargo test
```

Format and check the workspace with your normal Rust toolchain commands. Some
optional tools depend on external binaries:

- `rg` for file search and content search
- `sg` for structural search, file outlines, and symbol lookup

## License

Flatline is licensed under the [Apache License 2.0](LICENSE).
