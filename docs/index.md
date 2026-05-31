# Flatline Documentation

This directory contains user-facing documentation for Flatline.

Flatline is a terminal agent with a shared PTY, an agent conversation panel,
layered TOML configuration, built-in filesystem and search tools, optional web
tools, MCP client support, and headless execution.

## Start Here

- [Getting Started](getting-started.md): first run, config, API keys, headless
  mode
- [Interface](interface.md): TUI layout, hotkeys, slash commands, permission
  prompts
- [Configuration](configuration.md): config layers, model profiles,
  permissions, web, LSP, budget
- [Tools](tools.md): built-in tools, limits, when to use each category
- [Workflows](workflows.md): concrete examples for safe approvals, headless
  inspection, MCP setup, and LSP debugging
- [MCP](mcp.md): configure external MCP servers and run Flatline as an MCP
  server
- [Troubleshooting](troubleshooting.md): common failures and fixes

## Search Topics

Use these phrases when looking for a specific feature:

- API keys: `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`, `DEEPSEEK_API_KEY`
- Config files: `config.toml`, `.flatline/config.toml`,
  `.flatline/config.local.toml`, launch directory config,
  `examples/config.toml`
- Model profiles: `heavyProfile`, `lightProfile`, `utilityProfile`
- Permissions: `allowReadOnly`, `defaultMode`, permission rules
- Web tools: `EXA_API_KEY`, `web.searchKey`, `webSearch`, `webFetch`
- MCP: `.mcp.json`, `mcpToolSearch`, `mcp-serve`
- LSP: `diagnostics`, `.flatline/lsp.toml`, language servers
- Background work: `runInBackground`, `jobOutput`, `monitor`, `fileWatch`
