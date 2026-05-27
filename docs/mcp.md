# MCP

Flatline can act as an MCP client and can also expose a subset of its own tools
as an MCP server.

## MCP Client Configuration

Flatline discovers MCP servers from:

1. `~/.config/flatline/mcp.json`
2. `.mcp.json` at the project root

Project servers override user servers with the same name.

Basic stdio server:

```json
{
  "mcpServers": {
    "example": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-everything"]
    }
  }
}
```

HTTP server:

```json
{
  "mcpServers": {
    "docs": {
      "url": "https://example.com/mcp",
      "auth": "Bearer ${MCP_TOKEN}"
    }
  }
}
```

Flatline accepts standard `${VAR}` environment syntax and converts it to its
internal `{env:VAR}` form.

## Flatline Extensions

MCP entries can include Flatline-specific fields:

```json
{
  "mcpServers": {
    "example": {
      "command": "example-mcp",
      "enabled": true,
      "enabledTools": ["search", "fetch"],
      "disabledTools": ["delete"],
      "startupTimeout": 20,
      "toolTimeout": 120,
      "maxOutputTokens": 25000
    }
  }
}
```

Supported extensions:

- `enabled`: enable or disable the server
- `enabledTools`: allow only listed tools
- `disabledTools`: hide listed tools
- `startupTimeout`: server startup timeout in seconds
- `toolTimeout`: tool call timeout in seconds
- `maxOutputTokens`: maximum returned tokens per tool call
- `auth`: authorization header for HTTP transport

The standard `disabled` field is also supported.

## MCP Tool Discovery

Flatline registers MCP tools with qualified names so permissions can distinguish
servers. If the MCP tool definition payload exceeds the context budget,
Flatline exposes a search meta-tool:

```text
mcpToolSearch
```

The agent can use it to search tool names and descriptions before calling a
qualified MCP tool.

Use `/mcp` in the TUI to inspect connected servers, status, and tool counts.

## Strict MCP Mode

Headless `exec` accepts `--strict-mcp`. In strict mode, Flatline skips automatic
MCP discovery from config files.

## Flatline as an MCP Server

Run:

```sh
flatline mcp-serve
```

This starts an MCP server over stdio and exposes a curated subset of Flatline's
built-in tools to other MCP clients. The current server implementation exposes
core shell and file/search tools rather than the full interactive session.

