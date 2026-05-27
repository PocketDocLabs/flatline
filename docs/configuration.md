# Configuration

Flatline uses layered TOML configuration. Lower layers provide defaults and
higher layers override them.

## Config Layers

Resolution order:

1. Built-in defaults
2. User config: `~/.config/flatline/config.toml`
3. Project config: `<project-root>/.flatline/config.toml`
4. Local project config: `<project-root>/.flatline/config.local.toml`
5. Environment variables

Set `FLATLINE_CONFIG=/path/to/config.toml` to load one explicit file instead
of using normal layer discovery.

Project root discovery walks upward from the current directory until it finds
`.git`. If no `.git` is found, the current directory is used.

## Model Profiles

Model settings live in named profiles. Top-level fields choose which profile is
used for each tier:

```toml
heavyProfile = "opus"
lightProfile = "sonnet"
utilityProfile = "kimi"
compactRatio = 0.8

[profile.opus]
provider = "openrouter"
model = "anthropic/claude-opus-4.6"
promptThinking = true
providerOrder = ["Anthropic"]
contextWindow = 250000

[profile.sonnet]
provider = "openrouter"
model = "anthropic/claude-sonnet-4.6"
promptThinking = true
providerOrder = ["Anthropic"]
contextWindow = 250000

[profile.kimi]
provider = "openrouter"
model = "moonshotai/kimi-k2.6"
contextWindow = 256000
```

Profile fallback:

- `lightProfile` defaults to `heavyProfile`
- `utilityProfile` defaults to `lightProfile`, then `heavyProfile`

Profiles are atomic across config layers. If a higher layer defines
`[profile.foo]`, that table fully replaces lower-layer `profile.foo` instead of
merging field by field.

## Supported Providers

Documented provider modes:

- `openrouter`
- `fireworks`
- `deepseek`
- `openai`
- `openai-codex`

Provider defaults fill fields such as `baseUrl`, `maxTokens`, and
`providerOrder`.

API key environment variables:

- `OPENROUTER_API_KEY`: used by `openrouter`
- `FIREWORKS_API_KEY`: used by `fireworks`
- `DEEPSEEK_API_KEY`: used by `deepseek`
- `OPENAI_API_KEY`: used by `openai`
- `EXA_API_KEY`: used by web tools

The `openai-codex` provider uses ChatGPT/Codex OAuth instead of an API key:

```sh
flatline auth login openai-codex
```

Profile selection environment variables:

- `FLATLINE_HEAVY_PROFILE`
- `FLATLINE_LIGHT_PROFILE`
- `FLATLINE_UTILITY_PROFILE`

## Reasoning and Prompt Thinking

Profiles can set official reasoning options:

```toml
[profile.deepseekPro]
provider = "deepseek"
model = "deepseek-v4-pro"
contextWindow = 128000
reasoning = { effort = "max" }
```

OpenAI Responses API profile:

```toml
[profile.openaiGpt54]
provider = "openai"
model = "gpt-5.4"
contextWindow = 1050000
reasoning = { effort = "high" }
```

ChatGPT/Codex OAuth profile:

```toml
[profile.openaiCodex]
provider = "openai-codex"
model = "gpt-5.3-codex"
contextWindow = 400000
reasoning = { effort = "high", summary = "auto" }
```

Profiles can also set:

```toml
promptThinking = true
```

Prompt thinking asks the model to use Flatline's scratchpad format instead of a
provider-native reasoning API.

## Permissions

If no permissions are configured, interactive sessions use the built-in
read-only allow preset and ask for everything else.

Example project permissions:

```toml
[permissions]
defaultMode = "ask"

[[permissions.rules]]
tool = "readFile"
allow = true

[[permissions.rules]]
tool = "grep"
allow = true

[[permissions.rules]]
tool = "shell"
pattern = "git status*"
allow = true
```

`defaultMode` values:

- `ask`: ask the supervisor when no rule matches
- `deny`: deny unmatched tools and continue
- `abort`: deny unmatched tools and abort the turn

Rules are checked in order. `tool = "*"` matches all tools. A trailing `*` in
`pattern` performs a prefix match; otherwise the pattern is a substring match
against the tool's key argument.

## Web Tools

Web tools use Exa:

```toml
[web]
searchKey = "..."
```

You can also set:

```sh
export EXA_API_KEY=...
```

Configured web tools:

- `webSearch`
- `webFetch`
- `webSimilar`

Fetched page content is cached in memory for 15 minutes.

## Budget

Set a session cost warning threshold in config:

```toml
[budget]
sessionLimit = 5.00
```

When the session cost reaches `budget.sessionLimit`, Flatline emits a warning
once and keeps running.

Headless mode also supports a hard stop:

```sh
flatline exec --max-budget-usd 5 "do the task"
```

`--max-budget-usd` stops the session when the cost reaches the limit. It is not
the same as the config warning threshold.

## LSP

User-level LSP overrides live under `[lsp."server-id"]` in
`~/.config/flatline/config.toml`.

Project-level overrides live in `.flatline/lsp.toml`:

```toml
[lsp.rust-analyzer]
enabled = true
startup_timeout = 30
diagnostics_timeout = 3
```

Custom server example:

```toml
[lsp.my-language-server]
command = "my-language-server"
args = ["--stdio"]
extensions = [".mine"]
languageIds = ["mine"]
rootMarkers = ["mine.toml"]
```

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

## Project Context

Flatline adds these context files to the system prompt when present:

- `~/.config/flatline/AGENTS.md`
- the first `AGENTS.md` found while walking upward from the working directory

Use them for project-specific instructions, test commands, and conventions.
