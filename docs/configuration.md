# Configuration

Flatline uses layered TOML configuration. Lower layers provide defaults and
higher layers override them.

## Config Layers

Resolution order:

1. Built-in defaults
2. User config: `~/.config/flatline/config.toml`
3. Project config: `<project-root>/.flatline/config.toml`
4. Launch directory config: `<launch-dir>/.flatline/config.toml`
5. Local project config: `<project-root>/.flatline/config.local.toml`
6. Local launch directory config: `<launch-dir>/.flatline/config.local.toml`
7. Environment variables

Set `FLATLINE_CONFIG=/path/to/config.toml` to load one explicit file instead
of using normal layer discovery.

Set `FLATLINE_SHELL=/path/to/bash-or-zsh` to choose the embedded shared shell.
This is most useful on Windows when `bash.exe` is installed but not on `PATH`.
The shell setting is intentionally an environment variable because it affects
the PTY process that Flatline starts before project-level interaction begins.

For a complete starter file that mirrors the generated defaults, see
[examples/config.toml](examples/config.toml).

Project root discovery walks upward from the current directory until it finds
`.git`. If no `.git` is found, the current directory is used.

The launch directory is the directory where Flatline was started. When it is
below the project root, launch-scoped config lets a nested workspace override
repo-level defaults without editing the repo root config. Local config files are
gitignored by Flatline when they are created through the UI.

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

The generated starter config also includes inactive profiles for the other
supported providers: `deepseekPro`, `deepseekFlash`, `deepseekUtility`,
`openaiCodex`, `openaiCodexFrontier`, `openaiCodexMini`, `openaiGpt54`, and
`openaiGpt55`. Switch to them from `/model` after setting the matching API key
or Codex OAuth login.

Profile fallback:

- `lightProfile` defaults to `heavyProfile`
- `utilityProfile` defaults to `lightProfile`, then `heavyProfile`

Profiles are atomic across config layers. If a higher layer defines
`[profile.foo]`, that table fully replaces lower-layer `profile.foo` instead of
merging field by field.

## Supported Providers

Documented provider modes:

- `openrouter`
- `deepseek`
- `openai`
- `openai-codex`

Provider defaults fill fields such as `baseUrl`, `maxTokens`, and
`providerOrder`.

API key environment variables:

- `OPENROUTER_API_KEY`: used by `openrouter`
- `DEEPSEEK_API_KEY`: used by `deepseek`
- `OPENAI_API_KEY`: used by `openai`
- `EXA_API_KEY`: used by web tools

The `openai-codex` provider uses ChatGPT/Codex OAuth instead of an API key:

```sh
flatline auth
flatline auth login openai-codex
```

Bare `flatline auth` opens a small terminal helper for OAuth setup. Today it
walks through OpenAI Codex OAuth, with room for additional OAuth providers as
they become supported. The explicit subcommands remain available for scripts and
direct use.

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
contextWindow = 400000
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
contextWindow = 272000
reasoning = { effort = "high", summary = "auto" }
```

Flatline does not send `maxTokens` / `max_output_tokens` for `openai-codex`
profiles because the Codex OAuth backend uses a narrower contract than the
public OpenAI Responses API.

Profiles can also set:

```toml
promptThinking = true
maxContextWindow = 272000
```

Prompt thinking asks the model to use Flatline's scratchpad format instead of a
provider-native reasoning API. In the in-app editor, this is represented as a
single thinking mode: `off`, `provider`, or `prompt scratchpad`. Provider
effort and summary settings are active only in `provider` mode.
`contextWindow` is the usable profile budget. `maxContextWindow` is optional
model metadata used by the in-app editor so a profile can be lowered below the
model maximum and later raised back up without rediscovering the model.

## In-App Profile Switching

Use `/model` to inspect configured profiles and assign one to the heavy, light,
or utility tier. The panel has an explicit save target; press `[` or `]` to
cycle between user, project, project-local, launch-dir, and launch-local config
files when those scopes are distinct.

If the active provider is missing an API key or Codex OAuth session, Flatline
still starts so `/model` remains available. The first attempted model call shows
the auth error and opens the model panel automatically.

When Flatline is launched from a directory below the project root, the default
save target is launch-local config. Otherwise it defaults to project-local
config. Pressing Enter in the profile view saves the tier selection and applies
it to the live session for the next model turn.

Press `e` to open the selected profile's config view. This view exposes the
model picker, usable context budget, thinking mode, provider effort, reasoning
summary, profile create/rename, and profile deletion. Space cycles common
values and each change is saved to the active save target. Provider effort and
reasoning summary are dimmed unless thinking mode is set to `provider`.
Provider effort cycles through the reasoning levels Flatline knows for the
selected model; models without known provider reasoning support cannot enter
provider thinking mode. Reasoning summary cycles through `off`, `auto`,
`concise`, and `detailed`. The context
editor accepts raw token counts plus shorthand such as `128k` or `1.05m` and
caps the value at the model max when Flatline knows it. Rename and delete
operate on the selected config file; if a profile is inherited from another
scope, switch the save target to that scope before renaming or deleting it.
Profiles assigned to heavy, light, or utility cannot be deleted until those
tiers are moved to another profile.

The model picker uses OpenRouter, OpenAI, and DeepSeek provider model APIs.
Known OpenAI API models are enriched with documented context windows and
reasoning effort options when available.
The `openai-codex` provider refreshes the same Codex-style `/models` catalog
shape used by Codex CLI when ChatGPT OAuth is configured, including model
slugs, display names, effective context windows, and supported reasoning
levels. If that refresh fails, Flatline falls back to a small built-in Codex
catalog so the picker remains usable offline.

## Permissions

If no permissions are configured, interactive sessions use the built-in
read-only allow preset and ask for everything else.

The built-in read-only preset also auto-approves `shell` calls whose tool
arguments mark them as `impact = "read"`. Mutating shell calls still ask unless
an explicit rule allows them.

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
- `auto`: use the automatic reviewer for unmatched tools; it may ask the supervisor only when the reviewer allows escalation
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
