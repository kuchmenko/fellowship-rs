# fellowship (crate `fellowship-rs`)

Provider-independent single-agent runtime for Rust with built-in tools.

The crate is published on crates.io as `fellowship-rs` because the bare
`fellowship` name was taken. Inside Rust code it imports as `fellowship`
(via `[lib] name = "fellowship"` in Cargo.toml).

The repository is still `kuchmenko/agent-runtime` on GitHub — keeping
the URL stable; the library was renamed mid-life so old git-history
links keep working.

## Architecture

Single crate with modules:
- `agent` — Agent struct, builder, agent loop, AgentStream
- `approval` — ApprovalHandler trait, ApprovalDecision, AutoApprove
- `tool` — Tool trait, ToolContext, ToolOutput, ToolClass
- `executor` — ToolExecutor, ToolRegistry, ToolPolicy, AllowAll
- `provider` — LlmProvider trait, Request/Response
- `stream` — StreamEvent, ProviderEventStream
- `message` — Message, Content, Role, StopReason, Usage
- `error` — AgentError, ProviderError, ToolError
- `tools/` — Built-in tools: Read, Write, Edit, Glob, Grep, Bash, SubAgent, WebFetch
- `providers/` — Anthropic, OpenAICompatible, Mock

## Commands

- `cargo test` — run all tests
- `cargo clippy --all-targets -- -D warnings` — lint
- `cargo fmt --check` — format check

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature (bumps minor)
- `fix:` — bug fix (bumps patch)
- `feat!:` or `fix!:` — breaking change (bumps minor pre-1.0)
- `chore:`, `docs:`, `refactor:`, `test:` — no release

## Release process

Automated via [release-please](https://github.com/googleapis/release-please).
See `RELEASING.md` for details.

**Flow:** conventional commits on `main` → release-please creates Release PR → merge PR → GitHub Release + tag created automatically.

**Do NOT** manually edit `CHANGELOG.md` or bump version — release-please handles both.

**When to suggest merging the Release PR:**
- When meaningful features or fixes have accumulated
- After a breaking API change
- When the user asks about versioning or shipping
