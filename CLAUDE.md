# agent-runtime

Provider-independent single-agent runtime for Rust with built-in tools.

## Architecture

Single crate (`agent-runtime`) with modules:
- `agent` — Agent struct, builder, agent loop
- `tool` — Tool trait, ToolContext, ToolOutput
- `provider` — LlmProvider trait, Request/Response
- `message` — Message, Content, Role, StopReason, Usage
- `tools/` — Built-in tools: Read, Write, Edit, Glob, Grep, Bash, SubAgent, WebFetch
- `providers/` — Anthropic, Mock

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
