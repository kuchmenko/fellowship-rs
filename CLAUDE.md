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

## Release process

This project uses SemVer. See `RELEASING.md` for the full checklist.

**Quick version:**
1. Update `CHANGELOG.md` — move Unreleased items to new version section
2. Bump `version` in `Cargo.toml`
3. `git commit -m "release: vX.Y.Z"` → `git tag vX.Y.Z` → push
4. GitHub Actions creates the release automatically

**When to suggest a release to the user:**
- When multiple features or fixes have accumulated in Unreleased
- After a breaking API change
- When the user asks about versioning or shipping

**Pre-1.0 SemVer:**
- MINOR = breaking changes or significant features
- PATCH = bug fixes, small improvements
