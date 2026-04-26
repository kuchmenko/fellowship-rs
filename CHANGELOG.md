# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-26

Initial public release of `tkach` on crates.io.

A provider-independent agent runtime for Rust with a stateless agent
loop, pluggable LLM providers (Anthropic, OpenAI-compatible), built-in
file/shell tools, real SSE streaming, cooperative cancellation, and
per-call approval gating.

### Core API

- `Agent::run` — stateless agent loop; caller owns message history
- `Agent::stream` + `AgentStream` — live token streaming with atomic `ToolUse` events
- `LlmProvider` trait — providers: Anthropic, OpenAICompatible (OpenAI / OpenRouter / Ollama / Moonshot / DeepSeek / Together / Groq), Mock
- `Tool` trait + `ToolClass::{ReadOnly, Mutating}` — read-only batches run in parallel; mutating runs sequentially
- `ToolExecutor` + `ToolPolicy` + `ApprovalHandler` — two-gate execution model with cancel-aware approval
- `CancellationToken` — propagates through the loop, SSE pull, HTTP body, and `Bash` child processes via `kill_on_drop`
- `SubAgent` (Model 3) — nested agents inherit parent's executor; one approval handler / policy / registry gates the whole tree

### Built-in tools

`Read`, `Glob`, `Grep`, `WebFetch` (read-only) · `Write`, `Edit`, `Bash`, `SubAgent` (mutating)
