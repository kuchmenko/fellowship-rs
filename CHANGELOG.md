# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-04

### Added

- Core agent loop with sequential tool execution
- `LlmProvider` trait for provider-independent LLM integration
- `Tool` trait for custom tool implementation
- `Agent` builder with configurable model, system prompt, tools, and limits
- Built-in tools: Read, Write, Edit, Glob, Grep, Bash, SubAgent, WebFetch
- Anthropic Messages API provider
- Mock provider for testing
- Sub-agent spawning with depth limiting
- CI pipeline: fmt, clippy, test, audit, deny, MSRV
