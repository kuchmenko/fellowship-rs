//! Demonstrates parallel execution of read-only tools in a single batch.
//!
//! Three custom tools are registered:
//! - `fetch_a`, `fetch_b` — `ReadOnly`, each sleeps 200ms
//! - `save`            — `Mutating` (default)
//!
//! The mock provider asks the agent to call all three in one turn, in
//! the order `[fetch_a, save, fetch_b]`. The executor partitions the
//! batch into `[RO] [Mut] [RO]` — two RO runs of size 1, each run
//! degenerate to sequential. Then the mock asks for `[fetch_a, fetch_b]`
//! in the next turn: both are RO and consecutive, so they execute in
//! parallel via `join_all` and wall-time is ~200ms instead of ~400ms.
//!
//! Run with: `cargo run --example parallel_tools`

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fellowship::message::{Content, StopReason, Usage};
use fellowship::provider::Response;
use fellowship::providers::Mock;
use fellowship::{
    Agent, CancellationToken, Message, Tool, ToolClass, ToolContext, ToolError, ToolOutput,
};
use serde_json::{Value, json};

/// A read-only tool that sleeps then echoes its name.
struct SlowReader {
    label: &'static str,
    delay_ms: u64,
}

#[async_trait::async_trait]
impl Tool for SlowReader {
    fn name(&self) -> &str {
        self.label
    }
    fn description(&self) -> &str {
        "Read-only tool that simulates slow I/O"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }
    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        Ok(ToolOutput::text(format!("{} done", self.label)))
    }
}

/// A mutating tool — forced sequential.
struct Save;

#[async_trait::async_trait]
impl Tool for Save {
    fn name(&self) -> &str {
        "save"
    }
    fn description(&self) -> &str {
        "Mutating tool — must run sequentially"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    // class() defaults to Mutating.
    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text("saved"))
    }
}

#[tokio::main]
async fn main() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    // Mock provider scripts two turns of tool use, then a final text answer.
    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            // Turn 1: mixed batch [RO, Mut, RO] — partitioned into three runs.
            0 => Ok(Response {
                content: vec![
                    tool_use("t1", "fetch_a"),
                    tool_use("t2", "save"),
                    tool_use("t3", "fetch_b"),
                ],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            // Turn 2: pure RO batch [RO, RO] — one parallel run.
            1 => Ok(Response {
                content: vec![tool_use("t4", "fetch_a"), tool_use("t5", "fetch_b")],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            // Turn 3: done.
            _ => Ok(Response {
                content: vec![Content::text("All fetches complete.")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("mock")
        .tool(SlowReader {
            label: "fetch_a",
            delay_ms: 200,
        })
        .tool(SlowReader {
            label: "fetch_b",
            delay_ms: 200,
        })
        .tool(Save)
        .build();

    let started = Instant::now();
    let result = agent
        .run(
            vec![Message::user_text("fetch some stuff")],
            CancellationToken::new(),
        )
        .await
        .expect("agent run");
    let elapsed = started.elapsed();

    println!("result: {}", result.text);
    println!("delta messages: {}", result.new_messages.len());
    println!("wall time: {elapsed:?}");
    println!();
    println!("Turn 1 batch [RO, Mut, RO] = 3 serial runs ≈ 400ms+");
    println!("Turn 2 batch [RO, RO]      = 1 parallel run ≈ 200ms");
    println!("Total: ~600ms (vs. ~800ms if everything were sequential)");
}

fn tool_use(id: &str, name: &str) -> Content {
    Content::ToolUse {
        id: id.into(),
        name: name.into(),
        input: json!({}),
    }
}
