//! OpenAI Responses API provider.
//!
//! This is the provider to use for OpenAI reasoning/thinking streams.
//! `OpenAICompatible` targets Chat Completions (`/chat/completions`),
//! whose standard wire format has no reasoning-summary events. This
//! provider targets `/responses`, opts into reasoning summaries when
//! configured, and maps `response.reasoning_summary_text.*` events into
//! provider-neutral [`StreamEvent::ThinkingDelta`] /
//! [`StreamEvent::ThinkingBlock`] values.

use std::collections::{HashMap, HashSet, VecDeque};

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::error::ProviderError;
use crate::message::{
    Content, Message, Role, StopReason, ThinkingMetadata, ThinkingProvider, Usage,
};
use crate::provider::{LlmProvider, Request, Response};
use crate::stream::{ProviderEventStream, StreamEvent};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Provider for OpenAI's `/responses` API.
///
/// Use this when you want first-class OpenAI reasoning summaries. For
/// Chat Completions-compatible endpoints, use [`super::OpenAICompatible`]
/// instead; that provider deliberately does not expose non-standard
/// `reasoning_content` fields as thinking.
pub struct OpenAIResponses {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    reasoning: Option<ReasoningConfig>,
    include_encrypted_reasoning: bool,
}

#[derive(Debug, Clone)]
struct ReasoningConfig {
    effort: String,
    summary: String,
}

impl OpenAIResponses {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
            reasoning: None,
            include_encrypted_reasoning: true,
        }
    }

    /// Read `OPENAI_API_KEY` from the environment.
    pub fn from_env() -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var is required");
        Self::new(api_key)
    }

    /// Override the endpoint root, without trailing `/responses`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Request reasoning effort + summaries from reasoning models.
    ///
    /// Typical values: effort `low|medium|high`, summary
    /// `auto|concise|detailed`. OpenAI validates the exact combinations
    /// per model.
    pub fn with_reasoning(mut self, effort: impl Into<String>, summary: impl Into<String>) -> Self {
        self.reasoning = Some(ReasoningConfig {
            effort: effort.into(),
            summary: summary.into(),
        });
        self
    }

    /// Do not request encrypted reasoning replay blobs.
    ///
    /// Keeping this enabled is useful for stateless multi-turn replay:
    /// OpenAI can return opaque `reasoning.encrypted_content` that should
    /// be persisted but never displayed.
    pub fn without_encrypted_reasoning(mut self) -> Self {
        self.include_encrypted_reasoning = false;
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl LlmProvider for OpenAIResponses {
    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError> {
        let mut body = build_request_body(
            &request,
            self.reasoning.as_ref(),
            self.include_encrypted_reasoning,
        );
        body["stream"] = json!(true);

        let response = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();
        if status >= 400 {
            let retry_after_ms = parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(classify_error(status, text, retry_after_ms));
        }

        Ok(Box::pin(responses_event_stream(
            response.bytes_stream().eventsource(),
        )))
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let body = build_request_body(
            &request,
            self.reasoning.as_ref(),
            self.include_encrypted_reasoning,
        );

        let response = self
            .client
            .post(self.responses_url())
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();
        if status >= 400 {
            let retry_after_ms = parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(classify_error(status, text, retry_after_ms));
        }

        let text = response.text().await?;
        let value = serde_json::from_str::<Value>(&text)?;
        response_error(&value).map_or_else(|| convert_response_value(&value), Err)
    }
}

fn classify_error(status: u16, message: String, retry_after_ms: Option<u64>) -> ProviderError {
    match status {
        429 => ProviderError::RateLimit { retry_after_ms },
        503 => ProviderError::Overloaded { retry_after_ms },
        500 | 502 | 504 => ProviderError::Api {
            status,
            message,
            retryable: true,
        },
        s => ProviderError::Api {
            status: s,
            message,
            retryable: (500..600).contains(&s),
        },
    }
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    raw.trim().parse::<u64>().ok().map(|s| s * 1_000)
}

fn response_error(value: &Value) -> Option<ProviderError> {
    let error = value.get("error")?;
    if error.is_null() {
        return None;
    }
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("OpenAI Responses request failed")
        .to_string();
    Some(ProviderError::Api {
        status: 500,
        message,
        retryable: false,
    })
}

fn build_request_body(
    request: &Request,
    reasoning: Option<&ReasoningConfig>,
    include_encrypted_reasoning: bool,
) -> Value {
    let mut body = json!({
        "model": request.model,
        "store": false,
        "stream": false,
        "input": build_input(&request.messages),
        "max_output_tokens": request.max_tokens,
    });

    if let Some(instructions) = instructions(request) {
        body["instructions"] = json!(instructions);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(reasoning) = reasoning {
        body["reasoning"] = json!({
            "effort": reasoning.effort,
            "summary": reasoning.summary,
        });
    }
    if include_encrypted_reasoning {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }

    let tools = build_tools(&request.tools);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }

    body
}

fn instructions(request: &Request) -> Option<String> {
    let joined = request
        .system
        .as_ref()?
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!joined.is_empty()).then_some(joined)
}

fn build_tools(tools: &[crate::provider::ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": null,
            })
        })
        .collect()
}

fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        match message.role {
            Role::User => push_user_message(&mut input, message),
            Role::Assistant => push_assistant_message(&mut input, message),
        }
    }
    input
}

fn push_user_message(input: &mut Vec<Value>, message: &Message) {
    let mut text = Vec::new();

    for content in &message.content {
        match content {
            Content::Text { text: chunk, .. } => text.push(chunk.as_str()),
            Content::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                push_text_message(input, "user", &text.join("\n"));
                text.clear();

                let output = if *is_error {
                    format!("[error] {content}")
                } else {
                    content.clone()
                };
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_call_id_for_output(tool_use_id),
                    "output": output,
                }));
            }
            Content::Thinking { .. } | Content::ToolUse { .. } => {}
        }
    }

    push_text_message(input, "user", &text.join("\n"));
}

fn push_assistant_message(input: &mut Vec<Value>, message: &Message) {
    let mut text = Vec::new();

    for content in &message.content {
        match content {
            Content::Text { text: chunk, .. } => text.push(chunk.as_str()),
            Content::Thinking {
                text: thinking,
                provider: ThinkingProvider::OpenAIResponses,
                metadata:
                    ThinkingMetadata::OpenAIResponses {
                        item_id,
                        output_index: _,
                        summary_index: _,
                        encrypted_content,
                    },
            } => {
                push_text_message(input, "assistant", &text.join(""));
                text.clear();
                input.push(openai_reasoning_input(
                    item_id.as_deref(),
                    thinking,
                    encrypted_content.as_deref(),
                ));
            }
            Content::ToolUse {
                id,
                name,
                input: args,
            } => {
                push_text_message(input, "assistant", &text.join(""));
                text.clear();

                let (call_id, item_id) = split_tool_use_id(id);
                let mut item = json!({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args.to_string(),
                    "status": "completed",
                });
                if let Some(item_id) = item_id {
                    item["id"] = json!(item_id);
                }
                input.push(item);
            }
            Content::Thinking { .. } | Content::ToolResult { .. } => {}
        }
    }

    push_text_message(input, "assistant", &text.join(""));
}

fn openai_reasoning_input(
    item_id: Option<&str>,
    text: &str,
    encrypted_content: Option<&str>,
) -> Value {
    let mut item = json!({
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
        "status": "completed",
    });
    if let Some(item_id) = item_id {
        item["id"] = json!(item_id);
    }
    if let Some(encrypted_content) = encrypted_content {
        item["encrypted_content"] = json!(encrypted_content);
    }
    item
}

fn push_text_message(input: &mut Vec<Value>, role: &str, content: &str) {
    if !content.is_empty() {
        input.push(json!({ "role": role, "content": content }));
    }
}

fn split_tool_use_id(id: &str) -> (&str, Option<&str>) {
    id.split_once('|').map_or((id, None), |(call_id, item_id)| {
        (call_id, (!item_id.is_empty()).then_some(item_id))
    })
}

fn tool_call_id_for_output(id: &str) -> &str {
    split_tool_use_id(id).0
}

fn combined_tool_use_id(call_id: &str, item_id: &str) -> String {
    if item_id.is_empty() {
        call_id.to_string()
    } else {
        format!("{call_id}|{item_id}")
    }
}

fn convert_response_value(value: &Value) -> Result<Response, ProviderError> {
    let mut content = Vec::new();
    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for (output_index, item) in output.iter().enumerate() {
            convert_output_item(item, output_index, &mut content);
        }
    }

    if content.is_empty() {
        if let Some(text) = value.get("output_text").and_then(Value::as_str) {
            if !text.is_empty() {
                content.push(Content::text(text));
            }
        }
    }

    let has_tool_use = content.iter().any(|c| matches!(c, Content::ToolUse { .. }));
    Ok(Response {
        stop_reason: response_stop_reason(value, has_tool_use),
        usage: usage_from_response(value.get("usage")),
        content,
    })
}

fn convert_output_item(item: &Value, output_index: usize, content: &mut Vec<Content>) {
    match item.get("type").and_then(Value::as_str) {
        Some("message") => convert_message_item(item, content),
        Some("function_call") => {
            if let Some(tool) = tool_call_from_item(item) {
                content.push(tool_to_content(tool));
            }
        }
        Some("reasoning") => convert_reasoning_item(item, Some(output_index), content),
        _ => {}
    }
}

fn convert_message_item(item: &Value, content: &mut Vec<Content>) {
    let Some(parts) = item.get("content").and_then(Value::as_array) else {
        return;
    };
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        content.push(Content::text(text));
                    }
                }
            }
            Some("refusal") => {
                if let Some(text) = part.get("refusal").and_then(Value::as_str) {
                    if !text.is_empty() {
                        content.push(Content::text(text));
                    }
                }
            }
            Some("reasoning_text") => {}
            _ => {}
        }
    }
}

fn convert_reasoning_item(item: &Value, output_index: Option<usize>, content: &mut Vec<Content>) {
    let item_id = item.get("id").and_then(Value::as_str).map(str::to_string);
    let encrypted_content = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .map(str::to_string);

    let Some(summary) = item.get("summary").and_then(Value::as_array) else {
        return;
    };
    for (summary_index, part) in summary.iter().enumerate() {
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            continue;
        };
        content.push(Content::thinking(
            text,
            ThinkingProvider::OpenAIResponses,
            ThinkingMetadata::openai_responses(
                item_id.clone(),
                output_index,
                summary_index,
                encrypted_content.clone(),
            ),
        ));
    }
}

fn response_stop_reason(value: &Value, has_tool_use: bool) -> StopReason {
    if has_tool_use {
        return StopReason::ToolUse;
    }

    match value.get("status").and_then(Value::as_str) {
        Some("incomplete") => match value
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::EndTurn,
            _ => StopReason::EndTurn,
        },
        _ => StopReason::EndTurn,
    }
}

fn usage_from_response(value: Option<&Value>) -> Usage {
    let Some(value) = value else {
        return Usage::default();
    };
    Usage {
        input_tokens: value
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        output_tokens: value
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    }
}

#[derive(Default)]
struct ResponsesSseParser {
    usage: Usage,
    pending_tools: HashMap<String, PendingToolCall>,
    emitted_tool_items: HashSet<String>,
    pending_reasoning: HashMap<ReasoningKey, PendingReasoning>,
    saw_tool_use: bool,
    emitted_terminal: bool,
}

#[derive(Default)]
struct PendingToolCall {
    call_id: String,
    item_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReasoningKey {
    item_id: String,
    summary_index: usize,
}

struct PendingReasoning {
    item_id: String,
    output_index: Option<usize>,
    summary_index: usize,
    text: String,
    encrypted_content: Option<String>,
    emitted: bool,
}

struct ResponsesStreamState<S> {
    sse: S,
    parser: ResponsesSseParser,
    outbox: VecDeque<Result<StreamEvent, ProviderError>>,
    done: bool,
}

fn responses_event_stream<S>(
    sse: S,
) -> impl futures::Stream<Item = Result<StreamEvent, ProviderError>>
where
    S: futures::Stream<
            Item = Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<reqwest::Error>,
            >,
        > + Send
        + Unpin
        + 'static,
{
    futures::stream::unfold(
        ResponsesStreamState {
            sse,
            parser: ResponsesSseParser::default(),
            outbox: VecDeque::new(),
            done: false,
        },
        |mut state| async move {
            loop {
                if let Some(event) = state.outbox.pop_front() {
                    return Some((event, state));
                }
                if state.done {
                    return None;
                }

                let event = match state.sse.next().await {
                    Some(Ok(event)) => event,
                    Some(Err(err)) => {
                        return Some((
                            Err(ProviderError::Other(format!("SSE read error: {err}"))),
                            state,
                        ));
                    }
                    None => {
                        state.parser.finish(&mut state.outbox);
                        state.done = true;
                        continue;
                    }
                };

                let data = event.data.trim();
                if data == "[DONE]" {
                    state.parser.finish(&mut state.outbox);
                    state.done = true;
                    continue;
                }
                if data.is_empty() {
                    continue;
                }

                let Ok(value) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if state.parser.process_value(value, &mut state.outbox) {
                    state.done = true;
                }
            }
        },
    )
}

impl ResponsesSseParser {
    fn process_value(
        &mut self,
        value: Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) -> bool {
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => self.output_text_delta(&value, out),
            Some("response.reasoning_summary_text.delta") => {
                self.reasoning_summary_delta(&value, out);
            }
            Some("response.reasoning_summary_text.done")
            | Some("response.reasoning_summary_part.done") => self.reasoning_summary_done(&value),
            Some("response.output_item.added") => self.output_item_added(&value),
            Some("response.function_call_arguments.delta") => self.tool_arguments_delta(&value),
            Some("response.function_call_arguments.done") => self.tool_arguments_done(&value),
            Some("response.output_item.done") => self.output_item_done(&value, out),
            Some("response.completed") | Some("response.incomplete") => {
                self.terminal_response(&value, out);
                return true;
            }
            Some("response.failed") => {
                self.failed_response(&value, out);
                return true;
            }
            Some("error") | Some("response.error") => {
                out.push_back(Err(stream_error(&value)));
                return true;
            }
            Some("response.reasoning_text.delta") | Some("response.reasoning_text.done") => {}
            _ => {}
        }
        false
    }

    fn output_text_delta(
        &self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if let Some(delta) = value.get("delta").and_then(Value::as_str) {
            if !delta.is_empty() {
                out.push_back(Ok(StreamEvent::ContentDelta(delta.to_string())));
            }
        }
    }

    fn reasoning_summary_delta(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let Some((key, output_index)) = reasoning_key(value) else {
            return;
        };
        let Some(delta) = value.get("delta").and_then(Value::as_str) else {
            return;
        };

        self.pending_reasoning
            .entry(key.clone())
            .or_insert_with(|| PendingReasoning::new(key.clone(), output_index))
            .text
            .push_str(delta);

        if !delta.is_empty() {
            out.push_back(Ok(StreamEvent::ThinkingDelta {
                text: delta.to_string(),
            }));
        }
    }

    fn reasoning_summary_done(&mut self, value: &Value) {
        let Some((key, output_index)) = reasoning_key(value) else {
            return;
        };
        let text = value
            .get("text")
            .or_else(|| value.pointer("/part/text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let pending = self
            .pending_reasoning
            .entry(key.clone())
            .or_insert_with(|| PendingReasoning::new(key, output_index));
        if !text.is_empty() {
            pending.text = text;
        }
    }

    fn output_item_added(&mut self, value: &Value) {
        if let Some(tool) = pending_tool_from_item(value.get("item")) {
            self.pending_tools.insert(tool.item_id.clone(), tool);
        }
    }

    fn tool_arguments_delta(&mut self, value: &Value) {
        let (Some(item_id), Some(delta)) = (
            value.get("item_id").and_then(Value::as_str),
            value.get("delta").and_then(Value::as_str),
        ) else {
            return;
        };
        self.pending_tools
            .entry(item_id.to_string())
            .or_insert_with(|| PendingToolCall {
                item_id: item_id.to_string(),
                ..PendingToolCall::default()
            })
            .arguments
            .push_str(delta);
    }

    fn tool_arguments_done(&mut self, value: &Value) {
        let Some(item_id) = value.get("item_id").and_then(Value::as_str) else {
            return;
        };
        let tool = self
            .pending_tools
            .entry(item_id.to_string())
            .or_insert_with(|| PendingToolCall {
                item_id: item_id.to_string(),
                ..PendingToolCall::default()
            });
        if let Some(arguments) = value.get("arguments").and_then(Value::as_str) {
            tool.arguments = arguments.to_string();
        }
        if let Some(name) = value.get("name").and_then(Value::as_str) {
            tool.name = name.to_string();
        }
    }

    fn output_item_done(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        match value.pointer("/item/type").and_then(Value::as_str) {
            Some("function_call") => {
                if let Some(tool) = self.completed_tool_call(value.get("item")) {
                    self.emit_tool_use(tool, out);
                }
            }
            Some("reasoning") => self.reasoning_item_done(value, out),
            _ => {}
        }
    }

    fn terminal_response(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if let Some(response) = value.get("response") {
            self.usage = usage_from_response(response.get("usage"));
            self.capture_reasoning_from_response(response, out);
            let status = response.get("status").and_then(Value::as_str);
            let incomplete_reason = response
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str);
            self.emit_terminal(status, incomplete_reason, out);
        } else {
            self.emit_terminal(None, None, out);
        }
    }

    fn failed_response(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if let Some(response) = value.get("response") {
            self.usage = usage_from_response(response.get("usage"));
        }
        out.push_back(Err(stream_error(value)));
        self.emitted_terminal = true;
    }

    fn capture_reasoning_from_response(
        &mut self,
        response: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let Some(output) = response.get("output").and_then(Value::as_array) else {
            return;
        };
        for (output_index, item) in output.iter().enumerate() {
            if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                self.capture_reasoning_item(item, Some(output_index), out);
            }
        }
    }

    fn reasoning_item_done(
        &mut self,
        value: &Value,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let output_index = value
            .get("output_index")
            .and_then(Value::as_u64)
            .map(|i| i as usize);
        if let Some(item) = value.get("item") {
            self.capture_reasoning_item(item, output_index, out);
        }
    }

    fn capture_reasoning_item(
        &mut self,
        item: &Value,
        output_index: Option<usize>,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let item_id = item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let encrypted_content = item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .map(str::to_string);

        if let Some(summary) = item.get("summary").and_then(Value::as_array) {
            for (summary_index, part) in summary.iter().enumerate() {
                let key = ReasoningKey {
                    item_id: item_id.clone(),
                    summary_index,
                };
                let pending = self
                    .pending_reasoning
                    .entry(key.clone())
                    .or_insert_with(|| PendingReasoning::new(key, output_index));
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    pending.text = text.to_string();
                }
                pending.encrypted_content = encrypted_content.clone();
                pending.output_index = pending.output_index.or(output_index);
            }
        }

        self.emit_reasoning_for_item(&item_id, out);
    }

    fn emit_reasoning_for_item(
        &mut self,
        item_id: &str,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let mut keys = self
            .pending_reasoning
            .keys()
            .filter(|key| key.item_id == item_id)
            .cloned()
            .collect::<Vec<_>>();
        keys.sort_by_key(|key| key.summary_index);

        for key in keys {
            self.emit_reasoning_key(&key, out);
        }
    }

    fn emit_all_reasoning(&mut self, out: &mut VecDeque<Result<StreamEvent, ProviderError>>) {
        let mut keys = self.pending_reasoning.keys().cloned().collect::<Vec<_>>();
        keys.sort_by(|a, b| {
            a.item_id
                .cmp(&b.item_id)
                .then(a.summary_index.cmp(&b.summary_index))
        });
        for key in keys {
            self.emit_reasoning_key(&key, out);
        }
    }

    fn emit_reasoning_key(
        &mut self,
        key: &ReasoningKey,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        let Some(pending) = self.pending_reasoning.get_mut(key) else {
            return;
        };
        if pending.emitted {
            return;
        }
        pending.emitted = true;
        out.push_back(Ok(StreamEvent::ThinkingBlock {
            text: pending.text.clone(),
            provider: ThinkingProvider::OpenAIResponses,
            metadata: ThinkingMetadata::openai_responses(
                (!pending.item_id.is_empty()).then_some(pending.item_id.clone()),
                pending.output_index,
                pending.summary_index,
                pending.encrypted_content.clone(),
            ),
        }));
    }

    fn completed_tool_call(&mut self, item: Option<&Value>) -> Option<PendingToolCall> {
        let item = item?;
        let item_id = item
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !item_id.is_empty() && self.emitted_tool_items.contains(&item_id) {
            return None;
        }

        let mut tool = if item_id.is_empty() {
            PendingToolCall::default()
        } else {
            self.pending_tools.remove(&item_id).unwrap_or_default()
        };
        if tool.item_id.is_empty() {
            tool.item_id = item_id;
        }
        if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
            tool.call_id = call_id.to_string();
        }
        if let Some(name) = item.get("name").and_then(Value::as_str) {
            tool.name = name.to_string();
        }
        if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
            tool.arguments = arguments.to_string();
        }
        (!tool.call_id.is_empty() || !tool.name.is_empty()).then_some(tool)
    }

    fn emit_tool_use(
        &mut self,
        tool: PendingToolCall,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if !tool.item_id.is_empty() {
            self.emitted_tool_items.insert(tool.item_id.clone());
        }
        self.saw_tool_use = true;
        out.push_back(Ok(StreamEvent::ToolUse {
            id: combined_tool_use_id(&tool.call_id, &tool.item_id),
            name: tool.name,
            input: parse_tool_arguments(&tool.arguments),
        }));
    }

    fn emit_pending_tools(&mut self, out: &mut VecDeque<Result<StreamEvent, ProviderError>>) {
        let tools = std::mem::take(&mut self.pending_tools);
        for (_, tool) in tools {
            if !tool.item_id.is_empty() && self.emitted_tool_items.contains(&tool.item_id) {
                continue;
            }
            self.emit_tool_use(tool, out);
        }
    }

    fn emit_terminal(
        &mut self,
        status: Option<&str>,
        incomplete_reason: Option<&str>,
        out: &mut VecDeque<Result<StreamEvent, ProviderError>>,
    ) {
        if self.emitted_terminal {
            return;
        }
        self.emit_all_reasoning(out);
        self.emit_pending_tools(out);
        out.push_back(Ok(StreamEvent::Usage(self.usage.clone())));
        out.push_back(Ok(StreamEvent::MessageDelta {
            stop_reason: responses_stop_reason(status, incomplete_reason, self.saw_tool_use),
        }));
        out.push_back(Ok(StreamEvent::Done));
        self.emitted_terminal = true;
    }

    fn finish(&mut self, out: &mut VecDeque<Result<StreamEvent, ProviderError>>) {
        self.emit_terminal(None, None, out);
    }
}

impl PendingReasoning {
    fn new(key: ReasoningKey, output_index: Option<usize>) -> Self {
        Self {
            item_id: key.item_id,
            output_index,
            summary_index: key.summary_index,
            text: String::new(),
            encrypted_content: None,
            emitted: false,
        }
    }
}

fn reasoning_key(value: &Value) -> Option<(ReasoningKey, Option<usize>)> {
    let item_id = value.get("item_id")?.as_str()?.to_string();
    let summary_index = value.get("summary_index")?.as_u64()? as usize;
    let output_index = value
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|i| i as usize);
    Some((
        ReasoningKey {
            item_id,
            summary_index,
        },
        output_index,
    ))
}

fn pending_tool_from_item(item: Option<&Value>) -> Option<PendingToolCall> {
    let item = item?;
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return None;
    }
    let item_id = item
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Some(PendingToolCall {
        call_id: item
            .get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        item_id,
        name: item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        arguments: item
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

fn tool_call_from_item(item: &Value) -> Option<PendingToolCall> {
    pending_tool_from_item(Some(item))
}

fn tool_to_content(tool: PendingToolCall) -> Content {
    Content::ToolUse {
        id: combined_tool_use_id(&tool.call_id, &tool.item_id),
        name: tool.name,
        input: parse_tool_arguments(&tool.arguments),
    }
}

fn parse_tool_arguments(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()))
    }
}

fn responses_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    saw_tool_use: bool,
) -> StopReason {
    if saw_tool_use {
        return StopReason::ToolUse;
    }
    match (status, incomplete_reason) {
        (Some("incomplete"), Some("max_output_tokens")) => StopReason::MaxTokens,
        (Some("incomplete"), Some("content_filter")) => StopReason::EndTurn,
        _ => StopReason::EndTurn,
    }
}

fn stream_error(value: &Value) -> ProviderError {
    let message = value
        .pointer("/response/error/message")
        .or_else(|| value.pointer("/error/message"))
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("OpenAI Responses stream failed")
        .to_string();
    ProviderError::Api {
        status: 500,
        message,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::SystemBlock;

    #[test]
    fn request_includes_reasoning_and_encrypted_content() {
        let req = Request {
            model: "gpt-5".into(),
            system: Some(vec![SystemBlock::text("be brief")]),
            messages: vec![Message::user_text("solve")],
            tools: vec![],
            max_tokens: 128,
            temperature: None,
        };
        let reasoning = ReasoningConfig {
            effort: "medium".into(),
            summary: "auto".into(),
        };

        let body = build_request_body(&req, Some(&reasoning), true);

        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn request_replays_openai_reasoning_and_tool_state() {
        let req = Request {
            model: "gpt-5".into(),
            system: None,
            messages: vec![
                Message::assistant(vec![
                    Content::thinking(
                        "summary",
                        ThinkingProvider::OpenAIResponses,
                        ThinkingMetadata::openai_responses(
                            Some("rs_1".into()),
                            Some(0),
                            0,
                            Some("enc".into()),
                        ),
                    ),
                    Content::ToolUse {
                        id: "call_1|fc_1".into(),
                        name: "bash".into(),
                        input: json!({"command":"echo hi"}),
                    },
                ]),
                Message::user(vec![Content::tool_result("call_1|fc_1", "hi", false)]),
            ],
            tools: vec![],
            max_tokens: 128,
            temperature: None,
        };

        let body = build_request_body(&req, None, true);
        let input = body["input"].as_array().unwrap();

        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["id"], "rs_1");
        assert_eq!(input[0]["encrypted_content"], "enc");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["id"], "fc_1");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
    }

    #[test]
    fn non_streaming_response_decodes_reasoning_text_and_tools() {
        let raw = json!({
            "status": "completed",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type":"summary_text", "text":"checked constraints"}],
                    "encrypted_content": "opaque"
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "content": [{"type":"output_text", "text":"answer"}]
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{\"command\":\"echo hi\"}"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 7}
        });

        let response = convert_response_value(&raw).unwrap();

        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.usage.input_tokens, 10);
        assert!(matches!(
            &response.content[0],
            Content::Thinking {
                text,
                provider: ThinkingProvider::OpenAIResponses,
                metadata:
                    ThinkingMetadata::OpenAIResponses {
                        item_id: Some(item_id),
                        output_index: Some(0),
                        summary_index: 0,
                        encrypted_content: Some(encrypted),
                    },
            } if text == "checked constraints" && item_id == "rs_1" && encrypted == "opaque"
        ));
        assert!(matches!(&response.content[1], Content::Text { text, .. } if text == "answer"));
        assert!(matches!(
            &response.content[2],
            Content::ToolUse { id, name, input }
                if id == "call_1|fc_1" && name == "bash" && input["command"] == "echo hi"
        ));
    }

    #[test]
    fn streaming_reasoning_summary_emits_delta_and_final_block() {
        let mut parser = ResponsesSseParser::default();
        let mut out = VecDeque::new();

        parser.process_value(
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": "checked"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "text": "checked constraints"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type":"summary_text", "text":"checked constraints"}],
                    "encrypted_content": "opaque"
                }
            }),
            &mut out,
        );

        assert!(matches!(
            out.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingDelta { text } if text == "checked"
        ));
        assert!(matches!(
            out.pop_front().unwrap().unwrap(),
            StreamEvent::ThinkingBlock {
                text,
                provider: ThinkingProvider::OpenAIResponses,
                metadata:
                    ThinkingMetadata::OpenAIResponses {
                        item_id: Some(item_id),
                        output_index: Some(0),
                        summary_index: 0,
                        encrypted_content: Some(encrypted),
                    },
            } if text == "checked constraints" && item_id == "rs_1" && encrypted == "opaque"
        ));
        assert!(out.is_empty());
    }

    #[test]
    fn streaming_ignores_raw_reasoning_text_events() {
        let mut parser = ResponsesSseParser::default();
        let mut out = VecDeque::new();

        parser.process_value(
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "raw chain of thought"
            }),
            &mut out,
        );

        assert!(out.is_empty());
    }

    #[test]
    fn streaming_tool_call_emits_atomic_tool_use() {
        let mut parser = ResponsesSseParser::default();
        let mut out = VecDeque::new();

        parser.process_value(
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": ""
                }
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "{\"command\":"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "name": "bash",
                "arguments": "{\"command\":\"echo hi\"}"
            }),
            &mut out,
        );
        parser.process_value(
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{\"command\":\"echo hi\"}"
                }
            }),
            &mut out,
        );

        assert!(matches!(
            out.pop_front().unwrap().unwrap(),
            StreamEvent::ToolUse { id, name, input }
                if id == "call_1|fc_1" && name == "bash" && input["command"] == "echo hi"
        ));
        assert!(out.is_empty());
    }
}
