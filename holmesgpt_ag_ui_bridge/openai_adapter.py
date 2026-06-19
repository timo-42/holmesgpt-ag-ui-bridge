from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from ag_ui.core import (
    AssistantMessage,
    Context,
    EventType,
    FunctionCall,
    RunAgentInput,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from .agui import encode_event
from .holmes import HolmesSSEvent, parse_sse_lines


@dataclass(frozen=True)
class OpenAIClient:
    base_url: str
    api_key: str | None
    timeout_seconds: float

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def stream_chat_completions(self, payload: dict[str, Any]) -> AsyncIterator[HolmesSSEvent]:
        payload = {**payload, "stream": True}
        async for event in self._stream_post("/v1/chat/completions", payload):
            yield event

    async def stream_responses(self, payload: dict[str, Any]) -> AsyncIterator[HolmesSSEvent]:
        payload = {**payload, "stream": True}
        async for event in self._stream_post("/v1/responses", payload):
            yield event

    async def _stream_post(self, path: str, payload: dict[str, Any]) -> AsyncIterator[HolmesSSEvent]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url.rstrip('/')}{path}",
                json=payload,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for event in parse_sse_lines(response.aiter_lines()):
                    if event.data.get("content") == "[DONE]":
                        continue
                    yield event


@dataclass(frozen=True)
class AguiClient:
    url: str
    api_key: str | None
    timeout_seconds: float

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def stream(self, input_data: RunAgentInput) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds, connect=30.0)) as client:
            async with client.stream(
                "POST",
                self.url,
                json=input_data.model_dump(mode="json", by_alias=True, exclude_none=True),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for event in parse_sse_lines(response.aiter_lines()):
                    yield event.data


def agui_to_chat_completions(input_data: RunAgentInput, *, model: str) -> dict[str, Any]:
    state = input_data.state if isinstance(input_data.state, dict) else {}
    forwarded = input_data.forwarded_props if isinstance(input_data.forwarded_props, dict) else {}
    payload: dict[str, Any] = {
        "model": forwarded.get("model") or state.get("model") or model,
        "messages": [_agui_message_to_openai(message) for message in input_data.messages],
    }
    tools = [_agui_tool_to_chat_tool(tool) for tool in input_data.tools or []]
    if tools:
        payload["tools"] = tools
    for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "tool_choice", "response_format"):
        if key in forwarded:
            payload[key] = forwarded[key]
        elif key in state:
            payload[key] = state[key]
    return payload


def agui_to_responses(input_data: RunAgentInput, *, model: str) -> dict[str, Any]:
    state = input_data.state if isinstance(input_data.state, dict) else {}
    forwarded = input_data.forwarded_props if isinstance(input_data.forwarded_props, dict) else {}
    payload: dict[str, Any] = {
        "model": forwarded.get("model") or state.get("model") or model,
        "input": [_agui_message_to_response_input(message) for message in input_data.messages],
    }
    tools = [_agui_tool_to_response_tool(tool) for tool in input_data.tools or []]
    if tools:
        payload["tools"] = tools
    for key in ("temperature", "top_p", "max_output_tokens", "tool_choice", "text", "reasoning"):
        if key in forwarded:
            payload[key] = forwarded[key]
        elif key in state:
            payload[key] = state[key]
    return payload


def chat_completions_to_agui_input(payload: dict[str, Any]) -> RunAgentInput:
    messages = [_openai_message_to_agui(message) for message in payload.get("messages") or []]
    tools = [_chat_tool_to_agui_tool(tool) for tool in payload.get("tools") or []]
    return RunAgentInput(
        thread_id=str(payload.get("user") or payload.get("conversation_id") or uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        state={_k: payload[_k] for _k in ("model", "temperature", "top_p") if _k in payload},
        forwarded_props={"openai_request": {k: v for k, v in payload.items() if k not in {"messages", "tools"}}},
        messages=messages,
        tools=tools,
        context=[],
    )


def responses_to_agui_input(payload: dict[str, Any]) -> RunAgentInput:
    messages = _responses_input_to_agui_messages(payload.get("input"))
    tools = [_response_tool_to_agui_tool(tool) for tool in payload.get("tools") or []]
    return RunAgentInput(
        thread_id=str(payload.get("conversation") or payload.get("user") or uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        state={_k: payload[_k] for _k in ("model", "temperature", "top_p") if _k in payload},
        forwarded_props={"openai_request": {k: v for k, v in payload.items() if k not in {"input", "tools"}}},
        messages=messages,
        tools=tools,
        context=[],
    )


async def openai_chat_events_to_agui(openai_events: AsyncIterator[HolmesSSEvent], input_data: RunAgentInput) -> AsyncIterator[Any]:
    from ag_ui.core import RunErrorEvent, RunFinishedEvent, RunStartedEvent, TextMessageContentEvent, TextMessageEndEvent, TextMessageStartEvent, ToolCallArgsEvent, ToolCallEndEvent, ToolCallStartEvent

    yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=input_data.thread_id, run_id=input_data.run_id)
    message_id = str(uuid.uuid4())
    text_started = False
    tool_names: dict[str, str] = {}

    async for event in openai_events:
        if event.data.get("error"):
            yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(event.data["error"]))
            return
        for choice in event.data.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if not text_started:
                    text_started = True
                    yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant")
                yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=content)
            for tool_call in delta.get("tool_calls") or []:
                tool_id = str(tool_call.get("id") or tool_call.get("index") or uuid.uuid4())
                function = tool_call.get("function") or {}
                name = function.get("name")
                if name and tool_id not in tool_names:
                    tool_names[tool_id] = name
                    yield ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id=tool_id, tool_call_name=name)
                arguments = function.get("arguments")
                if arguments:
                    if tool_id not in tool_names:
                        tool_names[tool_id] = name or "tool"
                        yield ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id=tool_id, tool_call_name=tool_names[tool_id])
                    yield ToolCallArgsEvent(type=EventType.TOOL_CALL_ARGS, tool_call_id=tool_id, delta=arguments)

    if text_started:
        yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)
    for tool_id in tool_names:
        yield ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tool_id)
    yield RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=input_data.thread_id, run_id=input_data.run_id)


async def openai_response_events_to_agui(openai_events: AsyncIterator[HolmesSSEvent], input_data: RunAgentInput) -> AsyncIterator[Any]:
    from ag_ui.core import RunErrorEvent, RunFinishedEvent, RunStartedEvent, TextMessageContentEvent, TextMessageEndEvent, TextMessageStartEvent, ToolCallArgsEvent, ToolCallEndEvent, ToolCallStartEvent

    yield RunStartedEvent(type=EventType.RUN_STARTED, thread_id=input_data.thread_id, run_id=input_data.run_id)
    message_id = str(uuid.uuid4())
    text_started = False
    tool_names: dict[str, str] = {}

    async for event in openai_events:
        data = event.data
        event_type = data.get("type") or event.event
        if event_type in {"response.output_text.delta", "response.text.delta"}:
            delta = str(data.get("delta") or "")
            if delta:
                if not text_started:
                    text_started = True
                    yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant")
                yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=delta)
        elif event_type in {"response.function_call_arguments.delta"}:
            tool_id = str(data.get("item_id") or data.get("output_index") or uuid.uuid4())
            if tool_id not in tool_names:
                tool_names[tool_id] = str(data.get("name") or "tool")
                yield ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id=tool_id, tool_call_name=tool_names[tool_id])
            yield ToolCallArgsEvent(type=EventType.TOOL_CALL_ARGS, tool_call_id=tool_id, delta=str(data.get("delta") or ""))
        elif event_type in {"response.output_item.added"}:
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                tool_id = str(item.get("id") or item.get("call_id") or uuid.uuid4())
                tool_names[tool_id] = str(item.get("name") or "tool")
                yield ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id=tool_id, tool_call_name=tool_names[tool_id])
        elif event_type in {"response.failed", "error"}:
            yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(data.get("error") or data))
            return

    if text_started:
        yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)
    for tool_id in tool_names:
        yield ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tool_id)
    yield RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=input_data.thread_id, run_id=input_data.run_id)


async def agui_events_to_chat_response(events: AsyncIterator[dict[str, Any]], *, model: str, stream: bool) -> AsyncIterator[str] | dict[str, Any]:
    if stream:
        return _agui_events_to_chat_sse(events, model=model)

    text, tool_calls = await _collect_agui(events)
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            for call_id, call in tool_calls.items()
        ]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
    }


async def agui_events_to_response(events: AsyncIterator[dict[str, Any]], *, model: str, stream: bool) -> AsyncIterator[str] | dict[str, Any]:
    if stream:
        return _agui_events_to_response_sse(events, model=model)

    text, tool_calls = await _collect_agui(events)
    output: list[dict[str, Any]] = []
    if text:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    for call_id, call in tool_calls.items():
        output.append({"type": "function_call", "call_id": call_id, "name": call["name"], "arguments": call["arguments"]})
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
    }


async def _agui_events_to_chat_sse(events: AsyncIterator[dict[str, Any]], *, model: str) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    tool_names: dict[str, str] = {}
    async for event in events:
        event_type = event.get("type")
        if event_type == "TEXT_MESSAGE_CONTENT":
            yield _sse_data({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"content": event.get("delta", "")}, "finish_reason": None}]})
        elif event_type == "TOOL_CALL_START":
            tool_id = event["toolCallId"]
            tool_names[tool_id] = event.get("toolCallName", "tool")
            yield _sse_data({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": len(tool_names) - 1, "id": tool_id, "type": "function", "function": {"name": tool_names[tool_id], "arguments": ""}}]}, "finish_reason": None}]})
        elif event_type == "TOOL_CALL_ARGS":
            tool_id = event["toolCallId"]
            yield _sse_data({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": list(tool_names).index(tool_id) if tool_id in tool_names else 0, "function": {"arguments": event.get("delta", "")}}]}, "finish_reason": None}]})
        elif event_type == "RUN_ERROR":
            yield _sse_data({"error": {"message": event.get("message", "AG-UI upstream error"), "type": "agui_error"}})
            return
    yield _sse_data({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"


async def _agui_events_to_response_sse(events: AsyncIterator[dict[str, Any]], *, model: str) -> AsyncIterator[str]:
    response_id = f"resp_{uuid.uuid4().hex}"
    yield _sse_event("response.created", {"type": "response.created", "response": {"id": response_id, "model": model, "status": "in_progress"}})
    async for event in events:
        event_type = event.get("type")
        if event_type == "TEXT_MESSAGE_CONTENT":
            yield _sse_event("response.output_text.delta", {"type": "response.output_text.delta", "item_id": response_id, "delta": event.get("delta", "")})
        elif event_type == "TOOL_CALL_START":
            yield _sse_event("response.output_item.added", {"type": "response.output_item.added", "item": {"id": event["toolCallId"], "type": "function_call", "name": event.get("toolCallName", "tool"), "arguments": ""}})
        elif event_type == "TOOL_CALL_ARGS":
            yield _sse_event("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "item_id": event["toolCallId"], "delta": event.get("delta", "")})
        elif event_type == "RUN_ERROR":
            yield _sse_event("response.failed", {"type": "response.failed", "error": {"message": event.get("message", "AG-UI upstream error")}})
            return
    yield _sse_event("response.completed", {"type": "response.completed", "response": {"id": response_id, "model": model, "status": "completed"}})


async def _collect_agui(events: AsyncIterator[dict[str, Any]]) -> tuple[str, dict[str, dict[str, str]]]:
    text_parts: list[str] = []
    tool_calls: dict[str, dict[str, str]] = {}
    async for event in events:
        event_type = event.get("type")
        if event_type == "TEXT_MESSAGE_CONTENT":
            text_parts.append(str(event.get("delta", "")))
        elif event_type == "TOOL_CALL_START":
            tool_calls[event["toolCallId"]] = {"name": event.get("toolCallName", "tool"), "arguments": ""}
        elif event_type == "TOOL_CALL_ARGS":
            tool_calls.setdefault(event["toolCallId"], {"name": "tool", "arguments": ""})
            tool_calls[event["toolCallId"]]["arguments"] += str(event.get("delta", ""))
        elif event_type == "RUN_ERROR":
            raise RuntimeError(str(event.get("message", "AG-UI upstream error")))
    return "".join(text_parts), tool_calls


def _agui_message_to_openai(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", "user")
    if role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": getattr(message, "content", "")}
    result = {"role": "system" if role == "developer" else role, "content": _message_content_text(message)}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [_agui_tool_call_to_openai(call) for call in tool_calls]
    return result


def _agui_message_to_response_input(message: Any) -> dict[str, Any]:
    return _agui_message_to_openai(message)


def _openai_message_to_agui(message: dict[str, Any]) -> Any:
    role = message.get("role", "user")
    content = _openai_content_to_text(message.get("content", ""))
    if role == "assistant":
        return AssistantMessage(
            id=str(message.get("id") or uuid.uuid4()),
            role="assistant",
            content=content,
            tool_calls=[_openai_tool_call_to_agui(call) for call in message.get("tool_calls") or []] or None,
        )
    if role == "tool":
        return ToolMessage(id=str(message.get("id") or uuid.uuid4()), role="tool", content=content, tool_call_id=str(message.get("tool_call_id") or message.get("toolCallId") or uuid.uuid4()))
    return UserMessage(id=str(message.get("id") or uuid.uuid4()), role="user", content=content)


def _responses_input_to_agui_messages(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [UserMessage(id=str(uuid.uuid4()), role="user", content=value)]
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    messages.append(ToolMessage(id=str(item.get("id") or uuid.uuid4()), role="tool", content=str(item.get("output", "")), tool_call_id=str(item.get("call_id") or uuid.uuid4())))
                else:
                    messages.append(_openai_message_to_agui(item))
        return messages
    return [UserMessage(id=str(uuid.uuid4()), role="user", content=str(value or ""))]


def _message_content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if getattr(part, "type", None) == "text":
                parts.append(getattr(part, "text", ""))
            else:
                parts.append(f"[{getattr(part, 'type', 'attachment')} attachment]")
        return "\n".join(parts)
    return str(content)


def _openai_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _agui_tool_to_chat_tool(tool: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters or {"type": "object", "properties": {}}}}


def _agui_tool_to_response_tool(tool: Any) -> dict[str, Any]:
    return {"type": "function", "name": tool.name, "description": tool.description, "parameters": tool.parameters or {"type": "object", "properties": {}}}


def _chat_tool_to_agui_tool(tool: dict[str, Any]) -> Tool:
    function = tool.get("function") or {}
    return Tool(name=function.get("name") or tool.get("name") or "tool", description=function.get("description") or "", parameters=function.get("parameters"))


def _response_tool_to_agui_tool(tool: dict[str, Any]) -> Tool:
    return Tool(name=tool.get("name") or "tool", description=tool.get("description") or "", parameters=tool.get("parameters"))


def _agui_tool_call_to_openai(call: Any) -> dict[str, Any]:
    return {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}


def _openai_tool_call_to_agui(call: dict[str, Any]) -> ToolCall:
    function = call.get("function") or {}
    return ToolCall(id=str(call.get("id") or uuid.uuid4()), function=FunctionCall(name=function.get("name") or "tool", arguments=function.get("arguments") or "{}"))


def _sse_data(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
