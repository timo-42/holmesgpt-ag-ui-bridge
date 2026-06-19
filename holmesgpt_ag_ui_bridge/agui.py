from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

from .holmes import HolmesSSEvent


HOLMES_SYSTEM_PROMPT = (
    "You are Holmes, an AI assistant for observability. You use available "
    "metrics, alerts, logs, runbooks, and infrastructure context to perform "
    "root cause analysis."
)


def agui_to_holmes_chat(input_data: RunAgentInput) -> dict[str, Any]:
    messages = list(input_data.messages or [])
    last_user_index = _find_last_user_message(messages)
    last_tool_messages = _trailing_tool_messages(messages)

    ask = ""
    if last_user_index is not None and not last_tool_messages:
        ask = _message_text(messages[last_user_index])

    if last_tool_messages:
        history_limit = len(messages) - len(last_tool_messages)
    else:
        history_limit = last_user_index if last_user_index is not None else len(messages)
    conversation_history = _build_conversation_history(messages[:history_limit], input_data)
    tool_names = _tool_call_names(messages[:history_limit])

    payload: dict[str, Any] = {
        "ask": ask,
        "conversation_history": conversation_history,
        "stream": True,
        "conversation_id": input_data.thread_id,
        "request_type": "agui_chat",
        "frontend_tools": [_tool_to_holmes(tool) for tool in (input_data.tools or [])],
    }

    if last_user_index is not None and not last_tool_messages:
        images = _message_images(messages[last_user_index])
        if images:
            payload["images"] = images

    frontend_tool_results = [
        _tool_message_to_result(msg, tool_names) for msg in last_tool_messages
    ]
    if frontend_tool_results:
        payload["frontend_tool_results"] = frontend_tool_results

    state = input_data.state if isinstance(input_data.state, dict) else {}
    forwarded_props = (
        input_data.forwarded_props if isinstance(input_data.forwarded_props, dict) else {}
    )
    for key in (
        "model",
        "enable_tool_approval",
        "tool_decisions",
        "additional_system_prompt",
        "response_format",
        "behavior_controls",
        "user_id",
        "user_email",
        "request_source",
        "source_ref",
        "meta",
        "is_internal",
    ):
        if key in forwarded_props:
            payload[key] = forwarded_props[key]
        elif key in state:
            payload[key] = state[key]

    return {key: value for key, value in payload.items() if value not in (None, [], {})}


def encode_event(encoder: EventEncoder, event: Any) -> str:
    return encoder.encode(event)


async def holmes_to_agui_events(
    holmes_events: AsyncIterator[HolmesSSEvent],
    input_data: RunAgentInput,
) -> AsyncIterator[Any]:
    yield RunStartedEvent(
        type=EventType.RUN_STARTED,
        thread_id=input_data.thread_id,
        run_id=input_data.run_id,
        parent_run_id=input_data.parent_run_id,
    )

    message_id = str(uuid.uuid4())
    text_started = False

    async for holmes_event in holmes_events:
        event_type = holmes_event.event
        data = holmes_event.data

        if event_type in {"ai_message", "ai_answer_end"}:
            content = str(data.get("content") or data.get("analysis") or "")
            if content:
                if not text_started:
                    text_started = True
                    yield TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant",
                    )
                yield TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=content,
                )
            continue

        if event_type == "start_tool_calling":
            if text_started:
                yield TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
                text_started = False
                message_id = str(uuid.uuid4())
            tool_call_id = _tool_call_id(data)
            yield ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
                tool_call_name=_tool_name(data),
            )
            args = data.get("params") or data.get("arguments") or data.get("input")
            if args is not None:
                yield ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=tool_call_id,
                    delta=json.dumps(args),
                )
            continue

        if event_type == "tool_calling_result":
            yield ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=_tool_call_id(data),
            )
            continue

        if event_type == "approval_required":
            if text_started:
                yield TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
                text_started = False
            for call in data.get("pending_frontend_tool_calls") or []:
                tool_call_id = str(call.get("tool_call_id") or call.get("id") or uuid.uuid4())
                yield ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_call_id,
                    tool_call_name=str(call.get("tool_name") or call.get("name") or "frontend_tool"),
                )
                yield ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=tool_call_id,
                    delta=json.dumps(call.get("arguments") or call.get("params") or {}),
                )
                yield ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tool_call_id,
                )
            yield RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
                result=data,
            )
            return

        if event_type == "error":
            if text_started:
                yield TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(data.get("description") or data.get("msg") or data),
                code=str(data.get("error_code")) if data.get("error_code") is not None else None,
            )
            return

    if text_started:
        yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)

    yield RunFinishedEvent(
        type=EventType.RUN_FINISHED,
        thread_id=input_data.thread_id,
        run_id=input_data.run_id,
    )


def _build_conversation_history(messages: list[Any], input_data: RunAgentInput) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [{"role": "system", "content": HOLMES_SYSTEM_PROMPT}]

    for context in input_data.context or []:
        description = getattr(context, "description", "context")
        value = getattr(context, "value", "")
        if value:
            history.append({"role": "system", "content": f"{description}: {value}"})

    for message in messages:
        role = getattr(message, "role", "")
        if role in {"system", "developer"}:
            history.append({"role": "system", "content": _message_text(message)})
        elif role in {"user", "assistant"}:
            history.append({"role": role, "content": _message_text(message)})
        elif role == "tool":
            history.append({"role": "assistant", "content": _message_text(message)})

    return [item for item in history if item["content"]]


def _find_last_user_message(messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "role", None) == "user":
            return index
    return None


def _trailing_tool_messages(messages: list[Any]) -> list[Any]:
    result: list[Any] = []
    for message in reversed(messages):
        if getattr(message, "role", None) != "tool":
            break
        result.append(message)
    return list(reversed(result))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            part_type = getattr(part, "type", None)
            if part_type == "text":
                parts.append(str(getattr(part, "text", "")))
            elif part_type == "image":
                source = getattr(part, "source", None)
                value = getattr(source, "value", "") if source else ""
                if value:
                    parts.append(f"[image: {value}]")
            elif part_type:
                parts.append(f"[{part_type} attachment]")
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _tool_to_holmes(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "mode": "pause",
    }


def _message_images(message: Any) -> list[str | dict[str, Any]]:
    content = getattr(message, "content", "")
    if not isinstance(content, list):
        return []

    images: list[str | dict[str, Any]] = []
    for part in content:
        if getattr(part, "type", None) != "image":
            continue
        source = getattr(part, "source", None)
        if source is None:
            continue
        source_type = getattr(source, "type", None)
        value = getattr(source, "value", None)
        mime_type = getattr(source, "mime_type", None)
        if not value:
            continue
        if source_type == "data":
            images.append(f"data:{mime_type or 'image/png'};base64,{value}")
        elif source_type == "url":
            if mime_type:
                images.append({"url": value, "format": mime_type})
            else:
                images.append(value)
    return images


def _tool_call_names(messages: list[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", None)
            if name:
                names[str(tool_call.id)] = str(name)
    return names


def _tool_message_to_result(message: Any, tool_names: dict[str, str]) -> dict[str, Any]:
    tool_call_id = message.tool_call_id
    return {
        "tool_call_id": tool_call_id,
        "tool_name": getattr(message, "name", None)
        or tool_names.get(str(tool_call_id))
        or "frontend_tool",
        "result": _message_text(message),
    }


def _tool_call_id(data: dict[str, Any]) -> str:
    return str(data.get("tool_call_id") or data.get("id") or uuid.uuid4())


def _tool_name(data: dict[str, Any]) -> str:
    return str(data.get("tool_name") or data.get("name") or "tool")
