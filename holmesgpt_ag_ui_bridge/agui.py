from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    ActivityDeltaEvent,
    ActivitySnapshotEvent,
    EventType,
    Interrupt,
    RawEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunFinishedInterruptOutcome,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
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

    if input_data.resume:
        resume_entries = [
            entry.model_dump(mode="json", by_alias=False, exclude_none=True)
            for entry in input_data.resume
        ]
        payload["resume"] = resume_entries
        payload["tool_decisions"] = [
            _resume_entry_to_tool_decision(entry) for entry in input_data.resume
        ]

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
    reasoning_message_id: str | None = None

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
            parent_message_id = _tool_parent_message_id(data)
            if text_started:
                parent_message_id = parent_message_id or message_id
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
                parent_message_id=parent_message_id,
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
            tool_call_id = _tool_call_id(data)
            result_content = _tool_result_content(data)
            if result_content is not None:
                yield ToolCallResultEvent(
                    type=EventType.TOOL_CALL_RESULT,
                    message_id=_tool_result_message_id(data),
                    tool_call_id=tool_call_id,
                    content=result_content,
                    role="tool",
                )
            yield ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tool_call_id,
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
                tool_call_id = _tool_call_id(call)
                yield ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_call_id,
                    tool_call_name=str(call.get("tool_name") or call.get("name") or "frontend_tool"),
                    parent_message_id=_tool_parent_message_id(call),
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
                outcome=RunFinishedInterruptOutcome(
                    interrupts=[
                        _frontend_tool_call_to_interrupt(call)
                        for call in data.get("pending_frontend_tool_calls") or []
                    ]
                    or [_event_to_interrupt(data)]
                ),
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

        if event_type in {"state_snapshot", "state"} and "state" in data:
            yield StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=data["state"],
                raw_event={"event": event_type, "data": data},
            )
            continue

        if event_type == "state_delta" and "delta" in data:
            yield StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=data["delta"],
                raw_event={"event": event_type, "data": data},
            )
            continue

        if event_type in {"activity_snapshot", "activity"} and "content" in data:
            yield ActivitySnapshotEvent(
                type=EventType.ACTIVITY_SNAPSHOT,
                message_id=_event_message_id(data),
                activity_type=str(data.get("activity_type") or data.get("activityType") or "activity"),
                content=data["content"],
                replace=bool(data.get("replace", True)),
                raw_event={"event": event_type, "data": data},
            )
            continue

        if event_type == "activity_delta" and "patch" in data:
            yield ActivityDeltaEvent(
                type=EventType.ACTIVITY_DELTA,
                message_id=_event_message_id(data),
                activity_type=str(data.get("activity_type") or data.get("activityType") or "activity"),
                patch=data["patch"],
                raw_event={"event": event_type, "data": data},
            )
            continue

        if event_type in {"reasoning_start", "thinking_start"}:
            reasoning_message_id = _event_message_id(data)
            yield ReasoningStartEvent(
                type=EventType.REASONING_START,
                message_id=reasoning_message_id,
                raw_event={"event": event_type, "data": data},
            )
            yield ReasoningMessageStartEvent(
                type=EventType.REASONING_MESSAGE_START,
                message_id=reasoning_message_id,
                role="reasoning",
                raw_event={"event": event_type, "data": data},
            )
            continue

        if event_type in {"reasoning_message", "reasoning_delta", "thinking_delta"}:
            content = str(data.get("content") or data.get("delta") or data.get("analysis") or "")
            if content:
                if reasoning_message_id is None:
                    reasoning_message_id = _event_message_id(data)
                    yield ReasoningMessageStartEvent(
                        type=EventType.REASONING_MESSAGE_START,
                        message_id=reasoning_message_id,
                        role="reasoning",
                        raw_event={"event": event_type, "data": data},
                    )
                yield ReasoningMessageContentEvent(
                    type=EventType.REASONING_MESSAGE_CONTENT,
                    message_id=reasoning_message_id,
                    delta=content,
                    raw_event={"event": event_type, "data": data},
                )
            continue

        if event_type in {"reasoning_end", "thinking_end"}:
            if reasoning_message_id is not None:
                yield ReasoningMessageEndEvent(
                    type=EventType.REASONING_MESSAGE_END,
                    message_id=reasoning_message_id,
                    raw_event={"event": event_type, "data": data},
                )
                yield ReasoningEndEvent(
                    type=EventType.REASONING_END,
                    message_id=reasoning_message_id,
                    raw_event={"event": event_type, "data": data},
                )
                reasoning_message_id = None
            continue

        yield RawEvent(
            type=EventType.RAW,
            event={"event": event_type, "data": data},
            source="holmes",
        )

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
        elif role == "activity":
            activity_type = getattr(message, "activity_type", "activity")
            content = json.dumps(getattr(message, "content", {}), separators=(",", ":"))
            history.append({"role": "system", "content": f"activity:{activity_type}: {content}"})
        elif role == "reasoning":
            history.append({"role": "system", "content": f"reasoning: {_message_text(message)}"})

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
            elif part_type in {"audio", "video", "document"}:
                parts.append(_sourced_attachment_text(part, str(part_type)))
            elif part_type == "binary":
                parts.append(_binary_attachment_text(part))
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
    result = {
        "tool_call_id": tool_call_id,
        "tool_name": getattr(message, "name", None)
        or tool_names.get(str(tool_call_id))
        or "frontend_tool",
        "result": _message_text(message),
    }
    error = getattr(message, "error", None)
    if error:
        result["error"] = str(error)
    return result


def _tool_call_id(data: dict[str, Any]) -> str:
    nested_call = data.get("tool_call") if isinstance(data.get("tool_call"), dict) else {}
    return str(
        data.get("tool_call_id")
        or data.get("toolCallId")
        or data.get("call_id")
        or data.get("callId")
        or nested_call.get("tool_call_id")
        or nested_call.get("toolCallId")
        or nested_call.get("id")
        or data.get("id")
        or uuid.uuid4()
    )


def _tool_name(data: dict[str, Any]) -> str:
    return str(data.get("tool_name") or data.get("name") or "tool")


def _tool_parent_message_id(data: dict[str, Any]) -> str | None:
    value = data.get("parent_message_id") or data.get("parentMessageId") or data.get("message_id") or data.get("messageId")
    return str(value) if value else None


def _tool_result_message_id(data: dict[str, Any]) -> str:
    return str(
        data.get("message_id")
        or data.get("messageId")
        or data.get("tool_message_id")
        or data.get("toolMessageId")
        or data.get("result_id")
        or data.get("resultId")
        or uuid.uuid4()
    )


def _tool_result_content(data: dict[str, Any]) -> str | None:
    for key in ("result", "output", "content", "data", "error"):
        if key not in data:
            continue
        value = data[key]
        if value is None:
            continue
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))
    return None


def _resume_entry_to_tool_decision(entry: Any) -> dict[str, Any]:
    payload = getattr(entry, "payload", None)
    approved = getattr(entry, "status", None) == "resolved"
    if isinstance(payload, dict) and "approved" in payload:
        approved = bool(payload["approved"])
    return {
        "interrupt_id": entry.interrupt_id,
        "tool_call_id": entry.interrupt_id,
        "status": entry.status,
        "approved": approved,
        "payload": payload,
    }


def _frontend_tool_call_to_interrupt(call: dict[str, Any]) -> Interrupt:
    tool_call_id = _tool_call_id(call)
    interrupt_id = str(call.get("interrupt_id") or call.get("interruptId") or tool_call_id)
    metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {"tool_call": call}
    return Interrupt(
        id=interrupt_id,
        reason=str(call.get("reason") or "tool_approval"),
        message=str(
            call.get("message")
            or call.get("description")
            or f"Approval required for {call.get('tool_name') or call.get('name') or 'frontend_tool'}"
        ),
        tool_call_id=tool_call_id,
        response_schema=call.get("response_schema") or call.get("responseSchema") or call.get("schema"),
        expires_at=call.get("expires_at") or call.get("expiresAt"),
        metadata=metadata,
    )


def _event_to_interrupt(data: dict[str, Any]) -> Interrupt:
    interrupt_id = str(data.get("interrupt_id") or data.get("interruptId") or data.get("id") or uuid.uuid4())
    return Interrupt(
        id=interrupt_id,
        reason=str(data.get("reason") or "approval_required"),
        message=str(data.get("message") or data.get("description") or "Approval required"),
        response_schema=data.get("response_schema") or data.get("responseSchema") or data.get("schema"),
        expires_at=data.get("expires_at") or data.get("expiresAt"),
        metadata={"event": data},
    )


def _event_message_id(data: dict[str, Any]) -> str:
    return str(data.get("message_id") or data.get("messageId") or data.get("id") or uuid.uuid4())


def _sourced_attachment_text(part: Any, part_type: str) -> str:
    source = getattr(part, "source", None)
    if source is None:
        return f"[{part_type} attachment]"
    source_type = getattr(source, "type", None)
    value = getattr(source, "value", None)
    mime_type = getattr(source, "mime_type", None)
    label = mime_type or part_type
    if source_type == "url" and value:
        return f"[{part_type}: {value} ({label})]"
    if source_type == "data":
        return f"[{part_type}: inline {label}]"
    return f"[{part_type} attachment]"


def _binary_attachment_text(part: Any) -> str:
    mime_type = getattr(part, "mime_type", "binary")
    filename = getattr(part, "filename", None)
    url = getattr(part, "url", None)
    identifier = getattr(part, "id", None)
    if url:
        return f"[binary: {url} ({mime_type})]"
    if filename:
        return f"[binary: {filename} ({mime_type})]"
    if identifier:
        return f"[binary: {identifier} ({mime_type})]"
    return f"[binary: inline {mime_type}]"
