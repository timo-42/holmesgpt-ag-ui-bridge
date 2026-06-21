import json

import pytest
from ag_ui.core import (
    ActivityMessage,
    AssistantMessage,
    AudioInputContent,
    BinaryInputContent,
    Context,
    DeveloperMessage,
    DocumentInputContent,
    FunctionCall,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    ReasoningMessage,
    ResumeEntry,
    RunAgentInput,
    SystemMessage,
    TextInputContent,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from holmesgpt_ag_ui_bridge.agui import agui_to_holmes_chat, holmes_to_agui_events
from holmesgpt_ag_ui_bridge.holmes import HolmesSSEvent, parse_sse_lines


def test_agui_input_maps_to_holmes_chat_request():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={"model": "gpt-test", "request_source": "unit"},
        forwarded_props={"user_id": "user-1"},
        messages=[
            UserMessage(id="msg-1", content="What broke?", role="user"),
        ],
        tools=[
            Tool(
                name="open_dashboard",
                description="Open a dashboard",
                parameters={"type": "object", "properties": {"url": {"type": "string"}}},
            )
        ],
        context=[Context(description="page", value="Alert: api latency high")],
    )

    payload = agui_to_holmes_chat(input_data)

    assert payload["ask"] == "What broke?"
    assert payload["stream"] is True
    assert payload["conversation_id"] == "thread-1"
    assert payload["model"] == "gpt-test"
    assert payload["request_source"] == "unit"
    assert payload["user_id"] == "user-1"
    assert payload["frontend_tools"][0]["name"] == "open_dashboard"
    assert payload["conversation_history"][0]["role"] == "system"
    assert any("Alert: api latency high" in item["content"] for item in payload["conversation_history"])


def test_trailing_tool_message_maps_to_frontend_tool_results():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[
            UserMessage(id="msg-1", content="Open it", role="user"),
            AssistantMessage(
                id="assistant-1",
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=FunctionCall(name="open_dashboard", arguments="{}"),
                    )
                ],
            ),
            ToolMessage(
                id="tool-msg-1",
                role="tool",
                content="done",
                tool_call_id="call-1",
            ),
        ],
        tools=[],
        context=[],
    )

    payload = agui_to_holmes_chat(input_data)

    assert payload["ask"] == ""
    assert payload["frontend_tool_results"] == [
        {"tool_call_id": "call-1", "tool_name": "open_dashboard", "result": "done"}
    ]
    assert any(item == {"role": "user", "content": "Open it"} for item in payload["conversation_history"])


def test_tool_message_error_maps_to_frontend_tool_result():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[
            UserMessage(id="msg-1", content="Open it", role="user"),
            AssistantMessage(
                id="assistant-1",
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=FunctionCall(name="open_dashboard", arguments="{}"),
                    )
                ],
            ),
            ToolMessage(
                id="tool-msg-1",
                role="tool",
                content="failed",
                tool_call_id="call-1",
                error="permission denied",
            ),
        ],
        tools=[],
        context=[],
    )

    payload = agui_to_holmes_chat(input_data)

    assert payload["frontend_tool_results"] == [
        {
            "tool_call_id": "call-1",
            "tool_name": "open_dashboard",
            "result": "failed",
            "error": "permission denied",
        }
    ]


def test_system_and_developer_messages_map_to_holmes_system_history():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[
            SystemMessage(id="sys-1", role="system", content="Cluster: prod"),
            DeveloperMessage(id="dev-1", role="developer", content="Prefer concise answers"),
            UserMessage(id="msg-1", content="What broke?", role="user"),
        ],
        tools=[],
        context=[],
    )

    payload = agui_to_holmes_chat(input_data)

    assert {"role": "system", "content": "Cluster: prod"} in payload["conversation_history"]
    assert {"role": "system", "content": "Prefer concise answers"} in payload["conversation_history"]


def test_activity_and_reasoning_messages_map_to_holmes_system_history():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[
            ActivityMessage(
                id="activity-1",
                role="activity",
                activity_type="search",
                content={"query": "latency"},
            ),
            ReasoningMessage(id="reasoning-1", role="reasoning", content="Checking alerts"),
            UserMessage(id="msg-1", content="What broke?", role="user"),
        ],
        tools=[],
        context=[],
    )

    payload = agui_to_holmes_chat(input_data)

    assert {"role": "system", "content": 'activity:search: {"query":"latency"}'} in payload["conversation_history"]
    assert {"role": "system", "content": "reasoning: Checking alerts"} in payload["conversation_history"]


def test_user_image_parts_map_to_holmes_images():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[
            UserMessage(
                id="msg-1",
                role="user",
                content=[
                    TextInputContent(type="text", text="What is in this graph?"),
                    ImageInputContent(
                        type="image",
                        source=InputContentUrlSource(
                            type="url",
                            value="https://example.test/graph.png",
                            mime_type="image/png",
                        ),
                    ),
                ],
            ),
        ],
        tools=[],
        context=[],
    )

    payload = agui_to_holmes_chat(input_data)

    assert payload["ask"] == "What is in this graph?\n[image: https://example.test/graph.png]"
    assert payload["images"] == [{"url": "https://example.test/graph.png", "format": "image/png"}]


def test_non_image_multimodal_parts_map_to_attachment_text():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[
            UserMessage(
                id="msg-1",
                role="user",
                content=[
                    TextInputContent(type="text", text="Review these"),
                    AudioInputContent(
                        type="audio",
                        source=InputContentUrlSource(
                            type="url",
                            value="https://example.test/audio.wav",
                            mime_type="audio/wav",
                        ),
                    ),
                    DocumentInputContent(
                        type="document",
                        source=InputContentDataSource(
                            type="data",
                            value="ZG9j",
                            mime_type="application/pdf",
                        ),
                    ),
                    BinaryInputContent(
                        type="binary",
                        mime_type="application/octet-stream",
                        filename="dump.bin",
                        data="AAAA",
                    ),
                ],
            ),
        ],
        tools=[],
        context=[],
    )

    payload = agui_to_holmes_chat(input_data)

    assert payload["ask"] == "\n".join(
        [
            "Review these",
            "[audio: https://example.test/audio.wav (audio/wav)]",
            "[document: inline application/pdf]",
            "[binary: dump.bin (application/octet-stream)]",
        ]
    )


def test_resume_entries_map_to_holmes_resume_and_tool_decisions():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-2",
        state={},
        forwarded_props={},
        messages=[],
        tools=[],
        context=[],
        resume=[
            ResumeEntry(
                interrupt_id="call-1",
                status="resolved",
                payload={"approved": True, "note": "ok"},
            )
        ],
    )

    payload = agui_to_holmes_chat(input_data)

    assert payload["resume"] == [
        {
            "interrupt_id": "call-1",
            "status": "resolved",
            "payload": {"approved": True, "note": "ok"},
        }
    ]
    assert payload["tool_decisions"] == [
        {
            "interrupt_id": "call-1",
            "tool_call_id": "call-1",
            "status": "resolved",
            "approved": True,
            "payload": {"approved": True, "note": "ok"},
        }
    ]


async def _aiter(lines):
    for line in lines:
        yield line


@pytest.mark.asyncio
async def test_parse_sse_lines_parses_holmes_events():
    events = [
        event
        async for event in parse_sse_lines(
            _aiter(
                [
                    "event: ai_message",
                    'data: {"content":"hello"}',
                    "",
                    "event: ai_answer_end",
                    'data: {"analysis":"done"}',
                    "",
                ]
            )
        )
    ]

    assert events == [
        HolmesSSEvent(event="ai_message", data={"content": "hello"}),
        HolmesSSEvent(event="ai_answer_end", data={"analysis": "done"}),
    ]


@pytest.mark.asyncio
async def test_holmes_stream_maps_to_agui_events():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="hi", role="user")],
        tools=[],
        context=[],
    )

    async def holmes_events():
        yield HolmesSSEvent(event="ai_message", data={"content": "hello"})
        yield HolmesSSEvent(event="ai_answer_end", data={"analysis": " world"})

    events = [event async for event in holmes_to_agui_events(holmes_events(), input_data)]
    dumped = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) for event in events]

    assert [event["type"] for event in dumped] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert dumped[2]["delta"] == "hello"
    assert dumped[3]["delta"] == " world"


@pytest.mark.asyncio
async def test_holmes_tool_result_maps_to_agui_result_before_end():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="run tool", role="user")],
        tools=[],
        context=[],
    )

    async def holmes_events():
        yield HolmesSSEvent(
            event="tool_calling_result",
            data={"tool_call_id": "call-1", "result": {"status": "ok"}},
        )

    events = [event async for event in holmes_to_agui_events(holmes_events(), input_data)]
    dumped = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) for event in events]

    assert [event["type"] for event in dumped] == [
        "RUN_STARTED",
        "TOOL_CALL_RESULT",
        "TOOL_CALL_END",
        "RUN_FINISHED",
    ]
    assert dumped[1]["toolCallId"] == "call-1"
    assert dumped[1]["content"] == '{"status":"ok"}'
    assert dumped[2]["toolCallId"] == "call-1"


@pytest.mark.asyncio
async def test_holmes_tool_events_preserve_parent_and_result_message_ids():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="run tool", role="user")],
        tools=[],
        context=[],
    )

    async def holmes_events():
        yield HolmesSSEvent(
            event="start_tool_calling",
            data={
                "toolCallId": "call-1",
                "name": "lookup",
                "params": {"q": "latency"},
                "parentMessageId": "assistant-1",
            },
        )
        yield HolmesSSEvent(
            event="tool_calling_result",
            data={
                "toolCallId": "call-1",
                "toolMessageId": "tool-msg-1",
                "output": "done",
            },
        )

    events = [event async for event in holmes_to_agui_events(holmes_events(), input_data)]
    dumped = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) for event in events]

    assert dumped[1]["type"] == "TOOL_CALL_START"
    assert dumped[1]["toolCallId"] == "call-1"
    assert dumped[1]["parentMessageId"] == "assistant-1"
    assert dumped[3]["type"] == "TOOL_CALL_RESULT"
    assert dumped[3]["messageId"] == "tool-msg-1"
    assert dumped[3]["toolCallId"] == "call-1"


@pytest.mark.asyncio
async def test_approval_required_maps_to_interrupt_outcome():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="approve", role="user")],
        tools=[],
        context=[],
    )

    async def holmes_events():
        yield HolmesSSEvent(
            event="approval_required",
            data={
                "pending_frontend_tool_calls": [
                    {
                        "interrupt_id": "interrupt-1",
                        "tool_call_id": "call-1",
                        "tool_name": "deploy",
                        "arguments": {"service": "api"},
                        "message": "Deploy api?",
                        "response_schema": {"type": "object"},
                    }
                ]
            },
        )

    events = [event async for event in holmes_to_agui_events(holmes_events(), input_data)]
    dumped = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) for event in events]

    assert [event["type"] for event in dumped] == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "RUN_FINISHED",
    ]
    assert dumped[4]["outcome"]["type"] == "interrupt"
    assert dumped[4]["outcome"]["interrupts"] == [
        {
            "id": "interrupt-1",
            "reason": "tool_approval",
            "message": "Deploy api?",
            "toolCallId": "call-1",
            "responseSchema": {"type": "object"},
            "metadata": {
                "tool_call": {
                    "interrupt_id": "interrupt-1",
                    "tool_call_id": "call-1",
                    "tool_name": "deploy",
                    "arguments": {"service": "api"},
                    "message": "Deploy api?",
                    "response_schema": {"type": "object"},
                }
            },
        }
    ]


@pytest.mark.asyncio
async def test_holmes_state_activity_and_reasoning_events_map_to_agui_events():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="hi", role="user")],
        tools=[],
        context=[],
    )

    async def holmes_events():
        yield HolmesSSEvent(event="state_snapshot", data={"state": {"service": "api"}})
        yield HolmesSSEvent(
            event="activity_snapshot",
            data={"message_id": "activity-1", "activity_type": "search", "content": {"status": "running"}},
        )
        yield HolmesSSEvent(
            event="activity_delta",
            data={
                "message_id": "activity-1",
                "activity_type": "search",
                "patch": [{"op": "replace", "path": "/status", "value": "done"}],
            },
        )
        yield HolmesSSEvent(event="reasoning_start", data={"message_id": "reason-1"})
        yield HolmesSSEvent(event="reasoning_delta", data={"message_id": "reason-1", "delta": "thinking"})
        yield HolmesSSEvent(event="reasoning_end", data={"message_id": "reason-1"})

    events = [event async for event in holmes_to_agui_events(holmes_events(), input_data)]
    dumped = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) for event in events]

    assert [event["type"] for event in dumped] == [
        "RUN_STARTED",
        "STATE_SNAPSHOT",
        "ACTIVITY_SNAPSHOT",
        "ACTIVITY_DELTA",
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
        "RUN_FINISHED",
    ]
    assert dumped[1]["snapshot"] == {"service": "api"}
    assert dumped[2]["messageId"] == "activity-1"
    assert dumped[3]["patch"] == [{"op": "replace", "path": "/status", "value": "done"}]
    assert dumped[6]["messageId"] == "reason-1"
    assert dumped[6]["delta"] == "thinking"


@pytest.mark.asyncio
async def test_unknown_holmes_event_maps_to_raw_agui_event():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="hi", role="user")],
        tools=[],
        context=[],
    )

    async def holmes_events():
        yield HolmesSSEvent(event="diagnostic", data={"detail": "kept"})

    events = [event async for event in holmes_to_agui_events(holmes_events(), input_data)]
    dumped = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) for event in events]

    assert [event["type"] for event in dumped] == [
        "RUN_STARTED",
        "RAW",
        "RUN_FINISHED",
    ]
    assert dumped[1]["source"] == "holmes"
    assert dumped[1]["event"] == {"event": "diagnostic", "data": {"detail": "kept"}}
