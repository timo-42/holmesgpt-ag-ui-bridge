import json

import pytest
from ag_ui.core import (
    AssistantMessage,
    Context,
    FunctionCall,
    ImageInputContent,
    InputContentUrlSource,
    RunAgentInput,
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
