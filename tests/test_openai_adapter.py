import json

import pytest
from ag_ui.core import (
    AssistantMessage,
    FunctionCall,
    RunAgentInput,
    Tool,
    ToolCall,
    UserMessage,
)

from holmesgpt_ag_ui_bridge.openai_adapter import (
    HolmesSSEvent,
    agui_events_to_chat_response,
    agui_to_chat_completions,
    agui_to_responses,
    chat_completions_to_agui_input,
    openai_chat_events_to_agui,
    responses_to_agui_input,
)


def test_agui_to_chat_completions_maps_messages_and_tools():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={"temperature": 0.2},
        forwarded_props={"model": "gpt-test"},
        messages=[UserMessage(id="msg-1", role="user", content="hello")],
        tools=[Tool(name="lookup", description="Look up data", parameters={"type": "object"})],
        context=[],
    )

    payload = agui_to_chat_completions(input_data, model="fallback")

    assert payload["model"] == "gpt-test"
    assert payload["temperature"] == 0.2
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["tools"][0]["function"]["name"] == "lookup"


def test_agui_to_responses_maps_messages_and_tools():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", role="user", content="hello")],
        tools=[Tool(name="lookup", description="Look up data", parameters={"type": "object"})],
        context=[],
    )

    payload = agui_to_responses(input_data, model="gpt-test")

    assert payload["model"] == "gpt-test"
    assert payload["input"] == [{"role": "user", "content": "hello"}]
    assert payload["tools"][0]["name"] == "lookup"


def test_chat_completions_to_agui_input_maps_tools_and_messages():
    input_data = chat_completions_to_agui_input(
        {
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "description": "Look up data"},
                }
            ],
        }
    )

    assert isinstance(input_data.messages[0], UserMessage)
    assert isinstance(input_data.messages[1], AssistantMessage)
    assert input_data.messages[1].tool_calls[0].function.name == "lookup"
    assert input_data.messages[2].tool_call_id == "call-1"
    assert input_data.tools[0].name == "lookup"


def test_responses_to_agui_input_maps_string_input():
    input_data = responses_to_agui_input({"model": "gpt-test", "input": "hello"})

    assert isinstance(input_data.messages[0], UserMessage)
    assert input_data.messages[0].content == "hello"


@pytest.mark.asyncio
async def test_openai_chat_stream_maps_to_agui_events():
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", role="user", content="hello")],
        tools=[],
        context=[],
    )

    async def events():
        yield HolmesSSEvent(
            event=None,
            data={"choices": [{"delta": {"content": "hi"}}]},
        )
        yield HolmesSSEvent(
            event=None,
            data={
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "lookup", "arguments": "{\"q\""},
                                }
                            ]
                        }
                    }
                ]
            },
        )
        yield HolmesSSEvent(
            event=None,
            data={
                "choices": [
                    {"delta": {"tool_calls": [{"id": "call-1", "function": {"arguments": ":\"x\"}"}}]}}
                ]
            },
        )

    agui_events = [json.loads(event.model_dump_json(by_alias=True, exclude_none=True)) async for event in openai_chat_events_to_agui(events(), input_data)]

    assert [event["type"] for event in agui_events] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_ARGS",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_END",
        "RUN_FINISHED",
    ]


@pytest.mark.asyncio
async def test_agui_events_to_chat_non_streaming_response():
    async def events():
        yield {"type": "TEXT_MESSAGE_CONTENT", "delta": "hello"}
        yield {"type": "TOOL_CALL_START", "toolCallId": "call-1", "toolCallName": "lookup"}
        yield {"type": "TOOL_CALL_ARGS", "toolCallId": "call-1", "delta": "{}"}

    result = await agui_events_to_chat_response(events(), model="gpt-test", stream=False)

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"
