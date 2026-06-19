from fastapi.testclient import TestClient

from holmesgpt_ag_ui_bridge.config import Settings
from holmesgpt_ag_ui_bridge.holmes import HolmesSSEvent
from holmesgpt_ag_ui_bridge.openai_apps import create_agui_to_openai_app, create_openai_to_agui_app


class FakeOpenAIClient:
    async def stream_responses(self, payload):
        assert payload["input"][0]["content"] == "hello"
        yield HolmesSSEvent(event="response.output_text.delta", data={"type": "response.output_text.delta", "delta": "hi"})

    async def stream_chat_completions(self, payload):
        assert payload["messages"][0]["content"] == "hello"
        yield HolmesSSEvent(event=None, data={"choices": [{"delta": {"content": "hi"}}]})


class FakeAguiClient:
    async def stream(self, input_data):
        assert input_data.messages[0].content == "hello"
        yield {"type": "TEXT_MESSAGE_CONTENT", "delta": "hi"}


def test_agui_to_openai_endpoint_streams_agui_events():
    app = create_agui_to_openai_app(
        settings=Settings(openai_surface="responses"),
        client=FakeOpenAIClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/agui/chat",
        json={
            "threadId": "thread-1",
            "runId": "run-1",
            "state": {},
            "forwardedProps": {},
            "messages": [{"id": "msg-1", "role": "user", "content": "hello"}],
            "tools": [],
            "context": [],
        },
    )

    assert response.status_code == 200
    assert '"type":"TEXT_MESSAGE_CONTENT"' in response.text
    assert '"delta":"hi"' in response.text


def test_openai_to_agui_chat_non_streaming_endpoint():
    app = create_openai_to_agui_app(client=FakeAguiClient())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"


def test_openai_to_agui_rejects_invalid_bearer_token():
    app = create_openai_to_agui_app(
        settings=Settings(openai_compat_api_key="secret"),
        client=FakeAguiClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-test", "input": "hello"},
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
