from ag_ui.core import RunAgentInput, UserMessage
from fastapi.testclient import TestClient

from holmesgpt_ag_ui_bridge.app import create_app
from holmesgpt_ag_ui_bridge.holmes import HolmesSSEvent


class FakeHolmesClient:
    health_url = "http://fake/healthz"
    model_url = "http://fake/api/model"

    async def get_json(self, url):
        if url == self.model_url:
            return {"model_name": "fake"}
        return {"status": "healthy"}

    async def stream_chat(self, payload):
        assert payload["ask"] == "hello"
        yield HolmesSSEvent(event="ai_message", data={"content": "hi"})
        yield HolmesSSEvent(event="ai_answer_end", data={"analysis": ""})


def test_agui_chat_endpoint_streams_sse():
    app = create_app(client=FakeHolmesClient())
    client = TestClient(app)
    input_data = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        forwarded_props={},
        messages=[UserMessage(id="msg-1", content="hello", role="user")],
        tools=[],
        context=[],
    )

    response = client.post(
        "/api/agui/chat",
        json=input_data.model_dump(mode="json", by_alias=True),
        headers={"accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type":"RUN_STARTED"' in response.text
    assert '"type":"TEXT_MESSAGE_CONTENT"' in response.text
    assert '"delta":"hi"' in response.text
    assert '"type":"RUN_FINISHED"' in response.text
