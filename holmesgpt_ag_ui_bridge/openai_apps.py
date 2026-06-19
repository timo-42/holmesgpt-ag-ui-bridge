from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from ag_ui.core import EventType, RunAgentInput, RunErrorEvent
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agui import encode_event
from .config import Settings
from .openai_adapter import (
    AguiClient,
    OpenAIClient,
    agui_events_to_chat_response,
    agui_events_to_response,
    agui_to_chat_completions,
    agui_to_responses,
    chat_completions_to_agui_input,
    openai_chat_events_to_agui,
    openai_response_events_to_agui,
    responses_to_agui_input,
)

logger = logging.getLogger(__name__)


def create_agui_to_openai_app(
    settings: Settings | None = None,
    client: OpenAIClient | None = None,
) -> FastAPI:
    settings = settings or Settings()
    openai = client or OpenAIClient(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )

    app = FastAPI(title="AG-UI to OpenAI Bridge")
    _add_cors(app, settings)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/api/agui/chat/health")
    async def agui_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/agui/chat")
    async def agui_chat(input_data: RunAgentInput, request: Request):
        encoder = EventEncoder(accept=request.headers.get("accept"))
        surface = _surface_for_request(input_data, settings)
        if surface == "chat":
            payload = agui_to_chat_completions(input_data, model=settings.openai_model)
            upstream = openai.stream_chat_completions(payload)
            events = openai_chat_events_to_agui(upstream, input_data)
        else:
            payload = agui_to_responses(input_data, model=settings.openai_model)
            upstream = openai.stream_responses(payload)
            events = openai_response_events_to_agui(upstream, input_data)

        async def event_stream():
            try:
                async for event in events:
                    yield encode_event(encoder, event)
            except httpx.HTTPStatusError as exc:
                yield encode_event(
                    encoder,
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"OpenAI upstream returned HTTP {exc.response.status_code}: {exc.response.text}",
                        code="OPENAI_HTTP_ERROR",
                    ),
                )
            except httpx.HTTPError as exc:
                yield encode_event(
                    encoder,
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"Failed to reach OpenAI upstream: {exc}",
                        code="OPENAI_CONNECTION_ERROR",
                    ),
                )

        return StreamingResponse(event_stream(), media_type=encoder.get_content_type())

    return app


def create_openai_to_agui_app(
    settings: Settings | None = None,
    client: AguiClient | None = None,
) -> FastAPI:
    settings = settings or Settings()
    agui = client or AguiClient(
        url=settings.agui_url,
        api_key=settings.agui_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )

    app = FastAPI(title="OpenAI to AG-UI Bridge")
    _add_cors(app, settings)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
        _check_inbound_auth(settings, authorization)
        payload = await request.json()
        input_data = chat_completions_to_agui_input(payload)
        model = payload.get("model") or settings.openai_model
        stream = bool(payload.get("stream"))
        result = await agui_events_to_chat_response(agui.stream(input_data), model=model, stream=stream)
        if stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return JSONResponse(content=result)

    @app.post("/v1/responses")
    async def responses(request: Request, authorization: str | None = Header(default=None)):
        _check_inbound_auth(settings, authorization)
        payload = await request.json()
        input_data = responses_to_agui_input(payload)
        model = payload.get("model") or settings.openai_model
        stream = bool(payload.get("stream"))
        result = await agui_events_to_response(agui.stream(input_data), model=model, stream=stream)
        if stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return JSONResponse(content=result)

    return app


def _add_cors(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _surface_for_request(input_data: RunAgentInput, settings: Settings) -> str:
    state = input_data.state if isinstance(input_data.state, dict) else {}
    forwarded = input_data.forwarded_props if isinstance(input_data.forwarded_props, dict) else {}
    surface = forwarded.get("openai_surface") or state.get("openai_surface") or settings.openai_surface
    return "chat" if str(surface).lower() in {"chat", "chat_completions", "chat-completions"} else "responses"


def _check_inbound_auth(settings: Settings, authorization: str | None) -> None:
    expected = settings.openai_compat_api_key
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
