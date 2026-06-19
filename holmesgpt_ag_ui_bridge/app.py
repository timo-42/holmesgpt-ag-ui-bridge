from __future__ import annotations

import logging

import httpx
from ag_ui.core import EventType, RunAgentInput, RunErrorEvent
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .agui import agui_to_holmes_chat, encode_event, holmes_to_agui_events
from .config import Settings
from .holmes import HolmesClient

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, client: HolmesClient | None = None) -> FastAPI:
    settings = settings or Settings()
    holmes = client or HolmesClient(
        base_url=settings.holmes_base_url,
        api_key=settings.holmes_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )

    app = FastAPI(title="HolmesGPT AG-UI Bridge")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/readyz")
    async def readyz():
        try:
            return await holmes.get_json(holmes.health_url)
        except httpx.HTTPError as exc:
            logger.warning("Holmes readiness check failed: %s", exc)
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": str(exc)},
            )

    @app.get("/api/agui/chat/health")
    async def agui_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/model")
    async def get_model():
        try:
            return await holmes.get_json(holmes.model_url)
        except httpx.HTTPError as exc:
            return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.post("/api/agui/chat")
    async def agui_chat(input_data: RunAgentInput, request: Request):
        encoder = EventEncoder(accept=request.headers.get("accept"))
        payload = agui_to_holmes_chat(input_data)

        async def event_stream():
            try:
                async for event in holmes_to_agui_events(holmes.stream_chat(payload), input_data):
                    yield encode_event(encoder, event)
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text
                yield encode_event(
                    encoder,
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"HolmesGPT returned HTTP {exc.response.status_code}: {detail}",
                        code="HOLMES_HTTP_ERROR",
                    ),
                )
            except httpx.HTTPError as exc:
                yield encode_event(
                    encoder,
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"Failed to reach HolmesGPT: {exc}",
                        code="HOLMES_CONNECTION_ERROR",
                    ),
                )

        return StreamingResponse(event_stream(), media_type=encoder.get_content_type())

    return app
