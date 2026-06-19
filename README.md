# HolmesGPT AG-UI Bridge

Python HTTP proxy that exposes HolmesGPT's stable `/api/chat` endpoint as an AG-UI-compatible streaming endpoint.

## Run locally

```bash
uv sync
BRIDGE_HOLMES_BASE_URL=http://localhost:8080 uv run holmesgpt-ag-ui-bridge
```

The AG-UI endpoint is:

```text
POST http://localhost:8080/api/agui/chat
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIDGE_HOLMES_BASE_URL` | `http://localhost:8080` | Base URL for the HolmesGPT server running `server.py`. |
| `BRIDGE_HOLMES_API_KEY` | unset | Optional API key forwarded as `X-API-Key` to HolmesGPT. |
| `BRIDGE_REQUEST_TIMEOUT_SECONDS` | `300` | Timeout for HolmesGPT requests. |
| `BRIDGE_HOST` | `0.0.0.0` | Bind host for this bridge. |
| `BRIDGE_PORT` | `8080` | Bind port for this bridge. |

## Docker

```bash
docker build -t holmesgpt-ag-ui-bridge .
docker run --rm -p 8080:8080 \
  -e BRIDGE_HOLMES_BASE_URL=http://host.docker.internal:8080 \
  holmesgpt-ag-ui-bridge
```

## Behavior

The bridge uses the official `ag-ui-protocol` Python package for AG-UI request and event models. It converts `RunAgentInput` into a HolmesGPT `ChatRequest`, calls HolmesGPT `/api/chat` with `stream=true`, parses Holmes SSE events, and emits AG-UI lifecycle, text, tool-call, and error events.
