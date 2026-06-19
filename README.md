# HolmesGPT AG-UI Bridge

Python HTTP proxy that exposes HolmesGPT's stable `/api/chat` endpoint as an AG-UI-compatible streaming endpoint.

## Run locally

```bash
uv sync
uv run holmesgpt-ag-ui-bridge
```

The AG-UI endpoint is:

```text
POST http://localhost:8080/api/agui/chat
```

If HolmesGPT is also running on `localhost:8080`, run one process on a different port. For example, keep HolmesGPT on `8080` and run the bridge on `8090`:

```bash
BRIDGE_PORT=8090 BRIDGE_HOLMES_BASE_URL=http://localhost:8080 uv run holmesgpt-ag-ui-bridge
```

Or keep the bridge on `8080` and point it at a HolmesGPT server on another port:

```bash
BRIDGE_HOLMES_BASE_URL=http://localhost:18080 uv run holmesgpt-ag-ui-bridge
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

Prebuilt multi-arch images are published to Docker Hub:

```text
https://hub.docker.com/r/kenobi42/holmesgpt-ag-ui-bridge
```

Run the published image:

```bash
docker run --rm -p 8080:8080 \
  -e BRIDGE_HOLMES_BASE_URL=http://host.docker.internal:8080 \
  kenobi42/holmesgpt-ag-ui-bridge:latest
```

Or build locally:

```bash
docker build -t holmesgpt-ag-ui-bridge .
docker run --rm -p 8080:8080 \
  -e BRIDGE_HOLMES_BASE_URL=http://host.docker.internal:8080 \
  holmesgpt-ag-ui-bridge
```

## Behavior

The bridge uses the official `ag-ui-protocol` Python package for AG-UI request and event models. It converts `RunAgentInput` into a HolmesGPT `ChatRequest`, calls HolmesGPT `/api/chat` with `stream=true`, parses Holmes SSE events, and emits AG-UI lifecycle, text, tool-call, and error events.
