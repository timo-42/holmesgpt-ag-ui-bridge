# HolmesGPT AG-UI Bridge

Python HTTP proxy that exposes HolmesGPT's stable `/api/chat` endpoint as an AG-UI-compatible streaming endpoint.

## Run locally

```bash
uv sync
uv run holmesgpt-ag-ui-bridge agui-to-holmes
```

The AG-UI endpoint is:

```text
POST http://localhost:8080/api/agui/chat
```

If HolmesGPT is also running on `localhost:8080`, run one process on a different port. For example, keep HolmesGPT on `8080` and run the bridge on `8090`:

```bash
BRIDGE_PORT=8090 BRIDGE_HOLMES_BASE_URL=http://localhost:8080 uv run holmesgpt-ag-ui-bridge agui-to-holmes
```

Or keep the bridge on `8080` and point it at a HolmesGPT server on another port:

```bash
BRIDGE_HOLMES_BASE_URL=http://localhost:18080 uv run holmesgpt-ag-ui-bridge agui-to-holmes
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIDGE_HOLMES_BASE_URL` | `http://localhost:8080` | Base URL for the HolmesGPT server running `server.py`. |
| `BRIDGE_HOLMES_API_KEY` | unset | Optional API key forwarded as `X-API-Key` to HolmesGPT. |
| `BRIDGE_OPENAI_BASE_URL` | `https://api.openai.com` | Base URL for OpenAI or an OpenAI-compatible API. |
| `OPENAI_API_KEY` | unset | API key used by the `agui-to-openai` bridge. |
| `BRIDGE_OPENAI_MODEL` | `gpt-4.1-mini` | Default OpenAI model for OpenAI adapters. |
| `BRIDGE_OPENAI_SURFACE` | `responses` | Upstream OpenAI API surface for `agui-to-openai`: `responses` or `chat`. |
| `BRIDGE_AGUI_URL` | `http://localhost:8080/api/agui/chat` | Upstream AG-UI endpoint for `openai-to-agui`. |
| `BRIDGE_AGUI_API_KEY` | unset | Optional bearer token forwarded to the upstream AG-UI endpoint. |
| `BRIDGE_OPENAI_COMPAT_API_KEY` | unset | Optional inbound bearer token required by OpenAI-compatible endpoints. |
| `BRIDGE_REQUEST_TIMEOUT_SECONDS` | `300` | Timeout for HolmesGPT requests. |
| `BRIDGE_HOST` | `0.0.0.0` | Bind host for this bridge. |
| `BRIDGE_PORT` | `8080` | Bind port for this bridge. |

## Bridges

Expose AG-UI backed by HolmesGPT:

```bash
uv run holmesgpt-ag-ui-bridge agui-to-holmes \
  --holmes-base-url http://localhost:8080
```

Expose AG-UI backed by OpenAI Responses:

```bash
OPENAI_API_KEY=... uv run holmesgpt-ag-ui-bridge agui-to-openai \
  --openai-model gpt-4.1-mini \
  --openai-surface responses
```

Expose OpenAI-compatible Chat Completions and Responses endpoints backed by AG-UI:

```bash
uv run holmesgpt-ag-ui-bridge openai-to-agui \
  --agui-url http://localhost:8080/api/agui/chat
```

The reverse bridge exposes:

```text
POST /v1/chat/completions
POST /v1/responses
```

## Docker

Prebuilt multi-arch images are published to [Docker Hub](https://hub.docker.com/r/kenobi42/holmesgpt-ag-ui-bridge).

```bash
docker pull kenobi42/holmesgpt-ag-ui-bridge:latest
```

Run the published image:

```bash
docker run --rm -p 8080:8080 \
  -e BRIDGE_HOLMES_BASE_URL=http://host.docker.internal:8080 \
  kenobi42/holmesgpt-ag-ui-bridge:latest
```

The image runs as a non-root user by default and supports read-only container filesystems:

```bash
docker run --rm --read-only --tmpfs /tmp -p 8080:8080 \
  -e BRIDGE_HOLMES_BASE_URL=http://host.docker.internal:8080 \
  kenobi42/holmesgpt-ag-ui-bridge:latest
```

Run an OpenAI adapter from the same image by overriding the command:

```bash
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY=... \
  kenobi42/holmesgpt-ag-ui-bridge:latest \
  holmesgpt-ag-ui-bridge agui-to-openai
```

Or build locally:

```bash
docker build -t holmesgpt-ag-ui-bridge .
docker run --rm -p 8080:8080 \
  -e BRIDGE_HOLMES_BASE_URL=http://host.docker.internal:8080 \
  holmesgpt-ag-ui-bridge
```

## Behavior

The bridge uses the official `ag-ui-protocol` Python package for AG-UI request and event models. It can translate AG-UI to HolmesGPT, AG-UI to OpenAI-compatible APIs, and OpenAI-compatible Chat Completions or Responses requests back to AG-UI. Streaming text and function/tool call events are translated between protocols.

For HolmesGPT, frontend tool approval pauses are exposed as AG-UI interrupt outcomes and later `resume` entries are forwarded back as Holmes tool decisions. Holmes state, activity, and reasoning events are mapped to AG-UI events when their payloads match AG-UI fields; otherwise Holmes-specific events are preserved as `RAW` passthrough events.
