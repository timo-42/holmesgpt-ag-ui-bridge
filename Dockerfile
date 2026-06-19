FROM ghcr.io/astral-sh/uv:0.5.29-python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    BRIDGE_HOST=0.0.0.0 \
    BRIDGE_PORT=8080

COPY pyproject.toml uv.lock README.md ./
COPY holmesgpt_ag_ui_bridge ./holmesgpt_ag_ui_bridge

RUN uv sync --no-dev

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "holmesgpt-ag-ui-bridge"]
