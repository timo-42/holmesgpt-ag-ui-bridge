FROM ghcr.io/astral-sh/uv:0.5.29-python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY pyproject.toml uv.lock README.md ./
COPY holmesgpt_ag_ui_bridge ./holmesgpt_ag_ui_bridge

RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim-bookworm

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    BRIDGE_HOST=0.0.0.0 \
    BRIDGE_PORT=8080

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --system --gid 10001 bridge \
    && useradd --system --uid 10001 --gid 10001 --home-dir /tmp --shell /usr/sbin/nologin bridge

USER 10001:10001

EXPOSE 8080

CMD ["holmesgpt-ag-ui-bridge", "agui-to-holmes"]
