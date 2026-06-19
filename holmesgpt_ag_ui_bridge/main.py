from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .app import create_app
from .config import Settings
from .openai_apps import create_agui_to_openai_app, create_openai_to_agui_app


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    settings = Settings(
        request_timeout_seconds=args.timeout,
        holmes_base_url=getattr(args, "holmes_base_url", None) or os.getenv("BRIDGE_HOLMES_BASE_URL", "http://localhost:8080"),
        holmes_api_key=getattr(args, "holmes_api_key", None) or os.getenv("BRIDGE_HOLMES_API_KEY"),
        openai_base_url=getattr(args, "openai_base_url", None) or os.getenv("BRIDGE_OPENAI_BASE_URL", "https://api.openai.com"),
        openai_api_key=getattr(args, "openai_api_key", None) or os.getenv("OPENAI_API_KEY"),
        openai_model=getattr(args, "openai_model", None) or os.getenv("BRIDGE_OPENAI_MODEL", "gpt-4.1-mini"),
        openai_surface=getattr(args, "openai_surface", None) or os.getenv("BRIDGE_OPENAI_SURFACE", "responses"),
        agui_url=getattr(args, "agui_url", None) or os.getenv("BRIDGE_AGUI_URL", "http://localhost:8080/api/agui/chat"),
        agui_api_key=getattr(args, "agui_api_key", None) or os.getenv("BRIDGE_AGUI_API_KEY"),
        openai_compat_api_key=getattr(args, "api_key", None) or os.getenv("BRIDGE_OPENAI_COMPAT_API_KEY"),
    )
    app = args.app_factory(settings)
    uvicorn.run(app, host=args.host, port=args.port)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="holmesgpt-ag-ui-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_holmes_parser(subparsers)
    _add_agui_to_openai_parser(subparsers)
    _add_openai_to_agui_parser(subparsers)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=os.getenv("BRIDGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BRIDGE_PORT", "8080")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("BRIDGE_REQUEST_TIMEOUT_SECONDS", "300")))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))


def _add_holmes_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("agui-to-holmes", help="Expose an AG-UI endpoint backed by HolmesGPT /api/chat.")
    _add_common(parser)
    parser.add_argument("--holmes-base-url", default=os.getenv("BRIDGE_HOLMES_BASE_URL"))
    parser.add_argument("--holmes-api-key", default=os.getenv("BRIDGE_HOLMES_API_KEY"))
    parser.set_defaults(app_factory=create_app)


def _add_agui_to_openai_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("agui-to-openai", help="Expose an AG-UI endpoint backed by an OpenAI-compatible API.")
    _add_common(parser)
    parser.add_argument("--openai-base-url", default=os.getenv("BRIDGE_OPENAI_BASE_URL"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--openai-model", default=os.getenv("BRIDGE_OPENAI_MODEL"))
    parser.add_argument("--openai-surface", choices=["responses", "chat"], default=os.getenv("BRIDGE_OPENAI_SURFACE"))
    parser.set_defaults(app_factory=create_agui_to_openai_app)


def _add_openai_to_agui_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("openai-to-agui", help="Expose OpenAI-compatible endpoints backed by an AG-UI agent.")
    _add_common(parser)
    parser.add_argument("--agui-url", default=os.getenv("BRIDGE_AGUI_URL"))
    parser.add_argument("--agui-api-key", default=os.getenv("BRIDGE_AGUI_API_KEY"))
    parser.add_argument("--api-key", default=os.getenv("BRIDGE_OPENAI_COMPAT_API_KEY"), help="Optional inbound OpenAI-compatible bearer token.")
    parser.add_argument("--openai-model", default=os.getenv("BRIDGE_OPENAI_MODEL"))
    parser.set_defaults(app_factory=create_openai_to_agui_app)


if __name__ == "__main__":
    main()
