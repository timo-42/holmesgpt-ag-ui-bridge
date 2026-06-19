from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    host = os.getenv("BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("BRIDGE_PORT", "8080"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
