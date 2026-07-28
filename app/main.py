"""
Process entrypoint.

  python -m app.main
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    _bootstrap_paths()

    try:
        from dotenv import load_dotenv

        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
    except Exception:
        pass

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger("app")

    bind_host = (os.getenv("APP_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("APP_PORT") or "8090")
    reload = os.getenv("API_RELOAD", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    logger.info("embedding-service starting bind=%s port=%s", bind_host, port)
    logger.info(
        "EMBED_PROVIDER=%s EMBED_MODEL=%s",
        os.getenv("EMBED_PROVIDER", "openai"),
        os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    )

    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=bind_host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
