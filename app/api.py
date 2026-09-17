"""
FastAPI — embedding-only service.

  POST /v1/embed
  POST /v1/embed/batch
  GET  /health /ready /v1/info
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import __version__

logger = logging.getLogger("app")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _cors_origins() -> list[str]:
    raw = (os.getenv("CORS_ORIGINS") or "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _check_bearer(
    authorization: Optional[str] = Header(default=None),
) -> None:
    expected = (os.getenv("API_BEARER_TOKEN") or "").strip()
    if not expected:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")
    input_type: Literal["query", "document"] = Field(
        default="query",
        description="query vs document (for models that distinguish them)",
    )


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int
    provider: str
    model: str
    index_key: str
    input_type: str


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    input_type: Literal["query", "document"] = Field(default="document")


class EmbedBatchResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int
    provider: str
    model: str
    index_key: str
    input_type: str


def _validate_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(t) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds EMBED_MAX_CHARS={max_chars}",
        )
    return t


def create_app() -> FastAPI:
    app = FastAPI(
        title=os.getenv("APP_NAME", "embedding-service"),
        version=__version__,
        description=(
            "Standalone embedding microservice. "
            "OpenAI / local HuggingFace (e.g. BAAI/bge-m3). "
            "Does not store vectors — consumers write their own DB."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        logger.info("Starting embedding-service v%s", __version__)
        try:
            from app.embeddings import get_service

            rd = get_service().initialize()
            logger.info("Embedding readiness: %s", rd)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("Embedding init failed at startup (non-fatal): %s", exc)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": os.getenv("APP_NAME", "embedding-service"),
            "version": __version__,
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        from app.embeddings import get_service

        svc = get_service()
        if not svc.ready:
            svc.initialize()
        rd = svc.readiness()
        if not rd.get("ready"):
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", **rd},
            )
        return {"status": "ready", **rd}

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        from app.embeddings import get_service

        svc = get_service()
        rd = svc.readiness()
        return {
            "service": os.getenv("APP_NAME", "embedding-service"),
            "version": __version__,
            "design": "embed-only",
            "stores_vectors": False,
            **rd,
        }

    @app.post("/v1/embed", response_model=EmbedResponse)
    def embed(
        body: EmbedRequest,
        _: None = Depends(_check_bearer),
    ) -> EmbedResponse:
        from app.embeddings import get_service

        max_chars = _env_int("EMBED_MAX_CHARS", 8000)
        text = _validate_text(body.text, max_chars)
        svc = get_service()
        try:
            vec = svc.embed_texts([text], body.input_type)[0]
        except Exception as exc:  # noqa: BLE001
            logger.error("embed failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=502,
                detail={"error": str(exc), "error_code": "embed_failed"},
            ) from exc

        ident = svc.identity
        if ident is None:
            raise HTTPException(status_code=503, detail="not ready")
        return EmbedResponse(
            embedding=vec,
            dim=ident.dim,
            provider=ident.provider,
            model=ident.model,
            index_key=ident.index_key,
            input_type=body.input_type,
        )

    @app.post("/v1/embed/batch", response_model=EmbedBatchResponse)
    def embed_batch(
        body: EmbedBatchRequest,
        _: None = Depends(_check_bearer),
    ) -> EmbedBatchResponse:
        from app.embeddings import get_service

        batch_max = _env_int("EMBED_BATCH_MAX", 64)
        max_chars = _env_int("EMBED_MAX_CHARS", 8000)
        if len(body.texts) > batch_max:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"batch size {len(body.texts)} exceeds "
                    f"EMBED_BATCH_MAX={batch_max}"
                ),
            )
        cleaned: list[str] = []
        for i, t in enumerate(body.texts):
            try:
                cleaned.append(_validate_text(t, max_chars))
            except HTTPException as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"texts[{i}]: {exc.detail}",
                ) from exc

        svc = get_service()
        try:
            vectors = svc.embed_texts(cleaned, body.input_type)
        except Exception as exc:  # noqa: BLE001
            logger.error("embed/batch failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=502,
                detail={"error": str(exc), "error_code": "embed_failed"},
            ) from exc

        if len(vectors) != len(cleaned):
            raise HTTPException(
                status_code=502,
                detail="provider returned mismatched embedding count",
            )

        ident = svc.identity
        if ident is None:
            raise HTTPException(status_code=503, detail="not ready")
        return EmbedBatchResponse(
            embeddings=vectors,
            dim=ident.dim,
            count=len(vectors),
            provider=ident.provider,
            model=ident.model,
            index_key=ident.index_key,
            input_type=body.input_type,
        )

    return app


app = create_app()
