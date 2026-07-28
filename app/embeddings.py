"""
Embedding factory — env only, no hard-coded model in callers.

Switch provider/model via env; consumers must reindex when index_key changes.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("embeddings")

_lock = threading.RLock()
_client: Any = None
_identity: Optional["EmbedIdentity"] = None

_DEFAULT_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "baai/bge-m3": 1024,
    "bge-m3": 1024,
}


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _safe_token(value: str) -> str:
    s = (value or "").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    return s[:120] or "model"


@dataclass(frozen=True)
class EmbedIdentity:
    provider: str
    model: str
    dim: int
    api_key: str
    base_url: Optional[str]
    device: str

    @property
    def index_key(self) -> str:
        return f"{_safe_token(self.provider)}__{_safe_token(self.model)}__d{self.dim}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "index_key": self.index_key,
            "device": self.device,
            "base_url_set": bool(self.base_url),
            "api_key_set": bool(self.api_key),
        }


def resolve_dim(provider: str, model: str, explicit: Optional[int]) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    key = (model or "").strip().lower()
    if key in _DEFAULT_DIMS:
        return _DEFAULT_DIMS[key]
    if key.endswith("/bge-m3") or key.endswith("bge-m3"):
        return 1024
    if provider in {"local", "huggingface", "hf"}:
        return 1024
    if "large" in key:
        return 3072
    if "small" in key or "ada" in key:
        return 1536
    return 1536


def get_embed_identity() -> EmbedIdentity:
    provider = (_env("EMBED_PROVIDER") or "openai").lower()
    if provider in {"hf", "huggingface"}:
        if _env("EMBED_BASE_URL"):
            provider = "huggingface"
        else:
            provider = "local"

    model = _env("EMBED_MODEL") or "text-embedding-3-small"
    explicit_dim = _env_int("EMBED_DIM", None)
    dim = resolve_dim(provider, model, explicit_dim)

    api_key = (
        _env("EMBED_API_KEY")
        or _env("OPENAI_API_KEY")
        or _env("LLM_API_KEY")
    )
    base_url = _env("EMBED_BASE_URL") or _env("OPENAI_BASE_URL") or None
    if base_url == "":
        base_url = None
    device = _env("EMBED_DEVICE") or "cpu"

    return EmbedIdentity(
        provider=provider,
        model=model,
        dim=dim,
        api_key=api_key,
        base_url=base_url,
        device=device,
    )


def get_embeddings(identity: EmbedIdentity | None = None) -> Any:
    """Build embeddings client from env only."""
    ident = identity or get_embed_identity()
    provider = ident.provider

    if provider == "openai":
        return _openai_embeddings(ident)
    if provider in {"local", "huggingface"}:
        if provider == "huggingface" and ident.base_url:
            try:
                return _openai_embeddings(ident)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "huggingface base_url client failed (%s); falling back to local",
                    exc,
                )
        return _local_embeddings(ident)

    raise ValueError(
        f"Unknown EMBED_PROVIDER={provider!r}. Use: openai | local | huggingface"
    )


def _openai_embeddings(ident: EmbedIdentity) -> Any:
    from langchain_openai import OpenAIEmbeddings

    if not ident.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY / EMBED_API_KEY required for EMBED_PROVIDER=openai"
        )

    kwargs: dict[str, Any] = {
        "model": ident.model,
        "api_key": ident.api_key,
    }
    if ident.base_url:
        kwargs["base_url"] = ident.base_url
    explicit = _env_int("EMBED_DIM", None)
    if explicit and explicit > 0 and "ada-002" not in (ident.model or "").lower():
        kwargs["dimensions"] = explicit

    return OpenAIEmbeddings(**kwargs)


def _local_embeddings(ident: EmbedIdentity) -> Any:
    model_name = ident.model or "BAAI/bge-m3"
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require sentence-transformers. "
                "Install: pip install -r requirements-local.txt "
                f"(model={model_name})"
            ) from exc

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": ident.device or "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load local embedding model {model_name!r}: {exc}. "
            "Install: pip install -r requirements-local.txt"
        ) from exc


class EmbeddingService:
    """Process-wide client; re-init when env identity changes."""

    def __init__(self) -> None:
        self._ready = False
        self._last_error: str | None = None
        self.identity: EmbedIdentity | None = None
        self._client: Any = None

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize(self, force: bool = False) -> dict[str, Any]:
        with _lock:
            try:
                ident = get_embed_identity()
                if (
                    not force
                    and self._ready
                    and self.identity is not None
                    and self.identity.index_key == ident.index_key
                    and self._client is not None
                ):
                    return self.readiness()

                self._client = get_embeddings(ident)
                self.identity = ident
                self._ready = True
                self._last_error = None
                logger.info(
                    "Embedding ready provider=%s model=%s dim=%s",
                    ident.provider,
                    ident.model,
                    ident.dim,
                )
                return self.readiness()
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                self._ready = False
                self._client = None
                self.identity = None
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Embedding init failed: %s", self._last_error)
                return self.readiness()

    def readiness(self) -> dict[str, Any]:
        ident = self.identity
        try:
            if ident is None:
                ident = get_embed_identity()
            ident_d = ident.as_dict()
        except Exception as exc:  # noqa: BLE001
            ident_d = {"error": str(exc)}

        return {
            "ready": self._ready,
            "service": "embedding-service",
            "identity": ident_d,
            "error": self._last_error,
            "batch_max": _env_int("EMBED_BATCH_MAX", 64) or 64,
            "max_chars": _env_int("EMBED_MAX_CHARS", 8000) or 8000,
        }

    def embed_query(self, text: str) -> list[float]:
        self._ensure()
        vec = self._client.embed_query(text)
        return self._check_dim(list(vec))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        vectors = self._client.embed_documents(texts)
        out: list[list[float]] = []
        for v in vectors:
            out.append(self._check_dim(list(v)))
        return out

    def _ensure(self) -> None:
        if not self._ready or self._client is None:
            self.initialize()
        if not self._ready or self._client is None:
            raise RuntimeError(self._last_error or "Embedding service not ready")

    def _check_dim(self, vec: list[float]) -> list[float]:
        if self.identity and len(vec) != int(self.identity.dim):
            raise RuntimeError(
                f"Embedding dim mismatch: config={self.identity.dim} actual={len(vec)}. "
                f"Set EMBED_DIM={len(vec)} and restart."
            )
        return vec


_service: Optional[EmbeddingService] = None


def get_service() -> EmbeddingService:
    global _service
    with _lock:
        if _service is None:
            _service = EmbeddingService()
            _service.initialize()
        return _service
