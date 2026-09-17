"""
Embedding factory — env only, no hard-coded model in callers.

Switch provider/model via env; consumers must reindex when index_key changes.

Providers (EMBED_PROVIDER):
  openai      OpenAI or any OpenAI-compatible HTTP API (default)
  local       sentence-transformers on this machine (e.g. BAAI/bge-m3)
  huggingface OpenAI-compatible HF endpoint when EMBED_BASE_URL is set,
              otherwise treated as `local`
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, replace
from typing import Any, Optional

logger = logging.getLogger("embeddings")

_lock = threading.RLock()

_DEFAULT_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "baai/bge-m3": 1024,
    "bge-m3": 1024,
}

_DEFAULT_LOCAL_MODEL = "BAAI/bge-m3"
_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

_TRUTHY = {"1", "true", "yes", "on", "y"}
_FALSY = {"0", "false", "no", "off", "n"}

_DEVICE_ALIASES = {"gpu": "cuda", "nvidia": "cuda", "": "auto"}


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


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return default


def _safe_token(value: str) -> str:
    s = (value or "").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", s)
    return s[:120] or "model"


# ---------------------------------------------------------------------------
# Device — resolved only for the local provider, since importing torch is slow
# ---------------------------------------------------------------------------

_device_cache: dict[str, str] = {}


def resolve_device(requested: str) -> str:
    """Map EMBED_DEVICE (auto|cpu|cuda|cuda:N|mps) onto a device torch can use.

    Never raises: an unavailable device degrades to cpu with a warning, so the
    same .env keeps working on a host without a GPU.
    """
    req = (requested or "auto").strip().lower()
    req = _DEVICE_ALIASES.get(req, req)

    with _lock:
        cached = _device_cache.get(req)
    if cached:
        return cached

    resolved = _detect_device(req)
    with _lock:
        _device_cache[req] = resolved
    return resolved


def _detect_device(req: str) -> str:
    if req == "cpu":
        return "cpu"

    if req not in {"auto", "cuda", "mps"} and not req.startswith("cuda:"):
        logger.warning("Unknown EMBED_DEVICE=%r — falling back to auto", req)
        req = "auto"

    try:
        import torch
    except ImportError:
        if req != "auto":
            logger.warning(
                "EMBED_DEVICE=%s requested but torch is not installed — using cpu",
                req,
            )
        return "cpu"

    has_cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = bool(mps_backend is not None and mps_backend.is_available())

    if req == "auto":
        if has_cuda:
            logger.info("EMBED_DEVICE=auto -> cuda (%s)", torch.cuda.get_device_name(0))
            return "cuda"
        if has_mps:
            logger.info("EMBED_DEVICE=auto -> mps")
            return "mps"
        logger.info("EMBED_DEVICE=auto -> cpu (no GPU visible to torch)")
        return "cpu"

    if req.startswith("cuda"):
        if not has_cuda:
            logger.warning(
                "EMBED_DEVICE=%s requested but CUDA is unavailable to torch — "
                "using cpu (check the torch CUDA build and GPU passthrough)",
                req,
            )
            return "cpu"
        return req

    if req == "mps" and not has_mps:
        logger.warning("EMBED_DEVICE=mps requested but MPS is unavailable — using cpu")
        return "cpu"

    return req


def torch_runtime() -> dict[str, Any]:
    """What torch actually sees — surfaced on /ready to debug GPU passthrough."""
    try:
        import torch
    except ImportError:
        return {"torch_installed": False}

    info: dict[str, Any] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if info["cuda_available"]:
        info["cuda_build"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedIdentity:
    provider: str
    model: str
    dim: int
    api_key: str
    base_url: Optional[str]
    device: str
    device_requested: str = "auto"

    @property
    def is_local(self) -> bool:
        return self.provider == "local"

    @property
    def index_key(self) -> str:
        return f"{_safe_token(self.provider)}__{_safe_token(self.model)}__d{self.dim}"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "index_key": self.index_key,
            "device": self.device,
            "device_requested": self.device_requested,
            "base_url_set": bool(self.base_url),
            "api_key_set": bool(self.api_key),
        }
        if self.is_local:
            out["runtime"] = torch_runtime()
        return out


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

    default_model = _DEFAULT_LOCAL_MODEL if provider == "local" else _DEFAULT_OPENAI_MODEL
    model = _env("EMBED_MODEL") or default_model
    explicit_dim = _env_int("EMBED_DIM", None)
    dim = resolve_dim(provider, model, explicit_dim)

    api_key = _env("EMBED_API_KEY") or _env("OPENAI_API_KEY") or _env("LLM_API_KEY")
    base_url = _env("EMBED_BASE_URL") or _env("OPENAI_BASE_URL") or None
    if base_url == "":
        base_url = None

    device_requested = _env("EMBED_DEVICE") or "auto"
    device = resolve_device(device_requested) if provider == "local" else "remote"

    return EmbedIdentity(
        provider=provider,
        model=model,
        dim=dim,
        api_key=api_key,
        base_url=base_url,
        device=device,
        device_requested=device_requested,
    )


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


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


def _missing_local_deps() -> list[str]:
    import importlib.util

    return [
        name
        for name in ("torch", "sentence_transformers")
        if importlib.util.find_spec(name) is None
    ]


def _import_hf_embeddings(model_name: str) -> Any:
    # Check the real engine first: langchain_community is often present without
    # it, and its own error is far less actionable than this one.
    missing = _missing_local_deps()
    if missing:
        raise RuntimeError(
            f"EMBED_PROVIDER=local needs {' and '.join(missing)}, not installed. "
            "Docker: set WITH_LOCAL_MODELS=true in .env and rebuild "
            "(docker compose up -d --build). "
            "Host: pip install -r requirements-local.txt . "
            f"(EMBED_MODEL={model_name})"
        )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except ImportError:
        pass
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.warning(
            "Using the deprecated langchain_community HuggingFaceEmbeddings — "
            "install langchain-huggingface (requirements-local.txt)"
        )
        return HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings need langchain-huggingface. "
            "Install: pip install -r requirements-local.txt . "
            f"(EMBED_MODEL={model_name})"
        ) from exc


def _local_embeddings(ident: EmbedIdentity) -> Any:
    model_name = ident.model or _DEFAULT_LOCAL_MODEL
    hf_embeddings = _import_hf_embeddings(model_name)

    model_kwargs: dict[str, Any] = {"device": ident.device}
    if _env_bool("EMBED_TRUST_REMOTE_CODE", False):
        model_kwargs["trust_remote_code"] = True

    encode_kwargs: dict[str, Any] = {
        "normalize_embeddings": _env_bool("EMBED_NORMALIZE", True),
        "batch_size": _env_int("EMBED_LOCAL_BATCH_SIZE", 8) or 8,
    }

    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "encode_kwargs": encode_kwargs,
    }
    cache_dir = _env("EMBED_CACHE_DIR") or _env("SENTENCE_TRANSFORMERS_HOME")
    if cache_dir:
        kwargs["cache_folder"] = cache_dir

    logger.info(
        "Loading local embedding model=%s device=%s batch_size=%s "
        "(first run downloads weights — bge-m3 is ~2.2GB)",
        model_name,
        ident.device,
        encode_kwargs["batch_size"],
    )
    try:
        return hf_embeddings(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load local embedding model {model_name!r} on "
            f"device={ident.device}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


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
                self.identity = self._reconcile_dim(ident)
                self._ready = True
                self._last_error = None
                logger.info(
                    "Embedding ready provider=%s model=%s dim=%s device=%s",
                    self.identity.provider,
                    self.identity.model,
                    self.identity.dim,
                    self.identity.device,
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

    def _reconcile_dim(self, ident: EmbedIdentity) -> EmbedIdentity:
        """Adopt the model's real width for local models when EMBED_DIM is unset.

        Any HuggingFace model then works without the caller knowing its dim; an
        explicit EMBED_DIM is always respected as given.
        """
        if not ident.is_local:
            return ident
        if _env_int("EMBED_DIM", None) is not None:
            return ident
        if not _env_bool("EMBED_PROBE_DIM", True):
            return ident

        try:
            actual = len(self._client.embed_query("dim probe"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dim probe failed (%s) — keeping configured dim=%s", exc, ident.dim
            )
            return ident

        if actual == ident.dim:
            return ident

        logger.warning(
            "Configured dim=%s but %s returns %s — using %s (index_key changes)",
            ident.dim,
            ident.model,
            actual,
            actual,
        )
        return replace(ident, dim=actual)

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

    def embed_texts(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """Embed a batch, honouring the query/document distinction."""
        self._ensure()
        prepared = self._apply_prefix(texts, input_type)
        if input_type == "query" and len(prepared) == 1:
            return [self._check_dim(list(self._client.embed_query(prepared[0])))]
        vectors = self._client.embed_documents(prepared)
        return [self._check_dim(list(v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text], "query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(list(texts), "document")

    def _apply_prefix(self, texts: list[str], input_type: str) -> list[str]:
        """Instruction prefixes for models that want them (e5, gte, ...).

        bge-m3 needs none, so both default to empty. Read unstripped: e5 expects
        exactly "query: " / "passage: ", trailing space included.
        """
        name = "EMBED_QUERY_PREFIX" if input_type == "query" else "EMBED_DOCUMENT_PREFIX"
        prefix = os.getenv(name) or ""
        if not prefix:
            return list(texts)
        return [f"{prefix}{t}" for t in texts]

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
