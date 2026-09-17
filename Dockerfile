FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=10001
ARG APP_GID=10001

# Local model stack (torch + sentence-transformers). Off by default: it adds
# ~2-3GB to the image and is only needed for EMBED_PROVIDER=local.
ARG WITH_LOCAL_MODELS=false
# CPU wheels by default. For NVIDIA pick the index matching the driver's CUDA
# version, e.g. https://download.pytorch.org/whl/cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Optional: bake weights into the image (e.g. BAAI/bge-m3) so the first request
# is not a 2.2GB download. Requires WITH_LOCAL_MODELS=true.
ARG PRELOAD_MODEL=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8090 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid appuser --create-home --shell /usr/sbin/nologin appuser \
    && python -m venv /opt/venv

WORKDIR /build
COPY requirements.txt requirements-local.txt pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ "${WITH_LOCAL_MODELS}" = "true" ]; then \
         echo "Installing local model stack from ${TORCH_INDEX_URL}"; \
         pip install --index-url "${TORCH_INDEX_URL}" \
                     --extra-index-url https://pypi.org/simple \
                     -r requirements-local.txt; \
       fi \
    && pip install --no-deps .

RUN if [ -n "${PRELOAD_MODEL}" ] && [ "${WITH_LOCAL_MODELS}" = "true" ]; then \
      echo "Pre-downloading ${PRELOAD_MODEL} into ${HF_HOME}"; \
      EMBED_PRELOAD="${PRELOAD_MODEL}" python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_PRELOAD'], device='cpu')"; \
      chown -R appuser:appuser /home/appuser/.cache; \
    fi

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser requirements.txt requirements-local.txt pyproject.toml README.md ./

# The HF cache must exist in the image and be owned by appuser: a named volume
# mounted over a path that does not exist is created root-owned, and the
# non-root process then cannot download weights into it.
RUN mkdir -p "${HF_HOME}" && chown -R appuser:appuser /home/appuser/.cache

USER appuser
EXPOSE 8090

# start-period is generous: with EMBED_PROVIDER=local the first boot may download
# model weights before the app answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${APP_PORT:-8090}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "app.main"]
