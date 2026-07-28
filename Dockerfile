FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8090 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid appuser --create-home --shell /usr/sbin/nologin appuser \
    && python -m venv /opt/venv

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install .

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser requirements.txt pyproject.toml README.md ./
COPY --chown=appuser:appuser requirements-local.txt ./

USER appuser
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${APP_PORT:-8090}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "app.main"]
