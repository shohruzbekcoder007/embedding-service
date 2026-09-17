# embedding-service

Standalone **text → vector** microservice.  
Does **not** store vectors, parse PDF, chat, or run SQL. Consumers (e.g. `ai-agents` RAG) call this API and write Chroma/pgvector themselves.

Two backends, chosen entirely in `.env`:

| `EMBED_PROVIDER` | What runs | Needs |
|---|---|---|
| `openai` | OpenAI (or any OpenAI-compatible API) | API key, network |
| `local` | `sentence-transformers` in this process, e.g. `BAAI/bge-m3` | torch, ~2.2GB weights, no key |

## Architecture

```text
Consumer app  ──POST /v1/embed|/v1/embed/batch──►  embedding-service
                  ◄──── vectors + dim + index_key ────
```

## Quick start (Docker)

```bash
cd d:\GROK\STATGPT\embedding-service
copy .env.example .env
# set OPENAI_API_KEY in .env, or switch to the local model (below)
docker compose up -d --build
```

Swagger: http://127.0.0.1:8090/docs

### Without Docker

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m app.main
```

## Local model: BAAI/bge-m3

No API key, nothing leaves the machine. Set in `.env`:

```env
EMBED_PROVIDER=local
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=auto          # auto | cpu | cuda | cuda:0 | mps

WITH_LOCAL_MODELS=true     # build arg: put torch into the image
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

Then rebuild, because torch is installed at image build time:

```bash
docker compose up -d --build
curl -s http://127.0.0.1:8090/ready
```

The first boot downloads ~2.2GB of weights into the `embedding-service-hf-cache`
volume; later starts reuse it. Raise `HEALTHCHECK_START_PERIOD` if that download
is slow. To bake the weights into the image instead, set `PRELOAD_MODEL=BAAI/bge-m3`
before building.

Once the image contains torch, switching between `openai` and `local` is just
`EMBED_PROVIDER` plus a restart — no rebuild:

```bash
docker compose up -d --force-recreate
```

### NVIDIA GPU

Build against CUDA wheels matching the host driver (`nvidia-smi` prints its CUDA
version; RTX 50xx / Blackwell needs cu128 or newer):

```env
WITH_LOCAL_MODELS=true
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
EMBED_DEVICE=auto
EMBED_LOCAL_BATCH_SIZE=32
```

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
curl -s http://127.0.0.1:8090/ready     # identity.runtime.cuda_available -> true
```

CUDA wheels make the image large (~6.3GB vs ~1.5GB for CPU-only).
Needs the NVIDIA Container Toolkit (Docker Desktop on Windows: WSL2 backend).
`EMBED_DEVICE=auto` falls back to CPU when no GPU is visible, so the same `.env`
works on a GPU-less host — `/ready` reports which device was actually chosen.

### Host install (no Docker)

```bash
pip install -r requirements-local.txt ^
  --index-url https://download.pytorch.org/whl/cu130 ^
  --extra-index-url https://pypi.org/simple
```

Use `.../whl/cpu` instead for a CPU-only install.

## API

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | Liveness |
| GET | `/ready` | Model ready + identity + device |
| GET | `/v1/info` | Config / limits |
| POST | `/v1/embed` | One text → one vector |
| POST | `/v1/embed/batch` | Many texts → many vectors |

### Examples

```bash
curl -s http://127.0.0.1:8090/ready

curl -s -X POST http://127.0.0.1:8090/v1/embed ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"mehnat tatili\",\"input_type\":\"query\"}"

curl -s -X POST http://127.0.0.1:8090/v1/embed/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"texts\":[\"chunk a\",\"chunk b\"],\"input_type\":\"document\"}"
```

Every success response includes:

- `dim` — vector length  
- `provider` / `model`  
- `index_key` — e.g. `openai__text-embedding-3-small__d1536`  

**When `index_key` changes, consumers must full-reindex** (dimensions must not mix).

| Model | Typical dim | index_key |
|--------|-------------|-----------|
| text-embedding-3-small | 1536 | `openai__text-embedding-3-small__d1536` |
| text-embedding-3-large | 3072 | `openai__text-embedding-3-large__d3072` |
| BAAI/bge-m3 | 1024 | `local__BAAI_bge-m3__d1024` |

For local models the real width is probed at startup when `EMBED_DIM` is empty,
so any HuggingFace model works without knowing its dimension in advance. Set
`EMBED_DIM` explicitly to pin it (and to request reduced dimensions from OpenAI).

## Env reference

See `.env.example`. Important:

| Var | Default | Note |
|-----|---------|------|
| `EMBED_PROVIDER` | `openai` | `openai` \| `local` \| `huggingface` |
| `EMBED_MODEL` | `text-embedding-3-small` | defaults to `BAAI/bge-m3` when provider is `local` |
| `EMBED_DIM` | (model default) | empty = resolved/probed |
| `EMBED_DEVICE` | `auto` | local only; degrades to `cpu` if unavailable |
| `EMBED_NORMALIZE` | `true` | local only; L2-normalise vectors |
| `EMBED_LOCAL_BATCH_SIZE` | `8` | local only; GPU 32-64, CPU 4-8 |
| `EMBED_PROBE_DIM` | `true` | local only; ask the model its real width |
| `EMBED_QUERY_PREFIX` / `EMBED_DOCUMENT_PREFIX` | (empty) | for e5-style models; bge-m3 needs none |
| `HF_TOKEN` | (optional) | lifts HuggingFace Hub rate limits on first download |
| `HF_HUB_OFFLINE` | `0` | `1` = never fetch weights (needs a warm cache volume) |
| `EMBED_BATCH_MAX` | 64 | |
| `EMBED_MAX_CHARS` | 8000 | |
| `APP_PORT` | 8090 | |
| `API_BEARER_TOKEN` | (optional) | protects `/v1/embed*` |
| `WITH_LOCAL_MODELS` | `false` | **build-time**; install torch into the image |
| `TORCH_INDEX_URL` | `.../whl/cpu` | **build-time**; CPU or CUDA wheels |
| `PRELOAD_MODEL` | (empty) | **build-time**; bake weights into the image |

Runtime vars take effect on restart; the three build-time vars need
`docker compose build`.

## Non-goals

- No Chroma / pgvector  
- No document loaders  
- No LLM chat  
- No Hermes / SQL  

## Integration with ai-agents (later)

Point RAG app at:

```env
RAG_EMBED_PROVIDER=remote
RAG_EMBED_URL=http://embedding-service:8090
# or host: http://host.docker.internal:8090
```

Then `embed_query` / `embed_documents` become HTTP calls; Chroma write stays in `ai-agents`.
