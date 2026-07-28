# embedding-service

Standalone **text → vector** microservice.  
Does **not** store vectors, parse PDF, chat, or run SQL. Consumers (e.g. `ai-agents` RAG) call this API and write Chroma/pgvector themselves.

## Architecture

```text
Consumer app  ──POST /v1/embed|/v1/embed/batch──►  embedding-service
                  ◄──── vectors + dim + index_key ────
```

## Quick start

```bash
cd d:\GROK\embedding-service
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# set OPENAI_API_KEY in .env
python -m app.main
```

Swagger: http://127.0.0.1:8090/docs

### Docker

```bash
copy .env.example .env
docker compose up -d --build
```

## API

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | Liveness |
| GET | `/ready` | Model ready + identity |
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

## Switch embedding model (env only)

```env
# Default OpenAI small
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small

# OpenAI large
# EMBED_MODEL=text-embedding-3-large
# EMBED_DIM=1024   # optional reduced dim

# Local BAAI/bge-m3 (heavy)
# pip install -r requirements-local.txt
# EMBED_PROVIDER=local
# EMBED_MODEL=BAAI/bge-m3
# EMBED_DEVICE=cpu
```

Restart the process after env change.

| Model | Typical dim |
|--------|-------------|
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| BAAI/bge-m3 | 1024 |

## Env reference

See `.env.example`. Important:

| Var | Default |
|-----|---------|
| `EMBED_PROVIDER` | `openai` |
| `EMBED_MODEL` | `text-embedding-3-small` |
| `EMBED_DIM` | (model default) |
| `EMBED_BATCH_MAX` | 64 |
| `EMBED_MAX_CHARS` | 8000 |
| `APP_PORT` | 8090 |
| `API_BEARER_TOKEN` | (optional) |

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
