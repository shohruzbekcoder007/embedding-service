# Build prompt (archive)

The service in this repo was generated from the approved plan.  
Use this prompt to recreate or extend a similar service elsewhere.

---

You are building a standalone microservice: **embedding-service**.

## Mission
A small FastAPI service whose ONLY job is to turn text into embedding vectors.
It must NOT do: chat, RAG answers, SQL, PDF parsing, Chroma/pgvector storage, or Hermes.
Consumers (other apps) will call this HTTP API, then store vectors themselves.

## Stack
- Python 3.12, FastAPI, uvicorn, pydantic v2, langchain-openai
- Optional local: sentence-transformers / langchain-huggingface for BAAI/bge-m3
- Docker + docker-compose

## Core
- Factory reads ONLY EMBED_* env (provider, model, dim, api key, base_url, device)
- index_key = `{provider}__{safe_model}__d{dim}`
- Default dims: 3-small 1536, 3-large 3072, bge-m3 1024
- Switch model via env + restart; consumers reindex on index_key change

## API
- GET /health, /ready, /v1/info
- POST /v1/embed { text, input_type }
- POST /v1/embed/batch { texts, input_type }
- Response always includes embedding(s), dim, provider, model, index_key
- Optional Bearer auth

## Non-goals
No vector DB, no PDF, no chat, no multi-tenant model routing.
