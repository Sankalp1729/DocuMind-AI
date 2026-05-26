# Deployment

This guide describes how to deploy DocuMind AI as a production stack.

## Local Development

Run the backend and frontend separately when you want fast iteration:

```powershell
uvicorn backend.api:app --reload
streamlit run frontend/app.py
```

## Docker Compose

Use the full local stack when you want persistence and retrieval behavior that matches production more closely.

```powershell
docker compose up --build
```

Services exposed by the compose stack:

- Frontend on `http://localhost:8501`
- Backend on `http://localhost:8000`
- Ollama on `http://localhost:11434`

## Production Checklist

- Set `DOCUMIND_ADMIN_API_KEY` to a non-default secret
- Point `DOCUMIND_DATABASE_URL` to PostgreSQL in production
- Enable `DOCUMIND_REDIS_ENABLED` and `DOCUMIND_ENABLE_REDIS_CACHE` for caching
- Decide whether `DOCUMIND_ENABLE_AGENTIC_RAG` and `DOCUMIND_ENABLE_HYBRID_RETRIEVAL` should be on for the target tenant
- Route logs and metrics into your observability stack
- Preserve `data/` and `vector_store/` on durable storage

## Rollout Order

1. Deploy database and cache.
2. Deploy the backend with admin endpoints enabled.
3. Bring up the frontend and verify the admin cockpit.
4. Run the demo script and load test before stakeholder review.

## Operational Notes

- Retrieval caches are sized through TTL settings in backend config
- Embeddings are cached at process scope by the vector store layer
- Benchmark outputs are persisted under `data/evaluations/`
- Telemetry traces are persisted under `data/telemetry/`