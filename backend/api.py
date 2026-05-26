from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.startup import lifespan
from backend.core.middleware import AuthContextMiddleware, RateLimitMiddleware, RequestContextMiddleware
from backend.core.metrics_middleware import PrometheusMiddleware, metrics_endpoint
from .routes.auth import router as auth_router
from .routes.admin import router as admin_router
from .routes.chat import router as chat_router
from .routes.conversations import router as conversations_router
from .routes.documents import router as documents_router
from .routes.health import router as health_router
from .routes.workspaces import router as workspaces_router


app = FastAPI(title="DocuMind AI", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware and endpoint must be registered before app startup
app.add_middleware(PrometheusMiddleware)
app.add_api_route("/metrics", metrics_endpoint)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(AuthContextMiddleware)
app.add_middleware(RateLimitMiddleware)


app.include_router(health_router)
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(admin_router)
app.include_router(workspaces_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/")
def home():
    return {
        "message": "DocuMind AI Backend Running",
        "vector_store_ready": app.state.vector_store_service.is_ready() if hasattr(app.state, "vector_store_service") else False,
        "database_ready": hasattr(app.state, "database_service"),
        "redis_ready": app.state.cache_service.is_available() if hasattr(app.state, "cache_service") else False,
    }