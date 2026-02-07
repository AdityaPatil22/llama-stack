"""
Llama Stack API Server

A comprehensive FastAPI application that exposes all Llama Stack built-in APIs:
- Inference (chat completions)
- Models (list, get)
- Agents (create, sessions, turns)
- RAG (vector databases, document ingestion, retrieval)
- Embeddings
- Safety (shields, content moderation)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router as chat_router
from app.routes.models import router as models_router
from app.routes.agents import router as agents_router
from app.routes.rag import router as rag_router
from app.routes.embeddings import router as embeddings_router
from app.routes.safety import router as safety_router
from app.core.config import settings


app = FastAPI(
    title="Llama Stack API",
    description="""
A unified API for all Llama Stack capabilities:

- **Inference**: Chat completions with streaming support
- **Models**: List and manage available models
- **Agents**: Create intelligent agents with tool support
- **RAG**: Vector databases and retrieval-augmented generation
- **Embeddings**: Create text embeddings
- **Safety**: Content moderation with safety shields
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llama_stack_url": settings.LLAMA_STACK_BASE_URL,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Llama Stack API",
        "version": "1.0.0",
        "llama_stack_url": settings.LLAMA_STACK_BASE_URL,
        "endpoints": {
            "inference": "/api/chat, /api/chat/completions",
            "models": "/api/models",
            "agents": "/api/agents",
            "rag": "/api/rag",
            "embeddings": "/api/embeddings",
            "safety": "/api/safety",
        },
        "docs": "/docs",
    }


# Include all routers with /api prefix
app.include_router(chat_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(agents_router, prefix="/api")
app.include_router(rag_router, prefix="/api")
app.include_router(embeddings_router, prefix="/api")
app.include_router(safety_router, prefix="/api")
