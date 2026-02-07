"""
Routes package - All API route handlers
"""

from app.routes.chat import router as chat_router
from app.routes.models import router as models_router
from app.routes.agents import router as agents_router
from app.routes.rag import router as rag_router
from app.routes.embeddings import router as embeddings_router
from app.routes.safety import router as safety_router

__all__ = [
    "chat_router",
    "models_router",
    "agents_router",
    "rag_router",
    "embeddings_router",
    "safety_router",
]
