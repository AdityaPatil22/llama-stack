"""
Models Routes - List and manage available models
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from app.services.llama_stack import get_llama_stack_service


router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    List all available models from the Llama Stack server.
    
    Returns models organized by type (llm, embedding, etc.)
    """
    try:
        service = get_llama_stack_service()
        models = service.list_models()
        
        # Organize by type
        llm_models = [m for m in models if m.get("model_type") == "llm"]
        embedding_models = [m for m in models if m.get("model_type") == "embedding"]
        other_models = [m for m in models if m.get("model_type") not in ["llm", "embedding"]]
        
        return {
            "models": models,
            "llm_models": llm_models,
            "embedding_models": embedding_models,
            "other_models": other_models,
            "default_llm": service.get_default_llm(),
            "default_embedding": service.get_default_embedding_model(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/llm")
async def list_llm_models() -> List[Dict[str, Any]]:
    """List available LLM models."""
    try:
        service = get_llama_stack_service()
        models = service.list_models()
        return [m for m in models if m.get("model_type") == "llm"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/embedding")
async def list_embedding_models() -> List[Dict[str, Any]]:
    """List available embedding models."""
    try:
        service = get_llama_stack_service()
        models = service.list_models()
        return [m for m in models if m.get("model_type") == "embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
