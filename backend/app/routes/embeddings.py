"""
Embeddings Routes - Create text embeddings using Llama Stack
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llama_stack import get_llama_stack_service


router = APIRouter(tags=["embeddings"])


class EmbeddingsRequest(BaseModel):
    """Request to create embeddings."""
    texts: List[str] = Field(..., description="List of texts to embed")
    model: Optional[str] = Field(None, description="Embedding model to use")


class SingleEmbeddingRequest(BaseModel):
    """Request to create a single embedding."""
    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model to use")


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingsRequest) -> Dict[str, Any]:
    """
    Create embeddings for a list of texts.
    
    Returns embedding vectors for each input text.
    """
    try:
        service = get_llama_stack_service()
        
        result = service.create_embeddings(
            texts=request.texts,
            model=request.model,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/single")
async def create_single_embedding(request: SingleEmbeddingRequest) -> Dict[str, Any]:
    """
    Create an embedding for a single text.
    
    Convenience endpoint for single text embedding.
    """
    try:
        service = get_llama_stack_service()
        
        result = service.create_embeddings(
            texts=[request.text],
            model=request.model,
        )
        
        # Extract the single embedding
        embedding = None
        if result.get("embeddings") and len(result["embeddings"]) > 0:
            embedding = result["embeddings"][0].get("embedding")
        
        return {
            "model": result.get("model"),
            "embedding": embedding,
            "dimensions": len(embedding) if embedding else 0,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
