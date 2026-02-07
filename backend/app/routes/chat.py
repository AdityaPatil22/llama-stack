"""
Inference Routes - Chat completions using Llama Stack Inference API
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.llama_stack import get_llama_stack_service


router = APIRouter(tags=["inference"])


class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat completion request model."""
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(None, description="Model identifier (uses default if not specified)")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")


class SimpleChatRequest(BaseModel):
    """Simple chat request with just a message."""
    message: str = Field(..., description="User message")
    model: Optional[str] = Field(None, description="Model identifier")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")


@router.post("/chat/completions")
async def chat_completions(request: ChatRequest) -> Dict[str, Any]:
    """
    Create a chat completion using the Llama Stack inference API.
    
    This is the primary inference endpoint that supports:
    - Multi-turn conversations
    - System prompts
    - Streaming responses
    - Temperature and max_tokens control
    """
    try:
        service = get_llama_stack_service()
        
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        if request.stream:
            def generate():
                for chunk in service.chat_completion_stream(
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        result = service.chat_completion(
            messages=messages,
            model=request.model,
            stream=False,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def simple_chat(request: SimpleChatRequest) -> Dict[str, Any]:
    """
    Simple chat endpoint for single-turn conversations.
    
    This is a convenience endpoint that wraps the full chat completions API.
    """
    try:
        service = get_llama_stack_service()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})
        
        result = service.chat_completion(
            messages=messages,
            model=request.model,
            stream=False,
        )
        
        # Extract just the response content for simple API
        content = ""
        if result.get("choices") and len(result["choices"]) > 0:
            content = result["choices"][0].get("message", {}).get("content", "")
        
        return {
            "response": content,
            "model": result.get("model"),
            "usage": result.get("usage"),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
