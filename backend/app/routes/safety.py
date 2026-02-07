"""
Safety Routes - Content moderation with Llama Stack safety shields
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llama_stack import get_llama_stack_service


router = APIRouter(tags=["safety"])


class Message(BaseModel):
    """Message model for safety checks."""
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class SafetyCheckRequest(BaseModel):
    """Request to run safety checks on messages."""
    messages: List[Message] = Field(..., description="Messages to check")
    shields: Optional[List[str]] = Field(None, description="Specific shields to use")


class ContentCheckRequest(BaseModel):
    """Simple request to check content safety."""
    content: str = Field(..., description="Content to check")
    shields: Optional[List[str]] = Field(None, description="Specific shields to use")


@router.get("/safety/shields")
async def list_shields() -> Dict[str, Any]:
    """
    List available safety shields.
    
    Shields are safety mechanisms that check content for violations.
    """
    try:
        service = get_llama_stack_service()
        shields = service.list_shields()
        
        return {
            "shields": shields,
            "count": len(shields),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safety/check")
async def check_safety(request: SafetyCheckRequest) -> Dict[str, Any]:
    """
    Run safety shields on messages.
    
    Checks messages for content policy violations using configured shields.
    """
    try:
        service = get_llama_stack_service()
        
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        result = service.run_shields(
            messages=messages,
            shields=request.shields,
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safety/check/content")
async def check_content_safety(request: ContentCheckRequest) -> Dict[str, Any]:
    """
    Simple endpoint to check if content is safe.
    
    Wraps content in a user message and runs safety checks.
    """
    try:
        service = get_llama_stack_service()
        
        messages = [{"role": "user", "content": request.content}]
        
        result = service.run_shields(
            messages=messages,
            shields=request.shields,
        )
        
        return {
            "content": request.content,
            "is_safe": result.get("is_safe", True),
            "status": result.get("status"),
            "violations": [
                r for r in result.get("results", [])
                if not r.get("is_safe", True)
            ],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
