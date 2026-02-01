from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llama_stack import LlamaStackClient

router = APIRouter()
llama = LlamaStackClient()

class ChatRequest(BaseModel):
    message: str
    
@router.post("/chat")
async def chat(req: ChatRequest):
    result = await llama.chat(req.message)

    # Ollama native API response format
    return {
        "response": result["message"]["content"]
    }
