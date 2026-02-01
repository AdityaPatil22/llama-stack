import httpx
from app.core.config import settings

class LlamaStackClient:
    def __init__(self):
        self.base_url = settings.LLAMA_STACK_BASE_URL
        self.model = settings.OLLAMA_MODEL

    async def chat(self, message: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": message}
                    ],
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()