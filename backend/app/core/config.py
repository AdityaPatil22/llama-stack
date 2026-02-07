from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Llama Stack server settings
    LLAMA_STACK_BASE_URL: str = "http://localhost:8321"
    
    # Default model settings (use llama stack model identifier format)
    DEFAULT_MODEL: str = "ollama/llama3.2:3b"
    DEFAULT_EMBEDDING_MODEL: str = "ollama/nomic-embed-text:v1.5"
    
    # Agent settings
    DEFAULT_AGENT_INSTRUCTIONS: str = "You are a helpful assistant."
    
    # Safety settings
    ENABLE_SAFETY_SHIELDS: bool = True
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()