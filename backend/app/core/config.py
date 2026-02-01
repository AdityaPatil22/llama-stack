from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LLAMA_STACK_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

settings = Settings()