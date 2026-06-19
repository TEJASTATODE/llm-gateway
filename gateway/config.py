from pydantic_settings import BaseSettings
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    # Provider keys
    openai_api_key: str
    anthropic_api_key: str
    gemini_api_key: str

    # Databases
    postgres_url: str
    redis_url: str
    qdrant_url: str

    # Gateway
    gateway_api_key: str

    # OpenAI base URL — override for Groq
    openai_base_url: str = "https://api.groq.com/openai/v1"

    # Cache thresholds
    threshold_code: float = 0.96
    threshold_factual: float = 0.75
    threshold_conceptual: float = 0.78
    threshold_conversational: float = 0.70

    # Circuit breaker
    cb_failure_threshold: int = 5
    cb_recovery_timeout: float = 30.0

    class Config:
        env_file = str(ENV_FILE)

settings = Settings()