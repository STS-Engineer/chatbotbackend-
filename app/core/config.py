import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Absolute path to the .env file in the project root
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"

# Explicitly load the .env file into the environment
if _ENV_FILE.exists():
    print(f"Loading environment from: {_ENV_FILE}")
    load_dotenv(dotenv_path=_ENV_FILE)
else:
    print(f"WARNING: .env file NOT FOUND at: {_ENV_FILE}")


class Settings(BaseSettings):
    # Pydantic will now pick up the loaded environment variables
    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    APP_NAME: str = "Costing RAG Chatbot"
    ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    DATABASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str = "gpt-4-mini"  # Defaulting to most common mini model name
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CORS_ORIGINS: str = "http://localhost:5173"
    KNOWLEDGE_FILE: str = "./costing_kms_rag_knowledge_base_v2.json"


settings = Settings()
