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
    OPENAI_CHAT_MODEL: str = "gpt-4-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CORS_ORIGINS: str = "https://avo-kms-rag-knowledge.azurewebsites.net"
    KNOWLEDGE_FILE: str = "./costing_kms_rag_knowledge_base_v2.json"

    @property
    def api_v1_prefix(self) -> str:
        prefix = self.API_V1_PREFIX.strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        return prefix


settings = Settings()
