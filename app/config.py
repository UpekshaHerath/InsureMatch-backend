from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Groq LLM
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str

    # ChromaDB
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "vectordb")
    CHROMA_COLLECTION_NAME: str = "insurance_policies"

    # Embeddings (HuggingFace, free — no API key)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    RETRIEVAL_K: int = 8
    RETRIEVAL_FETCH_K: int = 20

    # Saved models
    MODEL_SAVE_DIR: str = str(BASE_DIR / "saved_models")

    # Raw data directory
    DATA_DIR: str = str(BASE_DIR / "data" / "raw")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
