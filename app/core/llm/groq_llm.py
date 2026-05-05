from langchain_groq import ChatGroq
from app.config import settings


def get_groq_llm(temperature: float = 0.1, max_tokens: int = 4096) -> ChatGroq:
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_groq_llm_creative(temperature: float = 0.4) -> ChatGroq:
    """Higher temperature for narrative generation."""
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=temperature,
        max_tokens=2048,
    )
