try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import ingest, recommend, recommendations, explain, chat, policies, riders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure directories exist and warm up the XGBoost model."""
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Warm up scorer (trains & saves model if not present)
    logger.info("Warming up XGBoost scorer…")
    from app.core.recommendation.scorer import load_model
    load_model()
    logger.info("XGBoost scorer ready.")

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Insurance Recommendation RAG",
    description=(
        "A RAG-powered insurance recommendation engine for Sri Lanka. "
        "Combines ChromaDB vector search, LangChain + Groq LLM, and XGBoost + SHAP "
        "to recommend the best insurance policy for a user and explain why."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.8.152:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(ingest.router)
app.include_router(recommend.router)
app.include_router(recommendations.router)
app.include_router(explain.router)
app.include_router(chat.router)
app.include_router(policies.router)
app.include_router(riders.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Insurance Recommendation RAG",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    from app.core.vectorstore.chroma_store import get_all_policies, load_policy_registry
    policies_count = len(get_all_policies())
    registry_count = len(load_policy_registry())
    return {
        "status": "healthy",
        "policies_indexed": policies_count,
        "policies_in_registry": registry_count,
    }
