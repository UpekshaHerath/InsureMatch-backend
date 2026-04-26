__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)

# Path for the policy registry (structured metadata for scorer)
POLICY_REGISTRY_PATH = Path(settings.MODEL_SAVE_DIR) / "policy_registry.json"
RIDER_REGISTRY_PATH = Path(settings.MODEL_SAVE_DIR) / "rider_registry.json"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def add_documents(chunks: List[Document]) -> int:
    """Add document chunks to ChromaDB. Returns number of chunks added."""
    store = get_vector_store()
    store.add_documents(chunks)
    logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    return len(chunks)


def similarity_search(query: str, k: int = None, fetch_k: int = None) -> List[Document]:
    """MMR-based retrieval for diverse, relevant results."""
    k = k or settings.RETRIEVAL_K
    fetch_k = fetch_k or settings.RETRIEVAL_FETCH_K
    store = get_vector_store()
    return store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def similarity_search_for_policy(query: str, policy_name: str, k: int = 5) -> List[Document]:
    """Retrieve chunks for a specific policy."""
    store = get_vector_store()
    return store.similarity_search(
        query,
        k=k,
        filter={"policy_name": policy_name},
    )


def get_all_policies() -> List[Dict[str, Any]]:
    """Return unique policies indexed in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        seen = {}
        for meta in results["metadatas"]:
            name = meta.get("policy_name", "unknown")
            if name not in seen:
                seen[name] = {
                    "policy_name": name,
                    "policy_type": meta.get("policy_type", "unknown"),
                    "company": meta.get("company"),
                    "source_file": meta.get("source_file", "unknown"),
                    "chunk_count": 1,
                }
            else:
                seen[name]["chunk_count"] += 1

        return list(seen.values())
    except Exception as e:
        logger.warning(f"Could not fetch policies: {e}")
        return []


def delete_policy(policy_name: str) -> bool:
    """Delete all chunks for a given policy."""
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
        results = collection.get(where={"policy_name": policy_name})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for '{policy_name}'")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete policy '{policy_name}': {e}")
        return False


# ─── Policy Registry (for scorer) ────────────────────────────────────────────

def save_policy_to_registry(policy_meta: Dict[str, Any]) -> None:
    """Persist structured policy metadata for the XGBoost scorer."""
    POLICY_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: Dict = {}
    if POLICY_REGISTRY_PATH.exists():
        with open(POLICY_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    registry[policy_meta["policy_name"]] = policy_meta
    with open(POLICY_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Saved '{policy_meta['policy_name']}' to policy registry")


def load_policy_registry() -> Dict[str, Any]:
    """Load all registered policy metadata."""
    if not POLICY_REGISTRY_PATH.exists():
        return {}
    with open(POLICY_REGISTRY_PATH, "r") as f:
        return json.load(f)


# ─── Rider Registry ──────────────────────────────────────────────────────────

def save_rider_to_registry(rider_meta: Dict[str, Any]) -> None:
    """Persist structured rider metadata for the rider scorer."""
    RIDER_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: Dict = {}
    if RIDER_REGISTRY_PATH.exists():
        with open(RIDER_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    registry[rider_meta["rider_code"]] = rider_meta
    with open(RIDER_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Saved rider '{rider_meta['rider_code']}' to rider registry")


def load_rider_registry() -> Dict[str, Any]:
    if not RIDER_REGISTRY_PATH.exists():
        return {}
    with open(RIDER_REGISTRY_PATH, "r") as f:
        return json.load(f)


def clear_rider_registry() -> None:
    """Replace the whole rider registry (used before re-ingesting the bundle)."""
    if RIDER_REGISTRY_PATH.exists():
        RIDER_REGISTRY_PATH.unlink()
        logger.info("Cleared rider registry")


def delete_all_rider_chunks() -> int:
    """Remove every chunk with doc_type='rider' from ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
        results = collection.get(where={"doc_type": "rider"})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} rider chunks")
            return len(results["ids"])
        return 0
    except Exception as e:
        logger.warning(f"delete_all_rider_chunks failed: {e}")
        return 0


def delete_rider(rider_code: str) -> bool:
    """Delete one rider's chunks + registry entry."""
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
        results = collection.get(where={"$and": [{"doc_type": "rider"}, {"rider_code": rider_code}]})
        if results["ids"]:
            collection.delete(ids=results["ids"])
        # Remove from registry
        if RIDER_REGISTRY_PATH.exists():
            with open(RIDER_REGISTRY_PATH, "r") as f:
                reg = json.load(f)
            if rider_code in reg:
                del reg[rider_code]
                with open(RIDER_REGISTRY_PATH, "w") as f:
                    json.dump(reg, f, indent=2)
                return True
        return False
    except Exception as e:
        logger.error(f"delete_rider '{rider_code}' failed: {e}")
        return False
