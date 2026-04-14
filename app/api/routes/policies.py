from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import PolicyListItem
from app.core.vectorstore.chroma_store import get_all_policies, load_policy_registry

router = APIRouter(prefix="/api/policies", tags=["Policies"])


@router.get("", response_model=List[PolicyListItem])
async def list_policies():
    """List all insurance policies currently indexed in ChromaDB."""
    policies = get_all_policies()
    if not policies:
        return []
    return [PolicyListItem(**p) for p in policies]


@router.get("/registry")
async def get_registry():
    """Return the full policy registry with structured metadata (used by the scorer)."""
    registry = load_policy_registry()
    if not registry:
        raise HTTPException(
            status_code=404,
            detail="No policies in registry. Upload documents via POST /api/ingest."
        )
    return registry


@router.get("/{policy_name}")
async def get_policy_details(policy_name: str):
    """Get structured metadata for a specific policy."""
    registry = load_policy_registry()
    if policy_name not in registry:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found.")
    return registry[policy_name]
