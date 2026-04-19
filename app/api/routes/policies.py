from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models.schemas import PolicyListItem
from app.core.vectorstore.chroma_store import get_all_policies, load_policy_registry
from app.core.auth.deps import require_admin, AuthUser

router = APIRouter(prefix="/api/policies", tags=["Policies"])


@router.get("", response_model=List[PolicyListItem])
async def list_policies(_: AuthUser = Depends(require_admin)):
    """List all insurance policies currently indexed in ChromaDB (admin only)."""
    policies = get_all_policies()
    if not policies:
        return []
    return [PolicyListItem(**p) for p in policies]


@router.get("/registry")
async def get_registry(_: AuthUser = Depends(require_admin)):
    """Return the full policy registry with structured metadata (admin only)."""
    registry = load_policy_registry()
    if not registry:
        raise HTTPException(
            status_code=404,
            detail="No policies in registry. Upload documents via POST /api/ingest."
        )
    return registry


@router.get("/{policy_name}")
async def get_policy_details(
    policy_name: str,
    _: AuthUser = Depends(require_admin),
):
    """Get structured metadata for a specific policy (admin only)."""
    registry = load_policy_registry()
    if policy_name not in registry:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found.")
    return registry[policy_name]
