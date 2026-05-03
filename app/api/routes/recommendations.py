import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth.deps import AuthUser, get_current_user
from app.core.db import supabase_client as db

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])
logger = logging.getLogger(__name__)


class RecommendationsPage(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


@router.get("", response_model=RecommendationsPage)
async def list_user_recommendations(
    page: int = Query(1, ge=1),
    page_size: int = Query(5, ge=1, le=100),
    user: AuthUser = Depends(get_current_user),
):
    """Paginated past recommendations for the authenticated user."""
    try:
        rows, total = await db.list_recommendations_paginated(
            user.user_id, page=page, page_size=page_size
        )
    except Exception as e:
        logger.error(f"list_recommendations_paginated failed for {user.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load recommendations")

    return RecommendationsPage(items=rows, total=total, page=page, page_size=page_size)
