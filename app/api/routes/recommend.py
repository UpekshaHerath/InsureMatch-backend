import logging
from fastapi import APIRouter, HTTPException, Depends

from app.models.schemas import RecommendationRequest, RecommendationResponse
from app.core.recommendation.ranker import recommend
from app.core.auth.deps import get_current_user, AuthUser
from app.core.db import supabase_client as db

router = APIRouter(prefix="/api/recommend", tags=["Recommendation"])
logger = logging.getLogger(__name__)


@router.post("", response_model=RecommendationResponse)
async def get_recommendation(
    request: RecommendationRequest,
    user: AuthUser = Depends(get_current_user),
):
    """Submit a user profile and receive ranked insurance policy recommendations."""
    try:
        response = await recommend(profile=request.user_profile, top_k=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

    # Persist profile + recommendation (best-effort; failures logged with full trace)
    try:
        await db.upsert_profile(user.user_id, request.user_profile)
    except Exception as e:
        logger.error(f"upsert_profile failed for {user.user_id}: {e}", exc_info=True)
    try:
        await db.ensure_session(user.user_id, response.session_id, title="Recommendation session")
    except Exception as e:
        logger.error(f"ensure_session failed for {user.user_id}: {e}", exc_info=True)
    try:
        await db.insert_recommendation(user.user_id, response.session_id, response)
    except Exception as e:
        logger.error(f"insert_recommendation failed for {user.user_id}: {e}", exc_info=True)

    return response
