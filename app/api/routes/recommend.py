import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import RecommendationRequest, RecommendationResponse
from app.core.recommendation.ranker import recommend

router = APIRouter(prefix="/api/recommend", tags=["Recommendation"])
logger = logging.getLogger(__name__)


@router.post("", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """
    Submit a user profile and receive ranked insurance policy recommendations
    with SHAP-based explainability and a natural language narrative.

    The response includes:
    - `ranked_policies`: Top-K policies with suitability scores
    - `explanations`: SHAP factor breakdown per policy (why it was selected)
    - `rag_narrative`: Full natural language recommendation from the LLM
    - `session_id`: Use this for follow-up chat via POST /api/chat
    """
    try:
        response = await recommend(profile=request.user_profile, top_k=request.top_k)
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
