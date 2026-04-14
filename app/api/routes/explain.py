import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ExplainRequest, PolicyExplanation
from app.core.recommendation.scorer import score_policies, extract_user_features, extract_policy_features, combine_features
from app.core.recommendation.explainer import explain_policy
from app.core.vectorstore.chroma_store import load_policy_registry

router = APIRouter(prefix="/api/explain", tags=["Explainability"])
logger = logging.getLogger(__name__)


@router.post("", response_model=PolicyExplanation)
async def explain_specific_policy(request: ExplainRequest):
    """
    Get a detailed SHAP explanation for why a specific policy is (or is not)
    suitable for the given user profile.

    Use this for deep-dive analysis on any policy — not just the top recommendation.
    """
    registry = load_policy_registry()
    if not registry:
        raise HTTPException(status_code=422, detail="No policies indexed. Upload documents first.")

    if request.policy_name not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Policy '{request.policy_name}' not found in registry. "
                   f"Available policies: {list(registry.keys())}",
        )

    policy_meta = registry[request.policy_name]
    user_feats = extract_user_features(request.user_profile)
    policy_feats = extract_policy_features(policy_meta)
    combined = combine_features(user_feats, policy_feats)

    # Use scorer to get the score
    scored = score_policies(request.user_profile, {request.policy_name: policy_meta})
    if not scored:
        raise HTTPException(status_code=500, detail="Scoring failed.")

    try:
        explanation = explain_policy(scored[0], top_n_factors=7)
        return explanation
    except Exception as e:
        logger.error(f"Explanation failed for '{request.policy_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
