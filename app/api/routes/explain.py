import logging
from fastapi import APIRouter, HTTPException, Depends

from app.models.schemas import ExplainRequest, PolicyExplanation
from app.core.recommendation.scorer import score_policies, extract_user_features, extract_policy_features, combine_features
from app.core.recommendation.explainer import explain_policy
from app.core.vectorstore.chroma_store import load_policy_registry
from app.core.auth.deps import get_current_user, AuthUser

router = APIRouter(prefix="/api/explain", tags=["Explainability"])
logger = logging.getLogger(__name__)


@router.post("", response_model=PolicyExplanation)
async def explain_specific_policy(
    request: ExplainRequest,
    _: AuthUser = Depends(get_current_user),
):
    """Deep SHAP explanation for a specific policy (authenticated users)."""
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

    scored = score_policies(request.user_profile, {request.policy_name: policy_meta})
    if not scored:
        raise HTTPException(status_code=500, detail="Scoring failed.")

    try:
        return explain_policy(scored[0], top_n_factors=7)
    except Exception as e:
        logger.error(f"Explanation failed for '{request.policy_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
