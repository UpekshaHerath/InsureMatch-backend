"""
Recommendation Ranker — Orchestrates the full pipeline:
  UserProfile → XGBoost scoring → SHAP explanation → RAG narrative
"""
import uuid
import logging
from typing import List, Dict, Any

from app.models.schemas import (
    UserProfile,
    RecommendationResponse,
    PolicyScore,
    RiderRecommendation,
)
from app.core.recommendation.scorer import score_policies
from app.core.recommendation.explainer import explain_multiple_policies
from app.core.recommendation.rider_scorer import rank_riders_for_policy
from app.core.rag.chain import generate_recommendation_narrative
from app.core.vectorstore.chroma_store import load_policy_registry, load_rider_registry

logger = logging.getLogger(__name__)


async def recommend(profile: UserProfile, top_k: int = 3) -> RecommendationResponse:
    """
    Full recommendation pipeline.

    Steps:
        1. Load policy registry (structured metadata)
        2. Score all policies with XGBoost
        3. Select top-K policies
        4. Generate SHAP explanations for each
        5. Generate LLM narrative (RAG + SHAP context)
        6. Return structured response
    """
    # 1. Load policy registry
    registry = load_policy_registry()
    if not registry:
        raise ValueError(
            "No insurance policies have been indexed yet. "
            "Please upload policy documents via POST /api/ingest first."
        )

    # 2. Score all policies
    scored = score_policies(profile, registry)

    # 3. Take top-K
    top_scored = scored[:top_k]

    # 4. SHAP explanations
    explanations = explain_multiple_policies(top_scored, top_n_factors=5)

    # 5. Build ranked policy list
    ranked_policies = []
    for rank, sp in enumerate(top_scored, start=1):
        meta = sp["policy_meta"]
        ranked_policies.append(PolicyScore(
            policy_name=sp["policy_name"],
            policy_type=meta.get("policy_type", "unknown"),
            company=meta.get("company"),
            suitability_score=round(sp["score"], 4),
            rank=rank,
        ))

    # 6. Generate narrative for top recommendation
    top = top_scored[0]
    top_explanation = explanations[0] if explanations else None
    shap_text = top_explanation.shap_summary if top_explanation else "SHAP analysis unavailable."

    runner_up_names = [sp["policy_name"] for sp in top_scored[1:]]
    narrative = await generate_recommendation_narrative(
        profile=profile,
        top_policy_name=top["policy_name"],
        top_score=top["score"],
        shap_explanation=shap_text,
        ranked_policy_names=runner_up_names,
    )

    # 7. Rider suggestions — gap-closers per top policy
    rider_registry = load_rider_registry()
    rider_suggestions: Dict[str, List[RiderRecommendation]] = {}
    if rider_registry:
        for sp in top_scored:
            meta = sp["policy_meta"]
            ranked_riders = rank_riders_for_policy(
                profile=profile,
                policy_meta=meta,
                rider_registry=rider_registry,
                top_n=3,
            )
            rider_suggestions[sp["policy_name"]] = [
                RiderRecommendation(**r) for r in ranked_riders
            ]

    session_id = str(uuid.uuid4())

    return RecommendationResponse(
        ranked_policies=ranked_policies,
        top_recommendation=top["policy_name"],
        explanations=explanations,
        rag_narrative=narrative,
        session_id=session_id,
        rider_suggestions=rider_suggestions,
    )
