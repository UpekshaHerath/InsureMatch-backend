"""
SHAP Explainability Module
──────────────────────────
Uses SHAP TreeExplainer on the XGBoost scorer to produce per-prediction
feature importance values, then maps them to human-readable insurance insights.
"""
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import shap

from app.core.recommendation.scorer import (
    load_model,
    ALL_FEATURE_NAMES,
    USER_FEATURE_NAMES,
    POLICY_FEATURE_NAMES,
)
from app.models.schemas import SHAPFactor, PolicyExplanation, UserProfile

logger = logging.getLogger(__name__)

# ─── Human-readable feature descriptions ─────────────────────────────────────

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # User features
    "age_norm":               "Your age",
    "gender_male":            "Your gender",
    "is_married":             "Marital status (married)",
    "num_dependents_norm":    "Number of dependents",
    "permanent_employment":   "Permanent employment status",
    "hazardous_level_norm":   "Hazardous work level",
    "income_norm":            "Monthly income level",
    "goal_cheap_quick":       "Goal: Affordable / quick coverage",
    "goal_protection":        "Goal: Life protection focus",
    "goal_savings":           "Goal: Savings & investment",
    "goal_health":            "Goal: Health coverage",
    "goal_retirement":        "Goal: Retirement planning",
    "has_any_condition":      "Having any health condition",
    "has_chronic":            "Chronic disease history",
    "has_cardiovascular":     "Cardiovascular health history",
    "has_cancer":             "Cancer / tumor history",
    "has_respiratory":        "Respiratory condition history",
    "has_neurological":       "Neurological / mental health history",
    "bmi_norm":               "BMI (body weight) level",
    "is_smoker":              "Smoking status",
    "is_alcohol":             "Alcohol consumption",
    # Policy features
    "pol_term_life":          "Policy type: Term Life",
    "pol_whole_life":         "Policy type: Whole Life",
    "pol_endowment":          "Policy type: Endowment / savings",
    "pol_health":             "Policy type: Health insurance",
    "pol_critical_illness":   "Policy type: Critical Illness",
    "pol_accident":           "Policy type: Personal Accident",
    "pol_min_age_norm":       "Policy minimum age requirement",
    "pol_max_age_norm":       "Policy maximum age limit",
    "pol_premium_level_norm": "Policy premium cost level",
    "pol_covers_health":      "Policy includes health benefits",
}

# Positive / negative reason templates keyed by feature name
REASON_TEMPLATES: Dict[str, Dict[str, str]] = {
    "age_norm": {
        "positive": "Your age falls within the ideal range for this policy.",
        "negative": "Your age is at the boundary of eligibility for this policy.",
    },
    "gender_male": {
        "positive": "Your gender profile aligns with this policy's risk assessment.",
        "negative": "Gender-based actuarial factors slightly reduce the fit.",
    },
    "is_married": {
        "positive": "Married status increases priority for family protection coverage.",
        "negative": "Single status slightly reduces the urgency for family coverage.",
    },
    "num_dependents_norm": {
        "positive": "Having dependents strongly supports the need for life coverage.",
        "negative": "Fewer dependents reduce the emphasis on life protection.",
    },
    "permanent_employment": {
        "positive": "Permanent employment makes premium payments more stable.",
        "negative": "Contract/freelance status may require more flexible premium terms.",
    },
    "hazardous_level_norm": {
        "positive": "Your hazardous work level aligns well with this policy's accident coverage.",
        "negative": "High hazardous work may attract loading premiums on this policy.",
    },
    "income_norm": {
        "positive": "Your income level comfortably supports this policy's premium structure.",
        "negative": "Premium costs may be relatively high for your current income level.",
    },
    "goal_cheap_quick": {
        "positive": "This policy aligns perfectly with your goal for affordable, accessible coverage.",
        "negative": "Your goal for cheap coverage is not the primary strength of this policy.",
    },
    "goal_protection": {
        "positive": "This policy strongly addresses your life protection goal.",
        "negative": "This policy type is less focused on pure protection benefits.",
    },
    "goal_savings": {
        "positive": "This policy combines insurance with savings/investment returns.",
        "negative": "Your savings goal is better served by endowment or whole life policies.",
    },
    "goal_health": {
        "positive": "This policy directly addresses your health coverage requirements.",
        "negative": "This policy has limited health coverage relative to your goal.",
    },
    "goal_retirement": {
        "positive": "This policy supports long-term retirement planning.",
        "negative": "This policy type is less optimal for retirement planning.",
    },
    "has_any_condition": {
        "positive": "This policy's health benefits are relevant given your medical history.",
        "negative": "Pre-existing conditions may result in exclusions on this policy.",
    },
    "has_chronic": {
        "positive": "Chronic disease coverage is a key benefit of this policy.",
        "negative": "Chronic conditions may attract higher premiums on this policy type.",
    },
    "has_cardiovascular": {
        "positive": "Cardiovascular-related critical illness coverage is relevant for your profile.",
        "negative": "Cardiovascular history may lead to medical underwriting adjustments.",
    },
    "has_cancer": {
        "positive": "Critical illness cover for cancer is especially relevant for your profile.",
        "negative": "Cancer history may affect eligibility or require medical underwriting.",
    },
    "has_respiratory": {
        "positive": "Respiratory condition coverage aligns with this policy's benefits.",
        "negative": "Respiratory conditions may be reviewed during underwriting.",
    },
    "has_neurological": {
        "positive": "Neurological condition coverage is addressed in this policy.",
        "negative": "Neurological conditions may be subject to exclusions.",
    },
    "bmi_norm": {
        "positive": "Your BMI is within a healthy range, reducing premium loading.",
        "negative": "Higher BMI may attract health loading on some policies.",
    },
    "is_smoker": {
        "positive": "Non-smoker status qualifies you for standard (lower) premium rates.",
        "negative": "Smoker status will result in higher premiums across all policies.",
    },
    "is_alcohol": {
        "positive": "Non-alcohol consumption is favorable for health-based underwriting.",
        "negative": "Regular alcohol consumption may impact health policy premiums.",
    },
    "pol_term_life": {
        "positive": "Term life provides cost-effective pure life protection.",
        "negative": "Term life has no savings component or maturity benefit.",
    },
    "pol_whole_life": {
        "positive": "Whole life provides lifelong coverage with cash value accumulation.",
        "negative": "Whole life premiums are significantly higher than term life.",
    },
    "pol_endowment": {
        "positive": "Endowment combines insurance protection with guaranteed savings.",
        "negative": "Endowment policies require longer commitment and higher premiums.",
    },
    "pol_health": {
        "positive": "Health insurance directly covers your medical expense needs.",
        "negative": "Health policy premiums may be higher given your health profile.",
    },
    "pol_critical_illness": {
        "positive": "Critical illness cover provides a lump-sum payout for major diseases.",
        "negative": "Critical illness cover may have exclusions relevant to your conditions.",
    },
    "pol_accident": {
        "positive": "Accident coverage is highly relevant for your occupation risk level.",
        "negative": "Accident-only coverage may be insufficient for your broader needs.",
    },
    "pol_min_age_norm": {
        "positive": "You meet the minimum age requirement comfortably.",
        "negative": "Age proximity to minimum limit affects policy terms.",
    },
    "pol_max_age_norm": {
        "positive": "You are well within the maximum age limit for this policy.",
        "negative": "Proximity to maximum age limit may restrict coverage term.",
    },
    "pol_premium_level_norm": {
        "positive": "The premium level is well within your affordability range.",
        "negative": "The premium level is relatively high for your income.",
    },
    "pol_covers_health": {
        "positive": "This policy includes health benefits that match your needs.",
        "negative": "Limited health coverage in this policy does not fully address your needs.",
    },
}


def _get_reason(feature_name: str, direction: str, shap_value: float) -> str:
    templates = REASON_TEMPLATES.get(feature_name, {})
    return templates.get(direction, f"This factor has a {'positive' if direction == 'positive' else 'negative'} effect on your recommendation.")


# ─── SHAP Explanation ─────────────────────────────────────────────────────────

def explain_policy(
    scored_policy: Dict[str, Any],
    top_n_factors: int = 5,
) -> PolicyExplanation:
    """
    Generate SHAP-based explanation for a single scored policy.

    Args:
        scored_policy: dict returned by scorer.score_policies() for one policy
        top_n_factors: how many top positive/negative factors to surface

    Returns:
        PolicyExplanation with ranked factors and summary
    """
    model = load_model()
    combined_features = scored_policy["combined_features"].reshape(1, -1)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(combined_features)[0]  # shape: (n_features,)
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        shap_values = np.zeros(len(ALL_FEATURE_NAMES))

    # Pair feature names with SHAP values
    feature_shap_pairs = list(zip(ALL_FEATURE_NAMES, shap_values))

    # Split into positive (beneficial) and negative (detrimental)
    positive = sorted(
        [(n, v) for n, v in feature_shap_pairs if v > 0.005],
        key=lambda x: x[1], reverse=True
    )[:top_n_factors]

    negative = sorted(
        [(n, v) for n, v in feature_shap_pairs if v < -0.005],
        key=lambda x: x[1]
    )[:top_n_factors]

    def make_factor(name: str, value: float) -> SHAPFactor:
        direction = "positive" if value > 0 else "negative"
        return SHAPFactor(
            feature=FEATURE_DESCRIPTIONS.get(name, name),
            impact_score=round(abs(value), 4),
            direction=direction,
            reason=_get_reason(name, direction, value),
        )

    pos_factors = [make_factor(n, v) for n, v in positive]
    neg_factors = [make_factor(n, v) for n, v in negative]

    # Build plain-text SHAP summary for the LLM prompt
    shap_summary = _build_shap_summary(positive, negative)

    return PolicyExplanation(
        policy_name=scored_policy["policy_name"],
        suitability_score=round(scored_policy["score"], 4),
        positive_factors=pos_factors,
        negative_factors=neg_factors,
        shap_summary=shap_summary,
    )


def _build_shap_summary(
    positive: List[Tuple[str, float]],
    negative: List[Tuple[str, float]],
) -> str:
    lines = ["Key factors driving this recommendation:\n"]
    lines.append("Positive influences:")
    for name, val in positive[:4]:
        desc = FEATURE_DESCRIPTIONS.get(name, name)
        lines.append(f"  + {desc}: +{val:.3f} impact")
    if negative:
        lines.append("\nNegative influences:")
        for name, val in negative[:3]:
            desc = FEATURE_DESCRIPTIONS.get(name, name)
            lines.append(f"  - {desc}: {val:.3f} impact")
    return "\n".join(lines)


def explain_multiple_policies(
    scored_policies: List[Dict[str, Any]],
    top_n_factors: int = 5,
) -> List[PolicyExplanation]:
    """Generate SHAP explanations for multiple policies."""
    explanations = []
    for sp in scored_policies:
        try:
            exp = explain_policy(sp, top_n_factors)
            explanations.append(exp)
        except Exception as e:
            logger.error(f"Failed to explain policy {sp.get('policy_name')}: {e}")
    return explanations
