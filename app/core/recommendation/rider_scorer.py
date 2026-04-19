"""
Rider scorer — rule-based gap-closer.

For a given (policy, rider, user_profile), compute a score in [0,1] and a list
of short human-readable reasons explaining why the rider closes a coverage gap
between the policy and the user's needs.

No ML training data exists for riders — rules ARE the explanation.
"""
from typing import Dict, Any, List, Tuple

from app.models.schemas import UserProfile

# Category constants (keep in sync with RIDERS_EXTRACTION_PROMPT)
CAT_CRITICAL_ILLNESS = "critical_illness"
CAT_ACCIDENTAL_DEATH = "accidental_death"
CAT_WAIVER_OF_PREMIUM = "waiver_of_premium"
CAT_HOSPITAL_CASH = "hospital_cash"
CAT_INCOME_PROTECTION = "income_protection"
CAT_PERMANENT_DISABILITY = "permanent_disability"
CAT_TERM_EXTENSION = "term_extension"
CAT_OTHER = "other"


def _user_has_any_health_condition(profile: UserProfile) -> bool:
    h = profile.health
    return any([
        h.has_chronic_disease, h.has_cardiovascular, h.has_cancer,
        h.has_respiratory, h.has_neurological, h.has_gastrointestinal,
        h.has_musculoskeletal, h.recent_treatment_surgery,
    ])


def _policy_covers(policy_meta: Dict[str, Any], dim: str) -> bool:
    return bool(policy_meta.get(f"covers_{dim}", False))


def score_rider(
    profile: UserProfile,
    rider: Dict[str, Any],
    policy_meta: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """
    Return (score, reasons). Score in [0,1]. Higher = better gap-closer for this
    user given the base policy's existing coverage.
    """
    reasons: List[str] = []
    score = 0.0

    # Hard filter: age eligibility
    age = profile.personal.age
    if age < int(rider.get("min_age", 18)) or age > int(rider.get("max_age", 65)):
        return 0.0, [f"Outside this rider's age range ({rider.get('min_age')}–{rider.get('max_age')})."]

    # Hard filter: applicability list must contain the base policy
    applicable = rider.get("applicable_policies") or []
    if applicable and policy_meta.get("policy_name") not in applicable:
        return 0.0, ["Not applicable to this policy."]

    category = (rider.get("category") or "other").lower()
    health_relevant = bool(rider.get("health_relevant", False))
    hazard_relevant = bool(rider.get("hazard_relevant", False))
    dependents_relevant = bool(rider.get("dependents_relevant", False))
    target_goals = [g.lower() for g in (rider.get("target_goals") or [])]

    # ── Gap signals ──
    has_health_risk = _user_has_any_health_condition(profile) or profile.lifestyle.is_smoker
    policy_covers_health = _policy_covers(policy_meta, "health")
    policy_covers_accident = _policy_covers(policy_meta, "accident")
    policy_covers_life = _policy_covers(policy_meta, "life")

    # Critical illness rider closes health gap
    if category == CAT_CRITICAL_ILLNESS:
        if has_health_risk and not policy_covers_health:
            score += 0.55
            reasons.append("You reported health concerns and the base policy doesn't include illness coverage — this rider closes that gap.")
        elif has_health_risk:
            score += 0.3
            reasons.append("You reported health concerns, so extra critical-illness coverage adds useful depth.")
        elif not policy_covers_health:
            score += 0.15
            reasons.append("The base policy doesn't include illness coverage — this rider adds it.")

    # Hospital cash rider
    if category == CAT_HOSPITAL_CASH:
        if has_health_risk and not policy_covers_health:
            score += 0.5
            reasons.append("Daily hospital cash helps cover out-of-pocket costs given your reported health risk factors.")
        elif not policy_covers_health:
            score += 0.2
            reasons.append("Your base policy has no hospitalisation cover — this rider pays a daily cash benefit.")

    # Accidental death rider — matters for hazardous occupations
    if category == CAT_ACCIDENTAL_DEATH:
        haz = profile.occupation.hazardous_level.value
        if haz in ("medium", "high") and not policy_covers_accident:
            score += 0.55
            reasons.append(f"Your occupation carries a {haz} hazardous level and the base policy has no accident coverage.")
        elif haz in ("medium", "high"):
            score += 0.3
            reasons.append(f"Extra accidental-death cover makes sense for a {haz}-hazard occupation.")
        elif not policy_covers_accident:
            score += 0.15
            reasons.append("Adds accident cover that the base policy doesn't include.")

    # Permanent disability rider — hazard-focused
    if category == CAT_PERMANENT_DISABILITY:
        haz = profile.occupation.hazardous_level.value
        if haz in ("medium", "high"):
            score += 0.5
            reasons.append(f"Permanent-disability protection is valuable for a {haz}-hazard occupation.")
        if profile.personal.num_dependents > 0:
            score += 0.15
            reasons.append("You have dependents who rely on your income — disability cover protects them if you can't work.")

    # Waiver of premium — helps people with dependents or lower income buffers
    if category == CAT_WAIVER_OF_PREMIUM:
        if profile.personal.num_dependents > 0:
            score += 0.4
            reasons.append("With dependents, waiver-of-premium keeps cover alive if you can't pay during illness/disability.")
        if profile.occupation.employment_type.value in ("contract", "freelance", "self_employed"):
            score += 0.2
            reasons.append("Your income pattern is variable — waiver protects cover during income dips.")

    # Income protection — dependents + income
    if category == CAT_INCOME_PROTECTION:
        if profile.personal.num_dependents > 0:
            score += 0.45
            reasons.append("Income protection replaces lost earnings to support your dependents.")
        if not policy_covers_life:
            score += 0.1
            reasons.append("The base policy doesn't replace income; this rider does.")

    # Term extension — only useful when policy is term-life-ish
    if category == CAT_TERM_EXTENSION:
        if policy_meta.get("policy_type") == "term_life":
            score += 0.3
            reasons.append("Lets you extend cover beyond the initial term without re-underwriting.")

    # ── Soft signals on the rider's declared relevance flags ──
    if health_relevant and has_health_risk:
        score += 0.1
        if "health" not in " ".join(reasons).lower():
            reasons.append("Particularly useful given your reported health situation.")

    if hazard_relevant and profile.occupation.hazardous_level.value in ("medium", "high"):
        score += 0.1
        if "hazard" not in " ".join(reasons).lower() and "occupation" not in " ".join(reasons).lower():
            reasons.append("Designed for people in higher-risk occupations like yours.")

    if dependents_relevant and profile.personal.num_dependents > 0:
        score += 0.1
        if "dependent" not in " ".join(reasons).lower():
            reasons.append("Designed to protect people who support dependents.")

    # Goal alignment
    primary_goal = profile.goals.primary_goal.value
    if target_goals and primary_goal in target_goals:
        score += 0.1
        reasons.append(f"Aligned with your primary goal ({primary_goal.replace('_', ' ')}).")

    # Clamp and de-duplicate reasons
    score = max(0.0, min(1.0, score))
    seen = set()
    deduped = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return score, deduped


def rank_riders_for_policy(
    profile: UserProfile,
    policy_meta: Dict[str, Any],
    rider_registry: Dict[str, Dict[str, Any]],
    top_n: int = 3,
    min_score: float = 0.25,
) -> List[Dict[str, Any]]:
    """Return up to top_n riders (best first) that close a real gap for this user."""
    scored = []
    for code, rider in rider_registry.items():
        s, reasons = score_rider(profile, rider, policy_meta)
        if s <= 0 or not reasons:
            continue
        if s < min_score:
            continue
        scored.append({
            "rider_name": rider.get("rider_name", code),
            "rider_code": code,
            "category": rider.get("category", "other"),
            "description": rider.get("description"),
            "premium_level": int(rider.get("premium_level", 1)),
            "score": round(s, 4),
            "reasons": reasons,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
