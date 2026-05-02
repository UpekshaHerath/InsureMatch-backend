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


# Per-policy inbuilt rider catalog. Sourced from the consolidated Union Assurance
# product reference (see project_client.md). Each entry is the canonical inbuilt
# rider record returned to the frontend AND used to suppress duplicate add-on
# suggestions. Keys match policy_name normalized via `_normalize_policy_key`.
_INBUILT_RIDERS_BY_POLICY: Dict[str, List[Dict[str, str]]] = {
    "advantage starter": [
        {"rider_name": "Death Cover", "rider_code": "DEATH-COVER", "category": "death_cover",
         "description": "Lump sum financial benefit to beneficiaries on the life assured's untimely demise during the policy term."},
        {"rider_name": "Continuous Fund Accumulation", "rider_code": "FUND-ACCUMULATION", "category": "investment",
         "description": "Built-in investment account where premiums and annual dividends accumulate toward a maturity payout."},
    ],
    "union advantage starter": [],  # alias filled below

    "flexlife": [
        {"rider_name": "Death Cover", "rider_code": "DEATH-COVER", "category": "death_cover",
         "description": "Lump sum payment to loved ones on the life assured's untimely demise."},
        {"rider_name": "Waiver of Premium on Death", "rider_code": "WAIVER-OF-PREMIUM-ON-DEATH", "category": CAT_WAIVER_OF_PREMIUM,
         "description": "Future premiums for the basic death cover are waived if the life assured passes away during the premium paying term."},
        {"rider_name": "Waiver of Premium on Total Permanent Disability", "rider_code": "WAIVER-OF-PREMIUM-ON-TPD", "category": CAT_WAIVER_OF_PREMIUM,
         "description": "Future premiums waived if the life assured becomes totally and permanently disabled during the premium paying term."},
    ],
    "union flexlife": [],  # alias
    "union assurance flexlife": [],  # alias

    "health 360": [
        {"rider_name": "Hospitalisation Benefit", "rider_code": "HOSPITALISATION", "category": "hospitalisation",
         "description": "Covers hospital room, board, and ICU expenses as per actuals."},
        {"rider_name": "Surgical Benefit", "rider_code": "SURGICAL", "category": "surgery",
         "description": "Fees for surgeons, anaesthetists, consultants, and specialists for medical procedures."},
        {"rider_name": "Miscellaneous Hospital Services", "rider_code": "MISC-HOSPITAL", "category": "hospitalisation",
         "description": "Covers operation theatre charges, oxygen, blood, and prescribed drugs."},
        {"rider_name": "In-Built Critical Illness Cover", "rider_code": "INBUILT-CI", "category": CAT_CRITICAL_ILLNESS,
         "description": "Coverage for critical illnesses included in the base plan."},
        {"rider_name": "Day Care Surgery Benefit", "rider_code": "DAYCARE-SURGERY", "category": "surgery",
         "description": "Surgeries and hospital stays under 24 hours for listed procedures."},
        {"rider_name": "Pre and Post Hospitalisation Benefits", "rider_code": "PRE-POST-HOSP", "category": "hospitalisation",
         "description": "Reimburses medical expenses 30 days before admission and 30 days after discharge."},
        {"rider_name": "Ambulance Fees", "rider_code": "AMBULANCE", "category": "hospitalisation",
         "description": "Licensed ambulance service charges up to 2% of the basic annual sum insured."},
        {"rider_name": "Organ Donor Expenses", "rider_code": "ORGAN-DONOR", "category": "specialist_medical",
         "description": "Donor's hospitalisation costs covered within the recipient's overall sum insured."},
        {"rider_name": "Prosthesis and Implants", "rider_code": "PROSTHESIS", "category": "specialist_medical",
         "description": "Medical implants and prosthetics up to 70% of the annual sum insured."},
        {"rider_name": "Annual Limit Reinstatement", "rider_code": "LIMIT-REINSTATE", "category": "hospitalisation",
         "description": "Full benefit amount reinstated for an unrelated medical emergency in the same year."},
        {"rider_name": "25% Claim-Free Year Bonus", "rider_code": "NCB-25", "category": "loyalty",
         "description": "Coverage limit increases by 25% for every claim-free year."},
        {"rider_name": "Wellbeing Cover", "rider_code": "WELLBEING", "category": "preventive",
         "description": "Free health check-ups (up to 2% of sum insured) after two consecutive claim-free years."},
    ],
    "health360": [],  # alias
    "union health 360": [],  # alias
    "union health360": [],  # alias

    "life plus": [
        {"rider_name": "Death Cover (Higher of BSA or Fund)", "rider_code": "DEATH-COVER-HIGHER-OF", "category": "death_cover",
         "description": "Beneficiaries receive either the Basic Sum Assured or the investment account balance, whichever is higher."},
        {"rider_name": "Continuous Fund Accumulation", "rider_code": "FUND-ACCUMULATION", "category": "investment",
         "description": "Built-in investment account that grows monthly through premium contributions and annual dividends."},
    ],
    "union life plus": [],  # alias
    "life+": [],  # alias

    "pension advantage": [
        {"rider_name": "Pension Fund Accumulation", "rider_code": "PENSION-FUND", "category": "investment",
         "description": "Dedicated retirement account that grows through contributions and dividends."},
        {"rider_name": "Pension Payout (Lump Sum or Monthly)", "rider_code": "PENSION-PAYOUT", "category": "pension_payout",
         "description": "Receive the fund as a single lump sum or as a structured monthly pension over 10, 15, or 20 years."},
        {"rider_name": "Premium Waiver on Death", "rider_code": "WAIVER-OF-PREMIUM-ON-DEATH", "category": CAT_WAIVER_OF_PREMIUM,
         "description": "Company continues paying premiums until maturity if the policyholder passes away."},
        {"rider_name": "Premium Waiver on Total Permanent Disability", "rider_code": "WAIVER-OF-PREMIUM-ON-TPD", "category": CAT_WAIVER_OF_PREMIUM,
         "description": "Future premiums waived if the policyholder becomes disabled due to accident or sickness."},
        {"rider_name": "Withdrawal Benefit", "rider_code": "WITHDRAWAL", "category": "withdrawal",
         "description": "One-time emergency withdrawal of up to 15% after three policy years."},
    ],
    "union pension advantage": [],  # alias

    "single premium advantage": [
        {"rider_name": "Death Cover", "rider_code": "DEATH-COVER-105", "category": "death_cover",
         "description": "Guaranteed benefit of 105% of the initial single premium."},
        {"rider_name": "Dedicated Investment Account", "rider_code": "INVESTMENT-ACCOUNT", "category": "investment",
         "description": "One-time premium compounds through annual dividends across the policy term."},
    ],
    "single primium advantage": [],  # alias (registry typo)
    "union single premium advantage": [],  # alias

    "sisumaga+": [
        {"rider_name": "Life Cover (5x Basic Annual Premium)", "rider_code": "LIFE-COVER-5X", "category": "death_cover",
         "description": "Immediate lump-sum payment to the family on the parent's demise."},
        {"rider_name": "Education Assistance Fee Benefit", "rider_code": "EDU-ASSIST", "category": "education_income",
         "description": "Consistent monthly income paid to the family until policy maturity."},
        {"rider_name": "Waiver of Premium on Death", "rider_code": "WAIVER-OF-PREMIUM-ON-DEATH", "category": CAT_WAIVER_OF_PREMIUM,
         "description": "Union Assurance continues paying premiums if the parent passes away."},
        {"rider_name": "Education Fund Accumulation", "rider_code": "EDU-FUND", "category": "investment",
         "description": "Dedicated fund growing toward higher-education milestones."},
        {"rider_name": "15% Loyalty Bonus", "rider_code": "LOYALTY-15", "category": "loyalty",
         "description": "Additional 15% bonus added to the fund at maturity."},
    ],
    "sisumaga plus": [],  # alias
    "union sisumaga plus": [],  # alias
}

# Fill aliases (any empty list copies the canonical entry).
_ALIAS_MAP = {
    "union advantage starter": "advantage starter",
    "union flexlife": "flexlife",
    "union assurance flexlife": "flexlife",
    "health360": "health 360",
    "union health 360": "health 360",
    "union health360": "health 360",
    "union life plus": "life plus",
    "life+": "life plus",
    "union pension advantage": "pension advantage",
    "single primium advantage": "single premium advantage",
    "union single premium advantage": "single premium advantage",
    "sisumaga plus": "sisumaga+",
    "union sisumaga plus": "sisumaga+",
}
for _alias, _canonical in _ALIAS_MAP.items():
    if not _INBUILT_RIDERS_BY_POLICY.get(_alias):
        _INBUILT_RIDERS_BY_POLICY[_alias] = _INBUILT_RIDERS_BY_POLICY[_canonical]


def _inbuilt_name_set(policy_key: str) -> set:
    """Lowercased rider names already inbuilt for this policy (used for add-on dedup)."""
    return {r["rider_name"].strip().lower() for r in _INBUILT_RIDERS_BY_POLICY.get(policy_key, [])}


def _inbuilt_category_set(policy_key: str) -> set:
    """Categories with at least one inbuilt rider (kept for back-compat)."""
    return {r["category"] for r in _INBUILT_RIDERS_BY_POLICY.get(policy_key, [])}

# Per-policy boosts. Each entry: (extra_score, reason). Drives differentiation
# even when `covers_*` flags are identical across policies in the registry.
_POLICY_NAME_BOOSTS: Dict[str, Dict[str, Tuple[float, str]]] = {
    "pension advantage": {
        CAT_INCOME_PROTECTION:    (0.25, "Pension Advantage is built for retirement income — this rider strengthens that focus."),
        CAT_PERMANENT_DISABILITY: (0.15, "Disability cover protects the contributions that fund your future pension."),
    },
    "union pension advantage": {
        CAT_INCOME_PROTECTION:    (0.25, "Pension Advantage is built for retirement income — this rider strengthens that focus."),
        CAT_PERMANENT_DISABILITY: (0.15, "Disability cover protects the contributions that fund your future pension."),
    },
    "flexlife": {
        CAT_CRITICAL_ILLNESS: (0.20, "FlexLife's flexible structure pairs well with broader illness protection."),
        CAT_HOSPITAL_CASH:    (0.15, "FlexLife does not include hospital cash natively — this rider plugs that gap."),
    },
    "union assurance flexlife": {
        CAT_CRITICAL_ILLNESS: (0.20, "FlexLife's flexible structure pairs well with broader illness protection."),
        CAT_HOSPITAL_CASH:    (0.15, "FlexLife does not include hospital cash natively — this rider plugs that gap."),
    },
    "life plus": {
        CAT_ACCIDENTAL_DEATH: (0.20, "Life+ has no built-in accident cover — this rider adds it."),
        CAT_HOSPITAL_CASH:    (0.15, "Life+ does not pay daily hospital cash — this rider plugs that gap."),
    },
    "union life plus": {
        CAT_ACCIDENTAL_DEATH: (0.20, "Life+ has no built-in accident cover — this rider adds it."),
        CAT_HOSPITAL_CASH:    (0.15, "Life+ does not pay daily hospital cash — this rider plugs that gap."),
    },
    "single primium advantage": {
        CAT_TERM_EXTENSION:   (0.25, "Single Premium Advantage's fixed term benefits from a term-extension option."),
        CAT_ACCIDENTAL_DEATH: (0.15, "Adds an accident layer to a plan that is otherwise investment-led."),
    },
    "single premium advantage": {
        CAT_TERM_EXTENSION:   (0.25, "Single Premium Advantage's fixed term benefits from a term-extension option."),
        CAT_ACCIDENTAL_DEATH: (0.15, "Adds an accident layer to a plan that is otherwise investment-led."),
    },
    "advantage starter": {
        CAT_CRITICAL_ILLNESS: (0.20, "Advantage Starter's lean base benefits from added illness protection."),
        CAT_HOSPITAL_CASH:    (0.15, "Adds daily hospital cash on top of an entry-level base policy."),
    },
    "sisumaga+": {
        CAT_ACCIDENTAL_DEATH:     (0.20, "Adds an accident payout on top of the education-protection base."),
        CAT_INCOME_PROTECTION:    (0.15, "Income protection keeps your child's education fund on track."),
    },
}

_POLICY_TYPE_BOOSTS: Dict[str, Dict[str, Tuple[float, str]]] = {
    "term_life": {
        CAT_TERM_EXTENSION:   (0.20, "Term-life plans benefit from extending cover beyond the original term."),
        CAT_ACCIDENTAL_DEATH: (0.10, "A term plan with added accident cover gives broader death protection."),
    },
    "whole_life": {
        CAT_CRITICAL_ILLNESS:     (0.10, "A whole-life base pairs well with critical-illness depth."),
        CAT_PERMANENT_DISABILITY: (0.10, "Disability cover complements whole-life protection."),
    },
    "endowment": {
        CAT_WAIVER_OF_PREMIUM: (0.10, "Endowment plans rely on completed premiums — waiver protects the maturity payout."),
    },
    "health": {
        CAT_HOSPITAL_CASH: (0.10, "Pairs daily cash with the policy's hospitalisation coverage."),
    },
}


def _normalize_policy_key(name: str) -> str:
    return (name or "").strip().lower()


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

    # Hard filter: skip riders already built-in to this policy — suggesting
    # them again would duplicate coverage the customer already has. Match
    # primarily by rider_name (authoritative per inbuilt catalog) and fall
    # back to category for legacy registry entries.
    policy_key = _normalize_policy_key(policy_meta.get("policy_name", ""))
    rider_name_norm = (rider.get("rider_name") or "").strip().lower()
    if rider_name_norm and rider_name_norm in _inbuilt_name_set(policy_key):
        return 0.0, [f"Already built into {policy_meta.get('policy_name')}."]
    if category in _inbuilt_category_set(policy_key):
        return 0.0, [f"Already built into {policy_meta.get('policy_name')}."]
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

    # ── Policy-specific boosts (drive per-policy differentiation) ──
    name_boost = _POLICY_NAME_BOOSTS.get(policy_key, {}).get(category)
    if name_boost:
        score += name_boost[0]
        reasons.append(name_boost[1])

    type_boost = _POLICY_TYPE_BOOSTS.get(
        (policy_meta.get("policy_type") or "").lower(), {}
    ).get(category)
    if type_boost:
        score += type_boost[0]
        if type_boost[1] not in reasons:
            reasons.append(type_boost[1])

    # Clamp and de-duplicate reasons
    score = max(0.0, min(1.0, score))
    seen = set()
    deduped = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return score, deduped


def get_inbuilt_riders(
    policy_meta: Dict[str, Any],
    rider_registry: Dict[str, Dict[str, Any]],  # kept for signature compat; unused
) -> List[Dict[str, Any]]:
    """
    Return riders bundled with this policy directly from the canonical inbuilt
    catalog (`_INBUILT_RIDERS_BY_POLICY`). Source: Union Assurance product
    reference. Independent of the rider_registry / ChromaDB ingestion state.
    """
    policy_key = _normalize_policy_key(policy_meta.get("policy_name", ""))
    return [dict(r) for r in _INBUILT_RIDERS_BY_POLICY.get(policy_key, [])]


def rank_riders_for_policy(
    profile: UserProfile,
    policy_meta: Dict[str, Any],
    rider_registry: Dict[str, Dict[str, Any]],
    top_n: int = 3,
    min_score: float = 0.1,
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
