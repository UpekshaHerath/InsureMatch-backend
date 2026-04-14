"""
XGBoost Policy Scorer
─────────────────────
Trains on synthetic data encoding insurance domain rules.
Produces a suitability score in [0, 1] for each (user, policy) pair.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

import xgboost as xgb
import joblib

from app.config import settings
from app.models.schemas import UserProfile

logger = logging.getLogger(__name__)

MODEL_PATH = Path(settings.MODEL_SAVE_DIR) / "xgb_scorer.pkl"

# ─── Feature names (must stay ordered) ───────────────────────────────────────
USER_FEATURE_NAMES = [
    "age_norm", "gender_male", "is_married", "num_dependents_norm",
    "permanent_employment", "hazardous_level_norm", "income_norm",
    "goal_cheap_quick", "goal_protection", "goal_savings", "goal_health", "goal_retirement",
    "has_any_condition", "has_chronic", "has_cardiovascular", "has_cancer",
    "has_respiratory", "has_neurological",
    "bmi_norm", "is_smoker", "is_alcohol",
]

POLICY_FEATURE_NAMES = [
    "pol_term_life", "pol_whole_life", "pol_endowment",
    "pol_health", "pol_critical_illness", "pol_accident",
    "pol_min_age_norm", "pol_max_age_norm",
    "pol_premium_level_norm", "pol_covers_health",
]

ALL_FEATURE_NAMES = USER_FEATURE_NAMES + POLICY_FEATURE_NAMES

HAZARDOUS_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3}
EMPLOYMENT_PERMANENT = {"permanent"}
GOAL_MAP = {
    "cheap_and_quick": (1, 0, 0, 0, 0),
    "protection":      (0, 1, 0, 0, 0),
    "savings_and_investment": (0, 0, 1, 0, 0),
    "health_coverage": (0, 0, 0, 1, 0),
    "retirement":      (0, 0, 0, 0, 1),
    "none":            (0, 0, 0, 0, 0),
}

# Policy library used for training & inference
POLICY_TEMPLATES: Dict[str, List[float]] = {
    # name: [term, whole, endow, health, ci, accident, min_age/70, max_age/70, premium/2, covers_health]
    "Basic Term Life":          [1, 0, 0, 0, 0, 0, 18/70, 65/70, 0/2, 0],
    "Whole Life Protection":    [0, 1, 0, 0, 0, 0, 18/70, 55/70, 2/2, 0],
    "Endowment Savings Plan":   [0, 0, 1, 0, 0, 0, 18/70, 50/70, 2/2, 0],
    "Comprehensive Health":     [0, 0, 0, 1, 0, 0, 18/70, 60/70, 1/2, 1],
    "Critical Illness Cover":   [0, 0, 0, 0, 1, 0, 18/70, 60/70, 1/2, 1],
    "Personal Accident":        [0, 0, 0, 0, 0, 1, 18/70, 65/70, 0/2, 0],
    "Term Life + CI Rider":     [1, 0, 0, 0, 1, 0, 18/70, 60/70, 1/2, 1],
    "Premium Health Shield":    [0, 0, 0, 1, 1, 0, 18/70, 60/70, 2/2, 1],
}


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_user_features(profile: UserProfile) -> np.ndarray:
    p = profile.personal
    o = profile.occupation
    g = profile.goals
    h = profile.health
    ls = profile.lifestyle

    goal_vals = GOAL_MAP.get(g.primary_goal.value, (0, 0, 0, 0, 0))
    hazardous_num = HAZARDOUS_MAP.get(o.hazardous_level.value, 0)
    has_any = any([h.has_chronic_disease, h.has_cardiovascular, h.has_cancer,
                   h.has_respiratory, h.has_neurological])

    feats = [
        p.age / 70,
        1.0 if p.gender.value == "male" else 0.0,
        1.0 if p.marital_status.value == "married" else 0.0,
        min(p.num_dependents, 5) / 5.0,
        1.0 if o.employment_type.value in EMPLOYMENT_PERMANENT else 0.0,
        hazardous_num / 3.0,
        min(o.monthly_income_lkr / 500_000, 1.0),
        *goal_vals,
        float(has_any),
        float(h.has_chronic_disease),
        float(h.has_cardiovascular),
        float(h.has_cancer),
        float(h.has_respiratory),
        float(h.has_neurological),
        min(ls.bmi / 40.0, 1.0),
        float(ls.is_smoker),
        float(ls.is_alcohol_consumer),
    ]
    return np.array(feats, dtype=np.float32)


def extract_policy_features(policy_meta: Dict[str, Any]) -> np.ndarray:
    ptype = policy_meta.get("policy_type", "term_life")
    feats = [
        1.0 if ptype == "term_life" else 0.0,
        1.0 if ptype == "whole_life" else 0.0,
        1.0 if ptype == "endowment" else 0.0,
        1.0 if ptype == "health" else 0.0,
        1.0 if ptype == "critical_illness" else 0.0,
        1.0 if ptype == "accident" else 0.0,
        policy_meta.get("min_age", 18) / 70.0,
        policy_meta.get("max_age", 65) / 70.0,
        policy_meta.get("premium_level", 1) / 2.0,
        1.0 if policy_meta.get("covers_health", False) else 0.0,
    ]
    return np.array(feats, dtype=np.float32)


def combine_features(user_feats: np.ndarray, policy_feats: np.ndarray) -> np.ndarray:
    return np.concatenate([user_feats, policy_feats])


# ─── Domain Scoring Function (for synthetic training) ────────────────────────

def _domain_score(user_f: np.ndarray, policy_f: np.ndarray, noise_std: float = 0.04) -> float:
    """Encodes insurance domain rules as a differentiable scoring function."""
    age = user_f[0] * 70
    is_married = user_f[2]
    dependents = user_f[3] * 5
    hazardous = user_f[5] * 3
    income_norm = user_f[6]
    goal_cheap = user_f[7]
    goal_protect = user_f[8]
    goal_savings = user_f[9]
    goal_health = user_f[10]
    goal_retire = user_f[11]
    has_any_cond = user_f[12]
    has_cardiovascular = user_f[14]
    has_cancer = user_f[15]
    bmi_norm = user_f[18]
    is_smoker = user_f[19]

    pol_term = policy_f[0]
    pol_whole = policy_f[1]
    pol_endow = policy_f[2]
    pol_health = policy_f[3]
    pol_ci = policy_f[4]
    pol_accident = policy_f[5]
    min_age = policy_f[6] * 70
    max_age = policy_f[7] * 70
    premium_level = policy_f[8] * 2
    covers_health = policy_f[9]

    score = 0.40  # base

    # Age eligibility — hard constraint
    if min_age <= age <= max_age:
        score += 0.10
    else:
        score -= 0.45  # strongly ineligible

    # Goal alignment (primary driver)
    if goal_cheap:
        if pol_term:
            score += 0.25
        if premium_level == 0:
            score += 0.10
        if premium_level == 2:
            score -= 0.15

    if goal_protect:
        if pol_whole or pol_endow:
            score += 0.22
        if pol_term:
            score += 0.10

    if goal_health:
        if covers_health or pol_health or pol_ci:
            score += 0.28

    if goal_savings:
        if pol_endow or pol_whole:
            score += 0.25

    if goal_retire:
        if pol_whole or pol_endow:
            score += 0.22

    # Family / dependents → life coverage matters
    if dependents > 0:
        if pol_term or pol_whole:
            score += 0.08
        if is_married:
            score += 0.04

    # Health conditions
    if has_any_cond:
        if covers_health or pol_ci:
            score += 0.18
        elif pol_term and not covers_health:
            score -= 0.05

    if has_cardiovascular or has_cancer:
        if pol_ci:
            score += 0.22
        elif pol_term and not covers_health:
            score -= 0.08

    # Hazardous occupation → accident coverage
    if hazardous >= 2:
        if pol_accident:
            score += 0.25
        elif pol_term:
            score -= 0.05

    # Smoker effects
    if is_smoker:
        score -= 0.05
        if pol_health or pol_ci:
            score -= 0.06

    # BMI > overweight
    bmi = bmi_norm * 40
    if bmi > 30:
        if covers_health or pol_health:
            score += 0.08
        else:
            score -= 0.04

    # Low income + high premium → mismatch
    if income_norm < 0.28 and premium_level > 1:
        score -= 0.15

    # Young age bonus for affordable entry policies
    if age <= 25 and (pol_term or pol_accident) and premium_level <= 1:
        score += 0.08

    noise = np.random.normal(0, noise_std)
    return float(np.clip(score + noise, 0.0, 1.0))


# ─── Synthetic Training Data ──────────────────────────────────────────────────

def _generate_training_data(n_samples: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    policy_list = list(POLICY_TEMPLATES.values())
    X, y = [], []

    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        age = rng.integers(18, 66)
        gender = rng.integers(0, 2)
        is_married = rng.integers(0, 2)
        dependents = rng.integers(0, 6)
        perm_emp = rng.integers(0, 2)
        hazardous = rng.integers(0, 4)
        income_norm = float(rng.uniform(0.1, 1.0))

        goal_idx = rng.integers(0, 5)
        goal_vals = [(1 if i == goal_idx else 0) for i in range(5)]

        has_any = int(rng.random() < 0.25)
        has_chronic = int(rng.random() < 0.15)
        has_cardio = int(rng.random() < 0.10)
        has_cancer = int(rng.random() < 0.05)
        has_resp = int(rng.random() < 0.10)
        has_neuro = int(rng.random() < 0.08)
        bmi_norm = float(rng.uniform(0.4, 0.85))
        is_smoker = int(rng.random() < 0.20)
        is_alcohol = int(rng.random() < 0.30)

        user_f = np.array([
            age / 70, gender, is_married, dependents / 5,
            perm_emp, hazardous / 3, income_norm,
            *goal_vals,
            has_any, has_chronic, has_cardio, has_cancer, has_resp, has_neuro,
            bmi_norm, is_smoker, is_alcohol,
        ], dtype=np.float32)

        policy_f = np.array(rng.choice(policy_list), dtype=np.float32)
        score = _domain_score(user_f, policy_f)

        X.append(combine_features(user_f, policy_f))
        y.append(score)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── Model Training & Loading ─────────────────────────────────────────────────

def train_and_save_model() -> xgb.XGBRegressor:
    logger.info("Training XGBoost scorer on synthetic data…")
    X, y = _generate_training_data(n_samples=8000)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"XGBoost scorer saved to {MODEL_PATH}")
    return model


def load_model() -> xgb.XGBRegressor:
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


# ─── Scoring Interface ────────────────────────────────────────────────────────

def score_policies(
    profile: UserProfile,
    policies: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score all policies for the given user profile.

    Returns list of dicts sorted by score descending:
        [{"policy_name": ..., "score": float, "policy_meta": {...}}, ...]
    """
    model = load_model()
    user_feats = extract_user_features(profile)
    results = []

    for policy_name, policy_meta in policies.items():
        policy_feats = extract_policy_features(policy_meta)
        combined = combine_features(user_feats, policy_feats).reshape(1, -1)
        score = float(model.predict(combined)[0])
        score = float(np.clip(score, 0.0, 1.0))
        results.append({
            "policy_name": policy_name,
            "score": score,
            "policy_meta": policy_meta,
            "user_features": user_feats,
            "policy_features": policy_feats,
            "combined_features": combined[0],
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
