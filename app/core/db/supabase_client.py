"""Thin async Supabase REST wrapper using service-role key (bypasses RLS).

Ownership enforced in code via user_id from validated JWT.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_BASE = f"{settings.SUPABASE_URL.rstrip('/')}/rest/v1"
_HEADERS = {
    "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}


async def _request(method: str, path: str, **kwargs) -> Any:
    url = f"{_BASE}{path}"
    headers = {**_HEADERS, **kwargs.pop("headers", {})}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.request(method, url, headers=headers, **kwargs)
    if r.status_code >= 400:
        logger.error(f"Supabase {method} {path} → {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    if r.status_code == 204 or not r.content:
        return None
    return r.json()


# ─── profiles ────────────────────────────────────────────────────────────────

def _flatten_profile(user_id: str, profile) -> Dict[str, Any]:
    p = profile.personal
    o = profile.occupation
    g = profile.goals
    h = profile.health
    l = profile.lifestyle
    return {
        "user_id": user_id,
        "age": p.age,
        "gender": p.gender.value if hasattr(p.gender, "value") else p.gender,
        "marital_status": p.marital_status.value if hasattr(p.marital_status, "value") else p.marital_status,
        "nationality": p.nationality,
        "country": p.country,
        "district": p.district,
        "city": p.city,
        "num_dependents": p.num_dependents,
        "occupation": o.occupation,
        "employment_type": o.employment_type.value if hasattr(o.employment_type, "value") else o.employment_type,
        "designation": o.designation,
        "hazardous_level": o.hazardous_level.value if hasattr(o.hazardous_level, "value") else o.hazardous_level,
        "hazardous_activities": o.hazardous_activities,
        "monthly_income_lkr": o.monthly_income_lkr,
        "has_existing_insurance": o.has_existing_insurance,
        "current_insurance_status": o.current_insurance_status.value if hasattr(o.current_insurance_status, "value") else o.current_insurance_status,
        "employer_insurance_scheme": o.employer_insurance_scheme,
        "primary_goal": g.primary_goal.value if hasattr(g.primary_goal, "value") else g.primary_goal,
        "secondary_goal": g.secondary_goal.value if (g.secondary_goal and hasattr(g.secondary_goal, "value")) else g.secondary_goal,
        "travel_history_high_risk": g.travel_history_high_risk,
        "dual_citizenship": g.dual_citizenship,
        "tax_regulatory_flags": g.tax_regulatory_flags,
        "insurance_history_issues": g.insurance_history_issues,
        "has_chronic_disease": h.has_chronic_disease,
        "has_cardiovascular": h.has_cardiovascular,
        "has_cancer": h.has_cancer,
        "has_respiratory": h.has_respiratory,
        "has_neurological": h.has_neurological,
        "has_gastrointestinal": h.has_gastrointestinal,
        "has_musculoskeletal": h.has_musculoskeletal,
        "has_infectious_sexual": h.has_infectious_sexual,
        "recent_treatment_surgery": h.recent_treatment_surgery,
        "covid_related": h.covid_related,
        "bmi": l.bmi,
        "is_smoker": l.is_smoker,
        "is_alcohol_consumer": l.is_alcohol_consumer,
    }


async def upsert_profile(user_id: str, profile) -> None:
    payload = _flatten_profile(user_id, profile)
    await _request(
        "POST", "/profiles?on_conflict=user_id",
        headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        json=payload,
    )


async def get_profile(user_id: str) -> Optional[Dict[str, Any]]:
    rows = await _request("GET", f"/profiles?user_id=eq.{user_id}&select=*")
    return rows[0] if rows else None


# ─── recommendations ────────────────────────────────────────────────────────

async def insert_recommendation(user_id: str, session_id: str, resp) -> None:
    rider_suggestions = {}
    raw = getattr(resp, "rider_suggestions", {}) or {}
    for policy_name, riders in raw.items():
        rider_suggestions[policy_name] = [r.model_dump() for r in riders]

    inbuilt_riders = {}
    raw_inbuilt = getattr(resp, "inbuilt_riders", {}) or {}
    for policy_name, riders in raw_inbuilt.items():
        inbuilt_riders[policy_name] = [r.model_dump() for r in riders]

    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "top_recommendation": resp.top_recommendation,
        "ranked_policies": [p.model_dump() for p in resp.ranked_policies],
        "explanations": [e.model_dump() for e in resp.explanations],
        "rag_narrative": resp.rag_narrative,
        "rider_suggestions": rider_suggestions,
        "inbuilt_riders": inbuilt_riders,
    }
    await _request(
        "POST", "/recommendations",
        headers={"Prefer": "return=minimal"},
        json=payload,
    )


async def list_recommendations(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    return await _request(
        "GET",
        f"/recommendations?user_id=eq.{user_id}&select=*&order=created_at.desc&limit={limit}",
    ) or []


# ─── chat sessions + messages ───────────────────────────────────────────────

async def ensure_session(user_id: str, session_id: str, title: Optional[str] = None) -> None:
    await _request(
        "POST", "/chat_sessions?on_conflict=id",
        headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        json={"id": session_id, "user_id": user_id, "title": title},
    )


async def touch_session(session_id: str, user_id: str) -> None:
    await _request(
        "PATCH",
        f"/chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}",
        headers={"Prefer": "return=minimal"},
        json={"last_active": "now()"},
    )


async def delete_session(session_id: str, user_id: str) -> None:
    await _request(
        "DELETE",
        f"/chat_sessions?id=eq.{session_id}&user_id=eq.{user_id}",
        headers={"Prefer": "return=minimal"},
    )


async def list_messages(session_id: str, user_id: str, limit: int = 24) -> List[Dict[str, Any]]:
    return await _request(
        "GET",
        f"/chat_messages?session_id=eq.{session_id}&user_id=eq.{user_id}"
        f"&select=role,content,sources,created_at&order=created_at.asc&limit={limit}",
    ) or []


async def insert_message(user_id: str, session_id: str, role: str, content: str, sources: Optional[List[str]] = None) -> None:
    await _request(
        "POST", "/chat_messages",
        headers={"Prefer": "return=minimal"},
        json={
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": sources or [],
        },
    )
