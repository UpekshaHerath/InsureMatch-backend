import json
import logging
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.llm.groq_llm import get_groq_llm, get_groq_llm_creative
from app.core.rag.prompts import (
    RECOMMENDATION_PROMPT,
    CHAT_PROMPT,
    METADATA_EXTRACTION_PROMPT,
)
from app.core.vectorstore.chroma_store import similarity_search, similarity_search_for_policy
from app.models.schemas import UserProfile, PolicyMetadata

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6  # Keep last 6 turns (12 messages)


# ─── Metadata Extraction ──────────────────────────────────────────────────────

def extract_policy_metadata_with_llm(document_text: str, filename: str) -> Dict[str, Any]:
    """Use Groq to extract structured policy metadata from document content."""
    llm = get_groq_llm(temperature=0.0)
    prompt = METADATA_EXTRACTION_PROMPT.format(
        document_excerpt=document_text[:3000],
        filename=filename,
    )
    try:
        result = llm.invoke(prompt)
        raw = result.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}. Using defaults.")
        stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        return {
            "policy_name": stem,
            "policy_type": "term_life",
            "company": None,
            "min_age": 18,
            "max_age": 65,
            "premium_level": 1,
            "covers_health": False,
            "covers_life": True,
            "covers_accident": False,
            "is_entry_level": False,
            "description": None,
        }


# ─── RAG Narrative Generation ─────────────────────────────────────────────────

def format_chat_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for role, msg in history[-MAX_HISTORY_TURNS * 2:]:
        label = "Client" if role == "human" else "Advisor"
        lines.append(f"{label}: {msg}")
    return "\n".join(lines)


def build_health_summary(profile: UserProfile) -> str:
    h = profile.health
    conditions = []
    if h.has_chronic_disease:
        conditions.append("Chronic Disease")
    if h.has_cardiovascular:
        conditions.append("Cardiovascular Issue")
    if h.has_cancer:
        conditions.append("Cancer/Tumor")
    if h.has_respiratory:
        conditions.append("Respiratory Condition")
    if h.has_neurological:
        conditions.append("Neurological/Mental Health")
    if h.has_gastrointestinal:
        conditions.append("Gastrointestinal")
    if h.has_musculoskeletal:
        conditions.append("Musculoskeletal")
    if h.recent_treatment_surgery:
        conditions.append("Recent Treatment/Surgery")
    return ", ".join(conditions) if conditions else "None reported"


async def generate_recommendation_narrative(
    profile: UserProfile,
    top_policy_name: str,
    top_score: float,
    shap_explanation: str,
    ranked_policy_names: List[str],
) -> str:
    """Generate the full recommendation narrative using RAG context + SHAP insights."""
    llm = get_groq_llm_creative(temperature=0.3)

    # Build user query from profile for RAG retrieval
    query = _build_user_query(profile)
    retrieved = similarity_search(query, k=settings_k())

    # Filter to top policy context + general context
    rag_context = _format_rag_context(retrieved, top_policy_name)

    p = profile.personal
    o = profile.occupation
    g = profile.goals

    prompt_values = {
        "age": p.age,
        "gender": p.gender.value,
        "marital_status": p.marital_status.value,
        "num_dependents": p.num_dependents,
        "city": p.city or "N/A",
        "district": p.district or "N/A",
        "occupation": o.occupation,
        "employment_type": o.employment_type.value,
        "monthly_income": f"{o.monthly_income_lkr:,.0f}",
        "hazardous_level": o.hazardous_level.value,
        "primary_goal": g.primary_goal.value.replace("_", " ").title(),
        "secondary_goal": g.secondary_goal.value.replace("_", " ").title() if g.secondary_goal else "None",
        "health_summary": build_health_summary(profile),
        "bmi": profile.lifestyle.bmi,
        "is_smoker": "Yes" if profile.lifestyle.is_smoker else "No",
        "is_alcohol": "Yes" if profile.lifestyle.is_alcohol_consumer else "No",
        "has_existing_insurance": "Yes" if o.has_existing_insurance else "No",
        "top_policy_name": top_policy_name,
        "top_score": top_score,
        "shap_explanation": shap_explanation,
        "rag_context": rag_context,
    }

    try:
        chain = RECOMMENDATION_PROMPT | llm | StrOutputParser()
        narrative = await chain.ainvoke(prompt_values)
        return narrative
    except Exception as e:
        logger.error(f"Narrative generation failed: {e}")
        return f"Based on your profile, {top_policy_name} is the most suitable insurance policy for you with a suitability score of {top_score:.0%}."


# ─── Chat Chain ───────────────────────────────────────────────────────────────

async def chat(
    session_id: str,
    user_id: str,
    message: str,
    user_profile: UserProfile = None,
    recommendation_context: str = None,
) -> Tuple[str, List[str]]:
    """Handle a chat message (history persisted in Supabase). Returns (response, sources)."""
    from app.core.db import supabase_client as db

    llm = get_groq_llm(temperature=0.2)

    # Ensure session row exists (upsert)
    try:
        await db.ensure_session(user_id, session_id)
    except Exception as e:
        logger.warning(f"ensure_session failed: {e}")

    # Load prior messages from DB
    try:
        rows = await db.list_messages(session_id, user_id, limit=MAX_HISTORY_TURNS * 2)
    except Exception as e:
        logger.warning(f"list_messages failed: {e}")
        rows = []
    history: List[Tuple[str, str]] = [(r["role"], r["content"]) for r in rows]

    # Build retrieval query (combine user message + profile context if available)
    retrieval_query = message
    if user_profile:
        retrieval_query = f"{message} | User context: {_build_user_query(user_profile)}"

    retrieved_docs = similarity_search(retrieval_query, k=6)
    context = _format_rag_context(retrieved_docs)
    sources = list({d.metadata.get("policy_name", "unknown") for d in retrieved_docs})
    chat_history_str = format_chat_history(history)

    profile_summary = _build_profile_summary(user_profile) if user_profile else "No profile on record."
    rec_summary = recommendation_context or "No prior recommendation results available in this session."

    try:
        chain = CHAT_PROMPT | llm | StrOutputParser()
        response = await chain.ainvoke({
            "user_profile_summary": profile_summary,
            "recommendation_summary": rec_summary,
            "context": context,
            "chat_history": chat_history_str,
            "question": message,
        })
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        response = "I'm sorry, I encountered an error processing your question. Please try again."

    # Persist turn
    try:
        await db.insert_message(user_id, session_id, "human", message)
        await db.insert_message(user_id, session_id, "ai", response, sources=sources)
        await db.touch_session(session_id, user_id)
    except Exception as e:
        logger.warning(f"persist chat turn failed: {e}")

    return response, sources


async def clear_session(session_id: str, user_id: str) -> None:
    from app.core.db import supabase_client as db
    try:
        await db.delete_session(session_id, user_id)
    except Exception as e:
        logger.warning(f"delete_session failed: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_profile_summary(profile: UserProfile) -> str:
    """Readable profile block for the chat prompt."""
    p = profile.personal
    o = profile.occupation
    g = profile.goals
    l = profile.lifestyle
    lines = [
        f"- Age: {p.age} | Gender: {p.gender.value} | Marital status: {p.marital_status.value}",
        f"- Dependents: {p.num_dependents} | Location: {p.city or 'N/A'}, {p.district or 'N/A'}",
        f"- Occupation: {o.occupation} ({o.employment_type.value}) | Monthly income: LKR {o.monthly_income_lkr:,.0f}",
        f"- Hazardous level: {o.hazardous_level.value}",
        f"- Primary goal: {g.primary_goal.value.replace('_', ' ')}",
    ]
    if g.secondary_goal:
        lines.append(f"- Secondary goal: {g.secondary_goal.value.replace('_', ' ')}")
    lines.append(f"- Health: {build_health_summary(profile)}")
    lines.append(f"- BMI: {l.bmi} | Smoker: {'Yes' if l.is_smoker else 'No'} | Alcohol: {'Yes' if l.is_alcohol_consumer else 'No'}")
    lines.append(f"- Has existing insurance: {'Yes' if o.has_existing_insurance else 'No'}")
    return "\n".join(lines)


def _build_user_query(profile: UserProfile) -> str:
    """Convert user profile to a natural language query for semantic retrieval."""
    p = profile.personal
    o = profile.occupation
    g = profile.goals
    h = profile.health

    parts = [
        f"insurance policy for {p.age} year old {p.gender.value}",
        f"goal: {g.primary_goal.value.replace('_', ' ')}",
        f"monthly income LKR {o.monthly_income_lkr:,.0f}",
        f"employment: {o.employment_type.value}",
    ]
    if p.num_dependents > 0:
        parts.append(f"{p.num_dependents} dependents")
    if o.hazardous_level.value not in ("none", "low"):
        parts.append(f"hazardous work: {o.hazardous_level.value}")
    conditions = []
    if h.has_chronic_disease:
        conditions.append("chronic disease")
    if h.has_cardiovascular:
        conditions.append("cardiovascular")
    if h.has_cancer:
        conditions.append("cancer")
    if conditions:
        parts.append(f"health conditions: {', '.join(conditions)}")

    return " | ".join(parts)


def _format_rag_context(docs: List[Document], policy_name: str = None) -> str:
    """Format retrieved documents into readable context for the LLM."""
    if not docs:
        return "No specific policy documents retrieved."

    sections = []
    for doc in docs:
        meta = doc.metadata
        name = meta.get("policy_name", "Unknown Policy")
        section = meta.get("section", "general")
        company = meta.get("company", "")
        header = f"[{name}{' - ' + company if company else ''} | {section}]"
        sections.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


def settings_k():
    from app.config import settings
    return settings.RETRIEVAL_K
