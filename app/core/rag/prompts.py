from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ─── System persona ───────────────────────────────────────────────────────────

SYSTEM_PERSONA = """You are an expert insurance advisor for Sri Lanka. You have deep knowledge of life insurance, health insurance, critical illness, endowment, and accident policies offered in Sri Lanka. You give clear, empathetic, and unbiased advice based on the client's personal circumstances.

Always:
- Be concise yet thorough
- Reference specific policy details from the provided context
- Use LKR (Sri Lankan Rupee) for monetary values
- Acknowledge health conditions sensitively
- Respect that Sri Lankan insurance products may have local regulatory nuances"""


# ─── Recommendation narrative prompt ─────────────────────────────────────────

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PERSONA),
    HumanMessagePromptTemplate.from_template("""
Based on the client's profile and the SHAP analysis, generate a clear insurance recommendation narrative.

## Client Profile Summary
- Age: {age} years | Gender: {gender} | Marital Status: {marital_status}
- Dependents: {num_dependents} | Location: {city}, {district}
- Occupation: {occupation} ({employment_type}) | Monthly Income: LKR {monthly_income}
- Hazardous Level: {hazardous_level}
- Primary Goal: {primary_goal} | Secondary Goal: {secondary_goal}
- Health Conditions: {health_summary}
- BMI: {bmi} | Smoker: {is_smoker} | Alcohol: {is_alcohol}
- Existing Insurance: {has_existing_insurance}

## Top Recommended Policy
Policy: {top_policy_name}
Suitability Score: {top_score:.0%}

## Why This Policy Was Recommended (SHAP Analysis)
{shap_explanation}

## Policy Details from Documents
{rag_context}

## Instructions
Write a 3-4 paragraph recommendation explaining:
1. Why this specific policy is the best fit for this client
2. How the key features align with their goals and circumstances
3. Any important considerations or caveats (health loading, exclusions)
4. A brief mention of the runner-up policy options

Keep the tone professional yet approachable. Address the client directly.
""")
])


# ─── Chat prompt ──────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are an expert insurance advisor for Sri Lanka specializing in life, health, critical illness, endowment, and accident insurance policies.

You help clients understand insurance policies, riders, premiums, and claims processes. Always base your answers on the retrieved policy documents provided as context. If you don't have enough information, say so clearly rather than speculating.

When discussing premiums or coverage amounts, use LKR (Sri Lankan Rupee).
Be concise, accurate, and helpful."""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CHAT_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("""
## Relevant Policy Information
{context}

## Conversation History
{chat_history}

## Client Question
{question}

Provide a clear, helpful answer based on the policy documents above.
""")
])


# ─── Policy metadata extraction prompt ───────────────────────────────────────

METADATA_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["document_excerpt", "filename"],
    template="""Extract structured metadata from this insurance policy document.

Filename: {filename}

Document excerpt:
\"\"\"
{document_excerpt}
\"\"\"

Return ONLY a valid JSON object with these exact fields:
{{
  "policy_name": "full name of the policy",
  "policy_type": "one of: term_life, whole_life, endowment, health, critical_illness, accident",
  "company": "insurance company name or null",
  "min_age": 18,
  "max_age": 65,
  "premium_level": 1,
  "covers_health": false,
  "covers_life": false,
  "covers_accident": false,
  "is_entry_level": false,
  "description": "one sentence description"
}}

premium_level: 0=low/affordable, 1=medium, 2=high/comprehensive
Return ONLY JSON, no explanation."""
)
