from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class MaritalStatus(str, Enum):
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"


class EmploymentType(str, Enum):
    PERMANENT = "permanent"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"


class HazardousLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InsuranceStatus(str, Enum):
    NONE = "none"
    HAS_INSURANCE = "has_insurance"


class InsuranceGoal(str, Enum):
    CHEAP_AND_QUICK = "cheap_and_quick"
    PROTECTION = "protection"
    SAVINGS_AND_INVESTMENT = "savings_and_investment"
    HEALTH_COVERAGE = "health_coverage"
    RETIREMENT = "retirement"
    NONE = "none"


# ─── User Profile ─────────────────────────────────────────────────────────────

class PersonalInfo(BaseModel):
    age: int = Field(..., ge=18, le=70, description="Age nearest birthday")
    gender: Gender
    marital_status: MaritalStatus
    nationality: str = "Sri Lankan"
    country: str = "Sri Lanka"
    district: Optional[str] = None
    city: Optional[str] = None
    num_dependents: int = Field(default=0, ge=0, le=20)


class OccupationInfo(BaseModel):
    occupation: str
    employment_type: EmploymentType
    designation: Optional[str] = None
    hazardous_level: HazardousLevel = HazardousLevel.NONE
    hazardous_activities: Optional[str] = None
    monthly_income_lkr: float = Field(..., gt=0, description="Monthly income in LKR")
    has_existing_insurance: bool = False
    current_insurance_status: InsuranceStatus = InsuranceStatus.NONE
    employer_insurance_scheme: Optional[str] = None


class InsuranceGoalInfo(BaseModel):
    primary_goal: InsuranceGoal
    secondary_goal: Optional[InsuranceGoal] = InsuranceGoal.NONE
    travel_history_high_risk: bool = False
    dual_citizenship: bool = False
    tax_regulatory_flags: bool = False
    insurance_history_issues: bool = False


class HealthInfo(BaseModel):
    has_chronic_disease: bool = False
    has_cardiovascular: bool = False
    has_cancer: bool = False
    has_respiratory: bool = False
    has_neurological: bool = False
    has_gastrointestinal: bool = False
    has_musculoskeletal: bool = False
    has_infectious_sexual: bool = False
    recent_treatment_surgery: bool = False
    covid_related: bool = False


class LifestyleInfo(BaseModel):
    bmi: float = Field(..., gt=0, le=60, description="Body Mass Index")
    is_smoker: bool = False
    is_alcohol_consumer: bool = False


class UserProfile(BaseModel):
    personal: PersonalInfo
    occupation: OccupationInfo
    goals: InsuranceGoalInfo
    health: HealthInfo
    lifestyle: LifestyleInfo


# ─── Policy Metadata (stored alongside ChromaDB chunks) ───────────────────────

class PolicyMetadata(BaseModel):
    policy_name: str
    policy_type: str  # term_life | whole_life | endowment | health | critical_illness | accident
    company: Optional[str] = None
    min_age: int = 18
    max_age: int = 65
    premium_level: int = Field(default=1, ge=0, le=2)  # 0=low, 1=medium, 2=high
    covers_health: bool = False
    covers_life: bool = False
    covers_accident: bool = False
    is_entry_level: bool = False
    description: Optional[str] = None


# ─── Rider Metadata (stored alongside ChromaDB rider chunks) ──────────────────

class RiderMetadata(BaseModel):
    rider_name: str
    rider_code: str  # stable ID
    category: str  # critical_illness | accidental_death | waiver_of_premium | hospital_cash | income_protection | permanent_disability | term_extension | other
    company: Optional[str] = None
    description: Optional[str] = None
    min_age: int = 18
    max_age: int = 65
    premium_level: int = Field(default=1, ge=0, le=2)
    applicable_policies: List[str] = Field(default_factory=list)  # policy_names this rider can attach to
    target_goals: List[str] = Field(default_factory=list)  # e.g. ["health_coverage","protection"]
    health_relevant: bool = False
    hazard_relevant: bool = False
    dependents_relevant: bool = False


# ─── Request / Response Models ────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    top_k: int = Field(default=3, ge=1, le=5)


class SHAPFactor(BaseModel):
    feature: str
    impact_score: float
    direction: str  # "positive" | "negative"
    reason: str


class PolicyScore(BaseModel):
    policy_name: str
    policy_type: str
    company: Optional[str]
    suitability_score: float
    rank: int


class PolicyExplanation(BaseModel):
    policy_name: str
    suitability_score: float
    positive_factors: List[SHAPFactor]
    negative_factors: List[SHAPFactor]
    shap_summary: str


class RiderRecommendation(BaseModel):
    rider_name: str
    rider_code: str
    category: str
    description: Optional[str] = None
    premium_level: int = 1
    score: float  # 0..1 gap-closer score
    reasons: List[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    ranked_policies: List[PolicyScore]
    top_recommendation: str
    explanations: List[PolicyExplanation]
    rag_narrative: str
    session_id: str
    # policy_name → ranked rider suggestions (best fit first)
    rider_suggestions: Dict[str, List[RiderRecommendation]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_profile: Optional[UserProfile] = None
    recommendation_context: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[str] = []


class IngestResponse(BaseModel):
    message: str
    policy_name: str
    chunks_indexed: int
    policy_metadata: Dict[str, Any]


class RiderIngestResponse(BaseModel):
    message: str
    riders_extracted: int
    chunks_indexed: int
    riders: List[RiderMetadata]


class PolicyListItem(BaseModel):
    policy_name: str
    policy_type: str
    company: Optional[str]
    source_file: str
    chunk_count: int


class ExplainRequest(BaseModel):
    user_profile: UserProfile
    policy_name: str
