from app.models.schemas import UserProfile


def profile_to_text_summary(profile: UserProfile) -> str:
    """Convert a UserProfile to a human-readable text summary."""
    p = profile.personal
    o = profile.occupation
    g = profile.goals
    h = profile.health
    ls = profile.lifestyle

    health_conditions = []
    if h.has_chronic_disease:
        health_conditions.append("Chronic Disease")
    if h.has_cardiovascular:
        health_conditions.append("Cardiovascular")
    if h.has_cancer:
        health_conditions.append("Cancer/Tumor")
    if h.has_respiratory:
        health_conditions.append("Respiratory")
    if h.has_neurological:
        health_conditions.append("Neurological")
    if h.has_gastrointestinal:
        health_conditions.append("Gastrointestinal")
    if h.has_musculoskeletal:
        health_conditions.append("Musculoskeletal")
    if h.recent_treatment_surgery:
        health_conditions.append("Recent Surgery/Treatment")

    lines = [
        f"Age: {p.age} | Gender: {p.gender.value} | Marital: {p.marital_status.value}",
        f"Location: {p.city or 'N/A'}, {p.district or 'N/A'}, {p.country}",
        f"Dependents: {p.num_dependents}",
        f"Occupation: {o.occupation} ({o.employment_type.value}) | Income: LKR {o.monthly_income_lkr:,.0f}/month",
        f"Hazardous Level: {o.hazardous_level.value}",
        f"Existing Insurance: {'Yes' if o.has_existing_insurance else 'No'}",
        f"Primary Goal: {g.primary_goal.value.replace('_', ' ').title()}",
        f"Health Conditions: {', '.join(health_conditions) if health_conditions else 'None'}",
        f"BMI: {ls.bmi} | Smoker: {'Yes' if ls.is_smoker else 'No'} | Alcohol: {'Yes' if ls.is_alcohol_consumer else 'No'}",
    ]
    return "\n".join(lines)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"
