from typing import Dict, Set, List, Tuple

try:
    from .skill_extractor import extract_skills, extract_skills_by_section
except ImportError:
    from ML_Models.skill_extractor import extract_skills, extract_skills_by_section

ROLE_WEIGHTS = {
    "software_engineer":   {"projects": 0.4, "experience": 0.35, "skills": 0.15, "education": 0.1},
    "backend_engineer":    {"projects": 0.35, "experience": 0.4, "skills": 0.15, "education": 0.1},
    "frontend_engineer":   {"projects": 0.45, "experience": 0.3, "skills": 0.15, "education": 0.1},
    "fullstack_engineer":  {"projects": 0.4, "experience": 0.35, "skills": 0.15, "education": 0.1},
    "ml_engineer":         {"projects": 0.35, "experience": 0.35, "skills": 0.2,  "education": 0.1},
    "data_scientist":      {"projects": 0.3,  "experience": 0.35, "skills": 0.2,  "education": 0.15},
    "data_analyst":        {"projects": 0.3,  "experience": 0.35, "skills": 0.2,  "education": 0.15},
    "devops_engineer":     {"projects": 0.3,  "experience": 0.45, "skills": 0.15, "education": 0.1},
    "cloud_engineer":      {"projects": 0.3,  "experience": 0.45, "skills": 0.15, "education": 0.1},
    "student":             {"projects": 0.5,  "experience": 0.15, "skills": 0.2,  "education": 0.15},
    "intern":              {"projects": 0.5,  "experience": 0.15, "skills": 0.2,  "education": 0.15},
}

DEFAULT_ROLE_WEIGHTS = {"projects": 0.4, "experience": 0.35, "skills": 0.15, "education": 0.1}


def compute_score(similarity: Dict[str, float]) -> Tuple[float, str]:
    if not similarity:
        return 0.0, "Low"

    values = list(similarity.values())

    strong   = [v for v in values if v >= 0.55]
    weak     = [v for v in values if v < 0.55]

    strong_avg = sum(strong) / len(strong) if strong else 0.0
    weak_avg   = sum(weak)   / len(weak)   if weak   else 0.0
    coverage   = len(strong) / len(values)

    raw_score = (strong_avg * 0.5) + (coverage * 0.35) + (weak_avg * 0.15)
    score_pct = round(raw_score * 100, 2)

    if coverage >= 0.75:
        confidence = "High"
    elif coverage >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    return score_pct, confidence


def role_weighted_score(section_similarities: Dict[str, float], role: str) -> float:
    weights = ROLE_WEIGHTS.get(role, DEFAULT_ROLE_WEIGHTS)
    SIM_LOW  = 0.15   
    SIM_HIGH = 0.72  
    total_weight = 0.0
    weighted_sum = 0.0

    for section, sim in section_similarities.items():
        w = weights.get(section, 0.1)
        rescaled = max(0.0, min(1.0, (sim - SIM_LOW) / (SIM_HIGH - SIM_LOW)))
        weighted_sum += rescaled * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    return round((weighted_sum / total_weight) * 100, 2)


def coverage_score(resume_skills: Set[str], jd_required_skills: Set[str]) -> float:
    if not jd_required_skills:
        return 1.0
    matched = resume_skills & jd_required_skills
    return round(len(matched) / len(jd_required_skills), 4)


def get_skill_gaps(resume_skills: Set[str], jd_required_skills: Set[str]) -> List[str]:
    return sorted(list(jd_required_skills - resume_skills))

def get_matched_skills(resume_skills: Set[str], jd_required_skills: Set[str]) -> List[str]:
    return sorted(list(resume_skills & jd_required_skills))
