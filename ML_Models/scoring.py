import numpy as np

def compute_score(similarity: dict) -> tuple:
    if not similarity:
        return 0.0, 0.0

    scores = np.array(list(similarity.values()))
    mean_score = scores.mean() * 100
    coverage = (scores >= 0.55).mean() * 100

    return round(mean_score, 2), round(coverage, 2)

def role_weighted_score(section_sims: dict, role: str) -> float:
    weights = {
        "ml_engineer": {"projects": 0.5, "experience": 0.4, "skills": 0.1},
        "data_scientist": {"projects": 0.45, "experience": 0.35, "skills": 0.2},
        "software_engineer": {"experience": 0.5, "projects": 0.4, "skills": 0.1},
        "student": {"projects": 0.6, "skills": 0.2, "experience": 0.2},
    }

    w = weights.get(role, weights["ml_engineer"])
    score = 0.0

    for sec, weight in w.items():
        if section_sims.get(sec) is not None:
            score += section_sims[sec] * weight

    return round(score * 100, 2)
