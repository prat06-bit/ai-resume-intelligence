import numpy as np

def compute_score(similarity: dict) -> tuple:
    if not similarity:
        return 0.0, 0.0

    scores = list(similarity.values())
    mean = np.mean(scores) * 100
    confidence = np.std(scores) * 100

    return round(mean, 2), round(confidence, 2)


def role_weighted_score(section_sims: dict, role: str) -> float:
    weights = {
        "software_engineer": {"experience": 0.5, "projects": 0.4, "skills": 0.1},
        "ml_engineer": {"experience": 0.4, "projects": 0.5, "skills": 0.1},
        "student": {"experience": 0.2, "projects": 0.6, "skills": 0.2},
    }

    w = weights[role]
    score = 0.0

    for k in w:
        if section_sims.get(k) is not None:
            score += section_sims[k] * w[k]

    return round(score * 100, 2)
