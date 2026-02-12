import numpy as np

FEATURE_NAMES = [
    "mean_skill_similarity",
    "jd_skill_coverage",
    "project_similarity",
    "experience_similarity",
    "skills_section_similarity"
]

def build_feature_vector(similarity: dict, section_sims: dict):
    if similarity:
        scores = np.array(list(similarity.values()))
        mean_sim = scores.mean()
        coverage = (scores >= 0.55).mean()
    else:
        mean_sim = 0.0
        coverage = 0.0

    return np.array([
        mean_sim,
        coverage,
        section_sims.get("projects", 0.0),
        section_sims.get("experience", 0.0),
        section_sims.get("skills", 0.0),
    ])
