import numpy as np

FEATURE_NAMES = [
    "mean_skill_similarity",
    "jd_skill_coverage",
    "project_similarity",
    "experience_similarity",
    "skills_section_similarity"
]

MATCH_THRESHOLD  = 0.30   
STRONG_THRESHOLD = 0.50  


def build_feature_vector(similarity: dict, section_sims: dict) -> np.ndarray:
    if similarity:
        scores = np.array(list(similarity.values()), dtype=float)

        raw_mean = scores.mean()
        mean_sim = float(np.clip(raw_mean / 0.80, 0.0, 1.0))

        coverage = float((scores >= MATCH_THRESHOLD).mean())

        strong_coverage = float((scores >= STRONG_THRESHOLD).mean())

        jd_coverage = round(0.6 * strong_coverage + 0.4 * coverage, 4)

    else:
        mean_sim    = 0.0
        jd_coverage = 0.0

    proj_sim  = float(section_sims.get("projects",    0.0))
    exp_sim   = float(section_sims.get("experience",  0.0))
    skill_sim = float(section_sims.get("skills",      0.0))

    return np.array([
        mean_sim,
        jd_coverage,
        proj_sim,
        exp_sim,
        skill_sim,
    ], dtype=float)