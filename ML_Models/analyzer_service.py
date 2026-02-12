from ML_Models.skill_extractor import extract_skills, load_skills
from ML_Models.semantic_matcher import semantic_similarity
from ML_Models.scoring import (
    skill_coverage_score,
    experience_signal,
    final_match_score,
    potential_score
)

def analyze_resume(resume_text: str, jd_text: str):
    skills_db = load_skills()

    resume_skills = extract_skills(resume_text, skills_db)
    jd_skills = extract_skills(jd_text, skills_db)

    semantic = semantic_similarity(resume_text, jd_text)
    skill_cov = skill_coverage_score(resume_skills, jd_skills)
    experience = experience_signal(resume_text)

    final_score = final_match_score(semantic, skill_cov, experience)

    return {
        "final_score": round(final_score, 2),  # ✅ FIXED
        "semantic_score": semantic,
        "skill_coverage": skill_cov,
        "experience_signal": experience,
        "potential_score": potential_score(resume_skills, jd_skills),
        "matched_skills": sorted(list(resume_skills & jd_skills)),
        "missing_skills": sorted(list(jd_skills - resume_skills)),
        "explanation": (
            f"Semantic relevance contributes {semantic}%, "
            f"skill coverage contributes {skill_cov}%, "
            f"experience contributes {experience} points."
        )
    }
