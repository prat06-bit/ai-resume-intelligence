from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict

from sklearn.metrics.pairwise import cosine_similarity

from backend.core.database import get_history_by_user_id, save_history
from backend.ml.recruiter_intelligence import build_recruiter_report, report_to_markdown
from backend.ml.roadmap import generate_roadmap
from backend.ml.semantic_matcher import SemanticMatcher, compute_score, role_weighted_score
from backend.ml.skill_extractor import (
    extract_skills,
    extract_skills_by_section,
    load_skills,
    parse_resume_sections,
)


@lru_cache(maxsize=1)
def get_matcher() -> SemanticMatcher:
    return SemanticMatcher()


@lru_cache(maxsize=1)
def get_skills_db():
    return load_skills()


def analyze_resume_match(
    *,
    user_id: str,
    email: str,
    role: str,
    resume_text: str,
    jd_text: str,
    persist: bool = True,
    include_roadmap: bool = True,
) -> Dict:
    matcher = get_matcher()
    skills_db = get_skills_db()

    resume_skills = sorted(extract_skills(resume_text, skills_db))
    jd_skills = sorted(extract_skills(jd_text, skills_db))
    resume_sections = parse_resume_sections(resume_text)
    skills_by_section = extract_skills_by_section(resume_sections, skills_db)

    similarity = matcher.match_skills(resume_skills, jd_skills) or {skill: 0.0 for skill in jd_skills}
    section_embeddings = matcher.embed_sections(resume_text)
    jd_emb = matcher.embed([jd_text])[0]

    section_similarities = {
        section: float(cosine_similarity([embedding], [jd_emb])[0][0])
        for section, embedding in section_embeddings.items()
        if embedding is not None
    }

    skill_score, confidence = compute_score(similarity)
    section_score = role_weighted_score(section_similarities, role)
    matched = {skill: score for skill, score in similarity.items() if score >= 0.30}
    missing = {skill: score for skill, score in similarity.items() if score < 0.30}
    matched_count = len(matched)
    total_jd = len(similarity) or 1
    coverage_pct = (matched_count / total_jd) * 100
    final_score = round((skill_score * 0.50) + (section_score * 0.30) + (coverage_pct * 0.20), 1)

    roadmap = []
    if include_roadmap:
        roadmap = generate_roadmap(
            missing_skills=list(missing.keys()),
            score=final_score,
            jd_text=jd_text,
            resume_text=resume_text,
        )

    history = get_history_by_user_id(user_id)
    recruiter_report = build_recruiter_report(
        final_score=final_score,
        confidence=confidence,
        skill_score=skill_score,
        section_score=section_score,
        coverage_pct=coverage_pct,
        similarity=similarity,
        matched=matched,
        missing=missing,
        skills_by_section=skills_by_section,
        section_similarities=section_similarities,
        roadmap=roadmap,
        history=history,
    )

    resume_hash = hashlib.sha256(resume_text.encode("utf-8")).hexdigest()
    jd_hash = hashlib.sha256(jd_text.encode("utf-8")).hexdigest()
    if persist:
        save_history(
            email=email,
            resume_hash=resume_hash,
            jd_hash=jd_hash,
            score=final_score,
            matched_skills=list(matched.keys()),
            missing_skills=list(missing.keys()),
            explanation=recruiter_report["score"]["explanation"],
            user_id=user_id,
        )

    return {
        "score": final_score,
        "confidence": confidence,
        "skill_score": skill_score,
        "section_score": section_score,
        "coverage_pct": coverage_pct,
        "matched": matched,
        "missing": missing,
        "matched_count": matched_count,
        "total_jd_skills": total_jd,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "section_similarities": section_similarities,
        "roadmap": roadmap,
        "recruiter_report": recruiter_report,
        "recruiter_markdown": report_to_markdown(recruiter_report),
        "resume_hash": resume_hash,
        "jd_hash": jd_hash,
        "embedding_backend": "fallback" if matcher.using_fallback else "sentence_transformer",
    }
