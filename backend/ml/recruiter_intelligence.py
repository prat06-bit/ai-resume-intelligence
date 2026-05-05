from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Set


SECTION_LABELS = {
    "projects": "Projects",
    "experience": "Experience",
    "skills": "Skills",
    "education": "Education",
}


def _title_skill(skill: str) -> str:
    return skill.replace("_", " ").title()


def _score_band(score: float) -> str:
    if score >= 80:
        return "strong"
    if score >= 65:
        return "competitive"
    if score >= 45:
        return "partial"
    return "weak"


def _section_evidence(
    skills_by_section: Dict[str, Set[str]],
    skills: Iterable[str],
) -> Dict[str, List[str]]:
    skill_set = set(skills)
    return {
        section: sorted(skill_set & set(section_skills))
        for section, section_skills in skills_by_section.items()
        if skill_set & set(section_skills)
    }


def _coverage_reason(skill: str, score: float, section_hits: Dict[str, List[str]]) -> str:
    label = _title_skill(skill)
    if score < 0.18:
        return f"{label} is explicitly needed by the JD but is not evidenced in resume skills, projects, or experience."
    if score < 0.30:
        return f"{label} has only adjacent semantic evidence; the resume does not show direct use in a delivered project or role."
    if "skills" in section_hits and len(section_hits) == 1:
        return f"{label} appears mainly in the skills list, which is weaker recruiter evidence than project or work usage."
    return f"{label} is present but could be stronger with impact, scale, or implementation detail."


def build_recruiter_report(
    *,
    final_score: float,
    confidence: str,
    skill_score: float,
    section_score: float,
    coverage_pct: float,
    similarity: Dict[str, float],
    matched: Dict[str, float],
    missing: Dict[str, float],
    skills_by_section: Dict[str, Set[str]],
    section_similarities: Dict[str, float],
    roadmap: List[Dict],
    history: List[Dict] | None = None,
) -> Dict:
    """Create a recruiter-style, section-aware explanation from analysis signals."""
    history = history or []
    section_hits = _section_evidence(skills_by_section, matched.keys())

    strongest_sections = sorted(
        section_similarities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    strong_skill_rows = sorted(matched.items(), key=lambda item: item[1], reverse=True)[:6]
    weak_skill_rows = sorted(missing.items(), key=lambda item: item[1])[:8]

    previous = history[0] if history else None
    score_delta = None
    if previous and previous.get("score") is not None:
        try:
            score_delta = round(final_score - float(previous["score"]), 1)
        except (TypeError, ValueError):
            score_delta = None

    missing_counter = Counter()
    for item in history[:5]:
        missing_counter.update(item.get("missing", []))
    repeated_gaps = [
        skill for skill, count in missing_counter.most_common()
        if skill in missing and count >= 2
    ][:5]

    strengths = []
    for skill, score in strong_skill_rows:
        evidence_sections = [
            SECTION_LABELS.get(section, section.title())
            for section, skills in section_hits.items()
            if skill in skills
        ]
        evidence = ", ".join(evidence_sections) if evidence_sections else "semantic resume evidence"
        strengths.append({
            "skill": skill,
            "evidence": f"{_title_skill(skill)} aligns at {score * 100:.1f}% and is supported by {evidence}.",
        })

    if strongest_sections:
        section, sim = strongest_sections[0]
        strengths.append({
            "skill": section,
            "evidence": f"{SECTION_LABELS.get(section, section.title())} is the strongest resume section against the JD at {sim * 100:.1f}% raw semantic similarity.",
        })

    gaps = []
    for skill, score in weak_skill_rows:
        gaps.append({
            "skill": skill,
            "coverage": round(score * 100, 1),
            "why_missing": _coverage_reason(skill, score, _section_evidence(skills_by_section, [skill])),
            "repeated": skill in repeated_gaps,
        })

    improvements = []
    for step in roadmap[:6]:
        skill = step.get("skill", "")
        action = step.get("action", "")
        why = step.get("why", "")
        priority = step.get("priority", "medium")
        improvements.append({
            "priority": priority,
            "skill": skill,
            "action": action,
            "why": why,
        })

    if not improvements and missing:
        for skill, score in weak_skill_rows[:4]:
            improvements.append({
                "priority": "high" if score < 0.20 else "medium",
                "skill": skill,
                "action": f"Add one bullet showing hands-on {_title_skill(skill)} usage, preferably in a project or work-experience section with a measurable outcome.",
                "why": f"The JD expects {_title_skill(skill)}, but the current resume evidence is weak.",
            })

    rewrite_suggestions = []
    for gap in gaps[:4]:
        label = _title_skill(gap["skill"])
        rewrite_suggestions.append({
            "section": "Projects or Experience",
            "suggestion": f"Add a bullet like: Built/optimized [system or feature] using {label}, improving [latency, accuracy, cost, reliability, or user outcome] by [metric].",
        })

    if coverage_pct < 50:
        rewrite_suggestions.append({
            "section": "Skills",
            "suggestion": "Group technical skills by category and mirror the JD's core stack only where you can back it up elsewhere in the resume.",
        })

    explanation = (
        f"The {final_score:.1f}% score is {_score_band(final_score)} because skill similarity contributes "
        f"{skill_score:.1f}%, section-aware alignment contributes {section_score:.1f}%, and JD skill coverage is "
        f"{coverage_pct:.1f}%. Recruiter confidence is {confidence.lower()}."
    )
    if score_delta is not None:
        direction = "up" if score_delta > 0 else "down" if score_delta < 0 else "unchanged"
        explanation += f" Compared with the latest stored analysis, the score is {direction} by {abs(score_delta):.1f} points."
    if repeated_gaps:
        repeated = ", ".join(_title_skill(skill) for skill in repeated_gaps)
        explanation += f" Repeated gaps to address first: {repeated}."

    return {
        "score": {
            "value": final_score,
            "confidence": confidence,
            "explanation": explanation,
            "components": {
                "skill_similarity": skill_score,
                "section_alignment": section_score,
                "jd_skill_coverage": coverage_pct,
            },
        },
        "key_strengths": strengths[:7],
        "skill_gaps": gaps,
        "actionable_improvements": improvements,
        "resume_rewrite_suggestions": rewrite_suggestions[:6],
        "history_summary": {
            "analyses_found": len(history),
            "score_delta": score_delta,
            "repeated_gaps": repeated_gaps,
        },
    }


def report_to_markdown(report: Dict) -> str:
    """Format report in the mandated user-facing structure."""
    lines = [
        "## Match Score",
        f"{report['score']['value']:.1f}% ({report['score']['confidence']} confidence)",
        report["score"]["explanation"],
        "",
        "## Key Strengths",
    ]
    strengths = report.get("key_strengths") or []
    lines.extend(
        f"- {item['evidence']}" for item in strengths
    )
    if not strengths:
        lines.append("- No strong recruiter-grade evidence was detected yet.")

    lines.extend(["", "## Skill Gaps"])
    gaps = report.get("skill_gaps") or []
    for gap in gaps:
        repeat = " Repeated from prior analyses." if gap.get("repeated") else ""
        lines.append(f"- {_title_skill(gap['skill'])}: {gap['coverage']:.1f}% coverage. {gap['why_missing']}{repeat}")
    if not gaps:
        lines.append("- No major JD skill gaps were detected at the current threshold.")

    lines.extend(["", "## Actionable Improvements"])
    improvements = report.get("actionable_improvements") or []
    for item in improvements:
        lines.append(f"- [{item['priority'].title()}] {_title_skill(item['skill'])}: {item['action']}")
    if not improvements:
        lines.append("- Keep strengthening measurable outcomes in the most relevant projects and work bullets.")

    lines.extend(["", "## Resume Rewrite Suggestions"])
    suggestions = report.get("resume_rewrite_suggestions") or []
    for item in suggestions:
        lines.append(f"- {item['section']}: {item['suggestion']}")
    if not suggestions:
        lines.append("- No rewrite suggestions are needed beyond tightening impact metrics.")

    return "\n".join(lines)
