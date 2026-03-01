import os
import json
import re
from typing import List, Dict
import anthropic


def generate_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    """
    Dynamically generate a personalized roadmap using Claude Haiku.
    Fully context-aware: uses actual JD + resume to tailor every step.
    Cost: ~$0.0005 per call (Claude Haiku, cheapest Anthropic model).
    """
    if not missing_skills:
        return []

    try:
        return _llm_roadmap(missing_skills, score, jd_text, resume_text)
    except Exception as e:
        print(f"[roadmap] LLM failed ({e}), using fallback.")
        return _fallback_roadmap(missing_skills, score, jd_text)


def _llm_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str,
    resume_text: str
) -> List[Dict]:
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    skills_list = "\n".join(f"- {s.replace('_', ' ')}" for s in missing_skills)

    prompt = f"""You are an expert technical career coach helping a candidate improve their resume.

CANDIDATE RESUME (excerpt):
{resume_text[:800] if resume_text else "Not provided"}

JOB DESCRIPTION:
{jd_text[:1200] if jd_text else "Not provided"}

MISSING SKILLS (skills required by JD but not found in resume):
{skills_list}

CURRENT MATCH SCORE: {score:.1f}%

For each missing skill above, write a specific, actionable improvement step that:
1. References the candidate's ACTUAL existing projects/tech stack from their resume
2. Uses context from the ACTUAL job description requirements
3. Gives a concrete, implementable action (not generic advice)
4. Explains WHY this skill matters for THIS specific role

Priority rules:
- "high"   = explicitly required in JD + completely missing from resume
- "medium" = mentioned in JD + candidate has partial/related experience
- "low"    = preferred/bonus skill in JD

Respond ONLY with a valid JSON array. No markdown, no extra text:
[
  {{
    "skill": "exact_skill_name_from_list",
    "action": "Specific 1-2 sentence action referencing their actual projects and JD context",
    "priority": "high|medium|low",
    "why": "One sentence: why this skill matters for THIS specific role"
  }}
]"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Cheapest model ~$0.25/M tokens
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    steps = json.loads(raw)

    # Validate and clean each step
    validated = []
    for step in steps:
        if all(k in step for k in ("skill", "action", "priority")):
            validated.append({
                "skill":    step["skill"],
                "action":   step["action"],
                "priority": step.get("priority", "medium"),
                "why":      step.get("why", "")
            })

    # Sort: high → medium → low
    order = {"high": 0, "medium": 1, "low": 2}
    validated.sort(key=lambda x: order.get(x["priority"], 1))

    return validated


def _fallback_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = ""
) -> List[Dict]:
    """Simple fallback if API unavailable — uses JD context extraction."""
    roadmap = []
    jd_lower = jd_text.lower() if jd_text else ""

    for skill in missing_skills:
        skill_clean = skill.replace("_", " ")
        mentions    = jd_lower.count(skill_clean.lower())
        priority    = "high" if mentions >= 2 else "medium" if mentions == 1 else "low"

        # Extract JD sentence mentioning this skill
        jd_context = ""
        for line in jd_text.splitlines():
            if skill_clean.lower() in line.lower():
                jd_context = line.strip()
                break

        action = (
            f"Add a project demonstrating {skill_clean}. "
            + (f"The JD mentions: '{jd_context[:100]}'. " if jd_context else "")
            + "Include specific tools, metrics, and outcomes in your resume bullet."
        )

        roadmap.append({
            "skill":    skill,
            "action":   action,
            "priority": priority,
            "why":      f"{skill_clean.title()} is required for this role but not clearly shown in your resume."
        })

    order = {"high": 0, "medium": 1, "low": 2}
    roadmap.sort(key=lambda x: order.get(x["priority"], 1))
    return roadmap