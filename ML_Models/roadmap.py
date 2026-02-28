import os
import json
import re
import anthropic
from typing import List, Dict


def _generate_steps_via_llm(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    client = anthropic.Anthropic()

    skills_str = ", ".join(missing_skills)
    
    prompt = f"""You are an expert technical career coach.

A candidate has the following skill gaps based on a job description analysis:
Missing skills: {skills_str}

Match score: {score:.1f}%

Job Description (excerpt):
{jd_text[:1500] if jd_text else "Not provided"}

Candidate Resume (excerpt):
{resume_text[:1000] if resume_text else "Not provided"}

For each missing skill, generate a specific, actionable improvement step.
The advice must be:
- Tailored to THIS specific JD and candidate background
- Concrete (mention specific tools, project ideas, or resources)
- Honest about the gap severity

Respond ONLY with a valid JSON array, no markdown, no preamble:
[
  {{
    "skill": "skill_name",
    "action": "Specific actionable advice in 1-2 sentences",
    "priority": "high" or "medium" or "low",
    "why": "One sentence explaining why this matters for the role"
  }},
  ...
]

Priority rules:
- "high"   → core requirement in JD, candidate has no evidence of it
- "medium" → mentioned in JD, candidate has partial/indirect experience  
- "low"    → preferred/bonus skill in JD
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    return json.loads(raw)

def generate_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    if not missing_skills:
        return []

    try:
        steps = _generate_steps_via_llm(missing_skills, score, jd_text, resume_text)

        validated = []
        for step in steps:
            if all(k in step for k in ("skill", "action", "priority")):
                validated.append({
                    "skill":    step["skill"],
                    "action":   step["action"],
                    "priority": step.get("priority", "medium"),
                    "why":      step.get("why", "")
                })

        return validated

    except Exception as e:
        print(f"[roadmap] LLM generation failed: {e}. Using fallback.")
        return _fallback_roadmap(missing_skills, score)


def _fallback_roadmap(missing_skills: List[str], score: float) -> List[Dict]:
    roadmap = []
    for i, skill in enumerate(missing_skills):
        if score < 40:
            priority = "high"
        elif score < 65:
            priority = "high" if i < len(missing_skills) // 2 else "medium"
        else:
            priority = "medium"

        roadmap.append({
            "skill":    skill,
            "action":   (
                f"Build a project that concretely demonstrates {skill}. "
                f"Add it to your resume with measurable outcomes and specific tools used."
            ),
            "priority": priority,
            "why":      f"{skill} is required for this role but is not clearly demonstrated in your resume."
        })

    return roadmap