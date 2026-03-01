import os
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file — works regardless of which terminal runs streamlit
load_dotenv()


def generate_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    if not missing_skills:
        return []
    try:
        return _gemini_roadmap(missing_skills, score, jd_text, resume_text)
    except Exception as e:
        print(f"[roadmap] Gemini failed ({e}), using fallback.")
        return _fallback_roadmap(missing_skills, score, jd_text)


def _gemini_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str,
    resume_text: str
) -> List[Dict]:

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ correct model name

    skills_list = "\n".join(f"- {s.replace('_', ' ')}" for s in missing_skills)

    prompt = f"""You are an expert technical career coach helping a candidate improve their resume.

CANDIDATE RESUME (excerpt):
{resume_text[:800] if resume_text else "Not provided"}

JOB DESCRIPTION:
{jd_text[:1200] if jd_text else "Not provided"}

MISSING SKILLS (required by JD but not found in resume):
{skills_list}

CURRENT MATCH SCORE: {score:.1f}%

For each missing skill, write a specific actionable improvement step that:
1. References the candidate's ACTUAL existing projects/tech stack from their resume
2. Uses context from the ACTUAL job description
3. Gives a concrete implementable action (not generic advice)
4. Explains WHY this skill matters for THIS specific role

Priority rules:
- "high"   = explicitly required in JD + completely missing from resume
- "medium" = mentioned in JD + candidate has partial/related experience
- "low"    = preferred/bonus skill in JD

Respond ONLY with a valid JSON array. No markdown, no extra text, no code fences:
[
  {{
    "skill": "exact_skill_name_from_list",
    "action": "Specific 1-2 sentence action referencing their actual projects and JD context",
    "priority": "high or medium or low",
    "why": "One sentence: why this skill matters for THIS specific role"
  }}
]"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    steps = json.loads(raw)

    validated = []
    for step in steps:
        if all(k in step for k in ("skill", "action", "priority")):
            validated.append({
                "skill":    step["skill"],
                "action":   step["action"],
                "priority": step.get("priority", "medium"),
                "why":      step.get("why", "")
            })

    order = {"high": 0, "medium": 1, "low": 2}
    validated.sort(key=lambda x: order.get(x["priority"], 1))
    return validated


def _fallback_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = ""
) -> List[Dict]:
    roadmap  = []
    jd_lower = jd_text.lower() if jd_text else ""

    for skill in missing_skills:
        skill_clean = skill.replace("_", " ")
        mentions    = jd_lower.count(skill_clean.lower())
        priority    = "high" if mentions >= 2 else "medium" if mentions == 1 else "low"

        jd_context = ""
        for line in jd_text.splitlines():
            if skill_clean.lower() in line.lower():
                jd_context = line.strip()
                break

        action = (
            f"Build a project demonstrating {skill_clean}."
            + (f" The JD mentions: '{jd_context[:100]}'." if jd_context else "")
            + " Include specific tools, metrics, and outcomes in your resume bullet."
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