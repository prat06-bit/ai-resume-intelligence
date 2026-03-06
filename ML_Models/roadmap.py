import os
import json
import re
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"  # free on Groq


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    if not missing_skills:
        return []

    # 1. Try local Ollama first (free, unlimited, fast on your GPU)
    try:
        return _ollama_roadmap(missing_skills, score, jd_text, resume_text)
    except Exception as e:
        print(f"[roadmap] Ollama failed ({e}), trying Groq...")

    # 2. Try Groq free tier
    if GROQ_API_KEY:
        try:
            return _groq_roadmap(missing_skills, score, jd_text, resume_text)
        except Exception as e:
            print(f"[roadmap] Groq failed ({e}), using fallback.")
    else:
        print("[roadmap] GROQ_API_KEY not set, using fallback.")

    # 3. Rule-based fallback (no AI needed)
    return _fallback_roadmap(missing_skills, score, jd_text)


# ── Shared prompt builder ─────────────────────────────────────────────────────

def _build_prompt(missing_skills, score, jd_text, resume_text) -> str:
    skills_list = "\n".join(f"- {s.replace('_', ' ')}" for s in missing_skills)
    return f"""You are an expert technical career coach helping a candidate improve their resume.

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


def _parse_and_validate(raw: str) -> List[Dict]:
    """Strip markdown fences, parse JSON, validate fields, sort by priority."""
    raw = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
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


# ── Ollama (local) ────────────────────────────────────────────────────────────

def _ollama_roadmap(missing_skills, score, jd_text, resume_text) -> List[Dict]:
    """Call local Ollama — runs on your RTX 4050, completely free."""
    prompt = _build_prompt(missing_skills, score, jd_text, resume_text)

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1500,
            }
        },
        timeout=120  # local inference can take a moment
    )
    response.raise_for_status()

    raw = response.json()["message"]["content"]
    return _parse_and_validate(raw)


# ── Groq (free cloud fallback) ────────────────────────────────────────────────

def _groq_roadmap(missing_skills, score, jd_text, resume_text) -> List[Dict]:
    """Call Groq free tier — llama3.1-8b-instant, no credit card needed."""
    prompt = _build_prompt(missing_skills, score, jd_text, resume_text)

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1500,
        },
        timeout=30
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    return _parse_and_validate(raw)


# ── Rule-based fallback (zero dependencies) ───────────────────────────────────

def _fallback_roadmap(missing_skills, score, jd_text="") -> List[Dict]:
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
