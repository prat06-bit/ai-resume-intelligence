import os
import json
import re
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")


#  Main entry point

def generate_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    if not missing_skills:
        return []

    if NVIDIA_API_KEY:
        try:
            result = _nvidia_roadmap(missing_skills, score, jd_text, resume_text)
            if result:
                print(f"[roadmap] NVIDIA ({NVIDIA_MODEL}) generated {len(result)} steps.")
                return result
        except Exception as e:
            print(f"[roadmap] NVIDIA error: {e}")
    else:
        print("[roadmap] NVIDIA_API_KEY not configured.")

    print("[roadmap]  Using rule-based fallback.")
    return _fallback_roadmap(missing_skills, score, jd_text)


#  Prompt builder

def _build_prompt(missing_skills, score, jd_text, resume_text) -> str:
    skills_list = "\n".join(f"- {s.replace('_', ' ')}" for s in missing_skills)
    return f"""You are an expert technical career coach. Create a personalized improvement plan.

CANDIDATE RESUME (excerpt):
{resume_text[:800] if resume_text else "Not provided"}

JOB DESCRIPTION:
{jd_text[:1200] if jd_text else "Not provided"}

MISSING SKILLS (required by JD but not found in resume):
{skills_list}

CURRENT MATCH SCORE: {score:.1f}%

Instructions:
- For EACH skill in the list, create one actionable step.
- Reference the candidate's actual resume tech stack when possible.
- Use context from the actual job description.
- Be specific and concrete, not generic.

Priority rules:
- "high"   = explicitly required in JD, completely absent from resume
- "medium" = mentioned in JD, candidate has partial/adjacent experience
- "low"    = nice-to-have or bonus skill in JD

Respond ONLY with a raw JSON array. No explanation, no markdown, no code fences.
Start with [ and end with ].

[
  {{
    "skill": "exact_skill_name_from_the_list_above",
    "action": "Specific 1-2 sentence action referencing their actual projects and JD context",
    "priority": "high or medium or low",
    "why": "One sentence: why this skill matters for this specific role"
  }}
]"""


#  JSON extractor

def _extract_json(raw: str) -> Optional[List]:
    # Strategy 1: strip markdown fences, parse directly
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first [ ... ] block
    match = re.search(r"(\[.*\])", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: fix trailing commas then parse
    try:
        fixed = re.sub(r",\s*([\]}])", r"\1", cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


def _parse_and_validate(raw: str, missing_skills: List[str]) -> List[Dict]:
    steps = _extract_json(raw)
    if not steps or not isinstance(steps, list):
        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:300]}")

    validated = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if not all(k in step for k in ("skill", "action", "priority")):
            continue
        priority = step.get("priority", "medium").lower()
        if priority not in ("high", "medium", "low"):
            priority = "medium"
        validated.append({
            "skill":    step["skill"],
            "action":   step["action"].strip(),
            "priority": priority,
            "why":      step.get("why", "").strip()
        })

    order = {"high": 0, "medium": 1, "low": 2}
    validated.sort(key=lambda x: order.get(x["priority"], 1))
    return validated


#  NVIDIA

def _nvidia_roadmap(missing_skills, score, jd_text, resume_text) -> List[Dict]:
    prompt = _build_prompt(missing_skills, score, jd_text, resume_text)

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a JSON-only API. Output ONLY a valid JSON array. "
                    "Never include explanations, markdown, or code fences. "
                    "Start with [ and end with ]."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": False,
    }

    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    return _parse_and_validate(raw, missing_skills)


#  Rule-based fallback

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
                jd_context = line.strip()[:120]
                break

        action = (
            f"Build a project demonstrating {skill_clean}."
            + (f" The JD mentions: '{jd_context}'." if jd_context else "")
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
