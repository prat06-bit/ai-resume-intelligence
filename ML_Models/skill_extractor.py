import json
import re
from typing import Dict, Set

# ---------------- LOAD SKILL ONTOLOGY ----------------
def load_skills(path: str = "data/skills.json") -> Dict[str, Set[str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text: str, skills_db: Dict[str, Set[str]]) -> Set[str]:
    text = text.lower()
    found = set()

    for variants in skills_db.values():
        for skill in variants:
            if re.search(rf"\b{re.escape(skill.lower())}\b", text):
                found.add(skill)

    return found
