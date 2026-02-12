import json
from typing import Dict, Set

def load_skills(path: str = "data/skills.json") -> Dict[str, Set[str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}

def extract_skills(text: str, skills_db: Dict[str, Set[str]]) -> Set[str]:
    text = text.lower()
    found = set()

    for canonical, variants in skills_db.items():
        for v in variants:
            if v.lower() in text:
                found.add(canonical)
                break

    return found
