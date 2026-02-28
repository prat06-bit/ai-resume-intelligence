import json
import re
from typing import Dict, Set, List


def load_skills(path: str = "data/skills.json") -> Dict[str, Set[str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: set(v) for k, v in raw.items()}


def extract_skills(text: str, skills_db: Dict[str, Set[str]]) -> Set[str]:

    text = text.lower()
    found = set()

    for canonical, variants in skills_db.items():
        for v in variants:
            pattern = r'\b' + re.escape(v.lower()) + r'\b'
            if re.search(pattern, text):
                found.add(canonical)
                break

    return found


def extract_skills_by_section(
    resume_sections: Dict[str, str],
    skills_db: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    return {
        section_name: extract_skills(section_text, skills_db)
        for section_name, section_text in resume_sections.items()
    }


def parse_resume_sections(full_text: str) -> Dict[str, str]:
    section_headers = {
        "experience": ["work experience", "experience", "internship"],
        "projects":   ["projects", "project"],
        "skills":     ["technical skills", "skills", "technologies"],
        "education":  ["education", "academic"],
    }

    lines = full_text.split("\n")
    sections: Dict[str, List[str]] = {k: [] for k in section_headers}
    current_section = "other"

    for line in lines:
        line_lower = line.strip().lower()
        matched = False
        for section, keywords in section_headers.items():
            if any(kw in line_lower for kw in keywords):
                current_section = section
                matched = True
                break
        if not matched and current_section in sections:
            sections[current_section].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}


def get_canonical_skill_name(raw_skill: str, skills_db: Dict[str, Set[str]]) -> str:
    raw_lower = raw_skill.lower()
    for canonical, variants in skills_db.items():
        for v in variants:
            pattern = r'\b' + re.escape(v.lower()) + r'\b'
            if re.search(pattern, raw_lower):
                return canonical
    return raw_skill