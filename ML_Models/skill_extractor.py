import re
import json
import os
from typing import Dict, Set, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_skills(path: str = "data/skills.json") -> Dict[str, Set[str]]:
    """Load skills database from JSON. Handles multiple path possibilities."""
    possible_paths = [
        path,
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(__file__), "../data/skills.json"),
        "skills.json",
        "../data/skills.json"
    ]

    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    raw = json.load(f)
                    return {k: set(v) for k, v in raw.items()}
            except:
                continue

    searched = ", ".join(possible_paths)
    raise FileNotFoundError(f"skills database not found. Checked: {searched}")


def extract_skills(text: str, skills_db: Dict[str, Set[str]]) -> Set[str]:
    """
    Extract skills using HYBRID approach:
    1. Regex-based exact matching (fast, reliable)
    2. Semantic fallback for contextual terms (catches "data preprocessing" â†’ numpy)
    """
    if not text or len(text) < 10:
        return set()

    text_lower = text.lower()
    found = set()

    #  PHASE 1: Exact regex matching
    for canonical, variants in skills_db.items():
        for v in variants:
            pattern = r'\b' + re.escape(v.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.add(canonical)
                break

    #  PHASE 2: Semantic matching for missed contextual terms
    # Extract common technical terms from text
    semantic_terms = _extract_technical_terms(text_lower)

    if semantic_terms and found:  # Only use semantic if regex already found some skills
        found.update(_semantic_skill_match(semantic_terms, skills_db))

    return found


def _extract_technical_terms(text: str) -> List[str]:
    """Extract technical terms from text (words after colons, commas, or standalone)."""
    # Remove common non-technical words
    stop_words = {
        "experience", "project", "skill", "worked", "built", "developed",
        "designed", "implemented", "created", "used", "the", "and", "or",
        "with", "for", "in", "on", "at", "to", "from", "by", "as"
    }

    # Extract compound technical phrases
    patterns = [
        r'data\s+\w+',  # "data preprocessing", "data analysis"
        r'\w+\s+learning',  # "machine learning", "deep learning"
        r'\w+\s+\w+\s+\w+',  # 3-word phrases
    ]

    terms = set()

    # Phrase patterns
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.update(matches)

    # Single words (4+ chars, likely technical)
    words = re.findall(r'\b[a-z]{4,}\b', text)
    terms.update([w for w in words if w not in stop_words and len(w) >= 4])

    return list(terms)


def _semantic_skill_match(terms: List[str], skills_db: Dict[str, Set[str]], threshold=0.6) -> Set[str]:
    """
    Match contextual terms to skills using semantic similarity.
    E.g., "data preprocessing" â†’ numpy, pandas
    """
    if len(terms) == 0:
        return set()

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode terms and skill names
        term_embeddings = model.encode(terms, normalize_embeddings=True)

        skill_names = list(skills_db.keys())
        skill_embeddings = model.encode(skill_names, normalize_embeddings=True)

        # Cosine similarity
        similarities = cosine_similarity(term_embeddings, skill_embeddings)

        matched_skills = set()
        for i, term in enumerate(terms):
            best_idx = np.argmax(similarities[i])
            best_score = similarities[i][best_idx]

            if best_score >= threshold:
                matched_skills.add(skill_names[best_idx])

        return matched_skills
    except Exception as e:
        print(f" Semantic matching failed: {e}")
        return set()


def extract_skills_by_section(
    resume_sections: Dict[str, str],
    skills_db: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """Extract skills from each section separately."""
    return {
        section_name: extract_skills(section_text, skills_db)
        for section_name, section_text in resume_sections.items()
    }


def parse_resume_sections(full_text: str) -> Dict[str, str]:
    """Split resume into work experience, projects, skills, education sections."""
    section_headers = {
        "experience": ["work experience", "experience", "internship", "employment"],
        "projects":   ["projects", "project", "portfolio"],
        "skills":     ["technical skills", "skills", "technologies", "technical"],
        "education":  ["education", "academic", "bachelor", "master"],
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

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def get_canonical_skill_name(raw_skill: str, skills_db: Dict[str, Set[str]]) -> str:
    """Convert raw skill mention to canonical name."""
    raw_lower = raw_skill.lower()
    for canonical, variants in skills_db.items():
        for v in variants:
            pattern = r'\b' + re.escape(v.lower()) + r'\b'
            if re.search(pattern, raw_lower):
                return canonical
    return raw_skill
