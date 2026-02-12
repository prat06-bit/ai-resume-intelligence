from typing import List, Dict

def generate_roadmap(missing_skills: List[str], score: float) -> List[Dict]:
    roadmap = []

    for skill in missing_skills:
        roadmap.append({
            "skill": skill,
            "action": f"Add a concrete project demonstrating {skill}",
            "priority": "high" if score < 60 else "medium" if score < 80 else "low"
        })

    if score >= 85:
        roadmap.append({
            "skill": "Interview Readiness",
            "action": "Focus on system design, ML trade-offs, and explainability",
            "priority": "low"
        })

    return roadmap
