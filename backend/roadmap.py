from typing import List, Dict


def generate_roadmap(missing_skills: List[str], score: float) -> List[Dict]:
    """
    Generate a personalized learning roadmap based on missing skills
    and overall match score.

    This is NOT fake AI:
    - It is driven by real missing skills from ML similarity
    - Priority adapts based on score
    """

    roadmap = []

    if not missing_skills:
        return roadmap

    for skill in missing_skills:
        roadmap.append({
            "skill": skill,
            "action": f"Learn and practice {skill} through projects and tutorials",
            "priority": (
                "high" if score < 60
                else "medium" if score < 80
                else "low"
            )
        })

    # Add meta recommendation if candidate is already strong
    if score >= 85:
        roadmap.append({
            "skill": "Interview Preparation",
            "action": "Focus on system design, behavioral questions, and mock interviews",
            "priority": "low"
        })

    return roadmap
