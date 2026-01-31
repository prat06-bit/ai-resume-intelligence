def analyze_resume(resume, jd):
    return {
        "match": 78,
        "skills": {
            "Python": 85,
            "ML": 70,
            "SQL": 60,
            "System Design": 50
        },
        "missing_skills": {
            "Docker": 60,
            "AWS": 70
        },
        "explanation": "Strong backend skills but missing cloud exposure.",
        "roadmap": [
            "Learn Docker fundamentals",
            "Deploy projects on AWS",
            "Improve system design depth"
        ]
    }
