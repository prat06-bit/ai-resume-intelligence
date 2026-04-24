import sys
import os

# Make sure ML_Models is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ML_Models"))

from roadmap import generate_roadmap

#  Fake test data  

MISSING_SKILLS = [
    "kubernetes",
    "pytorch",
    "rest_api",
    "microservices",
    "pandas"
]

SCORE = 58.0

JD_TEXT = """
We are looking for a Machine Learning Engineer to join our team.

Requirements:
- Strong proficiency in Python and machine learning libraries such as Scikit-learn, TensorFlow, or PyTorch
- Familiarity with MLOps tools and containerization (Docker, Kubernetes)
- Knowledge of model deployment, REST APIs, and cloud platforms
- Deploy ML models using APIs, microservices, or cloud platforms
- Experience with data manipulation using Pandas and NumPy
- Kubernetes experience is required for managing production workloads
"""

RESUME_TEXT = """
John Doe — ML Engineer
Skills: Python, Scikit-learn, TensorFlow, SQL, Git, Docker
Projects:
  - Built a sentiment analysis model using TensorFlow achieving 91% accuracy
  - Created a data pipeline using SQL and Python for ETL processing
  - Deployed a Flask web app on AWS EC2
Experience: 2 years at DataCorp as a junior data scientist
"""

#  Run 

if __name__ == "__main__":
    print("=" * 60)
    print(f"Testing generate_roadmap() with {len(MISSING_SKILLS)} missing skills")
    print(f"Model: {os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')} via Ollama")  
    print("=" * 60)

    result = generate_roadmap(
        missing_skills=MISSING_SKILLS,
        score=SCORE,
        jd_text=JD_TEXT,
        resume_text=RESUME_TEXT
    )

    print(f"\n Got {len(result)} roadmap steps:\n")
    for i, step in enumerate(result, 1):
        print(f"  Step {i}: {step['skill']}  [{step['priority'].upper()}]")
        print(f"  Action : {step['action']}")
        print(f"  Why    : {step['why']}")
        print()

    print("=" * 60)
    print("Done.")
