import re
from typing import List, Dict


def _extract_jd_context(skill: str, jd_text: str) -> str:
    """Find the sentence in JD that mentions this skill for context."""
    if not jd_text:
        return ""
    skill_clean = skill.replace("_", " ").lower()
    for line in jd_text.splitlines():
        if skill_clean in line.lower() or skill.lower() in line.lower():
            return line.strip()
    return ""


def _build_action(skill: str, jd_text: str, resume_text: str) -> Dict[str, str]:
    """
    Build a specific action + why for a missing skill
    using JD context and resume background.
    """
    skill_lower = skill.lower()
    jd_context  = _extract_jd_context(skill, jd_text)

    # Detect what the candidate already has (for tailored advice)
    has_flask    = "flask"    in resume_text.lower()
    has_node     = "node"     in resume_text.lower()
    has_python   = "python"   in resume_text.lower()
    has_docker   = "docker"   in resume_text.lower()
    has_projects = "project"  in resume_text.lower()

    # ── Specific skill advice ──────────────────────────────────────
    advice_map = {
        "jwt": (
            f"Implement JWT authentication in your {'Flask' if has_flask else 'Node.js' if has_node else 'backend'} project — "
            f"add /login, /refresh-token, and /logout endpoints with proper expiry handling.",
            "JWT is the standard auth mechanism for REST APIs and is explicitly required in this JD."
        ),
        "oauth": (
            "Integrate Google or GitHub OAuth2 login into one of your existing projects using Passport.js (Node) or Authlib (Python).",
            "OAuth2 is a core authentication requirement listed in this role's security practices."
        ),
        "redis": (
            f"Add Redis caching to your {'Flask' if has_flask else 'backend'} API — cache expensive DB queries or implement rate limiting.",
            "Redis is listed as a required caching system for this backend role."
        ),
        "spring_boot": (
            "Build a CRUD REST API with Spring Boot + JPA + MySQL — deploy it on a free Render or Railway instance and add it to GitHub.",
            "Spring Boot is one of the primary backend frameworks listed as required in this JD."
        ),
        "cicd": (
            f"Set up a GitHub Actions workflow for {'one of your existing' if has_projects else 'a'} project: lint → test → deploy on push to main.",
            "CI/CD is listed as a preferred qualification — even a basic pipeline significantly boosts your profile."
        ),
        "microservices": (
            "Refactor one of your existing projects into 2 services (e.g. auth service + main service) that communicate via REST.",
            "Microservices architecture is required knowledge for this backend role."
        ),
        "kafka": (
            "Add a Kafka producer/consumer to handle async tasks (e.g. email notifications, log processing) in a backend project.",
            "Kafka is listed as a preferred message queue technology for this role."
        ),
        "rabbitmq": (
            "Implement a task queue using RabbitMQ to process background jobs asynchronously in your backend.",
            "Message queue knowledge (Kafka/RabbitMQ) is a preferred qualification in this JD."
        ),
        "postgresql": (
            "Migrate one of your MySQL/MongoDB projects to PostgreSQL — write optimized queries and add indexes.",
            "PostgreSQL is one of the required SQL databases for this backend engineer role."
        ),
        "swagger": (
            "Add Swagger/OpenAPI docs to your REST API — use flask-swagger-ui (Python) or swagger-jsdoc (Node.js).",
            "API documentation with Swagger is listed as a required familiarity in this JD."
        ),
        "postman": (
            "Create a Postman collection for your REST API with environment variables, pre-request scripts, and automated tests.",
            "Postman is listed as a required API documentation/testing tool in this JD."
        ),
        "kubernetes": (
            f"Deploy your {'Dockerized' if has_docker else ''} app to a local Kubernetes cluster using minikube — write a Deployment + Service YAML.",
            "Container orchestration knowledge strengthens your DevOps alignment for this role."
        ),
        "aws": (
            "Deploy a backend service to AWS — use EC2 for a server, S3 for file storage, and RDS for a managed database.",
            "Basic cloud concepts (AWS/GCP/Azure) are listed as required architecture knowledge."
        ),
        "gcp": (
            "Deploy a containerized backend service to Google Cloud Run — connect it to Cloud SQL for managed PostgreSQL.",
            "Basic cloud concepts (AWS/GCP/Azure) are listed as required architecture knowledge."
        ),
        "authentication": (
            "Implement full auth in a project: registration, login, password reset, and session/token management.",
            "Authentication & authorization best practices are a core responsibility in this role."
        ),
        "rest_api": (
            "Build and deploy a REST API with proper HTTP methods, status codes, versioning, and error handling — document with Swagger.",
            "Building REST APIs is the primary technical responsibility listed in this JD."
        ),
        "data_structures": (
            "Solve 2–3 DSA problems per week on LeetCode (focus on arrays, hashmaps, trees) — mention DSA in your resume skills.",
            "Data Structures & Algorithms is an explicit technical requirement for this role."
        ),
        "system_design": (
            "Study and document the design of one system (e.g. URL shortener, rate limiter) — add it as a System Design project on GitHub.",
            "Scalable system design experience is a preferred qualification for this backend role."
        ),
        "graphql": (
            "Add a GraphQL endpoint to one of your existing REST APIs using Apollo Server (Node) or Strawberry (Python).",
            "GraphQL is a modern API paradigm increasingly expected in backend roles."
        ),
        "encryption": (
            "Implement data encryption at rest using AES-256 and in transit using HTTPS/TLS in one of your backend projects.",
            "Security and encryption are core responsibilities listed in this JD."
        ),
    }

    if skill_lower in advice_map:
        action, why = advice_map[skill_lower]
    else:
        # Dynamic fallback using JD context
        if jd_context:
            action = (
                f"Build a project demonstrating {skill.replace('_', ' ')} — "
                f"the JD specifically mentions: \"{jd_context[:120]}\". "
                f"Add measurable outcomes and tools to your resume bullet."
            )
        else:
            action = (
                f"Add a concrete project or work experience that demonstrates {skill.replace('_', ' ')}. "
                f"Include specific tools used, metrics achieved, and how it was applied in production."
            )
        why = f"{skill.replace('_', ' ').title()} is required for this role but is not clearly demonstrated in your current resume."

    return {"action": action, "why": why}


def _assign_priority(skill: str, score: float, jd_text: str) -> str:
    """Assign priority based on how strongly skill appears in JD and how low its score is."""
    skill_lower   = skill.lower().replace("_", " ")
    jd_lower      = jd_text.lower() if jd_text else ""

    # Count mentions in JD
    mentions = jd_lower.count(skill_lower) + jd_lower.count(skill.lower())

    if score < 0.15 and mentions >= 2:
        return "high"
    elif score < 0.20 or mentions >= 2:
        return "high"
    elif score < 0.28 or mentions == 1:
        return "medium"
    else:
        return "low"


def generate_roadmap(
    missing_skills: List[str],
    score: float,
    jd_text: str = "",
    resume_text: str = ""
) -> List[Dict]:
    """
    Generate a dynamic, context-aware improvement roadmap.

    - Uses JD text to find exact context for each missing skill
    - Uses resume text to tailor advice (e.g. "add to your Flask project")
    - No external API needed — fully self-contained
    - Returns sorted list: high priority first
    """
    if not missing_skills:
        return []

    roadmap = []
    for skill in missing_skills:
        sim_score = 0.20  # default if not available
        advice    = _build_action(skill, jd_text, resume_text)
        priority  = _assign_priority(skill, sim_score, jd_text)

        roadmap.append({
            "skill":    skill,
            "action":   advice["action"],
            "priority": priority,
            "why":      advice["why"],
        })

    # Sort: high → medium → low
    order = {"high": 0, "medium": 1, "low": 2}
    roadmap.sort(key=lambda x: order.get(x["priority"], 1))

    return roadmap
