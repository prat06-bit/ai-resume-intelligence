from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def match_skills(self, resume_skills, jd_skills):
        """
        For each JD skill, find the best matching resume skill via cosine similarity.

        Calibrated for all-MiniLM-L6-v2 where:
          - Exact same skill   → ~0.99
          - Same category      → ~0.65–0.80  (e.g. flask vs django)
          - Related domain     → ~0.45–0.65  (e.g. python vs java)
          - Unrelated          → ~0.10–0.40

        Scoring tiers:
          ≥ 0.80  → Exact / near-exact match          → full score
          0.60–0.80 → Same skill family, light penalty → × 0.80
          0.40–0.60 → Related but different            → × 0.55
          < 0.40  → Unrelated / missing                → × 0.15

        Target: 13 matched skills on a ~35% resume should give ~45–60% final score.
        """
        if not resume_skills or not jd_skills:
            return {}

        resume_list = list(resume_skills)
        jd_list     = list(jd_skills)

        res_emb = self.embed(resume_list)
        jd_emb  = self.embed(jd_list)

        sim_matrix = cosine_similarity(jd_emb, res_emb)  # (n_jd, n_resume)

        result = {}
        for i, jd_skill in enumerate(jd_list):
            best_score = float(sim_matrix[i].max())
            best_idx   = int(sim_matrix[i].argmax())
            best_match = resume_list[best_idx]

            # Exact canonical match → never penalise
            if jd_skill == best_match:
                result[jd_skill] = best_score

            # Strong semantic match (same skill, different label)
            elif best_score >= 0.80:
                result[jd_skill] = best_score

            # Same skill family — slight penalty
            elif best_score >= 0.60:
                result[jd_skill] = round(best_score * 0.80, 4)

            # Related domain — moderate penalty
            elif best_score >= 0.40:
                result[jd_skill] = round(best_score * 0.55, 4)

            # Unrelated / missing — treat as absent
            else:
                result[jd_skill] = round(best_score * 0.15, 4)

        return result

    def embed_sections(self, resume_text):
        """
        Split resume into sections and embed each one.
        Returns { section_name: embedding_vector | None }
        """
        sections = {
            "experience": [],
            "projects":   [],
            "skills":     [],
            "education":  [],
        }

        current = None

        for line in resume_text.splitlines():
            l = line.lower().strip()

            if any(kw in l for kw in ["work experience", "experience", "internship"]):
                current = "experience"
            elif "project" in l:
                current = "projects"
            elif any(kw in l for kw in ["technical skill", "skill", "technolog"]):
                current = "skills"
            elif any(kw in l for kw in ["education", "academic"]):
                current = "education"

            if current and current in sections:
                sections[current].append(line)

        embeddings = {}
        for sec, lines in sections.items():
            text = " ".join(lines).strip()
            embeddings[sec] = self.embed([text])[0] if text else None

        return embeddings
