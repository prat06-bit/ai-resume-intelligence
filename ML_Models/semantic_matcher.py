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
        
        Applies a calibrated penalty so that semantically-close-but-not-same
        skills don't inflate the score (e.g. 'flask' != 'spring_boot').

        Returns: { jd_skill: score (0.0–1.0) }
        """
        if not resume_skills or not jd_skills:
            return {}

        resume_list = list(resume_skills)
        jd_list     = list(jd_skills)

        res_emb = self.embed(resume_list)
        jd_emb  = self.embed(jd_list)

        sim_matrix = cosine_similarity(jd_emb, res_emb)  # shape: (n_jd, n_resume)

        result = {}
        for i, jd_skill in enumerate(jd_list):
            best_score = float(sim_matrix[i].max())
            best_idx   = int(sim_matrix[i].argmax())
            best_match = resume_list[best_idx]

            #  Exact / near-exact match (same canonical name)   
            if jd_skill == best_match:
                result[jd_skill] = best_score

            #  Strong semantic match    
            elif best_score >= 0.85:
                result[jd_skill] = best_score

            #  Moderate match — apply penalty   
            elif best_score >= 0.65:
                result[jd_skill] = best_score * 0.65

            #  Weak / unrelated match — treat as missing 
            else:
                result[jd_skill] = best_score * 0.3

        return result

    def embed_sections(self, resume_text):
        """
        Split resume into sections and embed each one.
        Returns { section_name: embedding_vector }
        """
        sections = {
            "experience": [],
            "projects":   [],
            "skills":     [],
            "education":  [],
        }

        current = None
        section_order = ["experience", "projects", "skills", "education"]

        for line in resume_text.splitlines():
            l = line.lower().strip()

            # Detect section header
            if any(kw in l for kw in ["work experience", "experience", "internship"]):
                current = "experience"
            elif any(kw in l for kw in ["project"]):
                current = "projects"
            elif any(kw in l for kw in ["technical skill", "skill", "technolog"]):
                current = "skills"
            elif any(kw in l for kw in ["education", "academic"]):
                current = "education"

            if current and current in sections:
                sections[current].append(line)

        embeddings = {}
        for sec in section_order:
            text = " ".join(sections[sec]).strip()
            if text:
                embeddings[sec] = self.embed([text])[0]
            else:
                embeddings[sec] = None

        return embeddings
