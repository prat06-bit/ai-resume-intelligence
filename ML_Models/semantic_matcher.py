from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class SemanticMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # ---------- CORE EMBEDDING ----------
    def embed(self, texts):
        """
        Public embedding method expected by analyzer.py
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    # ---------- SKILL MATCHING ----------
    def match_skills(self, resume_skills, jd_skills):
        if not resume_skills or not jd_skills:
            return {}

        resume_list = list(resume_skills)
        jd_list = list(jd_skills)

        res_emb = self.embed(resume_list)
        jd_emb = self.embed(jd_list)

        sim = cosine_similarity(res_emb, jd_emb)

        return {
            resume_list[i]: float(sim[i].max())
            for i in range(len(resume_list))
        }

    # ---------- SECTION-AWARE EMBEDDING ----------
    def embed_sections(self, resume_text):
        sections = {
            "experience": "",
            "projects": "",
            "skills": ""
        }

        current = None
        for line in resume_text.splitlines():
            l = line.lower()
            if "experience" in l:
                current = "experience"
            elif "project" in l:
                current = "projects"
            elif "skill" in l:
                current = "skills"

            if current:
                sections[current] += line + " "

        return {
            sec: self.embed([text])[0] if text.strip() else None
            for sec, text in sections.items()
        }
