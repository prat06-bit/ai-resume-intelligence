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
        if not resume_skills or not jd_skills:
            return {}

        resume_list = list(resume_skills)
        jd_list     = list(jd_skills)

        res_emb = self.embed(resume_list)
        jd_emb  = self.embed(jd_list)

        sim_matrix = cosine_similarity(jd_emb, res_emb) 

        result = {}
        for i, jd_skill in enumerate(jd_list):
            best_score = float(sim_matrix[i].max())
            best_idx   = int(sim_matrix[i].argmax())
            best_match = resume_list[best_idx]

            if jd_skill == best_match:
                result[jd_skill] = best_score

            elif best_score >= 0.80:
                result[jd_skill] = best_score

            elif best_score >= 0.60:
                result[jd_skill] = round(best_score * 0.80, 4)

            elif best_score >= 0.40:
                result[jd_skill] = round(best_score * 0.55, 4)

            else:
                result[jd_skill] = round(best_score * 0.15, 4)

        return result

    def embed_sections(self, resume_text):
        sections = {
            "experience": [],
            "projects":   [],
            "skills":     [],
            "education":  [],
        }

        HEADERS = {
            "experience": ["work experience", "experience", "internship", "employment"],
            "projects":   ["projects", "personal projects", "key projects"],
            "skills":     ["technical skills", "skills", "technologies", "competencies"],
            "education":  ["education", "academic", "qualifications", "certifications"],
        }

        current = None

        for line in resume_text.splitlines():
            l = line.lower().strip()

            if len(l) < 40:
                matched_section = None
                for sec, keywords in HEADERS.items():
                    if any(kw == l or l.startswith(kw) for kw in keywords):
                        matched_section = sec
                        break
                if matched_section:
                    current = matched_section
                    continue  

            if current and line.strip():
                sections[current].append(line.strip())

        embeddings = {}
        for sec, lines in sections.items():
            text = " ".join(lines).strip()
            embeddings[sec] = self.embed([text])[0] if len(text) > 20 else None

        return embeddings
