from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        jd_list = list(jd_skills)

        res_emb = self.embed(resume_list)
        jd_emb = self.embed(jd_list)

        sim = cosine_similarity(jd_emb, res_emb)

        # JD skill → best resume alignment
        return {
            jd_list[i]: float(sim[i].max())
            for i in range(len(jd_list))
        }

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
            sec: self.embed([txt])[0] if txt.strip() else None
            for sec, txt in sections.items()
        }
