import hashlib
import json
from typing import Dict, Set, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

from backend.core.database import get_cached_embedding, save_cached_embedding

ROLE_WEIGHTS = {
    "software_engineer":   {"projects": 0.4,  "experience": 0.35, "skills": 0.15, "education": 0.1},
    "backend_engineer":    {"projects": 0.35, "experience": 0.4,  "skills": 0.15, "education": 0.1},
    "frontend_engineer":   {"projects": 0.45, "experience": 0.3,  "skills": 0.15, "education": 0.1},
    "fullstack_engineer":  {"projects": 0.4,  "experience": 0.35, "skills": 0.15, "education": 0.1},
    "ml_engineer":         {"projects": 0.35, "experience": 0.35, "skills": 0.2,  "education": 0.1},
    "data_scientist":      {"projects": 0.3,  "experience": 0.35, "skills": 0.2,  "education": 0.15},
    "data_analyst":        {"projects": 0.3,  "experience": 0.35, "skills": 0.2,  "education": 0.15},
    "devops_engineer":     {"projects": 0.3,  "experience": 0.45, "skills": 0.15, "education": 0.1},
    "cloud_engineer":      {"projects": 0.3,  "experience": 0.45, "skills": 0.15, "education": 0.1},
    "student":             {"projects": 0.5,  "experience": 0.15, "skills": 0.2,  "education": 0.15},
    "intern":              {"projects": 0.5,  "experience": 0.15, "skills": 0.2,  "education": 0.15},
}

DEFAULT_ROLE_WEIGHTS = {"projects": 0.4, "experience": 0.35, "skills": 0.15, "education": 0.1}

SIM_FLOOR   = 0.20
SIM_CEILING = 0.65


class SemanticMatcher:
    """Semantic skill matching with a safe local fallback when torch is unavailable."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model."""
        self.model_name = model_name
        self.load_error = None
        self.using_fallback = False
        self.vectorizer = HashingVectorizer(
            n_features=768,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(3, 5),
        )
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception as e:
            self.model = None
            self.load_error = e
            self.using_fallback = True

    def _cache_key(self, text: str) -> tuple[str, str]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        backend = "sentence_transformer" if self.model is not None else "hashing_vectorizer"
        return f"{self.model_name}:{backend}:{text_hash}", text_hash

    def _embed_uncached(self, texts):
        if self.model is not None:
            try:
                return self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                )
            except Exception as e:
                self.load_error = e
                self.model = None
                self.using_fallback = True

        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def embed(self, texts, use_cache=True):
        """Embed list of texts into vectors."""
        texts = [str(text or "") for text in texts]
        if not use_cache:
            return self._embed_uncached(texts)

        vectors = []
        missing_indexes = []
        missing_texts = []

        for index, text in enumerate(texts):
            cache_key, _ = self._cache_key(text)
            cached = get_cached_embedding(cache_key)
            if cached:
                vectors.append(np.array(json.loads(cached["vector_json"]), dtype=np.float32))
            else:
                vectors.append(None)
                missing_indexes.append(index)
                missing_texts.append(text)

        if missing_texts:
            computed = self._embed_uncached(missing_texts)
            for offset, index in enumerate(missing_indexes):
                vector = np.asarray(computed[offset], dtype=np.float32)
                text = texts[index]
                cache_key, text_hash = self._cache_key(text)
                backend = "sentence_transformer" if self.model is not None else "hashing_vectorizer"
                save_cached_embedding(
                    cache_key=cache_key,
                    text_hash=text_hash,
                    model_name=self.model_name,
                    backend=backend,
                    dimensions=int(vector.shape[0]),
                    vector_json=json.dumps(vector.tolist()),
                )
                vectors[index] = vector

        return np.vstack(vectors).astype(np.float32)

    def match_skills(self, resume_skills, jd_skills):
        """
        Match resume skills to JD skills using semantic similarity.

        For each JD skill, find best match in resume skills.
        Applies penalties for partial matches (e.g., flask != django).

        Returns: {jd_skill: match_score (0.0-1.0)}
        """
        if not resume_skills or not jd_skills:
            return {}

        resume_list = list(resume_skills)
        jd_list = list(jd_skills)

        res_emb = self.embed(resume_list)
        jd_emb = self.embed(jd_list)

        # Similarity matrix: (n_jd, n_resume)
        sim_matrix = cosine_similarity(jd_emb, res_emb)

        result = {}
        for i, jd_skill in enumerate(jd_list):
            best_score = float(sim_matrix[i].max())
            best_idx = int(sim_matrix[i].argmax())
            best_match = resume_list[best_idx]

            if jd_skill == best_match:
                result[jd_skill] = best_score
            elif best_score >= 0.85:
                result[jd_skill] = best_score
            elif best_score >= 0.65:
                result[jd_skill] = best_score * 0.65
            else:
                result[jd_skill] = best_score * 0.3

        return result

    def embed_sections(self, resume_text):
        """
        Split resume into sections and embed each.

        Returns: {section_name: embedding_vector}
        """
        sections = {
            "experience": [],
            "projects": [],
            "skills": [],
            "education": [],
        }

        current = None
        section_order = ["experience", "projects", "skills", "education"]

        for line in resume_text.splitlines():
            l = line.lower().strip()

            # Detect section headers
            if any(kw in l for kw in ["work experience", "experience", "internship", "employment"]):
                current = "experience"
            elif any(kw in l for kw in ["project", "portfolio"]):
                current = "projects"
            elif any(kw in l for kw in ["technical skill", "skill", "technolog"]):
                current = "skills"
            elif any(kw in l for kw in ["education", "academic", "degree"]):
                current = "education"

            if current and current in sections and line.strip():
                sections[current].append(line)

        embeddings = {}
        for sec in section_order:
            text = " ".join(sections[sec]).strip()
            if text and len(text) > 10:
                try:
                    embeddings[sec] = self.embed([text])[0]
                except:
                    embeddings[sec] = None
            else:
                embeddings[sec] = None

        return embeddings


#  Scoring Functions

def compute_score(similarity):
    """
    Compute overall skill match score from similarity dict.

    Returns: (score_percentage, confidence_level)
    """
    if not similarity:
        return 0.0, "Low"

    values = list(similarity.values())
    strong = [v for v in values if v >= 0.55]
    weak = [v for v in values if v < 0.55]

    strong_avg = sum(strong) / len(strong) if strong else 0.0
    weak_avg = sum(weak) / len(weak) if weak else 0.0
    coverage = len(strong) / len(values) if values else 0.0

    raw_score = (strong_avg * 0.5) + (coverage * 0.35) + (weak_avg * 0.15)
    score_pct = round(raw_score * 100, 2)

    if coverage >= 0.75:
        confidence = "High"
    elif coverage >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    return score_pct, confidence


def role_weighted_score(section_similarities, role):
    """
    Weight section similarities based on role.

    E.g., frontend_engineer weights projects higher than experience.

    Returns: weighted_score_percentage
    """
    weights = ROLE_WEIGHTS.get(role, DEFAULT_ROLE_WEIGHTS)
    total_weight = 0.0
    weighted_sum = 0.0

    for section, sim in section_similarities.items():
        w = weights.get(section, 0.1)

        # Rescale similarity to 0-1 range
        rescaled = (sim - SIM_FLOOR) / (SIM_CEILING - SIM_FLOOR)
        rescaled = max(0.0, min(1.0, rescaled))

        # Education less important
        if section == "education":
            rescaled = min(rescaled, 0.5)

        weighted_sum += rescaled * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    return round((weighted_sum / total_weight) * 100, 2)


def coverage_score(resume_skills, jd_required_skills):
    """What fraction of JD skills are in resume."""
    if not jd_required_skills:
        return 1.0
    matched = resume_skills & jd_required_skills
    return round(len(matched) / len(jd_required_skills), 4)


def get_skill_gaps(resume_skills, jd_required_skills):
    """Skills missing from resume but required by JD."""
    return sorted(list(jd_required_skills - resume_skills))


def get_matched_skills(resume_skills, jd_required_skills):
    """Skills that appear in both resume and JD."""
    return sorted(list(resume_skills & jd_required_skills))
