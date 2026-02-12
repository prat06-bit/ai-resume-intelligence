
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

class SkillClusterer:
    def __init__(self, n_clusters=4):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.n_clusters = n_clusters

    def cluster(self, skills: list[str]):
        if len(skills) < self.n_clusters:
            return {0: skills}

        embeddings = self.model.encode(skills)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters = {}
        for skill, label in zip(skills, labels):
            clusters.setdefault(label, []).append(skill)

        return clusters
