import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class SkillClusterer:
    def __init__(self, n_clusters=4, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

    def embed(self, skills):
        return self.model.encode(skills, normalize_embeddings=True, show_progress_bar=False)

    def reduce_dim(self, embeddings, method="umap", n_components=2):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) < 2:
            return None

        if method == "umap":
            if len(embeddings) <= n_components:
                return None
            try:
                from umap import UMAP
            except ImportError:
                return None

            reducer = UMAP(
                n_components=n_components,
                n_neighbors=min(max(2, len(embeddings) // 2), len(embeddings) - 1),
                min_dist=0.08,
                metric="cosine",
                init="random",
                random_state=42,
            )
            return reducer.fit_transform(embeddings)

        n_safe = min(n_components, embeddings.shape[0], embeddings.shape[1])
        coords = PCA(n_components=n_safe, random_state=42).fit_transform(embeddings)
        if coords.shape[1] < n_components:
            padding = np.zeros((coords.shape[0], n_components - coords.shape[1]))
            coords = np.hstack([coords, padding])
        return coords

    def cluster(self, skills, reduction="umap"):
        if not skills or len(skills) < 2:
            return []

        embeddings = self.embed(skills)
        k = min(self.n_clusters, len(skills))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        coords = self.reduce_dim(embeddings, method=reduction)

        if coords is None:
            return []

        return [
            {
                "skill": skill,
                "x": float(coords[index][0]),
                "y": float(coords[index][1]),
                "cluster": int(labels[index]),
            }
            for index, skill in enumerate(skills)
        ]
