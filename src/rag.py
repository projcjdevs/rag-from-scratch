import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

class LightweightAG:

    def __init__(self, model_name="all-MiniLM-L6-v2", n_clusters=3):

        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

        self.documents = []
        self.embeddings = None
        self.centroids = None
        self.assignments = None
        self.adj_matrix = None

        print(f"LightweightRAG ready")
        print(f"  model:      {model_name}")
        print(f"  n_clusters: {n_clusters}")

    # Math Utilities (Sprint 1 & 2)

    def _l2_norm(self, v):
        return np.sqrt(np.sum(v ** 2))

    def _normalize(self, v):
        return v / self._l2_norm(v)

    def _cosine_similarity(self, a, b):
        return np.sum(a * b) / (self._l2_norm(a) * self._l2_norm(b))

    def _similarity_matrix(self, embeddings):
        n = len(embeddings)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self._cosine_similarity(
                    embeddings[i], embeddings[j]
                )
        return matrix
    
    # K-means clustering (Sprint 4)
    def _kmeans(self, vectors, max_iters=100, tol=1e-4):
        n, d   = vectors.shape
        k      = self.n_clusters
        idx       = np.random.choice(n, k, replace=False)
        centroids = vectors[idx].copy()
        assignments = np.zeros(n, dtype=int)

        for iteration in range(max_iters):

            distances = np.zeros((n, k))
            for j, centroid in enumerate(centroids):
                diff = vectors - centroid
                distances[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
            new_assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                members = vectors[new_assignments == j]
                new_centroids[j] = (
                    np.mean(members, axis=0)
                    if len(members) > 0
                    else centroids[j]
                )

            shift       = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            assignments = new_assignments
            centroids   = new_centroids

            if shift < tol:
                print(f"  K-Means converged in {iteration + 1} iterations")
                break

        return centroids, assignments
    
    # Graph Utilities (Sprint 5)

    def _build_adjacency(self, embeddings, threshold=0.75):
        n      = len(embeddings)
        sim    = self._similarity_matrix(embeddings)
        adj    = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                if i != j and sim[i, j] >= threshold:
                    adj[i, j] = 1

        n_edges = np.sum(adj) // 2
        print(f"  Graph built: {n} nodes, {n_edges} edges "
              f"(threshold={threshold})")
        return adj

    def _graph_traverse(self, query_node, hops=2):
        n         = self.adj_matrix.shape[0]
        reachable = {}
        A_power   = np.eye(n, dtype=int)

        for hop in range(1, hops + 1):
            A_power = A_power @ self.adj_matrix
            for j in range(n):
                if j != query_node and \
                   A_power[query_node, j] > 0 and \
                   j not in reachable:
                    reachable[j] = hop

        return reachable
    
    # Indexing

    def index(self, documents, adj_threshold=0.75):

        print(f"\nIndexing {len(documents)} documents...")
        self.documents = documents

        print("  Embedding...")
        raw = self.model.encode(documents, show_progress_bar=True)

        print("  Normalizing...")
        self.embeddings = np.array([
            self._normalize(v) for v in raw
        ])

        print("  Clustering...")
        self.centroids, self.assignments = self._kmeans(self.embeddings)

        print("  Building graph...")
        self.adj_matrix = self._build_adjacency(
            self.embeddings, threshold=adj_threshold
        )

        print(f"\nIndex ready — {len(documents)} documents across "
            f"{self.n_clusters} clusters\n")
        
    # Vector Search

    def search(self, query, top_k=3, use_ivf=True):
        raw_query = self.model.encode([query])[0]
        query_vec = self._normalize(raw_query)

        if use_ivf:
            centroid_dists = np.sqrt(
                np.sum((self.centroids - query_vec) ** 2, axis=1)
            )
            nearest_cluster = np.argmin(centroid_dists)

            mask = self.assignments == nearest_cluster
            indices = np.where(mask)[0]
            pool = self.embeddings[mask]

        else:
            indices = np.arange(len(self.embeddings))
            pool = self.embeddings

        scores = np.array([
            self._cosine_similarity(query_vec, doc_vec)
            for doc_vec in pool
        ])
        top_local = np.argsort(scores)[::-1][:top_k]
        top_global = indices[top_local]

        return [
            (self.documents[i], float(scores[top_local[j]]))
            for j, i in enumerate(top_global)
        ]
    
    # Hybrid Search
    def graph_search(self, query, top_k=3, hops=2):
        seed_results = self.search(query, top_k=top_k, use_ivf=False)
        seed_doc = seed_results[0][0]
        seed_index = self.documents.index(seed_doc)

        print(f"  Seed document: \"{seed_doc[:60]}...\"" 
            if len(seed_doc) > 60 else f"  Seed: \"{seed_doc}\"")
        
        reachable = self._graph_traverse(seed_index, hops=hops)
        candidates = {seed_index: 0, **reachable}

        print(f"  Graph expanded to {len(candidates)} candidates "
            f"({hops}-hop neighbourhood)")
        
        raw_query = self.model.encode([query])[0]
        query_vec = self._normalize(raw_query)

        ranked =sorted(
            [
                (idx, self._cosine_similarity(query_vec, self.embeddings[idx]))
                for idx in candidates
            ],
            key=lambda x: x[1],
            reverse=True
        )

        return [
            (self.documents[idx], float(score))
            for idx, score in ranked[:top_k]
        ]
        
    # Visualization (Sprint 3)

    def visualise(self, query=None, highlight_indices=None):
        X          = self.embeddings
        X_centered = X - np.mean(X, axis=0)
        cov        = (1 / (len(X) - 1)) * X_centered.T @ X_centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs    = eigvecs[:, ::-1]
        coords_2d  = X_centered @ eigvecs[:, :2]

        palette = ["#1D9E75", "#534AB7", "#E24B4A",
                   "#BA7517", "#2196F3", "#9C27B0"]
        colors  = [palette[a % len(palette)] for a in self.assignments]

        fig, ax = plt.subplots(figsize=(9, 7))

        for i, (x, y) in enumerate(coords_2d):
            c = colors[i]
            size = 160
            alpha = 0.85

            if highlight_indices and i in highlight_indices:
                ax.scatter(x, y, s=320, color=c, zorder=5,
                           edgecolors="black", linewidths=2)
            else:
                ax.scatter(x, y, s=size, color=c,
                           alpha=alpha, zorder=3)

            label = (self.documents[i][:32] + "..."
                     if len(self.documents[i]) > 32
                     else self.documents[i])
            ax.annotate(label, (x, y),
                        fontsize=7, alpha=0.8,
                        xytext=(5, 5), textcoords="offset points")

        if query:
            raw   = self.model.encode([query])[0]
            q_vec = self._normalize(raw)
            q_centered = q_vec - np.mean(X, axis=0)
            q_2d  = q_centered @ eigvecs[:, :2]
            ax.scatter(*q_2d, s=400, color="black",
                       marker="*", zorder=6, label=f"Query: '{query}'")
            ax.legend(fontsize=9)

        ax.set_title("LightweightRAG — Document Space (PCA 2D projection)\n"
                     "colours = clusters, ★ = query",
                     fontsize=11)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.grid(True, alpha=0.12)
        plt.tight_layout()
        plt.savefig("../data/rag_visualisation.png",
                    dpi=150, bbox_inches="tight")
        plt.show()