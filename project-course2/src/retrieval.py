# Yakira_S - IR Final Assignment - retrieval.py
# Experiment C: Dense (FAISS), Sparse (BM25), and Hybrid retrieval

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple


# ── Dense Retrieval: FAISS ────────────────────────────────────────────────────

class DenseRetriever:
    """FAISS-based dense retrieval using cosine similarity."""

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Build FAISS index from chunk embeddings.
        Args:
            chunks:     list of chunk dicts (must have 'id' and 'text')
            embeddings: np.ndarray of shape (N, dim), L2-normalized
        """
        import faiss

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner product = cosine on normalized vecs
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        print(f"[DenseRetriever] Built index: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-K chunks for a query embedding.
        Returns list of {chunk, score} dicts sorted by score descending.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "retriever": "dense",
            })
        return results

    def save(self, path: str):
        import faiss
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str):
        import faiss
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[DenseRetriever] Loaded index: {self.index.ntotal} vectors")


# ── Sparse Retrieval: BM25 ────────────────────────────────────────────────────

class SparseRetriever:
    """BM25-based sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: List[Dict]):
        """
        Build BM25 index from chunk texts.
        """
        from rank_bm25 import BM25Okapi

        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self.chunks = chunks
        print(f"[SparseRetriever] Built BM25 index over {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-K chunks for a string query using BM25.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "retriever": "sparse",
            })
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "bm25.pkl"), "wb") as f:
            pickle.dump((self.bm25, self.chunks), f)

    def load(self, path: str):
        with open(os.path.join(path, "bm25.pkl"), "rb") as f:
            self.bm25, self.chunks = pickle.load(f)
        print(f"[SparseRetriever] Loaded BM25 index over {len(self.chunks)} chunks")


# ── Hybrid Retrieval ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines DenseRetriever and SparseRetriever scores with a weighted sum.
    Both score distributions are normalized to [0,1] before combining.
    """

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, alpha: float = 0.5):
        """
        Args:
            alpha: weight for dense scores (1-alpha for sparse)
        """
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-K chunks by combining dense and sparse scores.
        """
        # Get a larger pool from each retriever
        pool_size = min(top_k * 4, len(self.dense.chunks))
        dense_results = self.dense.retrieve(query_embedding, top_k=pool_size)
        sparse_results = self.sparse.retrieve(query, top_k=pool_size)

        # Build score maps: chunk_id → score
        dense_scores = {r["chunk"]["id"]: r["score"] for r in dense_results}
        sparse_scores = {r["chunk"]["id"]: r["score"] for r in sparse_results}

        # Normalize each to [0,1]
        def normalize(scores: dict) -> dict:
            if not scores:
                return scores
            vals = np.array(list(scores.values()), dtype=float)
            mn, mx = vals.min(), vals.max()
            if mx == mn:
                return {k: 0.5 for k in scores}
            return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        # Combine over union of chunk ids
        all_ids = set(dense_norm) | set(sparse_norm)
        combined = {}
        chunk_map = {r["chunk"]["id"]: r["chunk"]
                     for r in dense_results + sparse_results}

        for cid in all_ids:
            d = dense_norm.get(cid, 0.0)
            s = sparse_norm.get(cid, 0.0)
            combined[cid] = self.alpha * d + (1 - self.alpha) * s

        # Sort and return top-K
        sorted_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
        results = []
        for cid in sorted_ids:
            results.append({
                "chunk": chunk_map[cid],
                "score": combined[cid],
                "retriever": "hybrid",
            })
        return results
