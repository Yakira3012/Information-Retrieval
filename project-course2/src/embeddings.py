# Yakira_S - IR Final Assignment - embeddings.py
# Experiment B: Compare embedding models for RAG

import os
import time
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()


# ── Model 1: Local sentence-transformers ─────────────────────────────────────

def embed_minilm(texts: List[str], batch_size: int = 64) -> Tuple[np.ndarray, Dict]:
    """
    Embed texts using all-MiniLM-L6-v2 (local, free).
    Returns:
        embeddings: np.ndarray of shape (N, 384)
        stats:      dict with timing and metadata
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    elapsed = time.time() - t0

    stats = {
        "model": "all-MiniLM-L6-v2",
        "dim": embeddings.shape[1],
        "n_texts": len(texts),
        "time_seconds": round(elapsed, 2),
        "cost_usd": 0.0,
        "source": "local",
    }
    return embeddings, stats


# ── Model 2: OpenAI text-embedding-3-small ────────────────────────────────────

def embed_openai(texts: List[str], batch_size: int = 100) -> Tuple[np.ndarray, Dict]:
    """
    Embed texts using OpenAI text-embedding-3-small.
    Requires OPENAI_API_KEY in environment.
    Returns:
        embeddings: np.ndarray of shape (N, 1536)
        stats:      dict with timing, cost estimate, and metadata
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    all_embeddings = []
    total_tokens = 0
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        total_tokens += response.usage.total_tokens
        print(f"  Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    elapsed = time.time() - t0
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)

    # Cost: $0.02 per 1M tokens for text-embedding-3-small
    cost = (total_tokens / 1_000_000) * 0.02

    stats = {
        "model": "text-embedding-3-small",
        "dim": embeddings.shape[1],
        "n_texts": len(texts),
        "total_tokens": total_tokens,
        "time_seconds": round(elapsed, 2),
        "cost_usd": round(cost, 5),
        "source": "openai_api",
    }
    return embeddings, stats


# ── Dispatcher ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], model: str = "minilm") -> Tuple[np.ndarray, Dict]:
    """
    Unified embedding interface.
    Args:
        texts: list of strings to embed
        model: 'minilm' | 'openai'
    """
    model = model.lower()
    if model == "minilm":
        return embed_minilm(texts)
    elif model == "openai":
        return embed_openai(texts)
    else:
        raise ValueError(f"Unknown model '{model}'. Choose: minilm | openai")


def cosine_similarity_matrix(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a query vector and all corpus vectors.
    Assumes vectors are already L2-normalized.
    Returns array of shape (N,).
    """
    return corpus_vecs @ query_vec
