# Yakira_S - IR Final Assignment - evaluation.py
# Evaluation: Precision@K, Recall@K, Faithfulness (LLM-as-judge), Latency

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

def precision_at_k(retrieved_chunks: List[Dict], relevant_chapter: str, k: int) -> float:
    """
    Compute Precision@K.
    A chunk is considered relevant if its source chapter matches the reference.
    Args:
        retrieved_chunks: list of retrieval result dicts (top-K already sliced)
        relevant_chapter: the ground-truth source chapter string
        k:                evaluation cutoff
    Returns:
        P@K as float in [0, 1]
    """
    top_k = retrieved_chunks[:k]
    relevant_count = sum(
        1 for r in top_k
        if relevant_chapter.lower() in r["chunk"].get("text", "").lower()
    )
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(retrieved_chunks: List[Dict], all_relevant_ids: List[str], k: int) -> float:
    """
    Compute Recall@K given a set of known relevant chunk IDs.
    """
    top_k_ids = {r["chunk"]["id"] for r in retrieved_chunks[:k]}
    relevant_retrieved = len(top_k_ids & set(all_relevant_ids))
    return relevant_retrieved / len(all_relevant_ids) if all_relevant_ids else 0.0


def average_precision(retrieved_chunks: List[Dict], relevant_chapter: str) -> float:
    """
    Compute Average Precision for a single query (area under P-R curve proxy).
    """
    hits = 0
    sum_precisions = 0.0
    for i, r in enumerate(retrieved_chunks, 1):
        if relevant_chapter.lower() in r["chunk"].get("text", "").lower():
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / hits if hits > 0 else 0.0


# ── Generation Metrics ────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are an objective evaluator assessing whether an AI-generated answer is grounded in the provided context.

Score the answer on a scale of 1 to 5:
5 - The answer is entirely and accurately supported by the provided context
4 - The answer is mostly supported with minor additions
3 - The answer is partially supported; some claims are not in the context
2 - The answer is mostly unsupported or contradicts the context
1 - The answer completely ignores or contradicts the context

Context:
{context}

Question: {question}

Generated Answer: {answer}

Respond with ONLY a JSON object in this exact format:
{{"score": <integer 1-5>, "reason": "<one sentence explanation>"}}"""


def score_faithfulness(question: str, retrieved_chunks: List[Dict],
                       generated_answer: str) -> Dict:
    """
    Use Claude as a judge to score faithfulness of the generated answer.
    Returns dict with 'score' (int 1-5) and 'reason' (str).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = "\n\n".join(r["chunk"]["text"] for r in retrieved_chunks)

    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=question,
        answer=generated_answer,
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheap judge model
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the fenced content (may start with "json\n")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        raw = inner.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"score": -1, "reason": f"Parse error: {raw}"}
    return result


# ── Full Evaluation Run ───────────────────────────────────────────────────────

def evaluate_retrieval(qa_set: List[Dict], retriever, embed_fn,
                       ks: List[int] = [3, 5]) -> pd.DataFrame:
    """
    Run retrieval evaluation over the full QA set.
    Args:
        qa_set:    list of QA dicts from qa_set.json
        retriever: a retriever object with .retrieve() method
        embed_fn:  function(str) -> np.ndarray for dense retrievers, or None for sparse
        ks:        list of K values to evaluate
    Returns:
        DataFrame with per-question and aggregate metrics
    """
    records = []
    for qa in qa_set:
        query = qa["question"]
        chapter = qa["source_chapter"]

        if embed_fn is not None:
            q_emb, _ = embed_fn([query])
            try:
                # HybridRetriever needs both query string and embedding
                results = retriever.retrieve(query, q_emb[0], top_k=max(ks))
            except TypeError:
                # DenseRetriever only needs the embedding
                results = retriever.retrieve(q_emb[0], top_k=max(ks))
        else:
            results = retriever.retrieve(query, top_k=max(ks))

        row = {"question_id": qa["id"], "type": qa["type"]}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(results, chapter, k)
        row["AP"] = average_precision(results, chapter)
        records.append(row)

    df = pd.DataFrame(records)
    # Add aggregate row
    agg = {"question_id": "MEAN", "type": "all"}
    for k in ks:
        agg[f"P@{k}"] = df[f"P@{k}"].mean()
    agg["AP"] = df["AP"].mean()  # = MAP
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    return df


def evaluate_generation(qa_set: List[Dict], full_results: List[Dict]) -> pd.DataFrame:
    """
    Score all generated answers for faithfulness and aggregate latency/cost.
    Args:
        qa_set:       list of QA dicts
        full_results: list of dicts with keys: question_id, retrieved_chunks,
                      answer, latency_seconds, cost_usd, model
    Returns:
        DataFrame with per-question scores and aggregate stats
    """
    records = []
    qa_map = {qa["id"]: qa for qa in qa_set}

    for result in full_results:
        qa = qa_map[result["question_id"]]
        faith = score_faithfulness(
            qa["question"], result["retrieved_chunks"], result["answer"]
        )
        records.append({
            "question_id": result["question_id"],
            "model": result["model"],
            "faithfulness": faith.get("score", -1),
            "faith_reason": faith.get("reason", ""),
            "latency_seconds": result["latency_seconds"],
            "cost_usd": result["cost_usd"],
        })

    df = pd.DataFrame(records)
    return df
