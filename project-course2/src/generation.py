# Yakira_S - IR Final Assignment - generation.py
# Experiment D: LLM generation backends — Ollama, Claude, GPT-4o-mini

import os
import time
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

RAG_PROMPT_TEMPLATE = """You are an expert in Information Retrieval. Use ONLY the provided context to answer the question.
If the answer is not contained in the context, respond with: "I don't know based on the provided context."
Do not add information from outside the context.

Context:
{context}

Question: {question}

Answer:"""


def _build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{r['chunk']['text']}"
        for i, r in enumerate(retrieved_chunks)
    )
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


# ── Backend 1: Ollama (local llama3) ─────────────────────────────────────────

def generate_ollama(question: str, retrieved_chunks: List[Dict],
                    model: str = "llama3") -> Dict:
    """
    Generate answer using a local Ollama model.
    Requires Ollama running: `ollama serve` and model pulled: `ollama pull llama3`
    """
    import ollama

    prompt = _build_prompt(question, retrieved_chunks)
    t0 = time.time()
    response = ollama.generate(model=model, prompt=prompt)
    elapsed = time.time() - t0

    return {
        "answer": response["response"].strip(),
        "model": f"ollama/{model}",
        "latency_seconds": round(elapsed, 3),
        "cost_usd": 0.0,
        "prompt_tokens": None,   # Ollama doesn't always expose this
        "completion_tokens": None,
    }


# ── Backend 2: Claude (Anthropic API) ────────────────────────────────────────

def generate_claude(question: str, retrieved_chunks: List[Dict],
                    model: str = "claude-sonnet-4-6") -> Dict:
    """
    Generate answer using Claude via Anthropic API.
    Requires ANTHROPIC_API_KEY in environment.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = _build_prompt(question, retrieved_chunks)

    t0 = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - t0

    answer = message.content[0].text.strip()
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    # Pricing: claude-sonnet-4 ~$3/M input, $15/M output
    cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0

    return {
        "answer": answer,
        "model": model,
        "latency_seconds": round(elapsed, 3),
        "cost_usd": round(cost, 6),
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }


# ── Backend 3: GPT-4o-mini (OpenAI API) ──────────────────────────────────────

def generate_openai(question: str, retrieved_chunks: List[Dict],
                    model: str = "gpt-4o-mini") -> Dict:
    """
    Generate answer using GPT-4o-mini via OpenAI API.
    Requires OPENAI_API_KEY in environment.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = _build_prompt(question, retrieved_chunks)

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    elapsed = time.time() - t0

    answer = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # Pricing: gpt-4o-mini ~$0.15/M input, $0.60/M output
    cost = (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60

    return {
        "answer": answer,
        "model": model,
        "latency_seconds": round(elapsed, 3),
        "cost_usd": round(cost, 6),
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: List[Dict],
                    backend: str = "claude") -> Dict:
    """
    Unified generation interface.
    Args:
        question:         the user's question string
        retrieved_chunks: list of retrieval result dicts (from retrieval.py)
        backend:          'ollama' | 'claude' | 'openai'
    """
    backend = backend.lower()
    if backend == "ollama":
        return generate_ollama(question, retrieved_chunks)
    elif backend == "claude":
        return generate_claude(question, retrieved_chunks)
    elif backend == "openai":
        return generate_openai(question, retrieved_chunks)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: ollama | claude | openai")
