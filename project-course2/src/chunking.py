# Yakira_S - IR Final Assignment - chunking.py
# Experiment A: Three chunking strategies for RAG preprocessing

import re
import nltk
from typing import List, Dict

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def clean_text(text: str) -> str:
    """Remove PDF artifacts, excessive whitespace, and page numbers."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # lone page numbers
    text = re.sub(r'-\n', '', text)   # dehyphenate line breaks
    return text.strip()


def approximate_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# ── Strategy 1: Fixed-Size ────────────────────────────────────────────────────

def chunk_fixed_size(text: str, chunk_size: int = 256, overlap: int = 50) -> List[Dict]:
    """
    Split text into fixed-size token windows with overlap.
    Args:
        text:       cleaned full document text
        chunk_size: target size in approximate tokens
        overlap:    token overlap between consecutive chunks
    Returns:
        List of dicts: {id, text, strategy, token_count, char_start}
    """
    words = text.split()
    chars_per_token = 4
    word_chunk = chunk_size          # approx words ≈ tokens for English
    word_overlap = overlap

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + word_chunk, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": f"fixed_{chunk_id}",
            "text": chunk_text,
            "strategy": "fixed_size",
            "token_count": approximate_token_count(chunk_text),
            "word_start": start,
        })
        chunk_id += 1
        start += word_chunk - word_overlap

    return chunks


# ── Strategy 2: Sentence-Based ────────────────────────────────────────────────

def chunk_sentence_based(text: str, target_tokens: int = 256) -> List[Dict]:
    """
    Group sentences until reaching target_tokens, then start a new chunk.
    Args:
        text:          cleaned full document text
        target_tokens: soft target size in approximate tokens
    Returns:
        List of dicts with same schema as chunk_fixed_size
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_tokens = 0
    chunk_id = 0

    for sentence in sentences:
        s_tokens = approximate_token_count(sentence)

        if current_tokens + s_tokens > target_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "id": f"sentence_{chunk_id}",
                "text": chunk_text,
                "strategy": "sentence_based",
                "token_count": approximate_token_count(chunk_text),
                "sentence_count": len(current_sentences),
            })
            chunk_id += 1
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += s_tokens

    # flush last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "id": f"sentence_{chunk_id}",
            "text": chunk_text,
            "strategy": "sentence_based",
            "token_count": approximate_token_count(chunk_text),
            "sentence_count": len(current_sentences),
        })

    return chunks


# ── Strategy 3: Paragraph-Based ───────────────────────────────────────────────

def chunk_paragraph_based(text: str, min_tokens: int = 50, max_tokens: int = 512) -> List[Dict]:
    """
    Split on paragraph breaks (double newlines). Merge short paragraphs,
    split overly long ones at sentence boundaries.
    Args:
        text:       cleaned full document text
        min_tokens: paragraphs shorter than this are merged with the next
        max_tokens: paragraphs longer than this are split at sentences
    Returns:
        List of dicts with same schema
    """
    raw_paragraphs = re.split(r'\n\n+', text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # merge short paragraphs
    merged = []
    buffer = ""
    for para in raw_paragraphs:
        if approximate_token_count(buffer + " " + para) < min_tokens:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)

    # split overly long paragraphs
    chunks = []
    chunk_id = 0
    for para in merged:
        if approximate_token_count(para) > max_tokens:
            sub_chunks = chunk_sentence_based(para, target_tokens=max_tokens)
            for sub in sub_chunks:
                sub["id"] = f"para_{chunk_id}"
                sub["strategy"] = "paragraph_based"
                chunks.append(sub)
                chunk_id += 1
        else:
            chunks.append({
                "id": f"para_{chunk_id}",
                "text": para,
                "strategy": "paragraph_based",
                "token_count": approximate_token_count(para),
            })
            chunk_id += 1

    return chunks


# ── Dispatcher ────────────────────────────────────────────────────────────────

def get_chunks(text: str, strategy: str = "fixed", **kwargs) -> List[Dict]:
    """
    Unified interface to all chunking strategies.
    Args:
        text:     cleaned document text
        strategy: one of 'fixed', 'sentence', 'paragraph'
        **kwargs: forwarded to the specific chunker
    """
    strategy = strategy.lower()
    if strategy == "fixed":
        return chunk_fixed_size(text, **kwargs)
    elif strategy == "sentence":
        return chunk_sentence_based(text, **kwargs)
    elif strategy == "paragraph":
        return chunk_paragraph_based(text, **kwargs)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: fixed | sentence | paragraph")
