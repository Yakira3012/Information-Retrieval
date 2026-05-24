# IR Final Assignment Course 2 — Complete Summary

## What was built
A full experimental RAG pipeline over an IR textbook corpus (`irbook.pdf`), comparing 4 components across 20 QA questions.

---

## Experiments run

### A — Chunking strategies
- Fixed-size (256 tokens, 50 overlap), Sentence-based, Paragraph-based
- Winner: **Paragraph** — best semantic coherence for structured textbook content

### B — Embedding models
- MiniLM-L6-v2 (local, 384-dim) vs OpenAI text-embedding-3-small (1536-dim)
- Winner: **MiniLM** — free, fast, competitive retrieval quality

### C — Retrieval methods (20 QA questions, P@K + MAP)

| Retriever | P@3 | P@5 | MAP |
|---|---|---|---|
| Dense (FAISS) | 0.800 | 0.820 | 0.846 |
| Sparse (BM25) | 0.700 | 0.730 | 0.839 |
| **Hybrid** | **0.800** | 0.770 | **0.883** |

### D — LLM backends (faithfulness scored by Claude Haiku as judge)

| Backend | Faithfulness | Latency | Cost |
|---|---|---|---|
| **Claude Sonnet** | **4.65/5** | 6.2s | $0.160 |
| GPT-4o-mini | 4.60/5 | 1.5s | $0.004 |
| Ollama/llama3 | 4.20/5 | 102s | $0.000 |

---

## Best pipeline
**Paragraph → MiniLM → Hybrid → Claude Sonnet** (faithfulness=4.65, latency~6s)

---

## Artifacts produced
- `outputs/charts/chunk_distributions.png` — Exp A chunk size histograms
- `outputs/charts/retrieval_comparison.png` — Exp C P@3/P@5/MAP bar chart
- `outputs/charts/llm_comparison.png` — Exp D faithfulness/latency/cost chart
- `outputs/embedding_comparison.csv` — Exp B model stats
- `outputs/retrieval_results.csv` — Exp C per-question metrics
- `outputs/generation_results.csv` — Exp D all 60 answers with faithfulness
- `outputs/best_pipeline_demo.csv` — Section 6 end-to-end demo (5 questions)
- `rag_experiments.ipynb` — full notebook with conclusions
- `src/` — chunking.py, embeddings.py, retrieval.py, generation.py, evaluation.py

## Issues fixed
- PyTorch CPU/CUDA DLL conflict → downgraded to `torch==2.3.1+cpu`
- Faithfulness scoring returning -1 → Claude Haiku wraps JSON in markdown fences; fixed by stripping ` ```json ``` ` before parsing in `src/evaluation.py`

## Git
Commit `db36bdb` on `main` — 20 files. Large binaries excluded via `.gitignore`.

## Remaining for submission
1. `git push` to GitHub
2. Upload `index/` and `data/irbook.pdf` to Google Drive
