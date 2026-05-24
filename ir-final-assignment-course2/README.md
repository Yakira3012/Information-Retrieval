# IR Final Assignment — Experimental RAG Research
### Course: Introduction to Information Retrieval (67023), HIT — Spring 2026
### Author: Yakira S.

---

## Overview

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** system using the Manning et al. *Introduction to Information Retrieval* textbook as the corpus. The goal is to systematically compare different RAG components and measure how each design choice affects retrieval quality and answer generation.

## Experiments

| # | Variable | Options Compared |
|---|----------|-----------------|
| A | Chunking Strategy | Fixed-size · Sentence-based · Paragraph-based |
| B | Embedding Model | `all-MiniLM-L6-v2` · `text-embedding-3-small` (OpenAI) |
| C | Retrieval Method | Dense (FAISS) · Sparse (BM25) · Hybrid |
| D | LLM Backend | `llama3` (local) · `claude-sonnet-4-6` · `gpt-4o-mini` |

## Project Structure

```
ir-final-assignment-course2/
├── rag_experiments.ipynb     # Main experiment notebook
├── src/
│   ├── chunking.py           # Experiment A — chunking strategies
│   ├── embeddings.py         # Experiment B — embedding models
│   ├── retrieval.py          # Experiment C — retrieval methods
│   ├── generation.py         # Experiment D — LLM generation
│   └── evaluation.py         # Metrics: P@K, faithfulness, latency
├── outputs/
│   ├── qa_set.json           # 20 evaluation questions
│   ├── results_table.csv     # All experiment results
│   └── charts/               # Generated visualizations
├── paper/
│   └── research_paper.docx   # ACL-format research write-up
├── scripts/
│   └── setup_ollama.sh       # One-time Ollama/llama3 setup
├── requirements.txt
└── .env.example              # API key template (never commit .env)
```

## Setup

### 0. Prerequisites

Before cloning, make sure you have the following:

**API keys & funds**
- **Anthropic API key** — create one at [console.anthropic.com](https://console.anthropic.com). Used for Claude Sonnet (Experiment D) and Claude Haiku (faithfulness judge). Add a small credit balance (~$1–2 is enough for one full run).
- **OpenAI API key** — create one at [platform.openai.com](https://platform.openai.com). Used for `text-embedding-3-small` (Experiment B) and `gpt-4o-mini` (Experiment D). Add credits (~$0.50 is enough for one full run).

**Tools to install**
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| pip | latest | Package manager |
| Jupyter | any | Running the notebook — `pip install jupyter` |
| Ollama | latest | Local LLM (Experiment D) — [ollama.com/download](https://ollama.com/download) |
| Git | any | Cloning the repo |

> **Ollama note:** Ollama is optional — if you skip it, Experiment D will only run the Claude and OpenAI backends. It works best on a machine with a GPU; on CPU it is very slow (~100s per query).

---

### 1. Clone & install
```bash
git clone https://github.com/Yakira3012/Information-Retrieval.git
cd Information-Retrieval/ir-final-assignment-course2
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your OpenAI and Anthropic keys
```

### 3. Add the corpus
Download the IR textbook PDF and place it at:
```
data/irbook.pdf
```
Link: http://nlp.stanford.edu/IRbook/pdf/irbookprint.pdf

### 4. Set up local LLM (for Experiment D — GPU PC only)
```bash
bash scripts/setup_ollama.sh
```

### 5. Run the notebook
```bash
jupyter notebook rag_experiments.ipynb
```

## Links
- 📓 Notebook: *(this repo)*
- 🗂️ Index files (FAISS/BM25 + embeddings pkl): [Google Drive](https://drive.google.com/drive/folders/1pZ95Gpqa4cO5RubbUtlqzG6qJ3poYxsn?usp=drive_link)
- 📊 Outputs (embeddings, chunking pkl): [Google Drive](https://drive.google.com/drive/folders/1lbAqrMuUccCIQzb6k1YTD4_OFu7xuJM_?usp=drive_link)
- 🎥 Presentation video: *TBD*

## References
- Manning, C., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.
