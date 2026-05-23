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
ir-final-assignment-course3/
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
├── HELPER.md                 # Background reading & concept explanations
├── requirements.txt
└── .env.example              # API key template (never commit .env)
```

## Setup

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/ir-final-assignment-course3.git
cd ir-final-assignment-course3
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
- 🗂️ Index files (FAISS/BM25): *[Google Drive link — add after running Stage 4]*
- 📊 Outputs: *[Google Drive link — add after running Stage 7]*
- 🎥 Presentation video: *[link — add after recording]*

## References
- Manning, C., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.
