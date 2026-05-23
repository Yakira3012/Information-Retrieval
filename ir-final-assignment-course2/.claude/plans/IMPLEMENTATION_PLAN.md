# IR Final Assignment — Full Implementation Plan for Claude Code
# Course: Introduction to Information Retrieval (67023), HIT Spring 2026
# Author: Yakira_S

## Context
This is a university final assignment implementing an Experimental RAG (Retrieval-Augmented Generation) 
research system in Python. The goal is to compare 4 RAG components and measure how each affects 
retrieval quality and answer generation quality, using the Manning et al. IR textbook as the corpus.

## Repo structure (already scaffolded)
ir-final-assignment-course3/
├── rag_experiments.ipynb        # Main notebook — all experiments run here
├── src/
│   ├── chunking.py              # Experiment A: 3 chunking strategies
│   ├── embeddings.py            # Experiment B: 2 embedding models
│   ├── retrieval.py             # Experiment C: FAISS / BM25 / Hybrid
│   ├── generation.py            # Experiment D: Ollama / Claude / GPT-4o-mini
│   └── evaluation.py            # P@K, faithfulness scoring, latency
├── outputs/
│   ├── qa_set.json              # 20 evaluation questions (already written)
│   └── charts/                  # PNG charts go here
├── data/
│   └── irbook.pdf               # IR textbook corpus (already placed)
├── index/
│   ├── faiss_index/             # Generated at runtime
│   └── bm25_index/              # Generated at runtime
├── scripts/
│   └── setup_ollama.sh          # Ollama setup (run once on GPU machine)
├── HELPER.md                    # Full background and concept explanations
├── README.md
├── requirements.txt
└── .env                         # Contains OPENAI_API_KEY and ANTHROPIC_API_KEY

## What is already done
- All src/ modules are fully written and ready
- Notebook is structured with all sections and cells
- 20 QA evaluation questions written in outputs/qa_set.json
- Git repo initialized and pushed to GitHub
- data/irbook.pdf placed in data/ folder
- .env created with both API keys

## What you need to do
Work through the following steps in order. Complete each step fully before moving to the next.
Fix all errors in the src/ files directly — do not work around them in the notebook.
After each step tell me: status, key result, and any number I should note.

---

## STEP 1 — Environment Setup
- Run pip install -r requirements.txt
- Verify .env loads and both API keys are readable
- Verify data/irbook.pdf exists and is accessible
- Run notebook Section 0 (Setup cell)
- Expected output: "API keys loaded", "Corpus found: data/irbook.pdf"
- Fix any missing dependency or import errors in requirements.txt or src/ files

---

## STEP 2 — Data Pipeline
- Run notebook Section 1
- Load irbook.pdf with pdfplumber, extract all page text
- Clean with chunking.clean_text() — remove page numbers, fix hyphenation, normalize whitespace
- Expected output: character count ~2M+, clean sample text with no artifacts
- If pdfplumber fails on any pages, handle gracefully and continue

---

## STEP 3 — Experiment A: Chunking Strategies
- Run notebook Section 2
- Run all 3 strategies on the cleaned full text:
  - fixed: 256 tokens, 50 overlap
  - sentence: target 256 tokens per chunk
  - paragraph: min 50, max 512 tokens
- Print stats table: chunk count, mean/median/min/max token sizes per strategy
- Generate outputs/charts/chunk_distributions.png (3-panel histogram)
- Commit chart to GitHub
- Report: which strategy has the most consistent chunk sizes, which has the most chunks

---

## STEP 4 — Experiment B: Embedding Models
- Run notebook Section 3
- Use paragraph chunks as the embedding corpus (best semantic boundaries)
- Embed with MiniLM (local): record time and confirm 384-dim output
- Embed with OpenAI text-embedding-3-small: record time, tokens used, cost in USD
- Save outputs/embedding_comparison.csv
- Run qualitative similarity test on 4 sample IR questions, print top-3 chunks per query
- Report: time difference, cost, and any noticeable quality difference in similarity results

---

## STEP 5 — Experiment C: Retrieval Methods
- Run notebook Section 4
- Build DenseRetriever (FAISS) from MiniLM embeddings, save to index/faiss_index/
- Build SparseRetriever (BM25) from paragraph chunk texts, save to index/bm25_index/
- Build HybridRetriever (alpha=0.5) combining both
- Run evaluate_retrieval() over all 20 QA questions for each retriever
- Metrics: P@3, P@5, Average Precision
- Save outputs/retrieval_results.csv
- Generate outputs/charts/retrieval_comparison.png (grouped bar chart)
- Commit chart and CSV to GitHub
- Report: P@3 and P@5 for each retriever, which one wins and by what margin

---

## STEP 6 — Experiment D: LLM Generation
- Run notebook Section 5
- Use the best retriever from Step 5 (or hybrid if tied) with top-3 chunks
- Run all 20 QA questions through 3 backends:
  - ollama/llama3 (local — tell me if Ollama needs to be started manually)
  - claude-sonnet-4-20250514 (Anthropic API)
  - gpt-4o-mini (OpenAI API)
- Score all answers with evaluate_generation() using Claude Haiku as faithfulness judge
- Save outputs/generation_results.csv
- Generate outputs/charts/llm_comparison.png (faithfulness bar + latency/cost scatter)
- Commit chart and CSV to GitHub
- Report: faithfulness score, avg latency, and total cost per backend

---

## STEP 7 — Best Pipeline Assembly
- Run notebook Section 6
- Based on results from Steps 3-6, update these variables with the winning config:
  BEST_CHUNKING   = "paragraph" or "fixed" or "sentence"
  BEST_EMBEDDING  = "minilm" or "openai"
  BEST_RETRIEVER  = dense or sparse or hybrid
  BEST_LLM        = "claude" or "openai" or "ollama"
- Run the 3 demo questions end-to-end through the best pipeline
- Confirm clean output: question, answer, latency, cost, source chunk IDs printed
- Report: the winning config combination and why

---

## STEP 8 — Final Results & Export
- Run notebook Section 7
- Print final summary tables for retrieval and generation
- Confirm all 4 charts exist in outputs/charts/:
  - chunk_distributions.png
  - retrieval_comparison.png
  - llm_comparison.png
  - (any additional chart generated)
- Confirm all CSVs exist in outputs/:
  - embedding_comparison.csv
  - retrieval_results.csv
  - generation_results.csv
- Save notebook with all outputs visible (all cells run, outputs showing)

---

## STEP 9 — Cleanup & Submission Prep
- Verify .gitignore is correct — these must NOT be in git:
  - .env
  - data/
  - index/
- These MUST be in git:
  - All src/ files
  - rag_experiments.ipynb (with outputs)
  - outputs/qa_set.json
  - outputs/charts/*.png
  - outputs/*.csv
  - README.md, HELPER.md, requirements.txt
- Commit everything clean with message: "Final experiments complete — Yakira_S"
- Push to GitHub
- Tell me exactly:
  - Which folders to upload to Google Drive (data/, index/, any large outputs)
  - The folder structure to use on Drive
  - What links to collect for the submission text box

---

## Rules for working with me
- Fix errors in src/ files directly, never patch around them in the notebook
- All modified src/ files must keep the # Yakira_S header at line 1
- If you need me to do something manually (start Ollama, check a key, upload a file) 
  say so with a clear "ACTION NEEDED:" label and exact instructions
- One step at a time — do not jump ahead
- If a step produces unexpected results (very low P@K, API error, empty output) 
  stop and explain before continuing
- Start with STEP 1 now