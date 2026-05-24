# Introduction to Information Retrieval
Holon Institute of Technology
Assignments and projects for course 67023, covering the full IR pipeline from
classical retrieval models to modern AI-driven architectures.

## Topics Covered
- IR Pipeline and Text Preprocessing — Boolean Model, tokenization, stemming
- Vectorization and Classical Models — Inverted Index, N-grams, BoW, VSM, BM25
- Evaluation — Precision, Recall, MAP, Query Expansion, Relevance Feedback
- Web Search — Crawling, Ranking, Q&A and Recommender Systems
- Non-Textual Data — OCR, Computer Vision, Voice and Speech Processing
- Semantics and Embeddings — Word2Vec, Contextual Embeddings, LLMs, Multimodal LLMs
- Dense Retrieval and RAG — Vector Databases, Re-ranking, Prompt Engineering, Ethics in IR

## Tech Stack
- Language: Python
- Libraries: NLTK / spaCy, scikit-learn, NumPy, Hugging Face Transformers

## Structure

```
notebooks/                        # Jupyter notebooks for weekly assignments
project-course2/      # Final project — Experimental RAG Research
```

## Final Project — Experimental RAG Research

The final project (`project-course2/`) implements a complete **Retrieval-Augmented Generation (RAG)** pipeline over the IR textbook corpus. It systematically experiments with 4 design axes:

| Experiment | Variable | Options |
|---|---|---|
| A | Chunking Strategy | Fixed-size · Sentence · Paragraph |
| B | Embedding Model | MiniLM (local) · OpenAI text-embedding-3-small |
| C | Retrieval Method | Dense (FAISS) · Sparse (BM25) · Hybrid |
| D | LLM Backend | Ollama/llama3 · Claude Sonnet · GPT-4o-mini |

**Key results:** Hybrid retrieval achieved the best MAP (0.883); Claude Sonnet produced the most faithful answers (4.65/5); MiniLM matched OpenAI embeddings at zero cost.

See [`project-course2/README.md`](project-course2/README.md) for full setup and reproduction instructions.

## Reference
Manning, Raghavan & Schutze — An Introduction to Information Retrieval
http://nlp.stanford.edu/IR-book/pdf/irbookprint.pdf
