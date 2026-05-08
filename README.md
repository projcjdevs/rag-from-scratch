# RAG from Scratch

A minimal, complete Retrieval-Augmented Generation system
built using only NumPy — no FAISS, no LangChain, no Scikit-learn.

## What This Is

Most RAG tutorials hand you a library and tell you to call `.search()`.
This project implements every component from mathematical first principles
so you understand what those libraries are doing internally.

## What's Implemented

| Component | Sprint | File |
|---|---|---|
| L2 Norm + Normalization | Sprint 1 | `src/rag.py` |
| Cosine Similarity | Sprint 2 | `src/rag.py` |
| PCA Visualisation | Sprint 3 | `src/rag.py` |
| K-Means + IVF Search | Sprint 4 | `src/rag.py` |
| Adjacency Matrix + Graph RAG | Sprint 5 | `src/rag.py` |
| Hybrid Retrieval | Sprint 5 | `src/rag.py` |

## Stack

- `numpy` — all math and matrix operations
- `sentence-transformers` — raw embedding vectors only
- `matplotlib` + `seaborn` — visualisation

## Usage

```python
from src.rag import LightweightRAG

rag = LightweightRAG(n_clusters=3)
rag.index(documents)

# Vector search
results = rag.search("how do neural networks learn?", top_k=3)

# Hybrid graph search
results = rag.graph_search("how do neural networks learn?", hops=2)

# Visualise the document space
rag.visualise(query="how do neural networks learn?")
```

## Project Structure
rag-from-scratch/
├── notebooks/          ← sprint walkthroughs with math + code
│   └── demo.ipynb      ← clean end-to-end demo
├── src/
│   └── rag.py          ← the complete model
├── data/
│   └── documents.txt   ← sample corpus   
└── README.md

## Key Ideas

- Normalization makes dot product equivalent to cosine similarity
- IVF search reduces O(n) to O(√n) by routing queries to clusters
- Graph traversal recovers relationship chains vector search misses
- Hybrid retrieval combines both: graph filters, vector ranks