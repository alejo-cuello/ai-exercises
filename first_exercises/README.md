# first_exercises

Notebooks progressing through RAG (retrieval-augmented generation) building
blocks with LangChain, using the *Speech and Language Processing* book
chapters (`data/raw/`) as the source corpus. I learned through Platzi courses.

## Notebooks

- [`01_splitting_by_chapter.ipynb`](01_splitting_by_chapter.ipynb) — splitting
  PDFs into chapter-aware chunks (plus a variant testing a custom
  `length_function` for the splitter).
- [`02_parent_retriever.ipynb`](02_parent_retriever.ipynb) — parent-document
  retrieval: index small chunks, return their larger parent documents.
- [`03_self_retriever.ipynb`](03_self_retriever.ipynb) — self-querying
  retriever that turns natural-language questions into structured metadata
  filters (plus a storage variant).
- [`04_multi_query_retriever.ipynb`](04_multi_query_retriever.ipynb) —
  generating multiple query rephrasings to broaden retrieval recall.
- [`05_ensemble_retriever.ipynb`](05_ensemble_retriever.ipynb) — combining
  BM25 and vector retrievers into a single ensemble.
- [`06_semantic_reranking.ipynb`](06_semantic_reranking.ipynb) — reranking
  retrieved documents by semantic relevance.
- [`07_maximum_marginal_relevance_reranking.ipynb`](07_maximum_marginal_relevance_reranking.ipynb)
  — MMR reranking to reduce redundancy ("lost in the middle") in results.
- [`08_chains.ipynb`](08_chains.ipynb) — foundational and sequential LangChain
  chains built on top of the Chroma-backed retriever.
- [`09_open_source_llm_and_embeddings.ipynb`](09_open_source_llm_and_embeddings.ipynb)
  — swapping in open-source LLMs and embedding models (e.g. Hugging Face)
  instead of hosted providers.
- [`10_chat_with_memory.ipynb`](10_chat_with_memory.ipynb) — adding
  conversation memory (windowing, trimming, summarization) to a chat chain.

## Data

- `data/raw/` — source PDFs used across the notebooks; `data/raw-unused/`
  holds chapters not currently referenced.
- `data/preprocessed/` — cached intermediate output (e.g. tagged chunks).
- `data/vectors/` — persisted vector stores (Chroma DB files).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Expects a `.env` file in this folder with the API keys used by the
notebooks (LLM/embeddings provider).
