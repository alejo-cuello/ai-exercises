# ai-exercises

Exercises and a final project built while learning RAG (retrieval-augmented
generation) techniques with LangChain.

## Structure

- [`first_exercises/`](first_exercises/) — notebooks progressing through
  RAG building blocks: chapter-based splitting, parent-document retrieval,
  self-querying retrieval, multi-query retrieval, ensemble retrieval,
  semantic reranking, MMR reranking, chains, open-source LLMs/embeddings,
  and chat with memory.
- [`final_project/`](final_project/) — a deployed chatbot: Streamlit UI,
  Postgres-backed auth/conversations, and a LangChain + Chroma retrieval
  chain, containerized and deployed to Cloud Run via Cloud Build. See
  [`final_project/README.md`](final_project/README.md) for setup,
  environment variables, and deployment instructions.

## Setup

Each part has its own `requirements.txt` (`first_exercises/requirements.txt`,
`final_project/requirements.txt`). Create a virtual environment and install
the one for the part you're working on:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r first_exercises/requirements.txt   # or final_project/requirements.txt
```

Both parts expect API keys (LLM/embeddings provider) via a `.env` file in
their respective folders.
