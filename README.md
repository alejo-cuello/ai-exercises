# ai-exercises

Exercises and a final project built while learning RAG (retrieval-augmented
generation) techniques with LangChain, plus a set of scripts exploring the
Claude API directly with the Anthropic Python SDK.

## Demo

[See how it works](https://youtu.be/t6Er_Ep7lXg)

## Credits

- Inspired by these Platzi courses:
  - [Langchain](https://platzi.com/cursos/langchain-chatbots/)
  - [Langchain for information management and retrieval](https://platzi.com/cursos/langchain-documents/)
  - [Claude API](https://platzi.com/cursos/claude-api/)

## Structure

- [`first_exercises/`](first_exercises/) — notebooks progressing through
  RAG building blocks: chapter-based splitting, parent-document retrieval,
  self-querying retrieval, multi-query retrieval, ensemble retrieval,
  semantic reranking, MMR reranking, chains, open-source LLMs/embeddings,
  and chat with memory. I learned through Platzi courses.
- [`final_project/`](final_project/) — a deployed chatbot: Streamlit UI,
  Postgres-backed auth/conversations, and a LangChain + Chroma retrieval
  chain, containerized and deployed to Cloud Run via Cloud Build. See
  [`final_project/README.md`](final_project/README.md) for setup,
  environment variables, and deployment instructions.
- [`claude_api_exercises/`](claude_api_exercises/) — standalone Python
  scripts, one per class, that use the Claude API without LangChain:
  messages, streaming, chat history, PDFs/images, JSON extraction, tool use,
  agent loops, prompt caching, and batches. They end in a FastAPI app that
  combines several of them. See
  [`claude_api_exercises/README.md`](claude_api_exercises/README.md) for a
  description of each script and how to run them.

## Setup

Each part has its own `requirements.txt` (`first_exercises/requirements.txt`,
`final_project/requirements.txt`, `claude_api_exercises/requirements.txt`).
Create a virtual environment and install the one for the part you're working on:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r first_exercises/requirements.txt   # or final_project/ or claude_api_exercises/
```

Every part expects API keys via a `.env` file in its own folder. For
`claude_api_exercises/`, that's `ANTHROPIC_API_KEY`.
