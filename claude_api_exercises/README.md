# Claude API exercises

Standalone Python scripts, one per class, that use the Claude API directly
with the Anthropic Python SDK (no LangChain). They end in a FastAPI app that
combines several of them.

Every script imports `common.py`, which loads `ANTHROPIC_API_KEY` from `.env`,
builds the `Anthropic` client, and sets the model all the scripts share
(`claude-haiku-4-5`). Prompts and output are in Spanish.

## Scripts

| Script | What it shows |
| --- | --- |
| `01_summarize_history.py` | Keeping context small: summarizes earlier turns, keeps only the most recent messages, and prints token usage. |
| `02_streaming.py` | Streaming a response token by token with `messages.stream`. |
| `03_chat_history.py` | An interactive terminal chatbot that streams replies and saves the conversation to `history.json` (`/reset` clears it, `/salir` exits). |
| `04_multimedia.py` | Sending a base64-encoded PDF or image (`sample.pdf`) and asking for a summary. |
| `05_json_extraction.py` | Turning invoice text into JSON that matches a Pydantic schema given in the system prompt, then validating it. |
| `06_tools.py` | A single tool call (`get_weather`): the model asks for the tool, the script runs it and sends back a `tool_result`, and the model writes the final answer. |
| `07_agent_loop.py` | An agent loop with a safe `ast`-based calculator tool, capped at `MAX_STEPS` iterations. |
| `08_loop_limit.py` | A local-only sketch (no API call) of a step cap and a tool allowlist with error handling. |
| `09_prompt_caching.py` | Caching a long system prompt with `cache_control` and checking the cache fields in `usage`. |
| `10_batch_processing.py` | Creating a Message Batch, checking its status, and reading results once it has ended. |
| `11_final_exercise.py` | Final project: a FastAPI app with an HTML frontend and three endpoints: `/api/chat`, `/api/extract` (structured output via `messages.parse`), and `/api/agent` (calculator agent). |

## Setup

Create a virtual environment, install the dependencies, and add a `.env` file
in this folder with your API key:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your-key" > .env
```

## Running

Run the scripts from this folder, because they look for `common.py`, `.env`,
`sample.pdf`, and `history.json` here:

```bash
python 02_streaming.py
uvicorn 11_final_exercise:app --reload   # final app at http://127.0.0.1:8000
```
