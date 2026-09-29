import json
from pathlib import Path

from anthropic import Anthropic

from common import MODEL, get_client

HISTORY_PATH = Path("history.json")

def load_history() -> list[dict[str, str]]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []

def save_history(messages: list[dict[str, str]]) -> None:
    HISTORY_PATH.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")

def stream_claude_response(client: Anthropic, messages: list[dict[str, str]]) -> str:
    assistant_text = ""
    with client.messages.stream(model=MODEL, max_tokens=500, messages=messages[-3:]) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            assistant_text += text
    print()
    return assistant_text

def main() -> None:
    client = get_client()
    messages = load_history()
    print("Chatbot listo. Comandos: /salir para terminar, /reset para borrar historial.")

    while True:
        user_input = input("\nTú: ").strip()

        if user_input == "/salir":
            break
        if user_input == "/reset":
            messages = []
            save_history(messages)
            print("Historial reiniciado.")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        print("Claude: ", end="", flush=True)

        try:
            assistant_text = stream_claude_response(client, messages)
        except Exception as error:
            messages.pop()
            print(f"Error llamando a Claude: {error}")
            continue

        messages.append({"role": "assistant", "content": assistant_text})
        save_history(messages)

if __name__ == "__main__":
    main()
