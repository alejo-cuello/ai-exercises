import os
from dotenv import load_dotenv
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"


def require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Define ANTHROPIC_API_KEY antes de ejecutar este script.")
    return api_key


def main() -> None:
    client = Anthropic(api_key=require_api_key())
    message = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system="Se breve y conciso, no hagas sugerencias y preguntas a no ser que se te pida explícitamente.",
        messages=[{"role": "user", "content": "Hola"}],
    )


    print("Message crudo: ")
    print(message)

    print("Mensajes separados: ")
    for block in message.content:
        if block.type == "text":
            print(block.text)


if __name__ == "__main__":
    main()