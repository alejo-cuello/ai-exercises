import os
from dotenv import load_dotenv
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_HISTORY_MESSAGES = 2


def require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Define ANTHROPIC_API_KEY antes de ejecutar este script.")
    return api_key

def keep_recent_messages(messages: list[dict[str, str]], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict[str, str]]:
    """Conserva solo los últimos turnos para controlar latencia, contexto y costo."""
    return messages[-max_messages:]

def summarize_history(client: Anthropic, messages: list[dict[str, str]]) -> str:
    """Resume la conversación cuando ya no conviene enviar todo el historial."""
    transcript = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system="Resume esta conversación.",
        messages=[{"role": "user", "content": transcript}],
    )
    return "".join(block.text for block in response.content if block.type == "text")

def exec_summarize_example(client: Anthropic) -> None:
    messages = [
        {"role": "user", "content": "Estoy creando un chatbot para soporte técnico."},
        {"role": "assistant", "content": "Perfecto. Lo enfocaremos en respuestas claras."},
        {"role": "user", "content": "El bot debe escalar casos urgentes."},
    ]

    summary = summarize_history(client, messages)
    controlled_history = [{"role": "user", "content": f"Resumen previo: {summary}"}]
    controlled_history.extend(keep_recent_messages(messages))
    controlled_history.append({"role": "user", "content": "¿Qué decisión importante debo recordar?"})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=controlled_history,
    )

    print("Respuesta:")
    print("".join(block.text for block in response.content if block.type == "text"))
    print("\nUso de tokens:")
    print({"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens})

def exec_streaming_example(client: Anthropic) -> None:
    with client.messages.stream(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": "Explícame streaming en Claude API con una analogía."}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("---")


def main() -> None:
    client = Anthropic(api_key=require_api_key())

    # exec_summarize_example(client)
    exec_streaming_example(client)


if __name__ == "__main__":
    main()