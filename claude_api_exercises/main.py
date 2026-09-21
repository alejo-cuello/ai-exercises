import os
import json
import base64
import mimetypes

from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic
from pydantic import BaseModel
from collections.abc import Callable

MODEL = "claude-haiku-4-5-20251001"
MAX_HISTORY_MESSAGES = 2
HISTORY_PATH = Path("history.json")

def get_weather(city: str) -> dict[str, object]:
    return {"city": city, "temperature": 18, "condition": "lluvia ligera"}

available_tools: dict[str, Callable[..., dict[str, object]]] = {"get_weather": get_weather}

class InvoiceItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    total: float

class InvoiceData(BaseModel):
    provider: str
    date: str
    currency: str
    total: float
    items: list[InvoiceItem]

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

def encode_file(path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return media_type, data

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

def exec_story_example(client: Anthropic) -> None:
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

def exec_multimedia_example(client: Anthropic) -> None:
    file_path = Path("sample.pdf")
    if not file_path.exists():
        raise FileNotFoundError("Agrega un archivo sample.pdf junto a este script.")

    media_type, data = encode_file(file_path)
    content_type = "document" if media_type == "application/pdf" else "image"

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {"type": content_type, "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": "Resume el contenido principal en 2 bullets."},
            ],
        }],
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)

def strip_json_fence(text: str) -> str:
    """Quita el bloque de código markdown (```json ... ```) si el modelo lo agrega."""
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return text

def exec_json_example(client: Anthropic) -> None:

    # Ejemplo 1
    # invoice_text = """
    # Factura de ACME S.A. emitida el 2026-05-01.
    # 2 horas de consultoría a 50 USD cada una. Total: USD 100.
    # """

    # Ejemplo 2
    invoice_text = "Factura ACME emitida el 2026-05-01. Servicio: soporte, total: USD 129.90. Impuesto del 10%"

    schema = json.dumps(InvoiceData.model_json_schema(), ensure_ascii=False)

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system= f"""
            Extrae la información de la factura.
            Reglas:
            - No inventes datos. Si un campo no aparece, usa null.
            - Normaliza montos como números, sin símbolos de moneda.
            - La moneda debe ser un código ISO si puedes inferirlo.
            - Si la factura no muestra moneda explícita, usa null.
            - Si hay impuestos separados, inclúyelos en items solo si aparecen como línea propia.
            - Si no puedes leer un campo, usa null en lugar de adivinar.
            - No agregues explicación fuera del JSON.
            - Responde únicamente JSON válido que siga este JSON Schema:
            {schema}
            """,
        messages=[{"role": "user", "content": invoice_text}],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text")
    print(f"raw text: \n {raw_text}")
    json_text = strip_json_fence(raw_text)
    invoice = InvoiceData.model_validate(json.loads(json_text))
    print(f"json: \n {invoice.model_dump_json(indent=2)}")

def exec_tools_example(client: Anthropic) -> None:
    tools = [{
        "name": "get_weather",
        "description": "Obtiene el clima actual de una ciudad.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }]

    messages=[{"role": "user", "content": "¿Cómo está el clima en Bogotá?"}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        tools=tools,
        messages=messages,
    )

    messages.append({"role": "assistant", "content": response.content})

    for block in response.content:
        if block.type == "tool_use":
            tool = available_tools[block.name]
            result = tool(**block.input)
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False),
            }]})

    print(f"messages: \n {messages}")

    final = client.messages.create(model=MODEL, max_tokens=500, messages=messages)
    print("".join(block.text for block in final.content if block.type == "text"))

def main() -> None:
    client = Anthropic(api_key=require_api_key())

    # exec_summarize_example(client)
    # exec_streaming_example(client)
    # exec_story_example(client)
    # exec_multimedia_example(client)
    # exec_json_example(client)
    exec_tools_example(client)

if __name__ == "__main__":
    main()