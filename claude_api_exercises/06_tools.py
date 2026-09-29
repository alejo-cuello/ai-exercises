import json
from collections.abc import Callable

from common import MODEL, get_client

def get_weather(city: str) -> dict[str, object]:
    return {"city": city, "temperature": 18, "condition": "lluvia ligera"}

available_tools: dict[str, Callable[..., dict[str, object]]] = {"get_weather": get_weather}

def main() -> None:
    client = get_client()
    tools = [{
        "name": "get_weather",
        "description": "Obtiene el clima actual de una ciudad.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }]

    messages = [{"role": "user", "content": "¿Cómo está el clima en Bogotá?"}]

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

if __name__ == "__main__":
    main()
