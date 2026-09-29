from common import MODEL, get_client

def main() -> None:
    client = get_client()
    long_policy = "Reglas internas del asistente. " * 400

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=[{"type": "text", "text": long_policy, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "Resume las 3 reglas principales."}],
    )
    print("".join(block.text for block in response.content if block.type == "text"))
    print(response.usage)

if __name__ == "__main__":
    main()
