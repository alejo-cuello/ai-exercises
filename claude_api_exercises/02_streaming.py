from common import MODEL, get_client

def main() -> None:
    client = get_client()
    with client.messages.stream(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": "Explícame streaming en Claude API con una analogía."}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("---")

if __name__ == "__main__":
    main()
