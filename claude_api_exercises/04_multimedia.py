import base64
import mimetypes
from pathlib import Path

from common import MODEL, get_client

def encode_file(path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return media_type, data

def main() -> None:
    client = get_client()
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

if __name__ == "__main__":
    main()
