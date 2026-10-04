import json

from pydantic import BaseModel

from common import MODEL, get_client

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

def strip_json_fence(text: str) -> str:
    """Quita el bloque de código markdown (```json ... ```) si el modelo lo agrega."""
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return text

def main() -> None:
    client = get_client()

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
        system=f"""
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

if __name__ == "__main__":
    main()
