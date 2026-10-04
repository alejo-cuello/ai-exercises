"""App FastAPI con frontend y tres mini-proyectos del curso en un solo hub."""

from __future__ import annotations

import ast

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from common import MODEL, get_client

MAX_AGENT_STEPS = 4

app = FastAPI(title="Claude API Hub", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


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


class ExtractResponse(BaseModel):
    data: InvoiceData


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)


class AgentResponse(BaseModel):
    answer: str
    steps: list[str]


INVOICE_PROMPT = "Extrae la información de la factura."

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Calculadora aritmética para sumas, restas, multiplicaciones y divisiones.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}

INDEX_HTML = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Claude API Hub</title>
    <style>
      :root {
        color-scheme: light;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #eef2ff;
        color: #172033;
      }
      body {
        margin: 0;
      }
      main {
        width: min(1120px, calc(100% - 32px));
        margin: 0 auto;
        padding: 40px 0;
      }
      header {
        background: linear-gradient(135deg, #172033, #5b4bff);
        border-radius: 28px;
        color: white;
        padding: 32px;
        box-shadow: 0 24px 70px rgba(23, 32, 51, 0.24);
      }
      h1 {
        font-size: clamp(2rem, 5vw, 4rem);
        line-height: 1;
        margin: 0 0 12px;
      }
      .grid {
        display: grid;
        gap: 20px;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        margin-top: 24px;
      }
      section {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(91, 75, 255, 0.14);
        border-radius: 24px;
        box-shadow: 0 18px 50px rgba(91, 75, 255, 0.10);
        padding: 22px;
      }
      label {
        display: block;
        font-weight: 700;
        margin-bottom: 8px;
      }
      textarea, input {
        width: 100%;
        border: 1px solid #c7d2fe;
        border-radius: 16px;
        box-sizing: border-box;
        font: inherit;
        min-height: 120px;
        padding: 14px;
        resize: vertical;
      }
      input {
        min-height: 0;
      }
      button {
        background: #5b4bff;
        border: 0;
        border-radius: 999px;
        color: white;
        cursor: pointer;
        font-weight: 800;
        margin-top: 12px;
        padding: 12px 18px;
      }
      pre {
        background: #0f172a;
        border-radius: 18px;
        color: #dbeafe;
        min-height: 120px;
        overflow: auto;
        padding: 16px;
        white-space: pre-wrap;
      }
    </style>
  </head>
  <body>
    <main>
      <header>
        <p>Clase 17 · Frontend + FastAPI</p>
        <h1>Todos los proyectos del curso en una sola app</h1>
        <p>Un frontend llama tres endpoints: chatbot, extractor JSON y agente con herramienta calculadora.</p>
      </header>

      <div class="grid">
        <section>
          <h2>Chatbot</h2>
          <label for="chat-message">Mensaje</label>
          <textarea id="chat-message">Dame 3 ideas para practicar Claude API.</textarea>
          <button data-run="chat">Enviar</button>
          <pre id="chat-output">Respuesta pendiente...</pre>
        </section>

        <section>
          <h2>Extractor JSON</h2>
          <label for="invoice-text">Factura</label>
          <textarea id="invoice-text">Factura de ACME S.A. emitida el 2026-05-01. 2 horas de consultoría a 50 USD cada una. Total: USD 100.</textarea>
          <button data-run="extract">Extraer</button>
          <pre id="extract-output">JSON pendiente...</pre>
        </section>

        <section>
          <h2>Agente con tools</h2>
          <label for="agent-question">Pregunta</label>
          <input id="agent-question" value="Calcula (128 * 7) + 34 y explica el resultado." />
          <button data-run="agent">Resolver</button>
          <pre id="agent-output">Respuesta pendiente...</pre>
        </section>
      </div>
    </main>

    <script>
      async function postJson(path, body) {
        const response = await fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.error || JSON.stringify(data));
        }
        return data;
      }

      function show(id, value) {
        document.getElementById(id).textContent =
          typeof value === 'string' ? value : JSON.stringify(value, null, 2);
      }

      document.querySelector('[data-run="chat"]').addEventListener('click', async () => {
        show('chat-output', 'Cargando...');
        try {
          const data = await postJson('/api/chat', {
            message: document.getElementById('chat-message').value,
          });
          show('chat-output', data.reply);
        } catch (error) {
          show('chat-output', error.message);
        }
      });

      document.querySelector('[data-run="extract"]').addEventListener('click', async () => {
        show('extract-output', 'Cargando...');
        try {
          const data = await postJson('/api/extract', {
            text: document.getElementById('invoice-text').value,
          });
          show('extract-output', data.data);
        } catch (error) {
          show('extract-output', error.message);
        }
      });

      document.querySelector('[data-run="agent"]').addEventListener('click', async () => {
        show('agent-output', 'Cargando...');
        try {
          const data = await postJson('/api/agent', {
            question: document.getElementById('agent-question').value,
          });
          show('agent-output', data);
        } catch (error) {
          show('agent-output', error.message);
        }
      });
    </script>
  </body>
</html>
"""

def response_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def evaluate_expression(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return evaluate_expression(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.UnaryOp):
        value = evaluate_expression(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    raise ValueError("Expresión no permitida.")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate_expression(tree))
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return f"Expresión rechazada: {error}"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": payload.message}],
    )
    return ChatResponse(reply=response_text(response))


@app.post("/api/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest) -> ExtractResponse:
    response = get_client().messages.parse(
        model=MODEL,
        max_tokens=700,
        system=INVOICE_PROMPT,
        messages=[{"role": "user", "content": payload.text}],
        output_format=InvoiceData,
    )
    if response.parsed_output is None:
        raise HTTPException(status_code=502, detail="Claude no devolvió una factura válida.")
    return ExtractResponse(data=response.parsed_output)


@app.post("/api/agent", response_model=AgentResponse)
def agent(payload: AgentRequest) -> AgentResponse:
    client = get_client()
    messages = [{"role": "user", "content": payload.question}]
    steps: list[str] = []

    for _step in range(MAX_AGENT_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=700,
            tools=[CALCULATOR_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use" and block.name == "calculator":
                expression = ""
                if isinstance(block.input, dict):
                    input_expression = block.input.get("expression")
                    if isinstance(input_expression, str):
                        expression = input_expression
                result = calculator(expression)
                steps.append(f"calculator({expression}) -> {result}")
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        if not tool_results:
            return AgentResponse(answer=response_text(response), steps=steps)
        messages.append({"role": "user", "content": tool_results})

    raise HTTPException(status_code=504, detail="El agente alcanzó el límite de pasos.")
