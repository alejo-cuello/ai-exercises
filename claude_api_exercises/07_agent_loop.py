import ast
import operator

from common import MODEL, get_client

MAX_STEPS = 4
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}

def evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluate(node.operand))
    raise ValueError("Expresión no permitida.")

def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree))
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return f"Expresión rechazada: {error}"

def main() -> None:
    client = get_client()
    messages = [{"role": "user", "content": "Calcula (128 * 7) + 34 y explica el resultado."}]
    tools = [{"name": "calculator", "description": "Calculadora aritmética.", "input_schema": {
        "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}]

    for _step in range(MAX_STEPS):
        response = client.messages.create(model=MODEL, max_tokens=500, tools=tools, messages=messages)
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": calculator(str(block.input["expression"]))})
        if not tool_results:
            print("".join(block.text for block in response.content if block.type == "text"))
            return
        messages.append({"role": "user", "content": tool_results})
        print(f"Herramientas usadas: {tool_results}")

    print("El agente alcanzó el límite de pasos.")

if __name__ == "__main__":
    main()
