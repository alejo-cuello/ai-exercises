MAX_STEPS = 4

def get_weather(city: str) -> dict[str, object]:
    return {"city": city, "temperature": 18, "condition": "lluvia ligera"}

available_tools = {"get_weather": get_weather}

def run_tool(name: str, args: dict[str, object]) -> str:
    if name not in available_tools:
        return "Error: herramienta no permitida."

    try:
        return str(available_tools[name](**args))
    except Exception as error:
        return f"Error ejecutando {name}: {error}"

def main() -> None:
    for step in range(MAX_STEPS):
        if step == MAX_STEPS - 1:
            print("El agente alcanzó el máximo de pasos permitidos.")
            # break
        print(run_tool("get_weather", {"city": "Bogotá"}))
        # break

if __name__ == "__main__":
    main()
