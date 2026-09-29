import os

from dotenv import load_dotenv
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

def require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Define ANTHROPIC_API_KEY antes de ejecutar este script.")
    return api_key

def get_client() -> Anthropic:
    return Anthropic(api_key=require_api_key())
