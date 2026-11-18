"""Part 2 of the GenAI app: the MODEL CALL.

This is the only file that talks to a language model. It sends a list of
messages to Ollama (running locally on this Mac) and yields the answer word by
word, so the app can show it as it is written ("streaming").

To use a different model, such as Claude, you would only change this file:
the app and the context stay the same.
"""
import json
from collections.abc import Iterator

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"


def chat(messages: list[dict], model: str = MODEL, temperature: float = 0.2) -> Iterator[str]:
    """Send messages and yield the reply in small pieces.

    messages: [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
    temperature: 0 = always the most likely words (factual), 1 = more varied (creative).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            yield chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                break


def is_available(model: str = MODEL) -> bool:
    """True if Ollama is running and the model is installed."""
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=3).json()
    except requests.RequestException:
        return False
    return any(m["name"] == model for m in tags.get("models", []))
