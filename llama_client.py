# llama_client.py
from __future__ import annotations

import json
from typing import Any, Dict

import requests

from llama_prompt import build_chat_messages, build_output_schema


class OllamaClientError(RuntimeError):
    pass


class LlamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def ask_with_chat(
        self,
        analysis_result: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": build_chat_messages(analysis_result),
            "format": build_output_schema(),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise OllamaClientError(
                f"Ollama chat failed: HTTP {resp.status_code} - {resp.text}"
            )

        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise OllamaClientError("Ollama returned empty content")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaClientError(
                f"Model output is not valid JSON.\nRaw output:\n{content}"
            ) from exc