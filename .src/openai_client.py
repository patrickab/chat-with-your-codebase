from __future__ import annotations  # noqa: I001

from typing import Dict, Iterable, List

from openai import OpenAI


class OpenAIBaseClient:
    """Thin wrapper around OpenAI's chat completions API."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI()

    def chat(
        self, messages: List[Dict[str, str]], system_prompt: str = ""
    ) -> Iterable[str]:
        """Yield the assistant's response as it streams from OpenAI."""
        request_messages = (
            [{"role": "system", "content": system_prompt}, *messages]
            if system_prompt
            else messages
        )
        stream = self.client.chat.completions.create(
            model=self.model, messages=request_messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.get("content"):
                yield delta["content"]
