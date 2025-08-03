from __future__ import annotations  # noqa: I001

import os
from typing import List, Union, Optional
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

API_KEY = os.getenv("OPENAI_API_KEY")

# The union of valid message types
Message = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
]


class OpenAIBaseClient:
    """Base client for OpenAI chat completions."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=API_KEY)
        self.messages: List[Message] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": content}]})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

    def set_system_prompt(self, system_prompt: str) -> None:
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = [{"type": "text", "text": system_prompt}]
        else:
            self.messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    def chat(self, user_message: str) -> str:
        self.add_user_message(user_message)
        response = self.client.chat.completions.create(model=self.model, messages=self.messages, stream=False)
        assistant_response = response.choices[0].message.content or ""
        self.add_assistant_message(assistant_response)
        return assistant_response
