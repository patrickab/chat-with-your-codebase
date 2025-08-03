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

OBSIDIAN_VAULT = "/home/noob/programs/Obsidian"


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
        self.messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    #    def reset_history(self) -> None:
    #        """Reset the chat history."""
    #        self.messages = []

    def write_to_md(self, filename: str) -> None:
        """Write the chat messages to a markdown file."""

        if not filename.endswith(".md"):
            filename += ".md"

        file_path = OBSIDIAN_VAULT + filename

        with open(file_path + "/" + filename, "w") as f:
            # select all assistant messages
            assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]
            latest_assistant_message = assistant_messages[-1]  # Get the latest assistant message
            content = latest_assistant_message["content"][0]["text"]
            f.write(content)

    def chat(self, user_message: str) -> str:
        self.add_user_message(user_message)
        response = self.client.chat.completions.create(model=self.model, messages=self.messages, stream=False)
        assistant_response = response.choices[0].message.content or ""
        self.add_assistant_message(assistant_response)
        return assistant_response
