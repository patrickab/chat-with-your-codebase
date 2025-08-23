from __future__ import annotations  # noqa: I001

import os
from typing import List, Union
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
        if not self.messages:
            # Initialize with system prompt if no messages exist
            self.messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        else:
            # Replace the existing system prompt
            self.messages[0] = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}

    def reset_history(self) -> None:
        """Reset the chat history."""
        self.messages = []

    def write_to_md(self, filename: str, idx: int) -> None:
        """Write a assistant response to .md."""

        if not filename.endswith(".md"):
            filename += ".md"

        # Omit first message (system prompt) & reverse order
        messages = self.messages[1:][::-1]
        assistant_message = messages[idx * 2]

        file_path = OBSIDIAN_VAULT + "/" + filename

        with open(file_path, "w") as f:
            # select all assistant messages
            content = assistant_message["content"][0]["text"]  # type: ignore
            f.write(content)

        md = "markdown"
        # create file in ./markdown/<filename>
        with open(os.path.join(md, filename), "w") as f:
            content = assistant_message["content"][0]["text"]  # type: ignore
            f.write(content)

    def chat(self, user_message: str) -> str:
        self.add_user_message(user_message)
        response = self.client.chat.completions.create(model=self.model, messages=self.messages, stream=False)
        assistant_response = response.choices[0].message.content or ""
        self.add_assistant_message(assistant_response)
        return assistant_response
