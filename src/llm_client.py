"""Thin LLM wrapper so the provider can be swapped later without touching
generation.py or app.py. Currently backed by OpenAI-SDK-compatible
OpenRouter, serving `openai/gpt-5.6-luna`.

To add another provider (direct OpenAI, Claude, a different OpenRouter
model, etc.), implement a class with the same
`stream_chat(messages) -> Iterator[str]` signature and swap what
`get_llm_client()` returns.
"""

from collections.abc import Iterator

from openai import OpenAI

from src.config import LLM_MODEL, OPENROUTER_BASE_URL


class OpenRouterChatClient:
    def __init__(
        self,
        api_key: str,
        model: str = LLM_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Yield response text incrementally."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta


def get_llm_client(api_key: str) -> OpenRouterChatClient:
    return OpenRouterChatClient(api_key=api_key)
