"""Ties retrieval + prompt template + LLM client into one call:
question in, streamed cited answer out.
"""

from collections.abc import Iterator

from src.llm_client import OpenRouterChatClient
from src.prompts import build_messages
from src.retrieval import RetrievedChunk, retrieve


def answer_question(
    question: str,
    collection,
    llm_client: OpenRouterChatClient,
) -> tuple[Iterator[str], list[RetrievedChunk]]:
    """Retrieve relevant chunks, build the prompt, and return a streaming
    answer alongside the chunks used (so the UI can show citations)."""
    chunks = retrieve(question, collection)

    if not chunks:
        def _empty() -> Iterator[str]:
            yield "I couldn't find that in the document."

        return _empty(), []

    messages = build_messages(question, chunks)
    return llm_client.stream_chat(messages), chunks
