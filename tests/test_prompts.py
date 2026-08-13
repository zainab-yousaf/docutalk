"""Unit tests that don't require an API key — prompt assembly only."""

from dataclasses import dataclass

from src.prompts import SYSTEM_PROMPT, build_context_block, build_messages


@dataclass
class FakeChunk:
    text: str
    page_number: int


def test_build_context_block_includes_page_tags():
    chunks = [FakeChunk(text="Revenue grew 12%.", page_number=3)]
    block = build_context_block(chunks)
    assert "[Page 3]" in block
    assert "Revenue grew 12%." in block


def test_build_messages_shape():
    chunks = [FakeChunk(text="Some fact.", page_number=1)]
    messages = build_messages("What happened?", chunks)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "What happened?" in messages[1]["content"]
    assert "[Page 1]" in messages[1]["content"]
