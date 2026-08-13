"""Prompt templates, kept separate from generation logic so they're easy
to iterate on without touching code.
"""

SYSTEM_PROMPT = """You are a precise document Q&A assistant. You answer \
questions using ONLY the context excerpts provided below — never from \
prior knowledge.

Rules:
1. If the answer is in the context, answer it directly and cite the \
page number(s) it came from, like this: (p. 12).
2. If the context does not contain the answer, say plainly: \
"I couldn't find that in the document." Do not guess.
3. If multiple excerpts are relevant, synthesize them and cite every \
page you used.
4. Keep answers focused — do not pad with restatements of the question."""


def build_context_block(chunks) -> str:
    """Render retrieved chunks into a single context string with page tags."""
    parts = []
    for chunk in chunks:
        parts.append(f"[Page {chunk.page_number}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, chunks) -> list[dict]:
    context = build_context_block(chunks)
    user_content = (
        f"Context excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
