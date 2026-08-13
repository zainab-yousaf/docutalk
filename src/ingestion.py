"""PDF parsing and chunking.

Extracts text per page with pdfplumber, then splits into overlapping
chunks sized in tokens (not characters) so chunks map cleanly onto the
embedding model's context window. Every chunk keeps the page number it
came from, which is what lets the chatbot cite sources later.
"""

from dataclasses import dataclass

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, TOKEN_ENCODING


@dataclass
class Chunk:
    text: str
    page_number: int  # 1-indexed
    chunk_index: int
    source_file: str


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) pairs, 1-indexed."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
    return pages


def chunk_pdf(pdf_path: str, source_file: str | None = None) -> list[Chunk]:
    """Parse a PDF and return page-aware, token-sized overlapping chunks."""
    source_file = source_file or pdf_path

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    chunks: list[Chunk] = []
    chunk_index = 0
    for page_number, page_text in extract_pages(pdf_path):
        for piece in splitter.split_text(page_text):
            chunks.append(
                Chunk(
                    text=piece,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    source_file=source_file,
                )
            )
            chunk_index += 1

    return chunks
