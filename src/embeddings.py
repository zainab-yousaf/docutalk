"""Embed chunks locally (sentence-transformers, free, no API key) and
store them in a local Chroma collection.

One Chroma collection per uploaded PDF (keyed by a hash of the file), so
switching documents in the UI doesn't mix contexts.
"""

import hashlib
from functools import lru_cache

import chromadb

from src.config import CHROMA_DIR, EMBED_BATCH_SIZE, EMBEDDING_MODEL
from src.ingestion import Chunk


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the local embedding model once and reuse it across calls.

    Imported lazily so the (fairly heavy) sentence-transformers / torch
    import only happens once an embedding is actually needed.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts locally. Used for both indexing and querying,
    so chunks and questions always land in the same vector space."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(model.encode(batch, convert_to_numpy=True).tolist())
    return vectors


def collection_name_for(file_bytes: bytes) -> str:
    """Derive a stable collection name from the PDF's content."""
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"pdf_{digest}"


def build_or_load_collection(
    chunks: list[Chunk],
    collection_name: str,
) -> chromadb.Collection:
    """Return a Chroma collection populated with the given chunks.

    If a collection with this name already exists (same PDF content seen
    before), it's reused as-is rather than re-embedding.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = {c.name for c in chroma_client.list_collections()}
    collection = chroma_client.get_or_create_collection(name=collection_name)

    if collection_name in existing and collection.count() > 0:
        return collection

    if not chunks:
        return collection

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[f"{c.source_file}-{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "source_file": c.source_file,
            }
            for c in chunks
        ],
    )
    return collection
