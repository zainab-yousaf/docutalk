"""Top-k similarity search over a Chroma collection."""

from dataclasses import dataclass

import chromadb

from src.config import TOP_K
from src.embeddings import embed_texts


@dataclass
class RetrievedChunk:
    text: str
    page_number: int
    distance: float


def retrieve(
    question: str,
    collection: chromadb.Collection,
    k: int = TOP_K,
) -> list[RetrievedChunk]:
    """Embed the question locally and return the k most similar chunks."""
    query_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()) or 1,
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        RetrievedChunk(
            text=doc,
            page_number=meta.get("page_number", -1),
            distance=dist,
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
