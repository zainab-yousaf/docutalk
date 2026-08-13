"""Centralized configuration. All env-driven constants live here so no
other module reaches into os.environ directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Chat/generation — routed through OpenRouter (OpenAI-compatible API).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.6-luna")

# Embeddings — local, free, no API key required. OpenRouter has no
# embeddings endpoint, so this runs on-device via sentence-transformers.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")

# Ingestion / chunking
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120
TOKEN_ENCODING = "cl100k_base"  # used for chunk sizing only, provider-agnostic

# Retrieval
TOP_K = 4

# Embedding
EMBED_BATCH_SIZE = 100


def require_openrouter_key() -> str:
    """Fail fast with a clear message instead of a cryptic SDK error."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one at https://openrouter.ai/keys)."
        )
    return OPENROUTER_API_KEY
