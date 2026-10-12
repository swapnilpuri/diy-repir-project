"""
Embedding client for Step 2 semantic deduplication.

Supports:
    EMBEDDING_PROVIDER=ollama
    EMBEDDING_PROVIDER=openai
"""

from typing import List

from config.settings import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
)

import ollama


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings using the configured embedding provider.

    The embedding provider/model is intentionally independent
    from the LLM used by Step 1 for Q&A generation.

    Args:
        texts:
            List of strings to embed.

    Returns:
        List of embedding vectors.
    """

    # Empty batches need no provider request and preserve the list return type.
    if not texts:
        return []

    provider = EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        return _create_openai_embeddings(texts)

    if provider == "ollama":
        return _create_ollama_embeddings(texts)

    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}"
    )


def _create_openai_embeddings(
    texts: List[str],
) -> List[List[float]]:
    """
    Generate embeddings using OpenAI.
    """

    from openai import OpenAI

    # Credentials and endpoint settings come from the SDK's environment setup.
    # This adapter does not apply the Ollama-specific host or timeout settings.
    client = OpenAI()

    # Submit the batch together; returned vector positions correspond to texts.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]

def _create_ollama_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings using a locally running Ollama instance.
    """

    # Use the native embedding endpoint, rather than the /v1 LLM endpoint.
    client = ollama.Client(
        host=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT,
    )

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # Normalize provider vectors to ordinary lists for downstream cosine checks.
    return [
            list(vector)
            for vector in response.embeddings
        ]
