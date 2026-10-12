# tests/test_embedding_client.py
from utils.embedding_client import create_embeddings


def test_create_embeddings_shape():
    texts = [
        "How do I fix a leaking faucet?",
        "How can I repair a faucet that leaks?",
        "How do I replace a broken light switch?",
    ]

    embeddings = create_embeddings(texts)
    print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")

    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) > 0
    assert all(len(e) == len(embeddings[0]) for e in embeddings)
