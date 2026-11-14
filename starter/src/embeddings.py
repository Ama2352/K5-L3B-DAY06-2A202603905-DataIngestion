"""Real local embeddings through FastEmbed/ONNX; no API or random fallback."""
import math
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = 'BAAI/bge-small-en-v1.5'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

class Embedder(Protocol):
    model_name: str
    dimensions: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, question: str) -> list[float]: ...

def validate_vectors(vectors: list[list[float]], dimensions: int) -> None:
    for vector in vectors:
        if len(vector) != dimensions or not all(math.isfinite(v) for v in vector):
            raise ValueError('Invalid embedding dimension or non-finite component')
        if not any(v != 0 for v in vector):
            raise ValueError('Zero embedding cannot be used for cosine similarity')

class MyEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding
        supported = {m['model']: m['dim'] for m in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError('Unsupported FastEmbed model')
        if model_name != DEFAULT_MODEL:
            raise ValueError('This experiment fixes BGE-small-en-v1.5 and its query preprocessing')
        self.model_name = model_name
        self.dimensions = supported[model_name]
        cache = cache_dir or str(Path(__file__).resolve().parents[2] / '.cache/models')
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache, threads=4)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError('Cannot embed empty text')
        vectors = [v.tolist() for v in self.model.embed(texts, batch_size=32)]
        if len(vectors) != len(texts):
            raise ValueError('Embedding count mismatch')
        validate_vectors(vectors, self.dimensions)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError('Query must not be empty')
        return self.embed_documents([QUERY_PREFIX + question])[0]
