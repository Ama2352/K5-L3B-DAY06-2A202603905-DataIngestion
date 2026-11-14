"""Persistent pgvector snapshots, exact cosine search and parameterized filters."""
import json
from typing import Any
import psycopg
from psycopg.types.json import Jsonb
from .embeddings import validate_vectors
from .models import Chunk

FILTER_FIELDS = {'chapter', 'language', 'version'}

class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError('dimensions must be positive')
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5)

    def create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute('CREATE EXTENSION IF NOT EXISTS vector')
            connection.execute("""CREATE TABLE IF NOT EXISTS lab6_chunks (
                strategy TEXT NOT NULL, chunk_id TEXT NOT NULL, text TEXT NOT NULL,
                embedding vector NOT NULL, metadata JSONB NOT NULL,
                PRIMARY KEY (strategy, chunk_id))""")
            connection.execute('CREATE INDEX IF NOT EXISTS lab6_metadata_idx ON lab6_chunks USING gin(metadata)')

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError('Chunk/vector count mismatch')
        validate_vectors(vectors, self.dimensions)
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError('Duplicate chunk IDs in snapshot')
        rows = []
        for chunk, vector in zip(chunks, vectors):
            metadata = {**chunk.metadata, 'document_id': chunk.document_id,
                        'source': chunk.source, 'page': chunk.page}
            rows.append((strategy, chunk.chunk_id, chunk.text, json.dumps(vector), Jsonb(metadata)))
        with self._connect() as connection:
            connection.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (strategy,))
            connection.execute('DELETE FROM lab6_chunks WHERE strategy = %s', (strategy,))
            if rows:
                with connection.cursor() as cursor:
                    cursor.executemany('INSERT INTO lab6_chunks VALUES (%s, %s, %s, %s::vector, %s)', rows)

    def count(self, strategy: str) -> int:
        with self._connect() as connection:
            return connection.execute('SELECT count(*) FROM lab6_chunks WHERE strategy = %s', (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        validate_vectors([query_vector], self.dimensions)
        if not 1 <= top_k <= 100:
            raise ValueError('top_k must be between 1 and 100')
        filters = filters or {}
        if set(filters) - FILTER_FIELDS or any(not isinstance(v, str) for v in filters.values()):
            raise ValueError('Unsupported metadata filter')
        vector = json.dumps(query_vector)
        with self._connect() as connection:
            rows = connection.execute("""SELECT chunk_id, text, metadata,
                1 - (embedding <=> %s::vector) AS score
                FROM lab6_chunks WHERE strategy = %s AND metadata @> %s
                ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s""",
                (vector, strategy, Jsonb(filters), vector, top_k)).fetchall()
        return [{'chunk_id': row[0], 'text': row[1], 'metadata': row[2], 'score': float(row[3])} for row in rows]
