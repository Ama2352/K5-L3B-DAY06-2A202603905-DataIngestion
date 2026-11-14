import math
import os
import pytest


@pytest.mark.parametrize('vectors,dimensions', [([[1, 2]], 3), ([[0, 0, 0]], 3),
                                               ([[1, math.nan, 2]], 3), ([[1, math.inf, 2]], 3)])
def test_reject_invalid_vectors_before_storage(vectors, dimensions):
    from starter.src.embeddings import validate_vectors
    with pytest.raises(ValueError):
        validate_vectors(vectors, dimensions)


def test_vector_validation_accepts_finite_nonzero_vectors():
    from starter.src.embeddings import validate_vectors
    validate_vectors([[1.0, -2.0, 3.0]], 3)


def test_pgvector_persistence_filter_and_snapshot_replacement():
    url = os.environ.get('TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set TEST_DATABASE_URL to an actual pgvector database')
    from starter.src.store import PgVectorStore
    from starter.src.models import Chunk
    store = PgVectorStore(url, 3)
    store.create_schema()
    chunks = [Chunk('unit-a', 'doc', 'source', 1, 'old', {'chapter': 'a'}),
              Chunk('unit-b', 'doc', 'source', 2, 'new', {'chapter': 'b'})]
    strategy = 'pytest-integration'
    try:
        store.upsert(strategy, chunks, [[1, 0, 0], [0, 1, 0]])
        assert store.count(strategy) == 2
        store.upsert(strategy, chunks, [[1, 0, 0], [0, 1, 0]])
        assert store.count(strategy) == 2
        reloaded = PgVectorStore(url, 3)
        hits = reloaded.search(strategy, [1, 0, 0], filters={'chapter': 'b'})
        assert len(hits) == 1 and hits[0]['chunk_id'] == 'unit-b'
        with pytest.raises(ValueError):
            store.upsert(strategy, chunks[:1], [[float('nan'), 0, 0]])
        assert store.count(strategy) == 2
        store.upsert(strategy, chunks[1:], [[0, 1, 0]])
        assert store.count(strategy) == 1
        assert reloaded.search(strategy, [0, 1, 0])[0]['text'] == 'new'
        assert store.search(strategy, [1, 0, 0], filters={'chapter': "b' OR 1=1 --"}) == []
    finally:
        store.upsert(strategy, [], [])
