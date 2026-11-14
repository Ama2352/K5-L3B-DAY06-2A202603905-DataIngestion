"""Opt-in check against the real downloaded ONNX model, never random vectors."""
import os
import math
import pytest


@pytest.mark.skipif(os.getenv('RUN_LOCAL_MODEL') != '1', reason='Set RUN_LOCAL_MODEL=1 after model download')
def test_real_local_model_batch_and_query_are_compatible():
    from starter.src.embeddings import MyEmbedder
    model = MyEmbedder()
    vectors = model.embed_documents(['Python sets remove duplicate elements.',
                                     'A file is closed when its with block exits.'])
    query = model.embed_query('What removes duplicates?')
    assert len(vectors) == 2
    assert len(query) == len(vectors[0]) == len(vectors[1]) == 384
    assert all(math.isfinite(x) for row in [*vectors, query] for x in row)
    assert vectors[0] != vectors[1]
    assert abs(sum(v*v for v in query) - 1) < .01
