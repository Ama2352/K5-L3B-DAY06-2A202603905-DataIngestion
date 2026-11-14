"""Demo boundaries: user-provided filters and one query embedding per comparison."""
import threading
from types import SimpleNamespace

import pytest

from starter.src.demo import DemoService, parse_search, read_artifact
from starter.src.embeddings import QUERY_PREFIX


@pytest.mark.parametrize('payload', [None, [], {'question': ''}, {'question': 12},
    {'question': 'x', 'strategy': 'D'}, {'question': 'x', 'chapter': 'invented'},
    {'question': 'x', 'strategy': 'A', 'chapter': 'errors'},
    {'question': 'x', 'chapter': None}])
def test_invalid_demo_input_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_search(payload, ['errors'])


def test_compare_embeds_once_and_filters_only_c():
    service = DemoService.__new__(DemoService)
    service.chapters = ['errors']
    service.lock = threading.Lock()
    embeddings, calls = [], []
    service.embedder = SimpleNamespace(embed_query=lambda q: embeddings.append(q) or [1, 0])
    def search(strategy, vector, filters):
        calls.append((strategy, vector, filters))
        return []
    service.store = SimpleNamespace(search=search)
    result = service.search({'question': '  What happens? ', 'strategy': 'all', 'chapter': 'errors'})
    assert embeddings == ['What happens?']
    assert [c[2] for c in calls] == [{}, {}, {'chapter': 'errors'}]
    assert len(result['results']) == 3
    assert 'answer' not in result
    assert result['process']['embedding_input'] == QUERY_PREFIX + 'What happens?'
    assert result['process']['dimensions'] == 2
    assert result['process']['vector_preview'] == [1, 0]
    assert result['process']['vector_norm'] == 1
    assert result['process']['model_loaded_now'] is False


@pytest.mark.parametrize('name', ['../starter/.env', '.env', 'chunks_A.json', '/etc/passwd'])
def test_artifact_viewer_rejects_paths_outside_allowlist(tmp_path, name):
    with pytest.raises(ValueError):
        read_artifact(name, tmp_path)


def test_artifact_viewer_reads_csv_values_without_recomputing(tmp_path):
    (tmp_path/'metrics.csv').write_text('strategy,n,hit1\nA,5,0.2\n', encoding='utf-8')
    result = read_artifact('metrics.csv', tmp_path)
    assert result['data'] == [{'strategy': 'A', 'n': '5', 'hit1': '0.2'}]


def test_artifact_viewer_reads_json_and_reports_missing_file(tmp_path):
    (tmp_path/'selection.json').write_text('{"selected_strategy":"B"}', encoding='utf-8')
    assert read_artifact('selection.json', tmp_path)['data'] == {'selected_strategy': 'B'}
    with pytest.raises(FileNotFoundError):
        read_artifact('traces_test.json', tmp_path)
