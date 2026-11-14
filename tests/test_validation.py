from copy import deepcopy
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def data():
    from starter.src.ingestion import load_pages
    return load_pages(ROOT/'data/pages.json'), json.loads((ROOT/'benchmark/questions.json').read_text(encoding='utf-8'))


def test_real_corpus_and_gold_validate_before_retrieval():
    from starter.src.validation import validate_benchmark
    pages, questions = data()
    result = validate_benchmark(pages, questions)
    assert result['dev'] == 15 and result['test'] == 5 and result['filter_questions'] == 4


@pytest.mark.parametrize('mutation', ['evidence', 'page', 'filter', 'split', 'duplicate'])
def test_corrupt_benchmark_is_rejected(mutation):
    from starter.src.validation import validate_benchmark
    pages, questions = data()
    questions = deepcopy(questions)
    if mutation == 'evidence':
        questions[0]['evidence_quote'] = 'This evidence was invented and is not in the source.'
    elif mutation == 'page':
        questions[0]['gold_page'] = 999
    elif mutation == 'filter':
        questions[5]['filter'] = {'chapter': 'fabricated'}
    elif mutation == 'split':
        questions[3]['split'] = 'dev'
    else:
        questions[0]['id'] = questions[1]['id']
    with pytest.raises(ValueError):
        validate_benchmark(pages, questions)
