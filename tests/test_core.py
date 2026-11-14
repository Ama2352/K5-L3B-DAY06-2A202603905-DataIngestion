import pytest

from starter.src.models import DocumentPage
from starter.src.ingestion import normalize_text
from starter.src.chunking import fixed_size, by_paragraph
from starter.src.evaluation import evaluate_query, summarize


def test_normalize_unicode_and_keep_paragraphs():
    assert normalize_text('  Cafe\u0301\t test\r\n\r\n next  ') == 'Café test\n\nnext'


def test_fixed_windows_traceability_and_stable_ids():
    p = DocumentPage('doc', 'url', 2, 'abcdefghijk', {'chapter': 'lists'})
    chunks = fixed_size(p, 5, 1)
    assert [c.text for c in chunks] == ['abcde', 'efghi', 'ijk']
    assert chunks[0].page == 2 and chunks[0].metadata['chapter'] == 'lists'
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in fixed_size(p, 5, 1)]
    assert chunks[0].chunk_id != fixed_size(DocumentPage('other', 'url', 2, p.text), 5, 1)[0].chunk_id


@pytest.mark.parametrize('size,overlap', [(0, 0), (5, 5), (5, -1)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        fixed_size(DocumentPage('d', 's', 1, 'abc'), size, overlap)


def test_paragraphs_remain_together_and_long_text_is_not_lost():
    p = DocumentPage('d', 's', 1, 'first.\n\nsecond.\n\n' + 'x' * 25)
    chunks = by_paragraph(p, 20)
    assert chunks[0].text == 'first.\n\nsecond.'
    assert ''.join(c.text for c in chunks[1:]) == 'x' * 25
    assert all(len(c.text) <= 20 for c in chunks)


def test_hit_requires_full_evidence_and_correct_page():
    q = {'id': 'q', 'split': 'dev', 'gold_source_id': 'd', 'gold_page': 2,
         'evidence_quote': 'complete evidence'}
    hit = lambda page, text: {'metadata': {'document_id': 'd', 'page': page}, 'text': text}
    out = evaluate_query(q, [hit(1, 'complete evidence'), hit(2, 'complete'), hit(2, 'Complete  evidence!')])
    assert out['hit1'] == 0 and out['hit3'] == 1
    assert out['hit_ranks'] == [3]


def test_alternative_gold_is_accepted():
    q = {'id': 'q', 'split': 'test', 'gold_source_id': 'd', 'gold_page': 1,
         'evidence_quote': 'primary', 'alternative_gold': [
             {'gold_source_id': 'e', 'gold_page': 3, 'evidence_quote': 'another answer'}]}
    out = evaluate_query(q, [{'metadata': {'document_id': 'e', 'page': 3}, 'text': 'another answer'}])
    assert out['hit1'] == out['hit3'] == 1


def test_metrics_do_not_mix_dev_test_or_strategies():
    out = summarize([{'strategy': 'A', 'split': 'dev', 'hit1': 1, 'hit3': 1},
                     {'strategy': 'A', 'split': 'dev', 'hit1': 0, 'hit3': 1},
                     {'strategy': 'A', 'split': 'test', 'hit1': 0, 'hit3': 0}])
    dev = next(row for row in out if row['split'] == 'dev')
    assert dev['n'] == 2 and dev['hit1_n'] == 1 and dev['hit3_n'] == 2
    assert dev['hit1'] == 0.5 and dev['hit3'] == 1.0
