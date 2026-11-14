def test_selection_uses_dev_only_with_declared_tiebreaks():
    from starter.src.experiment import choose_strategy
    rows = [{'strategy': 'A', 'split': 'dev', 'hit3': .5, 'hit1': .5},
            {'strategy': 'B', 'split': 'dev', 'hit3': .8, 'hit1': .4},
            {'strategy': 'C', 'split': 'dev', 'hit3': .8, 'hit1': .6},
            {'strategy': 'A', 'split': 'test', 'hit3': 1., 'hit1': 1.}]
    assert choose_strategy(rows, {'A': 4, 'B': 5, 'C': 6}) == 'C'


def test_selection_prefers_fewer_chunks_then_stable_strategy_name():
    from starter.src.experiment import choose_strategy
    rows = [{'strategy': s, 'split': 'dev', 'hit3': 1., 'hit1': 1.} for s in 'ABC']
    assert choose_strategy(rows, {'A': 10, 'B': 5, 'C': 5}) == 'B'


def test_audit_exports_review_inputs_without_fabricating_scores():
    from starter.src.experiment import prepare_review
    from starter.src.models import DocumentPage
    from starter.src.chunking import fixed_size
    page = DocumentPage('d', 'url', 1, 'abcdef', {'chapter': 'd'})
    chunks = fixed_size(page, 4, 0)
    out = prepare_review([page], {'A': chunks}, [], [{'document_id': 'd', 'page': 1, 'offset': 2}])
    assert out['coherence'][0]['text'] == 'abcd'
    assert out['coherence'][0]['score'] is None
    assert out['grounding'] == []


def test_rerun_preserves_review_for_unchanged_contexts(tmp_path):
    import json
    from copy import deepcopy
    from starter.src.experiment import ROOT, save_review
    existing = json.loads((ROOT/'results/review_labels.json').read_text(encoding='utf-8'))
    (tmp_path/'review_labels.json').write_text(json.dumps(existing), encoding='utf-8')
    pending = {kind: deepcopy(existing[kind]) for kind in ('coherence', 'grounding', 'metadata')}
    for kind, rows in pending.items():
        for row in rows:
            row['manual_correct' if kind == 'metadata' else 'score'] = None
            for key in ('reason', 'reviewer', 'method'):
                row[key] = ''
    # A similarity score change alone does not invalidate a qualitative judgement.
    pending['grounding'][0]['hits'][0]['score'] += 0.000001
    save_review(pending, tmp_path)
    result = json.loads((tmp_path/'review_labels.json').read_text(encoding='utf-8'))
    assert result['review_metadata'] == existing['review_metadata']
    assert result['coherence'] == existing['coherence']
    assert result['grounding'][0]['score'] == existing['grounding'][0]['score']
    assert (tmp_path/'audit_summary.csv').is_file()
    assert not list(tmp_path.glob('*_review_inputs.json'))


def test_changed_context_invalidates_label_and_stale_summary(tmp_path):
    import json
    from copy import deepcopy
    from starter.src.experiment import ROOT, save_review
    existing = json.loads((ROOT/'results/review_labels.json').read_text(encoding='utf-8'))
    (tmp_path/'review_labels.json').write_text(json.dumps(existing), encoding='utf-8')
    (tmp_path/'audit_summary.csv').write_text('stale', encoding='utf-8')
    pending = {kind: deepcopy(existing[kind]) for kind in ('coherence', 'grounding', 'metadata')}
    pending['coherence'][0].update(text='Changed evidence', score=None, reason='', reviewer='', method='')
    save_review(pending, tmp_path)
    result = json.loads((tmp_path/'review_labels.json').read_text(encoding='utf-8'))
    assert result['review_metadata']['human_review_status'] == 'pending'
    assert result['coherence'][0]['score'] is None
    assert result['coherence'][1]['score'] == existing['coherence'][1]['score']
    assert not (tmp_path/'audit_summary.csv').exists()
