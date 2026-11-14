"""Fail early on invalid source units, splits, evidence and metadata filters."""
from collections import Counter
from .evaluation import evidence_text, golds
from .models import DocumentPage


def validate_benchmark(pages: list[DocumentPage], questions: list[dict]) -> dict:
    lookup = {(p.document_id, p.page): p for p in pages}
    if len(lookup) != len(pages) or len(pages) < 30:
        raise ValueError('Require at least 30 unique source units')
    if any(len(p.text) < 1500 for p in pages):
        raise ValueError('HTML source units must have at least 1500 normalized characters')
    counts = Counter(q['split'] for q in questions)
    if counts['dev'] < 15 or counts['test'] < 5 or set(counts) - {'dev', 'test'}:
        raise ValueError('Require at least 15 dev and 5 test questions')
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    if len({q['question'] for q in questions}) != len(questions):
        raise ValueError('Duplicate questions across splits')
    filter_count = filter_test = 0
    for q in questions:
        if not q['question'].strip() or not q['expected_answer'].strip():
            raise ValueError('Missing question or expected answer')
        for gold in golds(q):
            page = lookup.get((gold['gold_source_id'], gold['gold_page']))
            quote = evidence_text(gold['evidence_quote'])
            if page is None or not quote or quote not in evidence_text(page.text):
                raise ValueError(f'Gold evidence not found: {q["id"]}')
        filters = q.get('filter', {})
        if filters:
            if set(filters) - {'chapter', 'language', 'version'}:
                raise ValueError('Unsupported filter key')
            if not q.get('filter_origin'):
                raise ValueError('Filter origin must be documented')
            if not any(all(p.metadata.get(k) == v for k, v in filters.items()) for p in pages):
                raise ValueError('Filter value absent from source metadata')
            gold_page = lookup[(q['gold_source_id'], q['gold_page'])]
            if not all(gold_page.metadata.get(k) == v for k, v in filters.items()):
                raise ValueError('Filter excludes gold evidence')
            filter_count += 1
            filter_test += q['split'] == 'test'
    if filter_count < 3 or filter_test < 1:
        raise ValueError('Require 3 filter questions including 1 test')
    return {'source_units': len(pages), 'dev': counts['dev'], 'test': counts['test'],
            'filter_questions': filter_count, 'filter_test': filter_test}
