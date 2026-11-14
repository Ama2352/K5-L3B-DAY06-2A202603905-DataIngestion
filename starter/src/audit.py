"""Aggregate explicit qualitative labels, never infer them from retrieval scores."""
import json
from collections import defaultdict
from statistics import mean
from .experiment import RESULTS, write_csv


def filter_compliance(rows: list[dict]) -> tuple[int, int]:
    questions = defaultdict(list)
    for row in rows:
        if row['filter_compliant'] is not None:
            questions[row['question_id']].append(row['filter_compliant'])
    return sum(all(values) for values in questions.values()), len(questions)


def summarize_review(review: dict) -> list[dict]:
    for kind in ['coherence', 'grounding', 'metadata']:
        for row in review[kind]:
            label = row.get('manual_correct') if kind == 'metadata' else row.get('score')
            if label is None or not all(row.get(k) for k in ['reason', 'reviewer', 'method']):
                raise ValueError('Unreviewed label: fill label, reason, reviewer and method')
            if kind != 'metadata' and (type(label) is not int or label not in (0, 1, 2)):
                raise ValueError('Qualitative scores must be integers 0, 1 or 2')
            if kind == 'metadata' and type(label) is not bool:
                raise ValueError('Metadata labels must be booleans')
    result = []
    strategies = sorted({r['strategy'] for r in review['coherence']})
    if strategies != ['A', 'B', 'C']:
        raise ValueError('Need reviewed rows for all three strategies')
    for strategy in strategies:
        coherence = [r for r in review['coherence'] if r['strategy'] == strategy]
        grounding = [r for r in review['grounding'] if r['strategy'] == strategy]
        metadata = [r for r in review['metadata'] if r['strategy'] == strategy]
        if len(coherence) != 10 or len(grounding) != 5 or len(metadata) != 15:
            raise ValueError('Require 10 coherence locations, 5 test grounding and 15 metadata rows per strategy')
        if len({r['sample'] for r in coherence}) != 10 or len({r['question_id'] for r in grounding}) != 5:
            raise ValueError('Duplicate review samples')
        if len({(r['question_id'], r['rank']) for r in metadata}) != 15:
            raise ValueError('Duplicate metadata review records')
        obeys, filtered = filter_compliance(metadata)
        result.append({'strategy': strategy, 'coherence_n': len(coherence),
                       'coherence_mean': mean(r['score'] for r in coherence),
                       'metadata_correct': sum(r['manual_correct'] for r in metadata), 'metadata_n': len(metadata),
                       'filter_questions_compliant': obeys, 'filter_questions_n': filtered,
                       'grounding_n': len(grounding),
                       'grounding_0': sum(r['score'] == 0 for r in grounding),
                       'grounding_1': sum(r['score'] == 1 for r in grounding),
                       'grounding_2': sum(r['score'] == 2 for r in grounding),
                       'grounding_mean': mean(r['score'] for r in grounding)})
    return result


if __name__ == '__main__':
    review = json.loads((RESULTS / 'review_labels.json').read_text(encoding='utf-8'))
    rows = summarize_review(review)
    write_csv(RESULTS / 'audit_summary.csv', rows)
    print(json.dumps(rows, indent=2))
