"""Source-aware full-evidence scoring, separate from qualitative review."""
import re
from collections import defaultdict
from typing import Any
from .ingestion import normalize_text

def evidence_text(text: str) -> str:
    return re.sub(r'\s+', ' ', normalize_text(text)).casefold()

def golds(question: dict) -> list[dict]:
    return [question, *question.get('alternative_gold', [])]

def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = []
    for rank, hit in enumerate(hits[:3], 1):
        meta = hit['metadata']
        if any(meta['document_id'] == gold['gold_source_id']
               and meta['page'] == gold['gold_page']
               and evidence_text(gold['evidence_quote']) in evidence_text(hit['text'])
               for gold in golds(question)):
            ranks.append(rank)
    return {'question_id': question['id'], 'split': question['split'],
            'question': question.get('question', ''), 'gold': question,
            'hits': hits[:3], 'hit_ranks': ranks,
            'hit1': int(1 in ranks), 'hit3': int(bool(ranks))}

def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for trace in traces:
        groups[(trace['strategy'], trace['split'])].append(trace)
    rows = []
    for (strategy, split), group in sorted(groups.items()):
        n = len(group)
        h1, h3 = sum(t['hit1'] for t in group), sum(t['hit3'] for t in group)
        rows.append({'strategy': strategy, 'split': split, 'n': n,
                     'hit1_n': h1, 'hit3_n': h3, 'hit1': h1/n, 'hit3': h3/n})
    return rows
