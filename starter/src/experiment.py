"""Controlled experiment: freeze inputs, select on dev, then evaluate held-out test."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .chunking import fixed_size, by_paragraph
from .embeddings import DEFAULT_MODEL, QUERY_PREFIX, MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pages
from .validation import validate_benchmark, matches_filter

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    pages = load_pages(ROOT / 'data/pages.json')
    questions = json.loads((ROOT / 'benchmark/questions.json').read_text(encoding='utf-8'))
    configs = json.loads((ROOT / 'artifacts/strategies.json').read_text(encoding='utf-8'))
    validation = validate_benchmark(pages, questions)
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    from .corpus import extract_pdf
    for source in manifest['sources']:
        raw = (ROOT / source['snapshot_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['snapshot_sha256']:
            raise ValueError('Source snapshot hash mismatch')
        expected = extract_pdf(ROOT / source['snapshot_path'])
        if hashlib.sha256(json.dumps(expected, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != source['normalized_sha256']:
            raise ValueError('Normalization hash mismatch')
        actual = [p for p in pages if p.document_id == source['source_id']]
        from dataclasses import asdict
        if expected != [asdict(p) for p in actual]:
            raise ValueError('Prepared pages differ from original source snapshot')
    return pages, questions, configs, validation


def make_chunks(pages, config):
    chunks = []
    for page in pages:
        if config['chunking'] == 'fixed':
            chunks.extend(fixed_size(page, config['size'], config['overlap']))
        elif config['chunking'] == 'paragraph':
            chunks.extend(by_paragraph(page, config['size']))
        else:
            raise ValueError('Unknown chunking method')
    return chunks


def choose_strategy(metrics: list[dict], chunk_counts: dict[str, int]) -> str:
    dev = [m for m in metrics if m['split'] == 'dev']
    if not dev:
        raise ValueError('No dev metrics')
    return min(dev, key=lambda m: (-m['hit3'], -m['hit1'], chunk_counts[m['strategy']], m['strategy']))['strategy']


def prepare_review(pages, chunks_by_strategy, traces, locations):
    lookup = {(p.document_id, p.page): p for p in pages}
    coherence, grounding, metadata = [], [], []
    for strategy, chunks in chunks_by_strategy.items():
        for i, loc in enumerate(locations, 1):
            page = lookup[(loc['document_id'], loc['page'])]
            candidates = [c for c in chunks if c.document_id == page.document_id and c.page == page.page
                          and c.metadata['start'] <= loc['offset'] < c.metadata['end']]
            if not candidates:
                raise ValueError('Coherence source location is not covered')
            # Multiple overlapping chunks: choose the earliest start, fixed before scoring.
            chunk = min(candidates, key=lambda c: (c.metadata['start'], c.chunk_id))
            coherence.append({'strategy': strategy, 'sample': i, **loc, 'chunk_id': chunk.chunk_id,
                              'text': chunk.text, 'source_text': page.text, 'score': None, 'reason': '',
                              'reviewer': '', 'method': ''})
    for trace in traces:
        if trace['split'] != 'test':
            continue
        grounding.append({'strategy': trace['strategy'], 'question_id': trace['question_id'],
                          'question': trace['question'], 'gold': trace['gold'], 'hits': trace['hits'],
                          'score': None, 'reason': '', 'reviewer': '', 'method': ''})
        for rank, hit in enumerate(trace['hits'], 1):
            meta = hit['metadata']
            page = lookup.get((meta.get('document_id'), meta.get('page')))
            expected_text = page.text[meta['start']:meta['end']] if page else None
            correct = page is not None and meta.get('source') == page.source and expected_text == hit['text']
            correct = correct and all(meta.get(k) == v for k, v in page.metadata.items())
            filters = trace['gold'].get('filter', {})
            obeys = matches_filter(meta, filters) if filters else None
            metadata.append({'strategy': trace['strategy'], 'question_id': trace['question_id'],
                             'rank': rank, 'chunk_id': hit['chunk_id'], 'metadata': meta,
                             'expected_metadata': {**page.metadata, 'source': page.source,
                                                   'document_id': page.document_id, 'page': page.page} if page else None,
                             'automated_correct': bool(correct), 'filter_compliant': obeys,
                             'manual_correct': None, 'reason': '', 'reviewer': '', 'method': ''})
    return {'coherence': coherence, 'grounding': grounding, 'metadata': metadata}


def model_info(embedder):
    cache = ROOT / '.cache/models'
    files = {str(p.relative_to(cache)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(cache.rglob('*')) if p.is_file() and p.suffix in {'.onnx', '.json'} and 'snapshots' in str(p)}
    versions = {p: importlib.metadata.version(p) for p in
                ['fastembed', 'onnxruntime', 'tokenizers', 'numpy', 'psycopg', 'pypdf', 'requests']}
    return {'provider': 'local FastEmbed/ONNX CPU', 'model': embedder.model_name,
            'dimensions': embedder.dimensions, 'query_prefix': QUERY_PREFIX,
            'document_prefix': '', 'files_sha256': files, 'library_versions': versions,
            'python': platform.python_version(), 'platform': platform.platform(),
            'api_cost_usd': 0, 'cost_note': 'No API used; electricity/hardware not monetized.'}


def save_review(review, result_dir=RESULTS):
    """Keep reviewed labels only when their source context is unchanged."""
    path = result_dir / 'review_labels.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    annotation = {'score', 'manual_correct', 'reason', 'reviewer', 'method'}
    def context(value):
        if isinstance(value, dict):
            return {k: context(v) for k, v in value.items() if k not in annotation}
        if isinstance(value, list):
            return [context(v) for v in value]
        return value
    unchanged = True
    for kind in ('coherence', 'grounding', 'metadata'):
        rows = review[kind]
        old_rows = previous.get(kind, [])
        old = {json.dumps(context(r), sort_keys=True): r for r in old_rows}
        unchanged = unchanged and len(rows) == len(old_rows)
        for row in rows:
            match = old.get(json.dumps(context(row), sort_keys=True))
            if match is None:
                unchanged = False
            else:
                row.update({k: match[k] for k in annotation if k in match})
    review['review_metadata'] = previous.get('review_metadata', {}) if unchanged else {
        'human_review_status': 'pending',
        'note': 'New or changed contexts require review. Unchanged contexts retain existing annotations.'}
    write_json(path, review)
    from .audit import summarize_review
    try:
        summary = summarize_review(review)
    except ValueError:
        (result_dir / 'audit_summary.csv').unlink(missing_ok=True)
    else:
        write_csv(result_dir / 'audit_summary.csv', summary)


def run_experiment(database_url: str, allow_rerun: bool = False) -> None:
    from .store import PgVectorStore
    pages, questions, configs, validation = load_inputs()
    if (RESULTS / 'selection.json').exists() and not allow_rerun:
        raise ValueError('Test was already evaluated. Use --allow-rerun to acknowledge reuse; do not claim a new independent holdout.')
    RESULTS.mkdir(exist_ok=True)
    # Probe the real database before loading/downloading the embedding model.
    store = PgVectorStore(database_url, 384)
    store.create_schema()
    started = datetime.now(timezone.utc).isoformat()
    frozen = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
              ['data/pages.json', 'data/manifest.json', 'benchmark/questions.json',
               'benchmark/coherence_locations.json', 'artifacts/strategies.json']}
    write_json(RESULTS / 'frozen_inputs.json', {'started_at': started, 'sha256': frozen,
               'selection_rule': 'dev Hit@3 DESC, dev Hit@1 DESC, chunk count ASC, strategy name ASC',
               'rerun_after_test_exposure': allow_rerun, 'validation': validation,
               'configs': configs, 'model': DEFAULT_MODEL})
    embedder = MyEmbedder()
    info = model_info(embedder)
    write_json(RESULTS / 'environment.json', info)
    from tokenizers import Tokenizer
    tokenizer_path = next((ROOT / '.cache/models').rglob('tokenizer.json'))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    chunks_by_strategy, stats, vector_cache = {}, [], {}
    for strategy, config in configs.items():
        start = time.perf_counter()
        chunks = make_chunks(pages, config)
        texts = [c.text for c in chunks]
        lengths = [len(t) for t in texts]
        tokens = [len(t.ids) for t in tokenizer.encode_batch(texts)]
        cache_key = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        embedding_start = time.perf_counter()
        reused = cache_key in vector_cache
        vectors = vector_cache[cache_key] if reused else embedder.embed_documents(texts)
        vector_cache[cache_key] = vectors
        embedding_seconds = time.perf_counter() - embedding_start
        store_start = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        store_seconds = time.perf_counter() - store_start
        first_count = store.count(strategy)
        ingest_seconds = time.perf_counter() - start
        check_start = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        second_count = PgVectorStore(database_url, embedder.dimensions).count(strategy)
        if first_count != len(chunks) or second_count != len(chunks):
            raise ValueError('Persistence/idempotency check failed')
        stats.append({'strategy': strategy, 'chunk_count': len(chunks), 'characters': sum(lengths),
                      'length_min': min(lengths), 'length_median': statistics.median(lengths),
                      'length_max': max(lengths), 'tokens_total': sum(tokens), 'tokens_max': max(tokens),
                      'chunks_over_512_tokens': sum(n > 512 for n in tokens),
                      'embedding_seconds': embedding_seconds, 'embedding_reused_from_B': reused,
                      'store_seconds': store_seconds, 'ingest_seconds': ingest_seconds,
                      'reload_check_seconds': time.perf_counter() - check_start,
                      'count_after_first_ingest': first_count, 'count_after_reload': second_count,
                      'api_cost_usd': 0})
        chunks_by_strategy[strategy] = chunks
        write_csv(RESULTS / 'ingest_stats.csv', stats)
        print(f'{strategy}: {len(chunks)} chunks; embedding {embedding_seconds:.2f}s; persisted/reload OK', flush=True)
    with store._connect() as connection:
        backend = connection.execute("SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname='vector'").fetchone()
    info.update({'postgresql_version': backend[0], 'pgvector_version': backend[1],
                 'search': 'exact cosine distance (<=>), no approximate vector index', 'table': 'lab6_chunks'})
    write_json(RESULTS / 'environment.json', info)
    traces = []
    query_vectors = {}
    counts = {row['strategy']: row['chunk_count'] for row in stats}
    for split in ['dev', 'test']:
        for strategy, config in configs.items():
            for q in questions:
                if q['split'] != split:
                    continue
                start = time.perf_counter()
                if q['id'] not in query_vectors:
                    query_vectors[q['id']] = embedder.embed_query(q['question'])
                filters = q.get('filter') if config['filter'] else None
                search_start = time.perf_counter()
                hits = store.search(strategy, query_vectors[q['id']], filters=filters)
                search_seconds = time.perf_counter() - search_start
                trace = evaluate_query(q, hits)
                trace.update({'strategy': strategy, 'applied_filter': filters or {},
                              'search_seconds': search_seconds, 'elapsed_seconds': time.perf_counter() - start})
                traces.append(trace)
        metrics = summarize(traces)
        write_json(RESULTS / f'traces_{split}.json', [t for t in traces if t['split'] == split])
        write_csv(RESULTS / 'metrics.csv', metrics)
        if split == 'dev':
            chosen = choose_strategy(metrics, counts)
            write_json(RESULTS / 'selection.json', {'selected_at': datetime.now(timezone.utc).isoformat(),
                       'selected_strategy': chosen, 'dev_metrics': metrics,
                       'rule': 'Hit@3, then Hit@1, fewer chunks, strategy name', 'test_evaluated': False})
            print(f'Dev selection frozen: {chosen}. Evaluating test next.', flush=True)
    selection = json.loads((RESULTS / 'selection.json').read_text(encoding='utf-8'))
    selection['test_evaluated'] = True
    selection['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(RESULTS / 'selection.json', selection)
    locations = json.loads((ROOT / 'benchmark/coherence_locations.json').read_text(encoding='utf-8'))
    review = prepare_review(pages, chunks_by_strategy, traces, locations)
    save_review(review)
    print(json.dumps(summarize(traces), indent=2))
