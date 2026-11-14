"""CLI for validation, controlled experiments and source-grounded retrieval."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from .embeddings import DEFAULT_MODEL, MyEmbedder
from .experiment import ROOT, RESULTS, load_inputs, run_experiment, write_json, model_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='experiment', choices=['validate', 'experiment', 'query'])
    parser.add_argument('--allow-rerun', action='store_true', help='Acknowledge that test was already exposed')
    parser.add_argument('--question', help='Retrieval query; never include gold evidence')
    parser.add_argument('--strategy', choices=['A', 'B', 'C'], default='C')
    parser.add_argument('--section', help='User-provided PDF main-section filter, only for C')
    args = parser.parse_args()
    load_dotenv(ROOT / 'starter/.env')
    try:
        if args.command == 'validate':
            print(json.dumps(load_inputs()[3], indent=2))
            return
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError('DATABASE_URL missing: copy starter/.env.example to starter/.env')
        if os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL) != DEFAULT_MODEL:
            raise ValueError('This frozen experiment requires BAAI/bge-small-en-v1.5')
        if args.command == 'experiment':
            run_experiment(url, allow_rerun=args.allow_rerun)
        else:
            from .store import PgVectorStore
            if not args.question or not args.question.strip():
                raise ValueError('query requires --question')
            if args.section and args.strategy != 'C':
                raise ValueError('Only strategy C applies metadata filters')
            if not (RESULTS / 'environment.json').exists():
                raise ValueError('Run experiment successfully before querying')
            embedder = MyEmbedder()
            saved = json.loads((RESULTS / 'environment.json').read_text(encoding='utf-8'))
            if saved['files_sha256'] != model_info(embedder)['files_sha256']:
                raise ValueError('Embedding model files differ from the stored experiment')
            store = PgVectorStore(url, embedder.dimensions)
            filters = {'section': args.section} if args.section else None
            hits = store.search(args.strategy, embedder.embed_query(args.question), filters=filters)
            print(json.dumps(hits, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Do not log raw provider/connection exceptions: they can contain credentials.
        message = str(exc) if isinstance(exc, ValueError) else 'External backend/model operation failed. Check Docker, DATABASE_URL and local model availability.'
        path = RESULTS / 'errors.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        previous.append({'at': datetime.now(timezone.utc).isoformat(), 'command': args.command,
                         'type': type(exc).__name__, 'message': message})
        write_json(path, previous)
        print(f'{type(exc).__name__}: {message}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
