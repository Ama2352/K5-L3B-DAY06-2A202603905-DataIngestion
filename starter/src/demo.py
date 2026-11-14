"""Local retrieval demo with read-only database access."""
import argparse
import csv
import hashlib
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv

from .embeddings import DEFAULT_MODEL, QUERY_PREFIX, MyEmbedder
from .experiment import ROOT, RESULTS, load_inputs, model_info
from .store import PgVectorStore


ARTIFACTS = {
    'metrics.csv': 'Điểm truy xuất dev / test',
    'ingest_stats.csv': 'Chunk, token, thời gian ingest',
    'traces_dev.json': 'Trace 15 câu dev × A/B/C',
    'traces_test.json': 'Trace 5 câu test × A/B/C',
    'audit_summary.csv': 'Tổng hợp audit',
    'review_labels.json': 'Nhãn và lý do audit',
    'errors.json': 'Log lỗi thực tế',
    'selection.json': 'Lựa chọn cấu hình trên dev',
    'environment.json': 'Model và môi trường',
    'frozen_inputs.json': 'Input đã chốt và hash',
    'scoring_amendment.json': 'Lịch sử hiệu chỉnh q12',
    'original_scoring/metrics.csv': 'Điểm trước hiệu chỉnh q12',
    'original_scoring/traces_dev.json': 'Trace dev trước hiệu chỉnh',
    'original_scoring/traces_test.json': 'Trace test trước hiệu chỉnh',
}


def read_artifact(name, root=RESULTS):
    # Fixed allowlist: never serve arbitrary paths, .env or model files.
    if name not in ARTIFACTS:
        raise ValueError('Tệp không nằm trong danh sách kết quả được phép xem.')
    path = root / name
    with path.open(encoding='utf-8', newline='') as stream:
        data = list(csv.DictReader(stream)) if path.suffix == '.csv' else json.load(stream)
    return {'name': name, 'label': ARTIFACTS[name], 'data': data}


def parse_search(payload, chapters):
    if not isinstance(payload, dict):
        raise ValueError('Yêu cầu phải là một JSON object.')
    question = payload.get('question')
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError('Nhập câu hỏi từ 1 đến 2.000 ký tự.')
    strategy = payload.get('strategy', 'all')
    if strategy not in ('A', 'B', 'C', 'all'):
        raise ValueError('Chọn A, B, C hoặc so sánh cả ba.')
    chapter = payload.get('chapter', '')
    if not isinstance(chapter, str) or (chapter and chapter not in chapters):
        raise ValueError('Chương không tồn tại trong corpus.')
    if chapter and strategy in ('A', 'B'):
        raise ValueError('Chỉ C áp dụng filter chapter; chọn C hoặc so sánh cả ba.')
    return question.strip(), strategy, chapter


class DemoService:
    def __init__(self):
        load_dotenv(ROOT / 'starter/.env')
        url = os.getenv('DATABASE_URL')
        if not url:
            raise ValueError('Thiếu DATABASE_URL trong starter/.env.')
        if os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL) != DEFAULT_MODEL:
            raise ValueError('Demo phải dùng cùng model BGE với thí nghiệm.')
        pages, _, _, _ = load_inputs()
        self.chapters = sorted({p.metadata['chapter'] for p in pages})
        self.store = PgVectorStore(url, 384)
        self.counts = {s: self.store.count(s) for s in 'ABC'}
        if any(n == 0 for n in self.counts.values()):
            raise ValueError('Database chưa đủ A/B/C. Chạy experiment trước khi demo.')
        frozen = json.loads((RESULTS / 'frozen_inputs.json').read_text(encoding='utf-8'))
        for name, expected in frozen['sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Input đã đổi so với lần experiment. Chạy lại thí nghiệm trước khi demo.')
        self.environment = json.loads((RESULTS / 'environment.json').read_text(encoding='utf-8'))
        self.embedder = None
        self.lock = threading.Lock()

    def config(self):
        return {'chapters': self.chapters, 'counts': self.counts, 'model': DEFAULT_MODEL}

    def search(self, payload):
        question, strategy, chapter = parse_search(payload, self.chapters)
        started = time.perf_counter()
        with self.lock:
            model_started = time.perf_counter()
            loaded_now = self.embedder is None
            if self.embedder is None:
                candidate = MyEmbedder()
                if model_info(candidate)['files_sha256'] != self.environment['files_sha256']:
                    raise ValueError('Model local khác model đã ingest. Kiểm tra cache và chạy lại experiment.')
                self.embedder = candidate
            model_ms = (time.perf_counter() - model_started) * 1000
            embed_started = time.perf_counter()
            vector = self.embedder.embed_query(question)
            embedding_ms = (time.perf_counter() - embed_started) * 1000
        results = []
        for name in ('ABC' if strategy == 'all' else strategy):
            filters = {'chapter': chapter} if chapter and name == 'C' else {}
            t = time.perf_counter()
            hits = self.store.search(name, vector, filters=filters)
            results.append({'strategy': name, 'applied_filter': filters, 'hits': hits,
                            'search_ms': round((time.perf_counter() - t) * 1000, 1)})
        output = {'question': question, 'results': results,
                  'retrieval_ms': round((time.perf_counter() - started) * 1000, 1),
                  'process': {'model': DEFAULT_MODEL, 'query_prefix': QUERY_PREFIX,
                              'embedding_input': QUERY_PREFIX + question,
                              'dimensions': len(vector), 'vector_preview': vector[:8],
                              'vector_norm': math.sqrt(sum(v * v for v in vector)),
                              'embedding_ms': round(embedding_ms, 1),
                              'model_loaded_now': loaded_now, 'model_load_ms': round(model_ms, 1),
                              'search': 'pgvector exact cosine, ORDER BY distance ASC, chunk_id; LIMIT 3',
                              'top_k': 3}}
        return output


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log user questions, credentials, or backend exceptions.

        def send_body(self, status, body, content_type='application/json; charset=utf-8'):
            raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(raw)

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            return host in allowed and (not origin or origin == f'http://{host}')

        def do_GET(self):
            if not self.local_request():
                self.send_body(403, {'error': 'Chỉ dùng demo qua địa chỉ localhost.'})
                return
            route = urlsplit(self.path)
            try:
                if route.path == '/':
                    self.send_body(200, Path(__file__).with_name('demo.html').read_bytes(), 'text/html; charset=utf-8')
                elif route.path == '/api/config':
                    self.send_body(200, service.config())
                elif route.path == '/api/artifacts':
                    self.send_body(200, [{'name': n, 'label': label, 'available': (RESULTS / n).is_file()}
                                         for n, label in ARTIFACTS.items()])
                elif route.path == '/api/artifact':
                    names = parse_qs(route.query).get('name', [])
                    if len(names) != 1:
                        raise ValueError('Chọn đúng một tệp kết quả.')
                    self.send_body(200, read_artifact(names[0]))
                else:
                    self.send_body(404, {'error': 'Không tìm thấy đường dẫn.'})
            except FileNotFoundError:
                self.send_body(404, {'error': 'Tệp kết quả này chưa có. Giao diện không tạo dữ liệu thay thế.'})
            except (ValueError, UnicodeDecodeError):
                self.send_body(400, {'error': 'Tệp không được phép xem hoặc nội dung không hợp lệ.'})
            except Exception:
                self.send_body(503, {'error': 'Không đọc được tệp kết quả trên máy.'})

        def do_POST(self):
            if not self.local_request():
                self.send_body(403, {'error': 'Nguồn yêu cầu không hợp lệ.'})
                return
            if self.path != '/api/search':
                self.send_body(404, {'error': 'Không tìm thấy đường dẫn.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Yêu cầu quá lớn hoặc rỗng.')
                payload = json.loads(self.rfile.read(size))
                self.send_body(200, service.search(payload))
            except (ValueError, UnicodeDecodeError) as exc:
                message = str(exc) if not isinstance(exc, json.JSONDecodeError) else 'JSON không hợp lệ.'
                self.send_body(400, {'error': message})
            except Exception:
                self.send_body(503, {'error': 'Truy xuất thất bại. Kiểm tra Docker/database và model local; không có kết quả giả thay thế.'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    try:
        service = DemoService()
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service))
    except ValueError as exc:
        parser.exit(1, f'{exc}\n')
    except Exception:
        parser.exit(1, 'Không khởi động được. Kiểm tra Docker, experiment đã chạy và cổng đang trống.\n')
    print(f'Demo: http://127.0.0.1:{server.server_port} — Ctrl+C để dừng.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
