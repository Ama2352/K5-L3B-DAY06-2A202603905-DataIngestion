"""Reproducible Python Tutorial corpus preparation (network only with --download)."""
import argparse
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from lxml import html
import requests
from .ingestion import normalize_text

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ['controlflow', 'datastructures', 'modules', 'inputoutput', 'errors', 'classes']
BASE = 'https://docs.python.org/3.13/'


def extract_html(content: bytes) -> str:
    tree = html.fromstring(content)
    bodies = tree.xpath('//div[@role="main"]')
    if not bodies:
        raise ValueError('Documentation main body missing')
    blocks = []
    for element in bodies[0].xpath('.//h1 | .//h2 | .//h3 | .//h4 | .//p | .//pre'):
        if element.xpath('ancestor::pre'):
            continue
        raw_text = element.text_content().replace('¶', '')
        if element.tag == 'pre':
            text = unicodedata.normalize('NFC', raw_text).replace('\r\n', '\n').strip('\n')
        else:
            text = normalize_text(raw_text)
        if text:
            blocks.append(text)
    return '\n\n'.join(blocks)


def source_units(text: str, target: int = 3500, minimum: int = 1500) -> list[dict]:
    if minimum <= 0 or target < minimum:
        raise ValueError('Invalid source unit bounds')
    boundaries = [0]
    start = 0
    while len(text) - start > target:
        end = text.find('\n\n', start + target)
        if end < 0 or len(text) - (end + 2) < minimum:
            break
        boundaries.append(end + 2)
        start = end + 2
    boundaries.append(len(text))
    return [{'start': a, 'end': b, 'text': text[a:b]}
            for a, b in zip(boundaries, boundaries[1:])]


def prepare(download: bool = False) -> None:
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    if download:
        for name in [*CHAPTERS, 'license']:
            url = BASE + ('license.html' if name == 'license' else f'tutorial/{name}.html')
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            (raw / f'{name}.html').write_bytes(response.content)
        (raw / 'downloaded_at.txt').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
    accessed = (raw / 'downloaded_at.txt').read_text(encoding='utf-8').strip()
    pages, sources = [], []
    for chapter in CHAPTERS:
        path = raw / f'{chapter}.html'
        text = extract_html(path.read_bytes())
        units = source_units(text)
        for i, unit in enumerate(units, 1):
            pages.append({'document_id': chapter, 'source': BASE + f'tutorial/{chapter}.html',
                          'page': i, 'text': unit['text'],
                          'metadata': {'chapter': chapter, 'language': 'en', 'version': '3.13',
                                       'unit_start': unit['start'], 'unit_end': unit['end']}})
        sources.append({'source_id': chapter, 'title': text.split('\n')[0],
                        'source_path_or_url': BASE + f'tutorial/{chapter}.html',
                        'snapshot_path': str(path.relative_to(ROOT)).replace('\\', '/'),
                        'snapshot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'normalized_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'license_or_permission': 'PSF License Version 2; code examples also 0BSD',
                        'license_url': BASE + 'license.html', 'accessed_at': accessed,
                        'physical_pages': None, 'content_units': len(units),
                        'units': [{'page': i, 'characters': len(u['text']),
                                   'start': u['start'], 'end': u['end']} for i, u in enumerate(units, 1)]})
    if len(pages) < 30 or any(len(p['text']) < 1500 for p in pages):
        raise ValueError('Corpus does not meet 30 units / 1500 characters per unit')
    manifest = {'counting_rule': 'HTML has no physical pages. Split each chapter in order at paragraph boundaries after 3500 normalized characters; merge a final tail shorter than 1500. No duplication. Page means 1-based unit within chapter.',
                'normalization': 'NFC, prose whitespace normalization; code indentation preserved. Main-body headings, paragraphs and code only. Navigation excluded.',
                'sources': sources, 'total_content_units': len(pages),
                'total_characters': sum(len(p['text']) for p in pages)}
    for name, data in [('pages.json', pages), ('manifest.json', manifest)]:
        (ROOT / 'data' / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {len(pages)} valid source units, {manifest["total_characters"]} characters')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Refresh source snapshots; invalidates frozen experiments')
    prepare(parser.parse_args().download)
