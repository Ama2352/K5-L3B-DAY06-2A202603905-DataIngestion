"""Prepare the CC BY 4.0 T5 paper, preserving physical PDF page numbers."""
import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = 't5_jmlr_2020'
TITLE = 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer'
URL = 'https://www.jmlr.org/papers/volume21/20-074/20-074.pdf'
PUBLICATION_URL = 'https://www.jmlr.org/papers/v21/20-074.html'
PDF_PATH = ROOT / 'data/raw/t5_jmlr_2020.pdf'
SELECTED_PAGES = list(range(4, 41))
LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/'
LIGATURES = str.maketrans({'\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl'})


def normalize_pdf_page(layout: str, number: int) -> str:
    """Join wrapped lines, keeping paragraphs indicated by indentation/layout."""
    text = unicodedata.normalize('NFC', layout).translate(LIGATURES)
    blocks, lines = [], []
    previous_indent = 0
    def flush():
        if lines:
            paragraph = ''
            for line in lines:
                paragraph += ('' if not paragraph or paragraph.endswith('-') else ' ') + line
            blocks.append(paragraph)
            lines.clear()
    for raw in text.splitlines():
        line = re.sub(r'\s+', ' ', raw.strip())
        if not line:
            flush()
            continue
        if line == str(number) or line == 'Exploring the Limits of Transfer Learning' or line.startswith('Raffel, Shazeer, Roberts, Lee, Narang,'):
            flush()
            continue
        heading = bool(re.match(r'^\d+(?:\.\d+)*\.\s+[A-Z]', line))
        indent = len(raw)-len(raw.lstrip())
        if heading or line.startswith('\u2022') or (0 < indent <= 4 and lines) or (previous_indent >= 6 and indent == 0 and lines):
            flush()
        lines.append(line)
        previous_indent = indent
        if heading:
            flush()
    flush()
    return '\n\n'.join(blocks)


def extract_pdf(path=PDF_PATH):
    reader = PdfReader(path)
    if len(reader.pages) != 67:
        raise ValueError('Expected the 67-page JMLR 2020 edition.')
    if 'CC-BY 4.0' not in (reader.pages[0].extract_text() or ''):
        raise ValueError('Expected CC-BY 4.0 notice on PDF page 1.')
    pages = []
    for number in SELECTED_PAGES:
        try:
            text = normalize_pdf_page(reader.pages[number-1].extract_text(extraction_mode='layout') or '', number)
        except Exception as exc:
            raise ValueError(f'PDF extraction failed on physical page {number}: {type(exc).__name__}') from None
        if len(text) < 500:
            raise ValueError(f'PDF page {number} has insufficient extracted content; inspect before counting.')
        sections = ['2. Setup'] if number < 10 else ['3. Experiments'] if number > 10 else ['2. Setup', '3. Experiments']
        pages.append({'document_id': SOURCE_ID, 'source': URL, 'page': number, 'text': text,
                      'metadata': {'section': sections, 'language': 'en', 'year': '2020',
                                   'document_type': 'research_paper', 'page_kind': 'pdf',
                                   'version': 'JMLR 21(140), 2020'}})
    if '2. Setup' not in pages[0]['text'] or '3. Experiments' not in pages[6]['text']:
        raise ValueError('Section headings no longer match the verified source boundaries.')
    return pages


def prepare(download=False):
    if download:
        response = requests.get(URL, timeout=90)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF-'):
            raise ValueError('Source did not return a PDF.')
        PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        PDF_PATH.write_bytes(response.content)
        PDF_PATH.with_name('downloaded_at.txt').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
    if not PDF_PATH.exists():
        raise ValueError('PDF missing. Run python -m starter.src.corpus --download.')
    pages = extract_pdf()
    source = {'source_id': SOURCE_ID, 'title': TITLE,
              'authors': 'Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu',
              'source_path_or_url': URL, 'publication_url': PUBLICATION_URL,
              'snapshot_path': 'data/raw/t5_jmlr_2020.pdf',
              'snapshot_sha256': hashlib.sha256(PDF_PATH.read_bytes()).hexdigest(),
              'normalized_sha256': hashlib.sha256(json.dumps(pages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
              'license_or_permission': 'CC BY 4.0; notice printed on PDF page 1', 'license_url': LICENSE_URL,
              'accessed_at': PDF_PATH.with_name('downloaded_at.txt').read_text(encoding='utf-8').strip(),
              'version': 'JMLR 21(140):1-67, 2020', 'physical_pages': 67,
              'selected_pages': SELECTED_PAGES, 'content_units': len(pages),
              'excluded_pages': [{'pages': [1,2,3], 'reason': 'Outside the selected Setup/Experiments scope.'},
                                 {'pages': list(range(41,68)), 'reason': 'Outside selected scope; includes Reflection, appendices, references (references start at page 58).'}],
              'section_provenance': {'2. Setup': 'Heading page 4, continues through first part of page 10.',
                                     '3. Experiments': 'Heading page 10, continues through selected page 40.'},
              'units': [{'page': p['page'], 'characters': len(p['text']), 'section': p['metadata']['section'],
                         'text_sha256': hashlib.sha256(p['text'].encode()).hexdigest()} for p in pages]}
    manifest = {'source_format': 'pdf',
                'counting_rule': 'Count physical PDF pages 4-40 inclusive: 37 pages of Setup and Experiments, including meaningful tables/figure captions. No duplicated, cover/contents/blank/reference-only pages. page is the original PDF page number.',
                'normalization': 'pypdf layout extraction; NFC; expand typographic ligatures; remove running headers/page numerals; join wrapped lines; keep blank/indented paragraphs. Line-end hyphens retained when joining; tables/equations are text approximations. No OCR or invented text.',
                'filter_rule': 'section is an array of real main-section titles. Page 10 belongs to both Setup and Experiments; filter uses array membership.',
                'sources': [source], 'total_content_units': len(pages),
                'total_characters': sum(len(p['text']) for p in pages)}
    for name,data in [('pages.json',pages),('manifest.json',manifest)]:
        (ROOT/'data'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Prepared {len(pages)} physical PDF pages; {manifest["total_characters"]} characters.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Download the fixed JMLR PDF edition; invalidates old experiments')
    prepare(parser.parse_args().download)
