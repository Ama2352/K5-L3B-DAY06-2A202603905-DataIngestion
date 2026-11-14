"""Normalize content without losing paragraph boundaries or source locations."""
import json
import logging
import re
import unicodedata
from pathlib import Path
from .models import DocumentPage

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()

def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    from pypdf import PdfReader
    pages = []
    for number, page in enumerate(PdfReader(path).pages, 1):
        try:
            text = normalize_text(page.extract_text() or '')
        except Exception as exc:
            logger.error('PDF extraction failed: %s page %s (%s)', path.name, number, type(exc).__name__)
            continue
        if not text:
            logger.warning('Empty PDF page: %s page %s', path.name, number)
        else:
            pages.append(DocumentPage(document_id, source, number, text))
    return pages

def load_pages(path: Path) -> list[DocumentPage]:
    return [DocumentPage(**row) for row in json.loads(path.read_text(encoding='utf-8'))]
