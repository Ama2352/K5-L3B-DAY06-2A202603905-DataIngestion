"""Deterministic chunking with content-addressed IDs and source offsets."""
import hashlib
from .models import Chunk, DocumentPage

def _chunk(page, text, start, end, method):
    identity = f'{page.document_id}:{page.page}:{method}:{start}:{end}:{text}'
    return Chunk(hashlib.sha256(identity.encode()).hexdigest()[:32], page.document_id,
                 page.source, page.page, text, {**page.metadata, 'start': start, 'end': end})

def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size')
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        end = min(start + size, len(page.text))
        chunks.append(_chunk(page, page.text[start:end], start, end, f'fixed:{size}:{overlap}'))
        if end == len(page.text):
            break
    return chunks

def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size <= 0:
        raise ValueError('max_size must be positive')
    chunks = []
    start = 0
    while start < len(page.text):
        end = min(start + max_size, len(page.text))
        if end < len(page.text):
            boundary = page.text.rfind('\n\n', start, end + 1)
            if boundary > start:
                end = boundary
        chunks.append(_chunk(page, page.text[start:end], start, end, f'paragraph:{max_size}'))
        start = end
        if page.text.startswith('\n\n', start):
            start += 2
    return chunks
