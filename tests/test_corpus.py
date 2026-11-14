from pathlib import Path
import json
from pypdf import PdfReader
from starter.src.corpus import normalize_pdf_page, PDF_PATH


def test_pdf_normalization_preserves_bullets_and_physical_page_text():
    text = "Exploring the Limits of Transfer Learning\n\n2. Setup\nA wrapped line\ncontinues here.\n\n   \u2022 A bullet starts\n       and continues without being split.\n\n4"
    out = normalize_pdf_page(text, 4)
    assert '2. Setup' in out
    assert 'A wrapped line continues here.' in out
    assert '\u2022 A bullet starts and continues without being split.' in out
    assert 'Exploring the Limits' not in out
    assert not out.endswith('4')


def test_ligatures_are_expanded_without_inventing_words():
    assert normalize_pdf_page('A \ufb01xed \ufb02ow.\n5', 5) == 'A fixed flow.'


def test_pdf_edition_license_and_counted_pages():
    root = Path(__file__).resolve().parents[1]
    reader = PdfReader(PDF_PATH)
    assert len(reader.pages) == 67
    assert 'CC-BY 4.0' in reader.pages[0].extract_text()
    pages = json.loads((root/'data/pages.json').read_text(encoding='utf-8'))
    assert [p['page'] for p in pages] == list(range(4,41))
    assert all(p['metadata']['page_kind'] == 'pdf' for p in pages)
    assert pages[6]['metadata']['section'] == ['2. Setup','3. Experiments']
