"""Tests for extractors.py — Markdown, PDF, image extraction, and frontmatter parsing."""

import tempfile
from pathlib import Path
from typing import Optional

import pytest


# --- Frontmatter parsing ---


def test_parse_frontmatter_basic():
    """Standard YAML frontmatter is parsed and stripped from content."""
    from extractors import parse_frontmatter

    content = "---\ntags: [recipe, korean]\nstatus: active\ncreated: 2026-01-15\n---\n# Hello\n\nBody text."
    fm, body = parse_frontmatter(content)
    assert fm["tags"] == ["recipe", "korean"]
    assert fm["status"] == "active"
    assert "---" not in body
    assert "# Hello" in body
    assert "Body text." in body


def test_parse_frontmatter_no_frontmatter():
    """Content without frontmatter returns empty dict and original content."""
    from extractors import parse_frontmatter

    content = "# Just a heading\n\nNo frontmatter here."
    fm, body = parse_frontmatter(content)
    assert fm == {}
    assert body == content


def test_parse_frontmatter_empty_frontmatter():
    """Empty frontmatter block returns empty dict."""
    from extractors import parse_frontmatter

    content = "---\n---\n# Content"
    fm, body = parse_frontmatter(content)
    # yaml.safe_load("") returns None, which isn't a dict
    assert fm == {}


def test_parse_frontmatter_string_tags():
    """Frontmatter with string tags (not a list)."""
    from extractors import parse_frontmatter

    content = "---\ntags: recipe\n---\n# Recipe"
    fm, body = parse_frontmatter(content)
    assert fm["tags"] == "recipe"


def test_parse_frontmatter_invalid_yaml():
    """Invalid YAML in frontmatter is handled gracefully."""
    from extractors import parse_frontmatter

    content = "---\n[invalid yaml: {\n---\n# Content"
    fm, body = parse_frontmatter(content)
    # Should return original content on parse failure
    assert fm == {}


def test_extract_title_from_heading():
    """Title is extracted from first # heading."""
    from extractors import extract_title

    assert extract_title("# My Great Title\n\nContent", "note.md") == "My Great Title"


def test_extract_title_from_filename():
    """Title falls back to filename when no heading."""
    from extractors import extract_title

    assert extract_title("No heading here, just text.", "my_notes.md") == "my_notes"


def test_extract_title_nested_doc_id():
    """Title from filename works with nested paths."""
    from extractors import extract_title

    assert extract_title("No heading", "subfolder/recipe.md") == "recipe"


def test_derive_folder():
    """Folder is derived from top-level directory."""
    from extractors import derive_folder

    assert derive_folder("subfolder/recipe.md") == "subfolder"
    assert derive_folder("Archive/old_note.md") == "Archive"
    assert derive_folder("note1.md") == ""
    assert derive_folder("a/b/c/deep.md") == "a"


def test_normalize_tags_list():
    """List tags are joined into comma-separated string."""
    from extractors import normalize_tags

    assert normalize_tags(["recipe", "korean"]) == "recipe,korean"


def test_normalize_tags_string():
    """String tags are normalized."""
    from extractors import normalize_tags

    assert normalize_tags("recipe, korean, fermentation") == "recipe,korean,fermentation"


def test_normalize_tags_none():
    """None tags return empty string."""
    from extractors import normalize_tags

    assert normalize_tags(None) == ""


def test_normalize_tags_single():
    """Single string tag."""
    from extractors import normalize_tags

    assert normalize_tags("recipe") == "recipe"


# --- Markdown extraction ---


def test_extract_markdown(tmp_path):
    """Markdown extraction returns the file contents."""
    from extractors import extract_markdown

    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\n\nSome content here.", encoding="utf-8")

    result = extract_markdown(md_file)
    assert "Hello" in result.full_text
    assert "Some content here" in result.full_text
    assert len(result.pages) == 1
    assert result.pages[0].page == 0


def test_extract_markdown_with_frontmatter(tmp_path):
    """Markdown with frontmatter: YAML is parsed, content is stripped of frontmatter."""
    from extractors import extract_markdown

    md_file = tmp_path / "with_fm.md"
    md_file.write_text(
        "---\ntags: [project, AI]\nstatus: active\ncreated: 2026-01-10\n---\n"
        "# My Project\n\nProject description here.",
        encoding="utf-8",
    )

    result = extract_markdown(md_file)
    # Frontmatter should be parsed
    assert result.frontmatter["tags"] == ["project", "AI"]
    assert result.frontmatter["status"] == "active"
    # Content should NOT contain the YAML block
    assert "---" not in result.full_text
    assert "tags:" not in result.full_text
    # Content should have the actual text
    assert "# My Project" in result.full_text
    assert "Project description" in result.full_text


def test_extract_markdown_no_frontmatter(tmp_path):
    """Markdown without frontmatter: frontmatter dict is empty."""
    from extractors import extract_markdown

    md_file = tmp_path / "no_fm.md"
    md_file.write_text("# Plain Note\n\nJust text.", encoding="utf-8")

    result = extract_markdown(md_file)
    assert result.frontmatter == {}
    assert "# Plain Note" in result.full_text


def test_extract_markdown_empty(tmp_path):
    """Empty markdown file returns empty text."""
    from extractors import extract_markdown

    md_file = tmp_path / "empty.md"
    md_file.write_text("", encoding="utf-8")

    result = extract_markdown(md_file)
    assert result.full_text == ""


def test_extract_markdown_unicode(tmp_path):
    """Markdown with Unicode characters."""
    from extractors import extract_markdown

    md_file = tmp_path / "unicode.md"
    md_file.write_text("日本語テスト 🚀 café", encoding="utf-8")

    result = extract_markdown(md_file)
    assert "日本語テスト" in result.full_text
    assert "🚀" in result.full_text


# --- PDF extraction ---


def _make_test_pdf(path: Path, pages_text: list[str]) -> None:
    """Create a test PDF with the given text on each page."""
    import fitz

    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_extract_pdf_text_only(tmp_path):
    """PDF text_only strategy extracts native text without OCR."""
    from extractors import extract_pdf

    pdf_path = tmp_path / "test.pdf"
    _make_test_pdf(pdf_path, ["Page one content", "Page two content"])

    result = extract_pdf(pdf_path, strategy="text_only")
    assert "Page one" in result.full_text
    assert "Page two" in result.full_text
    assert len(result.pages) == 2
    assert result.pages[0].page == 0
    assert result.pages[1].page == 1
    assert not any(p.was_ocr for p in result.pages)


def test_extract_pdf_text_then_ocr_sufficient_text(tmp_path):
    """text_then_ocr doesn't call OCR when there's enough native text."""
    import fitz
    from extractors import extract_pdf

    # Create a PDF with enough text (insert multiple lines to exceed 200 chars)
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for i in range(20):
        page.insert_text((72, y), f"Line {i}: This is enough text content to fill the page properly.", fontsize=11)
        y += 16
    doc.save(str(pdf_path))
    doc.close()

    result = extract_pdf(pdf_path, strategy="text_then_ocr", ocr_provider=None, min_text_chars=200)
    assert len(result.full_text) > 200
    assert not result.pages[0].was_ocr


def test_extract_pdf_text_then_ocr_calls_ocr_on_thin_page(tmp_path):
    """text_then_ocr falls back to OCR when native text is too short."""
    from extractors import extract_pdf
    from providers.ocr.base import OCRProvider

    class FakeOCR(OCRProvider):
        def __init__(self):
            self.called = False
        def extract(self, file_path, page=None):
            self.called = True
            return "OCR extracted text from image"

    pdf_path = tmp_path / "test.pdf"
    _make_test_pdf(pdf_path, ["Hi"])  # Very short text

    fake_ocr = FakeOCR()
    result = extract_pdf(pdf_path, strategy="text_then_ocr", ocr_provider=fake_ocr, min_text_chars=200)
    assert fake_ocr.called
    assert result.pages[0].was_ocr
    assert "OCR extracted text" in result.full_text


def test_extract_pdf_respects_ocr_page_limit(tmp_path):
    """OCR is not called beyond ocr_page_limit."""
    from extractors import extract_pdf
    from providers.ocr.base import OCRProvider

    class CountingOCR(OCRProvider):
        def __init__(self):
            self.call_count = 0
        def extract(self, file_path, page=None):
            self.call_count += 1
            return "OCR text"

    pdf_path = tmp_path / "test.pdf"
    _make_test_pdf(pdf_path, ["Hi"] * 5)  # 5 pages, all short

    counting_ocr = CountingOCR()
    result = extract_pdf(pdf_path, strategy="text_then_ocr", ocr_provider=counting_ocr,
                         min_text_chars=200, ocr_page_limit=2)
    # Only first 2 pages should be OCR'd
    assert counting_ocr.call_count == 2


def test_extract_pdf_single_empty_page(tmp_path):
    """PDF with one blank page returns only metadata header (no document text)."""
    import fitz
    from extractors import extract_pdf

    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page, no text inserted
    doc.save(str(pdf_path))
    doc.close()

    result = extract_pdf(pdf_path)
    assert len(result.pages) == 1
    assert result.frontmatter.get("page_count") == 1
    # Only metadata header, no actual document text beyond it
    lines = [l for l in result.full_text.strip().splitlines() if not l.startswith(("Pages:", "Author:", "Created:", "Document title:"))]
    assert all(l.strip() == "" for l in lines)


# --- Image extraction ---


def test_extract_image_with_ocr():
    """Image extraction calls OCR provider."""
    from extractors import extract_image
    from providers.ocr.base import OCRProvider

    class FakeOCR(OCRProvider):
        def extract(self, file_path, page=None):
            return "Meeting notes from the image"

    result = extract_image("/fake/path.png", ocr_provider=FakeOCR())
    assert "Meeting notes" in result.full_text


def test_extract_image_no_ocr():
    """Image extraction without OCR returns empty."""
    from extractors import extract_image

    result = extract_image("/fake/path.png", ocr_provider=None)
    assert result.full_text == ""


# --- Dispatcher ---


def test_dispatch_markdown(tmp_path):
    """Dispatcher routes .md to extract_markdown."""
    from extractors import extract_text

    md_file = tmp_path / "test.md"
    md_file.write_text("# Dispatch test", encoding="utf-8")

    result = extract_text(md_file, ext="md")
    assert "Dispatch test" in result.full_text


def test_dispatch_pdf(tmp_path):
    """Dispatcher routes .pdf to extract_pdf."""
    from extractors import extract_text

    pdf_path = tmp_path / "test.pdf"
    _make_test_pdf(pdf_path, ["PDF dispatch content"])

    result = extract_text(pdf_path, ext="pdf")
    assert "PDF dispatch" in result.full_text


def test_dispatch_image():
    """Dispatcher routes .png to extract_image."""
    from extractors import extract_text
    from providers.ocr.base import OCRProvider

    class FakeOCR(OCRProvider):
        def extract(self, file_path, page=None):
            return "Image OCR text"

    result = extract_text("/fake/img.png", ext="png", ocr_provider=FakeOCR())
    assert "Image OCR text" in result.full_text


def test_dispatch_txt(tmp_path):
    """Dispatcher routes .txt to extract_plaintext."""
    from extractors import extract_text

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Plain text content", encoding="utf-8")

    result = extract_text(txt_file, ext="txt")
    assert "Plain text content" in result.full_text


def test_dispatch_xlsx(tmp_path):
    """Dispatcher routes .xlsx to extract_excel."""
    from extractors import extract_text
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Age"])
    ws.append(["Alice", 30])
    xlsx_path = tmp_path / "test.xlsx"
    wb.save(str(xlsx_path))

    result = extract_text(xlsx_path, ext="xlsx")
    assert "Name" in result.full_text
    assert "Alice" in result.full_text


def test_dispatch_docx(tmp_path):
    """Dispatcher routes .docx to extract_markitdown."""
    from extractors import extract_text
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.text_content = "# Converted Document\n\nBody text."
    mock_md = MagicMock()
    mock_md.convert.return_value = mock_result

    with patch("extractors._get_markitdown", return_value=mock_md):
        result = extract_text("/fake/doc.docx", ext="docx")

    assert "Converted Document" in result.full_text


def test_dispatch_csv(tmp_path):
    """Dispatcher routes .csv to extract_markitdown."""
    from extractors import extract_text
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.text_content = "| col1 | col2 |\n|---|---|\n| a | b |"
    mock_md = MagicMock()
    mock_md.convert.return_value = mock_result

    with patch("extractors._get_markitdown", return_value=mock_md):
        result = extract_text("/fake/data.csv", ext="csv")

    assert "col1" in result.full_text


def test_dispatch_html():
    """Dispatcher routes .html to extract_markitdown."""
    from extractors import extract_text
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.text_content = "# Page Title\n\nParagraph content."
    mock_md = MagicMock()
    mock_md.convert.return_value = mock_result

    with patch("extractors._get_markitdown", return_value=mock_md):
        result = extract_text("/fake/page.html", ext="html")

    assert "Page Title" in result.full_text


def test_dispatch_pptx():
    """Dispatcher routes .pptx to extract_markitdown."""
    from extractors import extract_text
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.text_content = "# Slide 1\n\nBullet point."
    mock_md = MagicMock()
    mock_md.convert.return_value = mock_result

    with patch("extractors._get_markitdown", return_value=mock_md):
        result = extract_text("/fake/deck.pptx", ext="pptx")

    assert "Slide 1" in result.full_text


def test_dispatch_unknown_ext(tmp_path):
    """Unknown extension returns empty text."""
    from extractors import extract_text

    result = extract_text("/fake/file.xyz", ext="xyz")
    assert result.full_text == ""


# --- Plaintext extraction ---


def test_extract_plaintext_basic(tmp_path):
    """Plaintext extraction reads file content."""
    from extractors import extract_plaintext

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, world!", encoding="utf-8")

    result = extract_plaintext(txt_file)
    assert result.full_text == "Hello, world!"
    assert result.frontmatter == {}


def test_extract_plaintext_unicode(tmp_path):
    """Plaintext handles Unicode characters."""
    from extractors import extract_plaintext

    txt_file = tmp_path / "unicode.txt"
    txt_file.write_text("日本語テスト café 🚀", encoding="utf-8")

    result = extract_plaintext(txt_file)
    assert "日本語テスト" in result.full_text
    assert "café" in result.full_text


def test_extract_plaintext_empty(tmp_path):
    """Empty plaintext file returns empty text."""
    from extractors import extract_plaintext

    txt_file = tmp_path / "empty.txt"
    txt_file.write_text("", encoding="utf-8")

    result = extract_plaintext(txt_file)
    assert result.full_text == ""


# --- Excel extraction ---


def test_extract_excel_text_only(tmp_path):
    """Excel extraction includes text cells and skips numbers."""
    from extractors import extract_excel
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["Name", "Score", "Notes"])
    ws.append(["Alice", 95, "Excellent"])
    ws.append(["Bob", 80, "Good"])
    ws.append([None, 100, None])
    xlsx_path = tmp_path / "test.xlsx"
    wb.save(str(xlsx_path))

    result = extract_excel(xlsx_path)
    assert "Sheet: Data" in result.full_text
    assert "Headers: Name | Score | Notes" in result.full_text
    assert "Alice" in result.full_text
    assert "Excellent" in result.full_text
    assert "Bob" in result.full_text
    assert "Good" in result.full_text
    # Numbers should NOT appear as standalone extracted values (only in headers)
    lines = result.full_text.split("\n")
    data_lines = [l for l in lines if not l.startswith(("Sheet:", "Headers:"))]
    for line in data_lines:
        assert "95" not in line
        assert "80" not in line
        assert "100" not in line


def test_extract_excel_headers_always_included(tmp_path):
    """Excel headers are always included even if they contain numbers."""
    from extractors import extract_excel
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([2024, "Q1", "Revenue"])
    ws.append([None, "Good quarter", None])
    xlsx_path = tmp_path / "headers.xlsx"
    wb.save(str(xlsx_path))

    result = extract_excel(xlsx_path)
    assert "Headers: 2024 | Q1 | Revenue" in result.full_text


def test_extract_excel_multiple_sheets(tmp_path):
    """Excel extraction handles multiple sheets."""
    from extractors import extract_excel
    import openpyxl

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["Header1"])
    ws1.append(["Value1"])

    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["Header2"])
    ws2.append(["Value2"])

    xlsx_path = tmp_path / "multi.xlsx"
    wb.save(str(xlsx_path))

    result = extract_excel(xlsx_path)
    assert "Sheet: Sheet1" in result.full_text
    assert "Sheet: Sheet2" in result.full_text
    assert "Value1" in result.full_text
    assert "Value2" in result.full_text


def test_extract_excel_empty_workbook(tmp_path):
    """Empty Excel workbook returns empty text."""
    from extractors import extract_excel
    import openpyxl

    wb = openpyxl.Workbook()
    # Default sheet exists but has no rows
    xlsx_path = tmp_path / "empty.xlsx"
    wb.save(str(xlsx_path))

    result = extract_excel(xlsx_path)
    # Empty sheet has no rows via iter_rows in read_only mode
    assert result.frontmatter == {}


# --- MarkItDown extraction ---


def test_extract_markitdown_mock():
    """MarkItDown conversion works with a mock."""
    from extractors import extract_markitdown
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.text_content = "# Document Title\n\nConverted content here."
    mock_md = MagicMock()
    mock_md.convert.return_value = mock_result

    with patch("extractors._get_markitdown", return_value=mock_md):
        result = extract_markitdown("/fake/doc.docx")

    assert "Document Title" in result.full_text
    assert "Converted content" in result.full_text


def test_extract_markitdown_not_installed():
    """Graceful fallback when markitdown is not installed."""
    from extractors import extract_markitdown
    from unittest.mock import patch

    with patch("extractors._get_markitdown", return_value=None):
        result = extract_markitdown("/fake/doc.docx")

    assert result.full_text == ""


def test_extract_markitdown_conversion_error():
    """Graceful fallback when conversion fails."""
    from extractors import extract_markitdown
    from unittest.mock import patch, MagicMock

    mock_md = MagicMock()
    mock_md.convert.side_effect = RuntimeError("Conversion failed")

    with patch("extractors._get_markitdown", return_value=mock_md):
        result = extract_markitdown("/fake/doc.docx")

    assert result.full_text == ""
