"""extract_text records how each document's primary content was produced.

``text_origin`` tells a document.indexed consumer whether the text is the
file's own (``text_layer``) or written by a model (``ocr``, ``vision``,
``transcription``), and is None when no content was obtained.
"""

from pathlib import Path

import pytest

from extractors import (
    TEXT_ORIGIN_OCR,
    TEXT_ORIGIN_TEXT_LAYER,
    TEXT_ORIGIN_TRANSCRIPTION,
    TEXT_ORIGIN_VISION,
    TEXT_ORIGINS,
    extract_text,
)
from providers.ocr.base import OCRProvider


class PageOCR(OCRProvider):
    def __init__(self, text="Scanned lease addendum, signed by both parties."):
        self.text = text
        self.calls = 0

    def extract(self, file_path, page=None) -> str:
        self.calls += 1
        return self.text

    def describe(self, file_path) -> str:
        return "A photo of a water-stained ceiling."


class FailingVision(OCRProvider):
    def extract(self, file_path, page=None) -> str:
        raise TimeoutError("timed out")

    def describe(self, file_path) -> str:
        raise TimeoutError("timed out")


class FakeMedia:
    def transcribe_audio(self, file_path):
        return "Tenant reports the boiler is off."

    def analyze_video(self, file_path):
        return "Walkthrough of a kitchen with a leaking sink."


def _pdf(path: Path, pages: list[str]) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        for line_no, line in enumerate(text.split(". ")):
            if line.strip():
                page.insert_text((72, 100 + 14 * line_no), line, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _jpeg(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (64, 48), color=(120, 130, 140)).save(path, format="JPEG")
    return path


def test_origins_are_the_four_documented_values():
    assert TEXT_ORIGINS == {"text_layer", "ocr", "vision", "transcription"}


def test_pdf_with_a_text_layer_is_text_layer(tmp_path):
    pdf = _pdf(tmp_path / "lease.pdf", ["Lease term: twelve months. " * 20])
    ocr = PageOCR()
    result = extract_text(pdf, ext="pdf", ocr_provider=ocr)
    assert result.text_origin == TEXT_ORIGIN_TEXT_LAYER
    assert ocr.calls == 0


def test_scanned_pdf_read_by_ocr_is_ocr(tmp_path):
    pdf = _pdf(tmp_path / "scan.pdf", [""])
    result = extract_text(pdf, ext="pdf", ocr_provider=PageOCR())
    assert "Scanned lease addendum" in result.full_text
    assert result.text_origin == TEXT_ORIGIN_OCR


def test_pdf_with_one_ocr_page_among_text_pages_is_ocr(tmp_path):
    pdf = _pdf(tmp_path / "mixed.pdf", ["Typed page with plenty of text. " * 20, ""])
    result = extract_text(pdf, ext="pdf", ocr_provider=PageOCR())
    assert result.text_origin == TEXT_ORIGIN_OCR


def test_described_image_is_vision(tmp_path):
    result = extract_text(_jpeg(tmp_path / "ceiling.jpg"), ext="jpg", ocr_provider=PageOCR())
    assert "water-stained ceiling" in result.full_text
    assert result.text_origin == TEXT_ORIGIN_VISION


def test_image_whose_description_failed_has_no_origin(tmp_path):
    result = extract_text(_jpeg(tmp_path / "ceiling.jpg"), ext="jpg", ocr_provider=FailingVision())
    assert result.primary_content is False
    assert result.text_origin is None


@pytest.mark.parametrize("name,ext,origin", [
    ("call.mp3", "mp3", TEXT_ORIGIN_TRANSCRIPTION),
    ("walkthrough.mp4", "mp4", TEXT_ORIGIN_VISION),
])
def test_media_origin(tmp_path, name, ext, origin):
    media = tmp_path / name
    media.write_bytes(b"media")
    result = extract_text(media, ext=ext, media_provider=FakeMedia())
    assert result.text_origin == origin


def test_text_files_are_text_layer(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# Repair log\n\nBoiler serviced.")
    plain = tmp_path / "note.txt"
    plain.write_text("Boiler serviced.")
    assert extract_text(note, ext="md").text_origin == TEXT_ORIGIN_TEXT_LAYER
    assert extract_text(plain, ext="txt").text_origin == TEXT_ORIGIN_TEXT_LAYER


def test_empty_or_unsupported_documents_have_no_origin(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    unknown = tmp_path / "thing.xyz"
    unknown.write_text("whatever")
    assert extract_text(empty, ext="txt").text_origin is None
    assert extract_text(unknown, ext="xyz").text_origin is None
