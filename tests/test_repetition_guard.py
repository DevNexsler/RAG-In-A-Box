"""Runaway-repetition collapse — the four junk patterns found polluting the index."""
from extractors import collapse_runaway_repetition


def _uniq_ratio(text: str) -> float:
    w = text.split()
    return len(set(w)) / max(len(w), 1)


def test_collapses_ocr_token_loop():
    # "Off Hi Off OO OO OO ..." — a PaddleOCR/vision decode loop.
    text = "Off\nHi\n" + "OO\n" * 800
    out = collapse_runaway_repetition(text)
    assert out.count("OO") <= 3
    assert len(out) < len(text) / 10
    assert "Off" in out and "Hi" in out  # real content preserved


def test_collapses_repeated_sentence_loop():
    sentence = "In managed 54 B ROAD ST LLC and Northampton County Housing Authority on behalf of the MERCADANTE family."
    text = (sentence + " ") * 120
    out = collapse_runaway_repetition(text)
    assert out.count("MERCADANTE") <= 3
    assert "MERCADANTE" in out  # first occurrences kept
    assert len(out) < len(text) / 10


def test_collapses_marketing_email_invisible_padding():
    text = "Summer, Built By You --> 96 Summer, Built By You " + "&#847; &zwnj; " * 500
    out = collapse_runaway_repetition(text)
    assert out.count("&zwnj;") <= 3
    assert "Summer, Built By You" in out
    assert len(out) < len(text) / 10


def test_collapses_empty_spreadsheet_rows():
    text = "Header A | Header B\n" + ("|  " * 400)
    out = collapse_runaway_repetition(text)
    assert len(out) < len(text) / 5
    assert "Header A" in out


def test_healthy_text_is_untouched():
    text = (
        "The tenant reported a leak under the kitchen sink at 163 Washington Unit 2. "
        "Maintenance replaced the P-trap and verified no further drips. "
        "Photos were sent to the owner for the turnover file, and the work order was closed. "
    ) * 6
    assert _uniq_ratio(text) >= 0.12
    assert collapse_runaway_repetition(text) == text


def test_short_text_is_untouched():
    text = "OO OO OO OO OO"
    assert collapse_runaway_repetition(text) == text
