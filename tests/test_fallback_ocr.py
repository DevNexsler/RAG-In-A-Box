import pytest
from core.resilience import TransientError
from providers.ocr.base import OCRProvider
from providers.ocr.fallback import FallbackOCRProvider


class _Stub(OCRProvider):
    def __init__(self, describe_fn, extract_fn=None):
        self._d, self._e = describe_fn, extract_fn or (lambda p, pg=None: "")
    def describe(self, file_path): return self._d(file_path)
    def extract(self, file_path, page=None): return self._e(file_path, page)


def _fb(text):
    def run(path): return text
    return run


def test_describe_recovers_from_reachable_empty():
    w = FallbackOCRProvider(_Stub(lambda p: ""), describe_fallback=_fb("desc"))
    assert w.describe("/x.png") == "desc"


def test_describe_confirmed_blank_returns_empty():
    w = FallbackOCRProvider(_Stub(lambda p: ""), describe_fallback=_fb(""))
    assert w.describe("/x.png") == ""


def test_describe_primary_unreachable_raises_and_fallback_not_called():
    calls = {"n": 0}
    def fb(path): calls["n"] += 1; return "x"
    def boom(p): raise TransientError("down")
    w = FallbackOCRProvider(_Stub(boom), describe_fallback=fb)
    with pytest.raises(TransientError):
        w.describe("/x.png")
    assert calls["n"] == 0


def test_describe_dark_mode_empty_raises_transient():
    w = FallbackOCRProvider(_Stub(lambda p: ""), describe_fallback=None)
    with pytest.raises(TransientError):
        w.describe("/x.png")


def test_extract_passthrough_when_primary_has_text():
    w = FallbackOCRProvider(_Stub(lambda p: "", lambda p, pg=None: "text"),
                            extract_fallback=_fb("fb"))
    assert w.extract("/x.pdf", page=1) == "text"
