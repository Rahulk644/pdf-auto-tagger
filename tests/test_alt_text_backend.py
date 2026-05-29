"""Stage 9 alt-text VLM backend selection (Gemma-E4B endpoint vs Qwen).

The pipeline still ships placeholder alt text by default; these cover the optional
VLM path: the E4B endpoint backend, its graceful fallback, and the dispatcher
defaulting to E4B. Network + figure cropping are mocked / use a tiny real PDF, so
no model or endpoint is needed.
"""
import fitz

from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage9_alttext import alt_text_generator as atg


def _pdf(tmp_path):
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    p = tmp_path / "fig.pdf"
    doc.save(str(p))
    doc.close()
    return str(p)


def _fig():
    return TaggedElement(element_id="f0", page_num=1, pdf_tag=PDFTag.FIGURE,
                         text="", bbox=(100, 100, 400, 400))


def test_e4b_sets_caption(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA_ALT_ENDPOINT", "http://dummy/endpoint")
    monkeypatch.setattr(atg, "_e4b_caption", lambda ep, img: "A bar chart of revenue by year.")
    fig = _fig()
    n = atg.generate_alt_text_e4b([fig], _pdf(tmp_path))
    assert n >= 1
    assert fig.alt_text == "A bar chart of revenue by year."
    assert fig.needs_review is False


def test_e4b_no_endpoint_falls_back_to_placeholder(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMMA_ALT_ENDPOINT", raising=False)
    fig = _fig()
    atg.generate_alt_text_e4b([fig], _pdf(tmp_path))
    assert "[Figure" in (fig.alt_text or "") and fig.needs_review is True


def test_e4b_endpoint_error_leaves_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA_ALT_ENDPOINT", "http://dummy/endpoint")
    def boom(ep, img):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(atg, "_e4b_caption", boom)
    fig = _fig()
    atg.generate_alt_text_e4b([fig], _pdf(tmp_path))
    # figure that got no caption gets a placeholder + review flag, never crashes
    assert "[Figure" in (fig.alt_text or "") and fig.needs_review is True


def test_e4b_fans_out_one_request_per_figure(tmp_path, monkeypatch):
    """Concurrent fan-out: every figure gets captioned via exactly one request
    each (never batched into a single multi-image call)."""
    import threading
    monkeypatch.setenv("GEMMA_ALT_ENDPOINT", "http://dummy/endpoint")
    calls = []
    lock = threading.Lock()

    def fake_caption(ep, img):
        with lock:
            calls.append(ep)
        return f"caption {len(calls)}"

    monkeypatch.setattr(atg, "_e4b_caption", fake_caption)
    figs = [TaggedElement(element_id=f"f{i}", page_num=1, pdf_tag=PDFTag.FIGURE,
                          text="", bbox=(100, 100 + i * 50, 400, 140 + i * 50))
            for i in range(3)]
    n = atg.generate_alt_text_e4b(figs, _pdf(tmp_path))
    assert n == 3
    assert len(calls) == 3                       # one request per figure, not batched
    assert all(f.alt_text and f.needs_review is False for f in figs)


def test_dispatcher_defaults_to_e4b(tmp_path, monkeypatch):
    called = {}

    def fake_e4b(els, pdf):
        called["e4b"] = True
        return 7

    def fake_qwen(els, pdf):
        called["qwen"] = True
        return 0

    monkeypatch.setattr(atg, "generate_alt_text_e4b", fake_e4b)
    monkeypatch.setattr(atg, "generate_alt_text_qwen", fake_qwen)
    assert atg.ALT_TEXT.vlm_backend == "gemma_e4b"  # default per config
    n = atg.generate_alt_text_vlm([_fig()], _pdf(tmp_path))
    assert n == 7 and called == {"e4b": True}
