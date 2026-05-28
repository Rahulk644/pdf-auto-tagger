"""Adjacent axis — font_embedding (PDF/UA 7.21.4.1).

NOT a benchmark criterion (the benchmark's `fonts_readability` is the size/style/
spacing RENDERING axis, which a tagger doesn't address -> MissingFeature in base
dispatch). font_embedding is the distinct FILE-CONFORMANCE axis our gate detects,
surfaced as its own scorecard column so the pipeline gets credit for what it does
without conflating it with readability. Reuses the gate's detector (detection
addressed in all modes; auto-repair gated/out-of-scope).
"""
from tagger.benchmark.verdicts.base import Verdict


def verdict(pdf, pdf_path) -> Verdict:
    from tagger.stage10_writeback.content_stream_writer import detect_unembedded_fonts

    findings = detect_unembedded_fonts(pdf)
    detail = dict(unembedded_fonts=len(findings))
    return Verdict.passed(**detail) if not findings else Verdict.failed(**detail)
