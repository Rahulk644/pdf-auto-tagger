"""Semantic quality sub-axes (Gemma) — interfaces only; impl deferred.

The two criteria with a quality judgment above the structural primary:
  - alt_text quality (5b): does the /Alt actually DESCRIBE the figure?
  - link descriptiveness (2.4.9): is the link text meaningful vs "click here"?
Both are reported as SEPARATE sub-metrics so structural compliance is unaffected.
Stubs return cannot_derive(MissingFeature) so the harness shape is complete and
the real Gemma calls drop in later (the benchmark ships figure JPGs for 5b).
"""
from tagger.benchmark.verdicts.base import CannotDeriveReason, Verdict


def judge_alt_quality(figure_image_path: str, alt_text: str) -> Verdict:
    """Does `alt_text` adequately describe the figure image? (Gemma — stubbed.)"""
    return Verdict.cannot(CannotDeriveReason.MissingFeature,
                          note="gemma alt-quality not implemented")


def judge_link_descriptiveness(link_text: str) -> Verdict:
    """Is the link text meaningful (vs 'click here')? (Gemma — stubbed.)"""
    return Verdict.cannot(CannotDeriveReason.MissingFeature,
                          note="gemma link-descriptiveness not implemented")
