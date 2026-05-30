"""Stage 9 helper — zero-shot figure-type classification with SigLIP.

SigLIP (google/siglip-base-patch16-224, Apache-2.0, ~370 MB) classifies a figure
crop into a small set of buckets without any task-specific training. The buckets
mirror the McGraw-Hill Alt-Text Writing Guidelines image categories:

  decorative | logo | photograph | chart | diagram | schematic | map |
  screenshot | illustration

Bucket -> alt-text template (see `bucket_to_alt_text`):
  decorative -> reclassify the figure as ARTIFACT (no /Alt at all — PDF4 / H67;
                screen readers skip)
  logo       -> "Logo." (we don't OCR the brand here; v2 can)
  photograph -> "Photograph." (technical/historical-type prefix per the guidelines)
  chart      -> "Chart. Refer to long description." (guidelines' fallback for >150c)
  diagram    -> "Diagram. Refer to long description."
  schematic  -> "Schematic. Refer to long description."
  map        -> "Map. Refer to long description."
  screenshot -> "Screenshot."
  illustration -> "Illustration."

Self-gating: missing transformers/weights -> classifier returns ("other", 0.0),
the caller falls back to the legacy placeholder generator.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_NAME = "google/siglip-base-patch16-224"
_MAX_ALT_LEN = 150  # McGraw-Hill hard cap

# Zero-shot prompts — kept descriptive and unambiguous so SigLIP can distinguish
# them. Order is the bucket order returned by the model logits.
_PROMPTS: list[tuple[str, str]] = [
    ("decorative",   "a decorative graphic, ornament, divider, or background pattern"),
    ("logo",         "a company logo, brand mark, or simple icon"),
    ("photograph",   "a photograph of a real-world scene, person, or object"),
    ("chart",        "a data chart, graph, bar chart, line chart, or pie chart"),
    ("diagram",      "a technical diagram, flowchart, or labelled illustration"),
    ("schematic",    "an engineering schematic, blueprint, or circuit diagram"),
    ("map",          "a geographic map or floor plan"),
    ("screenshot",   "a screenshot of a software interface, website, or app"),
    ("illustration", "an illustration, drawing, sketch, painting, or cartoon"),
]
_DECORATIVE_THRESHOLD = 0.5  # softmax prob to reclassify as /Artifact
_LOGO_THRESHOLD = 0.5        # similar — only collapse to "Logo." if confident

_model = None
_processor = None
_load_failed = False


def _load() -> bool:
    global _model, _processor, _load_failed
    if _model is not None or _load_failed:
        return _model is not None
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoProcessor
        _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()
        logger.info("SigLIP figure classifier loaded (%s)", _MODEL_NAME)
        return True
    except Exception as e:
        _load_failed = True
        logger.info("SigLIP unavailable (%s) — figures keep the legacy placeholder", e)
        return False


def classify_figure(image) -> Tuple[str, float]:
    """Zero-shot classify a PIL figure crop into one of the bucket labels.
    Returns (bucket, confidence). On any failure returns ("other", 0.0) so the
    caller falls back gracefully."""
    if image is None:
        return ("other", 0.0)
    if not _load():
        return ("other", 0.0)
    try:
        import torch
        labels = [p for _, p in _PROMPTS]
        names = [n for n, _ in _PROMPTS]
        inputs = _processor(text=labels, images=image, padding="max_length",
                            return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        # logits_per_image: (1, N). Softmax across labels gives bucket probs.
        probs = outputs.logits_per_image.softmax(dim=-1)[0].tolist()
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return (names[best_idx], float(probs[best_idx]))
    except Exception as e:
        logger.warning("SigLIP classification failed: %s", e)
        return ("other", 0.0)


def figure_labels(image, max_labels: int = 4) -> list:
    """OCR a figure crop and return short TEXT labels (axis titles, legend
    terms) — deliberately NOT data values: pure-number / symbol-only tokens are
    dropped so the alt text can name what is *labelled* without ever implying we
    read the data points (the failure mode small VLMs hit). Reuses the scanned-
    page RapidOCR singleton; returns [] on any failure (no new model load path).
    """
    if image is None:
        return []
    try:
        from tagger.stage1_extraction import scanned_extractor as se
        if not se._load_ocr():
            return []
        import numpy as np
        res, _ = se._ocr(np.array(image))
        if not res:
            return []
        seen, out = set(), []
        for item in res:
            txt = str(item[1]).strip()
            if not (2 <= len(txt) <= 40):
                continue
            if not any(c.isalpha() for c in txt):  # drop pure numbers/symbols (no data values)
                continue
            key = txt.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(txt)
            if len(out) >= max_labels:
                break
        return out
    except Exception as e:
        logger.debug("figure OCR failed: %s", e)
        return []


# Data-bearing buckets where OCR'd labels make the alt text concrete + honest.
_DATA_BEARING = {"chart": "Chart", "diagram": "Diagram",
                 "schematic": "Schematic", "map": "Map"}


def bucket_to_alt_text(bucket: str, confidence: float = 1.0,
                       has_caption: bool = False,
                       labels: Optional[list] = None) -> Optional[str]:
    """Apply McGraw-Hill guideline templates per bucket.
    Returns None for `decorative` (above threshold) -> caller reclassifies as
    ARTIFACT (PDF4 technique — screen readers must skip).

    has_caption=True trims the "Refer to long description" suffix on complex
    types (chart/diagram/schematic/map) — the guidelines explicitly forbid
    duplicating the caption, and a screen reader will already read the
    Caption element next to the Figure. Bucket label alone is enough."""
    if bucket == "decorative" and confidence >= _DECORATIVE_THRESHOLD:
        return None
    # Data-bearing figure WITH OCR'd labels: name what's labelled, but never
    # claim data values (the small-VLM hallucination failure mode). Honest +
    # concrete + screen-reader-useful, all on CPU with no generative model.
    if bucket in _DATA_BEARING and labels:
        joined = ", ".join(labels[:4])
        alt = f"{_DATA_BEARING[bucket]}. Labelled: {joined}. Data values not detailed; see surrounding text."
        if len(alt) > _MAX_ALT_LEN:
            alt = alt[:_MAX_ALT_LEN - 3].rstrip() + "..."
        return alt
    # Type-prefixed templates. The "Refer to long description" suffix on the
    # complex types signals to a downstream reviewer that the figure needs a
    # narrative description; when an in-PDF caption is present (Stage 8's caption
    # detector tagged it) the caption fills that role, so we drop the suffix
    # to avoid the redundancy the guidelines call out.
    complex_with_suffix = {
        "chart":     ("Chart",    "Refer to long description."),
        "diagram":   ("Diagram",  "Refer to long description."),
        "schematic": ("Schematic","Refer to long description."),
        "map":       ("Map",      "Refer to long description."),
    }
    bare = {
        "logo":         "Logo.",
        "photograph":   "Photograph.",
        "screenshot":   "Screenshot.",
        "illustration": "Illustration.",
    }
    if bucket in complex_with_suffix:
        label, suffix = complex_with_suffix[bucket]
        alt = f"{label}." if has_caption else f"{label}. {suffix}"
    elif bucket in bare:
        alt = bare[bucket]
    else:
        alt = "Figure." if has_caption else "Figure. Refer to long description."
    # Hard cap per guideline (we're already well under, but enforce).
    if len(alt) > _MAX_ALT_LEN:
        alt = alt[: _MAX_ALT_LEN - 3].rstrip() + "..."
    return alt
