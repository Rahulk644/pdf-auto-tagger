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


def bucket_to_alt_text(bucket: str, confidence: float = 1.0) -> Optional[str]:
    """Apply McGraw-Hill guideline templates per bucket.
    Returns None for `decorative` (above threshold) -> caller reclassifies as
    ARTIFACT (PDF4 technique — screen readers must skip)."""
    if bucket == "decorative" and confidence >= _DECORATIVE_THRESHOLD:
        return None
    # All non-decorative cases: a short, type-prefixed alt text. Guidelines:
    # "Note the image format only when important to the content" — for accessibility
    # the type prefix IS important (a chart needs a long description, a photo
    # doesn't), so we keep them. Decorative is the only case that gets no /Alt.
    templates = {
        "logo":         "Logo.",
        "photograph":   "Photograph.",
        "chart":        "Chart. Refer to long description.",
        "diagram":      "Diagram. Refer to long description.",
        "schematic":    "Schematic. Refer to long description.",
        "map":          "Map. Refer to long description.",
        "screenshot":   "Screenshot.",
        "illustration": "Illustration.",
    }
    alt = templates.get(bucket, "Figure. Refer to long description.")
    # Hard cap per guideline (we're already well under, but enforce so v2 with
    # caption-stitching can't accidentally blow past it).
    if len(alt) > _MAX_ALT_LEN:
        alt = alt[: _MAX_ALT_LEN - 3].rstrip() + "..."
    return alt
