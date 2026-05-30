"""
Centralized configuration for the PDF auto-tagger pipeline.

All thresholds, model paths, DPI settings, and confidence cutoffs live here.
Import from this module — never hardcode magic numbers in stage code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# DPI & coordinate system
# ---------------------------------------------------------------------------

STANDARD_DPI: int = 150
"""Every bbox in the pipeline is normalized to this DPI, origin top-left."""

PDF_NATIVE_DPI: int = 72
"""PDF user-space units are 1/72 inch."""


# ---------------------------------------------------------------------------
# Stage 0 — Page classifier thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PageClassifierConfig:
    """Heuristic thresholds for native / scanned / mixed / corrupt detection."""

    # Minimum extractable characters for "native" classification
    min_native_chars: int = 50

    # If char count is between 1 and this, page may be "mixed"
    mixed_char_upper: int = 50

    # Fraction of page area covered by image XObjects
    # Above this → likely scanned
    scanned_image_coverage: float = 0.70

    # Below this → likely native (text-dominant)
    native_image_coverage: float = 0.30

    # Minimum fraction of characters that must be valid Unicode
    # Below this for non-zero char pages → corrupt or garbled OCR
    min_unicode_validity: float = 0.95

    # Character density (chars / page_area_sq_inches)
    # Pages with density below this AND some chars → mixed
    min_char_density: float = 0.001

    # Char-density floor for a "real text page" — a normal text page sits around
    # 30+ chars/sq.in.; PREP-tagged image-of-text docs with only a visible header
    # come out at ~1-3 (e.g. the MOU appendix: 120 chars + 60% image + 1.28
    # density). Below this with significant image coverage we override the
    # "many chars + some images = NATIVE" path to MIXED so OCR runs on the image.
    sparse_text_density: float = 5.0
    sparse_text_image_coverage: float = 0.5


PAGE_CLASSIFIER = PageClassifierConfig()


@dataclass(frozen=True)
class OCRConfig:
    """RapidOCR (PP-OCRv4) quality vs speed dial for the scanned-page extractor.

    quality presets map to internal score thresholds; "balanced" is the
    `rapidocr-onnxruntime` default. "quality" raises the recogniser confidence
    bar (fewer low-confidence garbage lines) and is the right pick for noisy
    scans where precision matters more than recall.

    Override at runtime with `TAGGER_OCR_QUALITY=quality` (or "speed" /
    "balanced"). Honored by tagger/stage1_extraction/scanned_extractor.py.
    """
    quality: str = field(default_factory=lambda: os.environ.get(
        "TAGGER_OCR_QUALITY", "balanced"))


OCR = OCRConfig()


# Map preset name -> RapidOCR kwargs. Threshold names match the upstream
# rapidocr-onnxruntime config.yaml.
def ocr_kwargs_for(preset: str) -> dict:
    if preset == "quality":
        # Higher confidence floor on the recogniser, slightly higher det box
        # threshold -> drops noisy false-positive lines.
        return {"text_score": 0.6, "box_thresh": 0.6}
    if preset == "speed":
        # Looser thresholds, more low-confidence lines but faster overall.
        return {"text_score": 0.3, "box_thresh": 0.5}
    return {}  # balanced -> RapidOCR defaults (text_score=0.5)


# ---------------------------------------------------------------------------
# Stage 2 — Text merger thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextMergerConfig:
    """Controls how characters are grouped into words, lines, paragraphs."""

    # Horizontal gap (in standard-DPI points) below which adjacent chars
    # are merged into the same word.  Expressed as a multiplier of the
    # average character width on the line.
    word_gap_multiplier: float = 1.0

    # Vertical overlap fraction required for two words to be on the
    # "same line".  1.0 = perfect overlap, 0.5 = half overlap.
    line_overlap_threshold: float = 0.5

    # Horizontal gap multiplier for words on the same line. If the gap between
    # two words exceeds this multiple of the average character width, they are
    # split into separate line elements (e.g. separate table cells).
    line_gap_multiplier: float = 3.0

    # Line clustering (Pass 1): a char joins a line-cluster if its baseline is
    # within baseline_tol_fraction * min(char_size, line_modal_size) of the
    # line's median baseline. 0.5 keeps superscripts (observed shift ~0.38x font)
    # while splitting stacked rows (observed leading ~1.13x font).
    baseline_tol_fraction: float = 0.5

    # Small-char attachment: a char smaller than this fraction of a neighbor
    # line's modal size that x-continues + y-overlaps the line attaches to it
    # regardless of baseline shift (sub/superscripts, footnote markers).
    small_char_size_ratio: float = 0.8

    # Tags that should NEVER be merged across MCIDs
    # (table cells, list labels, etc.)
    no_merge_tags: frozenset[str] = field(default_factory=lambda: frozenset({
        "TD", "TH", "TR", "Lbl", "THead", "TBody", "TFoot",
    }))

    # pdfplumber extract_words settings
    use_text_flow: bool = True
    keep_blank_chars: bool = False
    x_tolerance: int = 3
    y_tolerance: int = 3


TEXT_MERGER = TextMergerConfig()


# ---------------------------------------------------------------------------
# Stage 3 — Layout detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutConfig:
    """MinerU2.5 layout detection settings."""

    model_name: str = "opendatalab/MinerU2.5-Pro-2604-1.2B"

    # Layout backend: "mineru" (GPU VLM), "cpu" (Docling Heron + pdfplumber/xy-cut,
    # no GPU), or "picodet" (PP-DocLayout-V3 region detector via HF transformers,
    # no GPU). The CPU backend resolves the MinerU/GPU dependency for native PDFs;
    # scanned pages have no text layer and still need MinerU/OCR. Override with
    # TAGGER_LAYOUT_BACKEND=cpu to run the whole pipeline (and the test suite) locally
    # without ever spawning MinerU — MinerU must never run on the dev M1.
    #
    # "picodet" shares the CPU detector path but swaps the region SOURCE from Heron
    # to PP-DocLayout-V3 (drop-in via picodet_layout.py). EVALUATED 2026-05-30 and
    # NOT made default: it lost the dp-bench MHS gate (heading quality — Heron-
    # additive headings are what beat the GPU pipeline) and ran ~50% SLOWER on CPU
    # (it's a 33M RT-DETR served through transformers, not Paddle's 5MB native
    # PicoDet runtime). Retained as an option for re-eval if a faster runtime or a
    # table-detection-only hybrid is wanted (its one win was TEDS +0.04).
    backend: str = field(
        default_factory=lambda: os.environ.get("TAGGER_LAYOUT_BACKEND", "mineru")
    )

    # Categories MinerU outputs (canonical names)
    categories: tuple[str, ...] = (
        "Title",
        "Section-header",
        "Text",
        "List-item",
        "Table",
        "Formula",
        "Picture",
        "Caption",
        "Footnote",
        "Page-header",
        "Page-footer",
    )

    # Minimum confidence to accept a detected region
    min_region_confidence: float = 0.5


LAYOUT = LayoutConfig()


# ---------------------------------------------------------------------------
# Stage 5 — Specialist models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableConfig:
    """Table extraction settings."""

    # Structure-model tier in the extract_table_native cascade (lattice -> MODEL
    # -> text). env TAGGER_TABLE_ENGINE:
    #   "tableformer" — Docling TableFormer (default; current behaviour)
    #   "slanet"      — SLANet via rapid_table (ONNX image->HTML). Measured
    #                   better on a dp-bench single-table TEDS A/B (0.843 vs
    #                   0.750) and rescues TableFormer's 0.000 collapses; opt-in
    #                   until a full-corpus TEDS run promotes it to default.
    engine: str = field(default_factory=lambda: os.environ.get(
        "TAGGER_TABLE_ENGINE", "tableformer"))

    # Use pdfplumber for native PDFs
    use_pdfplumber_for_native: bool = True

    # Fall back to MinerU's built-in table extraction for scanned
    use_mineru_for_scanned: bool = True

    # Minimum cells for a valid table (below → likely misclassified paragraph)
    min_cells: int = 2

    # Minimum dimensions (standard DPI) to consider a table region
    min_width_px: int = 40
    min_height_px: int = 20


TABLE = TableConfig()


@dataclass(frozen=True)
class FormulaConfig:
    """Formula → LaTeX → MathML extraction (PDF/UA-2 Associated File on /Formula).

    recognizer dial (env TAGGER_FORMULA_RECOGNIZER):
      - "text"  — DEFAULT, no ML: build LaTeX from the born-digital text layer
                  (formula_extractor raw-text mode). Always available, CPU-free.
                  Produces structurally-valid MathML; on garbled math glyphs it
                  falls back to <mtext> (readable but not semantic).
      - "vlm"   — image→LaTeX via an isolated recogniser venv (UniMERNet /
                  pix2tex). Real LaTeX → semantic MathML. Runs the recogniser in a
                  SUBPROCESS because pix2tex/UniMERNet pin old x-transformers/timm
                  that conflict with our transformers/torch — never install them
                  into the main venv. Activation = provision the venv (see
                  formula_extractor._find_unimernet_python); no-op fallback to
                  "text" if the venv is absent, so it's safe to leave on.
    """
    recognizer: str = field(default_factory=lambda: os.environ.get(
        "TAGGER_FORMULA_RECOGNIZER", "text"))

    model_name: str = "wanderkid/unimernet_base"

    # Empty or invalid LaTeX fallback tag
    fallback_tag: str = "P"


FORMULA = FormulaConfig()


# ---------------------------------------------------------------------------
# Stage 6 — Consistency validator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatorConfig:
    """Thresholds for the consistency validator rule engine."""

    # Elements failing validation get this confidence cap
    failed_confidence_cap: float = 0.55

    # Minimum figure dimensions (standard DPI) — below → Artifact
    min_figure_width_px: int = 20
    min_figure_height_px: int = 20

    # IoU threshold for detecting overlapping regions
    overlap_iou_threshold: float = 0.80


VALIDATOR = ValidatorConfig()


# ---------------------------------------------------------------------------
# Stage 7 — Cross-page merge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossPageConfig:
    """Heuristics for detecting elements spanning page boundaries."""

    # Column width tolerance for table continuation (fraction)
    table_column_width_tolerance: float = 0.05

    # Confidence assigned to cross-page merged elements
    cross_page_confidence: float = 0.70


CROSS_PAGE = CrossPageConfig()


# ---------------------------------------------------------------------------
# Stage 8 — Semantic refinement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticConfig:
    """Controls for heading ranking, TOC, artifact, caption detection."""

    # Heading font-size tolerance: sizes within this many points are same level
    heading_size_tolerance_pt: float = 1.0

    # Maximum heading levels (H1–H6)
    max_heading_levels: int = 6

    # Heading rarity: a font size occurring in more than this fraction of all
    # document elements is treated as body text, not a distinct heading level.
    heading_body_frequency_fraction: float = 0.10

    # TOC: pages in the first N% of the document are candidates
    toc_page_fraction: float = 0.10

    # Artifact: text must appear at same Y-position on at least N pages
    artifact_min_page_occurrences: int = 3

    # Artifact: Y-position tolerance (standard DPI points)
    artifact_y_tolerance_px: float = 5.0

    # Artifact: page-furniture margin band as a fraction of page height. An
    # element whose vertical center falls within the top or bottom band is a
    # running-header/footer/page-number candidate. Single-page (no cross-page
    # repetition required), so it generalizes to short excerpts and recto/verso
    # docs. 0.09 cleanly separates furniture (observed <=0.07) from real
    # headings/body (observed >=0.11) on the clean corpus.
    artifact_margin_band_fraction: float = 0.09

    # Artifact: a margin-band element with more words than this is treated as
    # real content (a body line that begins inside the band), not furniture.
    artifact_max_furniture_words: int = 12

    # Artifact: repeated vertical-margin watermark detection (e.g. the rotated
    # "NIH-PA Author Manuscript" running up the side of HHS/NIH manuscripts). A
    # candidate is a tall, narrow (rotated/vertical) text element sitting in the
    # left/right margin that recurs on multiple pages in the same x-band. Such
    # furniture gets mis-tagged /P and jams the assistive reading order. Three
    # signals must hold together to avoid catching legitimate vertical marginalia
    # (rotated page numbers — too small; sidebar callouts — vary per page):
    #   aspect (h/w) >= watermark_min_aspect          (orientation: vertical)
    #   x-center within watermark_margin_x_fraction of an edge  (in the margin)
    #   recurs on >= artifact_min_page_occurrences pages in the same x-band
    # Calibrated on the NIH reading-order docs: watermark aspect 8-50, xc-frac
    # ~0.04; body aspect <=0.3, xc-frac >=0.28 — orders-of-magnitude separation.
    watermark_min_aspect: float = 3.0
    watermark_margin_x_fraction: float = 0.15
    watermark_x_tolerance: float = 0.03

    # Caption regex patterns
    caption_patterns: tuple[str, ...] = (
        r"^(Figure|Fig\.?|Table|Tbl\.?)\s*\d+",
        r"^(Exhibit|Chart|Diagram|Graph|Plate)\s*\d+",
    )


SEMANTIC = SemanticConfig()


# ---------------------------------------------------------------------------
# Stage 9 — Alt text generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AltTextConfig:
    """Figure alt-text VLM settings.

    Two interchangeable VLM backends (the optional, non-default VLM mode; the
    pipeline still ships placeholder alt text by default):
      - "gemma_e4b": calls the already-deployed Gemma-4-E4B vLLM endpoint (the QA
        auditor model) — consolidates the stack on one VLM. DEFAULT.
      - "qwen": loads Qwen2.5-VL-7B in-process (kept for the head-to-head quality
        comparison when the alt-text stage is actually tackled).
    Quality between the two is NOT yet measured (the alt-text quality eval axis is
    deferred) — see project memory project-alt-text-e4b-swap.
    """

    # Alt-text generation mode (env override: TAGGER_ALT_TEXT_MODE):
    #   "siglip"      — zero-shot SigLIP classifies the figure into a type bucket
    #                   (Chart/Diagram/Photo/Logo/Map/Decorative/...), then a McGraw-
    #                   Hill-aligned template produces the /Alt. Decorative figures are
    #                   reclassified to /Artifact (PDF4 technique). CPU, MIT/Apache.
    #                   DEFAULT — real product win over the legacy review-required
    #                   placeholder, with no GPU dependency. Falls back to placeholder
    #                   if transformers/the SigLIP weights are missing.
    #   "placeholder" — legacy behaviour: every figure gets the review-required string.
    #   "vlm"         — the optional Gemma-E4B / Qwen path below (GPU).
    mode: str = field(default_factory=lambda: __import__("os").environ.get(
        "TAGGER_ALT_TEXT_MODE", "siglip"))

    # Active VLM backend (only when mode="vlm"): "gemma_e4b" (default) | "qwen"
    vlm_backend: str = "gemma_e4b"

    # Qwen backend model (retained for the future A/B comparison)
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Gemma E4B backend: served model id + env var holding the vLLM endpoint URL.
    # The endpoint MUST be the vLLM winner (modal_gemma_vllm.py: serialized per
    # container, Triton backend, warmup, container fan-out) — NOT the abandoned
    # transformers.generate path. Parallelism comes from firing requests
    # CONCURRENTLY (one image per request, never batched) so Modal fans out
    # containers — mirrors the QA runner's PARALLEL fan-out, the proven-fast path.
    gemma_model_name: str = "google/gemma-4-E4B-it"
    gemma_endpoint_env: str = "GEMMA_ALT_ENDPOINT"
    gemma_parallel: int = 10            # concurrent in-flight requests (container fan-out)
    # Thinking is the big speed knob: OFF ~3x fewer output tokens on this decode-bound
    # model. Kept ON is the load-bearing accuracy lever for the QA auditor; for alt-text
    # captioning we default OFF for speed and revisit ON in the deferred quality A/B.
    gemma_enable_thinking: bool = False

    # Maximum tokens for alt text output
    max_output_tokens: int = 150

    # Temperature for alt text generation
    temperature: float = 0.3

    # System prompt
    system_prompt: str = (
        "You are an accessibility expert. Describe this figure for a "
        "screen reader user. Be concise (1-2 sentences), factual, and "
        "focus on what the figure communicates. Do not describe decorative "
        "elements. Do not start with 'This figure shows'."
    )


ALT_TEXT = AltTextConfig()


# ---------------------------------------------------------------------------
# Stage 10 — Struct tree writeback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WritebackConfig:
    """pikepdf struct tree writeback settings."""

    # PDF tag role map — maps our canonical tags to PDF struct types
    tag_role_map: dict[str, str] = field(default_factory=lambda: {
        "H1": "H1", "H2": "H2", "H3": "H3", "H4": "H4", "H5": "H5", "H6": "H6",
        "P": "P", "Span": "Span",
        "L": "L", "LI": "LI", "Lbl": "Lbl", "LBody": "LBody",
        "Table": "Table", "TR": "TR", "TH": "TH", "TD": "TD",
        "THead": "THead", "TBody": "TBody", "TFoot": "TFoot",
        "Figure": "Figure", "Formula": "Formula",
        "Caption": "Caption",
        "TOC": "TOC", "TOCI": "TOCI",
        "BlockQuote": "BlockQuote", "Quote": "Quote",
        "Note": "Note", "Reference": "Reference",
        "Code": "Code",
        "Artifact": "Artifact",
    })


WRITEBACK = WritebackConfig()


# ---------------------------------------------------------------------------
# Pipeline-wide
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline settings."""

    # Confidence threshold below which elements are flagged for human review
    review_threshold: float = 0.60

    # Default output directory
    default_output_dir: Path = Path("output")

    # Flask API settings
    flask_host: str = "0.0.0.0"
    flask_port: int = 5002  # 5001 is PREP-QA-Tool
    flask_debug: bool = False

    # ---- Remediation policy ------------------------------------------------
    # POLICY: adding STRUCTURE (the tag tree, /Alt, MathML, reading order) is our
    # core function and always runs — that's what makes a doc accessible and it
    # never alters how the page looks. Anything that modifies the SOURCE document
    # (embedding/substituting fonts, changing colours for contrast, rewriting
    # content) is OFF by default and only applied when the user opts in — we
    # detect-and-report such issues, we don't silently change the document.
    #
    # The two halves:
    #   DETECT (always on, non-modifying): tagger/audit/ (act_rules, matterhorn,
    #     screen_reader) report what's wrong without touching the file.
    #   FIX (opt-in, gated): source-modifying remediations run only when enabled.
    #
    # Structural-repair gating (see stage10_writeback/repair_gate.py) for the
    # font repairs that DO exist today:
    #   "auto"      — apply all modifying repairs (default; differentiator vs PREP)
    #   "confirm"   — apply only repairs whose finding_id is in repair_approval_file
    #   "flag-only" — never apply; only report them
    # Opt-in fixers for the currently detect-only axes (font embedding, colour
    # contrast, table-header promotion, descriptive link text) are added behind
    # their own off-by-default flags as they land — never auto-applied.
    repair_mode: str = "auto"
    repair_approval_file: Path | None = None


PIPELINE = PipelineConfig()
