# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run full pipeline locally on one PDF — CPU backend (no MinerU, no GPU)
TAGGER_LAYOUT_BACKEND=cpu python -m tagger.cli tag input.pdf -o output.pdf --report report.json

# Run only page classification
python -m tagger.cli classify input.pdf

# Run Stages 0-2 (extract + merge, no ML)
python -m tagger.cli extract input.pdf

# Start local Flask API (port 5002)
python -m tagger.cli serve

# Read-only ACT + PDF/UA-1 conformance audit on any tagged PDF
python -m tagger.audit.act_rules output.pdf [output2.pdf ...]  # --json for aggregate

# Run on Modal GPU (A10G) — processes all 5 corpus PDFs (legacy MinerU path)
/Users/rahulkhatri/Library/Python/3.9/bin/modal run run_modal.py

# Run on Modal GPU — Miramar + Summary of Revenues only (legacy MinerU path)
/Users/rahulkhatri/Library/Python/3.9/bin/modal run scratch/run_modal_targeted.py

# QA evaluation (legacy 31B auditor — retired; prefer the E4B/vLLM path below)
/Users/rahulkhatri/Library/Python/3.9/bin/modal run scratch/run_qa_modal.py
python analyze_qa_report.py

# E4B/vLLM QA auditor (current) — deploy then drive with the prompt-v2 client
/Users/rahulkhatri/Library/Python/3.9/bin/modal deploy tagger/qa/modal_gemma_vllm.py
MODAL_URL=<endpoint> PDFS_DIR=<tagged-output-dir> PARALLEL=10 \
  /Users/rahulkhatri/"PREP QA Tool"/venv/bin/python "/Users/rahulkhatri/PREP QA Tool/run_corpus_modal.py" [filter]

# Benchmark substrate — checker (PDF-A-B, CPU/free, all 125 docs)
PYTHONPATH=. python scratch/run_benchmark.py <benchmark_root> [--remediation-dir DIR]

# dp-bench scoring (CPU/free)
PYTHONPATH=. python scratch/run_dpbench.py --gt-dir <gt> --pred-dir <out> --out card.json

# Run tests — LOCAL CPU BACKEND (full suite green, 266 passing, ~30s)
TAGGER_LAYOUT_BACKEND=cpu pytest -q
# single file / verbose:
TAGGER_LAYOUT_BACKEND=cpu pytest tests/test_stage5.py -v
```

**Python environments:**
- `.venv3/` — has `pikepdf`, `pdfplumber`, `pillow`, `modal`, `torch`, `transformers`, `docling-ibm-models`, `rapidocr-onnxruntime`, `sentencepiece` (use for local pipeline + audit work)
- `/Users/rahulkhatri/Library/Python/3.9/bin/` — has `modal` CLI
- Modal remote runs Python 3.11 with the `tagger/` directory synced as `/root/tagger`

## Environment flags (set at session level)

| env var | maps to | values | default | when to set |
|---|---|---|---|---|
| `TAGGER_LAYOUT_BACKEND` | `LAYOUT.backend` | `cpu` / `mineru` | `mineru` (frozen-dataclass default; CI/local always sets `cpu`) | always `cpu` locally — `mineru` is the GPU/Modal-only path |
| `TAGGER_ALT_TEXT_MODE` | `ALT_TEXT.mode` | `siglip` / `placeholder` / `vlm` | `siglip` | leave default unless reproducing the legacy review-required placeholders |
| `TAGGER_OCR_QUALITY` | `OCR.quality` | `speed` / `balanced` / `quality` | `balanced` | `quality` for noisy scans |

## Architecture

### Stage tree (`tagger/pipeline.py`)

`AutoTaggerPipeline.run()` calls stages sequentially. Each stage receives `DocumentData` (which holds a `pages` dict of `PageData`) and modifies it in place.

```
Stage 0   page_classifier      Native vs scanned vs mixed vs corrupt. The
                               sparse-text-density override catches image-of-text
                               docs where a software (PREP) injected a header-only
                               text layer — they'd otherwise classify as NATIVE.
Stage 1a  native_extractor     Character-level extraction via pdfplumber; assigns
                               p{N}_c{idx} IDs.
Stage 1b  scanned_extractor    RapidOCR (PP-OCRv4 ONNX, Apache-2.0) on
                               SCANNED/MIXED pages. Renders at STANDARD_DPI so
                               pixel coords == standard 150-DPI space — no
                               transform needed downstream. PageElement.source =
                               "rapidocr". Lazy singleton; missing package =
                               graceful no-op.
Stage 2   text_merger          Merges chars → words → line elements (PageElement).
Stage 3   layout_detector      Pluggable LayoutModelAdapter, selected by
                               LAYOUT.backend:
                               - cpu (default for local) — cpu_layout_detector
                                 + Docling Heron (RT-DETRv2 layout, MIT) +
                                 Docling TableFormer (MIT) for borderless
                                 tables. On MIXED/SCANNED pages Heron is the
                                 entire layout source; on NATIVE pages Heron-
                                 detected Title/Section-header regions are
                                 UNION'd into the pdfplumber-line heading set
                                 (additive, never removes).
                               - mineru — legacy MinerU2.5-Pro on Modal A10G.
Stage 4+5 content_router       Maps Stage 2 PageElements into Stage 3 regions;
          + specialists        specialists (Docling TableFormer for tables,
                               figure / formula) produce TaggedElements.
Stage 6   consistency_validator Rule engine; converts bad elements to Artifact.
Stage 7   cross_page_merger    Merges elements split across page boundaries.
Stage 8a  heading_ranker       H1–H6 assignment by font-tier rarity.
Stage 8a' heading_hierarchy_enforcer  Deterministic PDF/UA-1 7.4.2 + WCAG
                               rules: no level skips (H1→H3 collapses to
                               H1→H2), first heading = H1, empty heading →
                               /Artifact, punctuation-only heading → /Artifact.
Stage 8b  toc_detector
Stage 8c  artifact_detector    Running headers/footers/page numbers + repeated
                               vertical-margin watermarks ("NIH-PA Author
                               Manuscript" etc.).
Stage 8d  caption_detector
Stage 8e  list_builder         L > LI > Lbl + LBody nesting.
Stage 8f  pdfua_structural_enforcer  Empty/punct-only /P|Caption|Note →
                               /Artifact, every /Figure has /Alt, floating
                               /Caption (no adjacent /Figure or /Table) → /P.
Stage 9   alt_text_generator   Mode = ALT_TEXT.mode (env TAGGER_ALT_TEXT_MODE):
                               - siglip (default) — google/siglip-base-patch16-224
                                 zero-shot bucket (chart / photograph / logo /
                                 schematic / map / decorative / ...). McGraw-
                                 Hill template per bucket; decorative figures
                                 reclassified to /Artifact (PDF4 / H67).
                                 Caption-aware: drops "Refer to long description"
                                 suffix when Stage-8d tagged an adjacent
                                 /Caption (do-not-duplicate per the guidelines).
                               - placeholder — legacy review-required string.
                               - vlm — Gemma-4-E4B or Qwen2.5-VL (GPU).
Stage 10  struct_tree_writer   Builds PDF struct tree (in READING order, not
                               geometric) + injects BDC/EMC markers via the
                               font-aware glyph counter (Type0 fonts use 2-byte
                               codes; len(str)/len(bytes) would over-count and
                               desync the char↔glyph mapping).
```

### Conformance audit layer (`tagger/audit/`)

Read-only checker — separate from the tagging pipeline. Reports per-rule pass / fail / N/A for the eight rules our pipeline cares about (`ACT-6cfa84`, `ACT-36b590`, `ACT-b40fd1`, `PDFUA-7.4.2`, `PDFUA-7.1-10`, `PDFUA-7.5.2`, `PDFUA-7.5.3`, `PDFUA-7.1-1`). Use this to compare our tagged output against PREP / PDFix / any other tool's tagged output deterministically.

### Key data flow invariants

**Coordinate spaces** — Two spaces coexist and must never be mixed:
- **Standard (150 DPI, top-left origin):** `PageElement.bbox`, `TaggedElement.bbox`, `LayoutRegion.bbox`. Used for all inter-stage comparisons. The CPU layout backend's `_image_boxes` / `_heading_lineboxes` / Heron all emit 150-DPI directly.
- **Native pdfplumber (72 DPI, top-left origin):** Only used inside Stage 1, Stage 5 table extraction, and Stage 10 BDC injection. Convert with `coord_transformer.py`.

**Element ID scheme** — Stage 1 assigns `element_id = f"p{page_num}_c{char_idx}"` where `char_idx` is the raw `enumerate(page.chars)` index (skipping only blank/zero-size chars). OCR'd elements (Stage 1b) use `p{N}_o{idx}` instead (`source = "rapidocr"`). This same ID scheme is used in `merged_from` lists all the way to Stage 10's `inject_bdc_markers` which maps char indices back to MCIDs.

**`merged_from`** — Every `PageElement` and `TaggedElement` carries a `merged_from: list[str]` of Stage-1 char IDs. Stage 10 uses these to decide which content-stream characters belong to each struct tree element. If `merged_from` is empty for a non-table element AND there is no associated content-stream glyph, Stage 10 now emits the element with `/ActualText` only (no `/K`, no `/Pg`) — this is the canonical path for OCR'd scanned text and is PDF/UA-valid.

**Table cells** are NOT `TaggedElement` instances — they live inside `el.specialist_data["cells"]` as dicts with keys `row_idx`, `col_idx`, `text`, `merged_from`, `is_header`, `is_row_header`. Stage 10 reads these directly to build `TR > TH/TD` structure.

### Stage 10 BDC injection (`content_stream_writer.py`)

`inject_bdc_markers` rewrites the page content stream entirely. It strips all existing `BDC`/`BMC`/`EMC` operators from the original stream (to prevent phantom MCID conflicts) then injects new `BDC`/`EMC` around text operators using `char_to_mcid` — a map from Stage-1 char index → MCID built from `merged_from` lists.

**Font-aware glyph counting:** the rewriter's positional counter `current_char_idx` advances by `len(bytes(operand)) // bytes_per_code` where `bytes_per_code` is resolved from the current Tf operator's font subtype (Type0 = 2, simple = 1). Using `len(str(operand))` instead — the previous bug — over-counts on Type0 fonts and desyncs the entire char↔glyph mapping for that page (table data falls to `/Artifact`, screen readers miss it, veraPDF still passes).

### CPU layout backend (`tagger/stage3_layout/cpu_layout_detector.py`)

Drop-in `LayoutModelAdapter` for Stage 3 that derives `LayoutRegion[]` from Stage-2 PageElements + pdfplumber primitives + Docling Heron. Branches on Stage-0 `page_type`:

- **NATIVE** — pdfplumber lattice → tables; Docling-table merge (TableFormer adds borderless); `_heading_lineboxes` for headings from pdfplumber `extract_text_lines`; `_merge_docling_headings` adds Heron-detected Title/Section-header regions that pdfplumber missed (additive only); `_image_boxes` with the loose `0.7` page-area threshold; XY-cut reading order.
- **MIXED / SCANNED** — Heron is the entire region source via `_detect_via_heron`. Page-spanning images (≥40% of page) are deliberately dropped — that's the page-image background of a scan, not a real Picture, and would otherwise `_center_inside`-block every OCR PageElement.

### Stage 1 scanned extractor (`tagger/stage1_extraction/scanned_extractor.py`)

RapidOCR PP-OCRv4 ONNX singleton with quality preset (`OCR.quality` → `text_score` / `box_thresh`). Renders pages at `STANDARD_DPI` so output pixel coords are already in 150-DPI standard space; no transform needed. Polygon → bbox conversion is axis-aligned (`min/max` over the 4 corners). `source = "rapidocr"` so Stage 4 can route OCR text differently if needed.

### Configuration (`tagger/config.py`)

All magic numbers are in frozen dataclasses exported as singletons. The dataclasses that read environment overrides via `field(default_factory=...)`:

- `LayoutConfig.backend` ← `TAGGER_LAYOUT_BACKEND`
- `AltTextConfig.mode` ← `TAGGER_ALT_TEXT_MODE`
- `OCRConfig.quality` ← `TAGGER_OCR_QUALITY`

Stage code imports from `tagger.config` — never hardcode thresholds.

### Running on Modal (legacy GPU path)

`run_modal.py` defines a Modal app with an A10G GPU. The `tagger/` directory is synced via `add_local_dir`. The pipeline runs remotely and returns `(tagged_pdf_bytes, report_bytes)`. The correct modal binary is `/Users/rahulkhatri/Library/Python/3.9/bin/modal`. Use this only when you specifically want the MinerU layout output for comparison; the CPU backend beats it on every dp-bench metric.

## Known architectural constraints

- **MinerU always on Modal** — Never run MinerU (Stage 3 `mineru` backend) locally; it pegs CPU on M1 8GB. Always use `modal run`. The CPU backend exists precisely so you don't need it.
- **Test suite always under `TAGGER_LAYOUT_BACKEND=cpu`** — the pipeline tests gate MinerU-output-specific assertions to the `mineru` backend, so a default-backend `pytest` will spawn MinerU and lag. The full suite is green locally on the CPU backend.
- **One ML model at a time** — Pipeline is designed for M1 8GB. Docling Heron + TableFormer + SigLIP coexist (small enough); MinerU was the one that needed isolation. Don't hold large model references across stages.
- **Stage 6 runs before Stage 8** — Any element created or reclassified by Stage 8 (heading ranker, list builder, etc.) bypasses Stage 6 validation.
- **pdfplumber `page.chars` skips** — Stage 1 skips chars where `text.isspace()` or bbox width/height < 0.1. These chars have no `p{N}_c{idx}` ID and are invisible to Stage 10's BDC injection.
- **QA runner needs `modal run`** — `scratch/run_qa_modal.py` must be invoked via `modal run`, not plain `python`. Running it directly causes `ClientClosed` errors when `.generate.remote()` tries to hold a connection without a Modal app context.
</content>
