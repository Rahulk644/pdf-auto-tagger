# Auto-Tagger Architecture & Technical Blueprint

Comprehensive technical foundation: the pipeline shape, the two interchangeable layout backends (CPU-native default + legacy MinerU GPU), the Stage-8 conformance enforcers, and the read-only audit layer.

## 1. System overview

The Auto-Tagger is a 10-stage pipeline that ingests an untagged or poorly tagged PDF and outputs a PDF/UA-1-conformant tagged PDF (H1–H6, P, L > LI > Lbl + LBody, Figure with /Alt, Table with TR > TH/TD, Caption colocated, page furniture artifacted).

It runs in two interchangeable layout backends, selected by `LAYOUT.backend` (env: `TAGGER_LAYOUT_BACKEND`):

- **`cpu` (default for local + CI):** Docling Heron (RT-DETRv2, MIT) for region detection, Docling TableFormer (MIT) for table structure, SigLIP (Apache 2.0) zero-shot for figure-type classification, RapidOCR PP-OCRv4 ONNX (Apache 2.0) for scanned-page OCR. No MinerU. No AGPL. Runs locally on M1 in ~2.2 s/doc.
- **`mineru` (legacy GPU on Modal):** MinerU2.5-Pro on Modal A10G. Retained for the throughput / Modal-fleet story (see `THROUGHPUT_ARCHITECTURE.md`) and dp-bench baseline comparisons.

### The 10-stage pipeline

```
0  Page classifier         Native / scanned / mixed / corrupt (with sparse-text-
                           density override for image-of-text docs)
1a Native extractor        pdfplumber chars on born-digital pages
1b Scanned extractor       RapidOCR PP-OCRv4 on scanned / mixed pages
2  Text merger             chars → words → line elements
3  Layout detector         Pluggable adapter (CPU-native Heron + TableFormer
                           default; MinerU on Modal as a fallback)
4+5 Router + specialists   Map PageElements into regions; specialists (Docling
                           TableFormer for tables, figure, formula) produce
                           TaggedElements
6  Consistency validator   Rule engine, deterministic safety checks
7  Cross-page merger       Splits/joins elements spanning page boundaries
8a Heading ranker          H1–H6 from font-tier rarity
8a' Heading-hierarchy enforcer  PDF/UA-1 7.4.2 — no level skips, first heading
                           is H1, empty/punct-only → /Artifact
8b TOC detector
8c Artifact detector       Running headers/footers, page numbers, repeated
                           margin watermarks
8d Caption detector
8e List builder            L > LI > Lbl + LBody
8f PDF/UA structural enforcer  Empty/punct-only body → /Artifact, every Figure
                           has /Alt, floating Caption → /P
9  Alt-text generator      SigLIP buckets + McGraw-Hill templates with
                           caption-aware suffix logic; decorative figures →
                           /Artifact (PDF4 / H67); placeholder + VLM modes
                           retained
10 Struct tree writeback   Builds StructTreeRoot in reading order; font-aware
                           glyph counting for the BDC/EMC injection
```

### Read-only conformance audit layer (`tagger/audit/`)

Separate from the tagging pipeline. Takes any tagged PDF (ours, PREP, PDFix, anything) and reports per-rule pass / fail / N/A for the eight rules we explicitly cover:

```
ACT-6cfa84   /  WCAG 1.1.1   Figure has /Alt or /ActualText
ACT-36b590   /  WCAG 1.3.1   Heading is non-empty
ACT-b40fd1   /  WCAG 3.1.1   /Lang valid BCP-47 tag
PDFUA-7.4.2  /  WCAG 1.3.1   No heading-level skips
PDFUA-7.1-10 /  WCAG 2.4.2   /Info /Title + /DisplayDocTitle = true
PDFUA-7.5.2  /  WCAG 1.3.1   /Caption colocates with /Figure or /Table
PDFUA-7.5.3  /  WCAG 1.3.1   /LI inside /L
PDFUA-7.1-1                  /MarkInfo /Marked = true
```

CLI: `python -m tagger.audit.act_rules <pdf> [...]` — per-doc summary; `--json` for the raw aggregate. Used in the audit-batch comparison to validate that the in-pipeline Stage-8 enforcers catch what they're supposed to.

## 2. Infrastructure

### CPU-native (default)

Everything runs locally on M1 / commodity CPU. No GPU, no Modal, no AGPL deps. The whole `.venv3` is ~1.3 GB including torch + transformers + Docling + RapidOCR. ~2.2 s/doc on dp-bench. Test suite (266 passing) runs in under 30 s.

### Modal (legacy MinerU layout + production QA auditor)

Modal is still used for two things even in a CPU-first deployment:

- **Layout-only legacy path** (`run_modal.py`): swap to `TAGGER_LAYOUT_BACKEND=mineru` and route Stage 3 to MinerU on A10G if you specifically need the MinerU output for comparison or for a workflow that already standardized on it.
- **QA semantic validator** (Gemma-4-E4B on Modal H100, `tagger/qa/modal_gemma_vllm.py`): a separate eval-time service that scores tag quality against PDF/UA + WCAG rules. The pipeline output is what's evaluated; the QA layer doesn't change the tagging output.

The same E4B endpoint is reused as the optional Stage 9 alt-text backend (`ALT_TEXT.mode = "vlm"`) when a GPU is available and richer chart/diagram descriptions are wanted.

## 3. Headline measured numbers

**dp-bench (200 docs, CPU backend, no GPU, ~2.2 s/doc):**

| metric | CPU pipeline | GPU pipeline (MinerU + V2 fixes) | Δ vs GPU |
|---|---|---|---|
| overall | **0.823** | 0.802 | **+0.022** |
| NID (reading order) | **0.888** | 0.874 | +0.014 |
| TEDS (tables) | **0.581** | 0.429 | **+0.152** |
| MHS (headings) | **0.720** | 0.716 | +0.005 |

**Audit batch (14 real-world tagged PDFs):**

| validator | OURS | PREP | PDFix |
|---|---|---|---|
| veraPDF UA-1 compliant | **9/14** | 6/14 | 9/14 |
| W3C ACT pass / fail / N/A | **84 / 0 / 28** | 84 / **2** / 26 | 77 / **3** / 32 |

The CPU pipeline beats GPU on every dp-bench metric and is the only one of the three with zero ACT-rule failures across the audit batch — the Stage-8 enforcers catch exactly the failure modes PREP and PDFix exhibit.

## 4. Architecture notes

### Pluggable layout backend (Stage 3)

Stage 3 is selected by `LAYOUT.backend` in `tagger/config.py` (env `TAGGER_LAYOUT_BACKEND`). The default is `cpu` for local work; `mineru` is the legacy GPU path. Both backends produce the same `LayoutRegion[]` interface, so Stages 4–10 are agnostic.

### CPU layout detector branches on Stage-0 classification

`cpu_layout_detector.detect_regions(pdf_path, page_num, elements, page_type)` branches:

- **NATIVE pages** — pdfplumber primitives (lattice tables, text-line heading detection, image bboxes), augmented with Heron-detected tables (`_merge_docling_tables`) and Heron-detected semantic headings (`_merge_docling_headings`, UNION-only — never removes a pdfplumber heading, so TEDS/NID can't structurally regress; this is what closed the MHS gap to GPU).
- **MIXED / SCANNED pages** — Heron is the entire region source via `_detect_via_heron`. Page-spanning images (≥40% of page) are dropped — that's the scan background, not a real Picture, and would otherwise `_center_inside`-block every OCR PageElement.

### Stage 1 split: native vs scanned

Stage 1 is two co-routines that run in parallel and merge their results by page number:

- `native_extractor` — pdfplumber chars (zero on a scanned page).
- `scanned_extractor` — RapidOCR PP-OCRv4 ONNX on SCANNED / MIXED pages. PageElements get `source = "rapidocr"` and IDs `p{N}_o{idx}` so Stage 4 can route them differently from native chars when needed.

### Stage 8 conformance enforcers

Two new deterministic enforcers run inside Stage 8 after the existing semantic passes:

- `heading_hierarchy_enforcer` — PDF/UA-1 7.4.2: no level skips (H1 → H3 collapses to H1 → H2), first heading = H1, empty heading → /Artifact, punctuation-only heading → /Artifact. R3 + R4 run before R1 + R2 so an empty H2 between H1 and H3 doesn't falsely look like a present-but-skipped H2.
- `pdfua_structural_enforcer` — S1: empty body element (/P, /Caption, /Note, /BlockQuote) → /Artifact. S2: punctuation-only body → /Artifact. S3: every surviving Figure gets a placeholder /Alt + needs_review flag (belt-and-braces). S4: floating /Caption (no adjacent /Figure or /Table on the same page within 80 px) → /P.

Both enforcers return a stats dict so the pipeline can log what was changed and tests can assert on it. 14 unit tests cover R1–R4 + S1–S4.

### Stage 10 font-aware glyph counting

The content-stream rewriter's positional counter (`current_char_idx` in `_rewrite_stream`) advances by `len(bytes(operand)) // bytes_per_code`. `bytes_per_code` is resolved from the current `Tf` operator's font subtype — Type0 = 2, simple = 1. Using `len(str(operand))` instead (the previous behavior) over-counts on Type0 fonts and desyncs the entire char↔glyph mapping for that page; the table data falls to `/Artifact`, screen readers miss it, veraPDF still passes. The fix recovered substantial table content on Type0-font pages and is one of the reasons the CPU pipeline now beats GPU on TEDS.

### Stage 9 alt-text and the PDF/UA-2 formula MathML path

Stage 9 default is SigLIP zero-shot bucket → McGraw-Hill template. Formulas now carry MathML as a PDF 2.0 Associated File: Stage 3 merges Heron `Formula` regions on native pages (`_merge_docling_formulas`), Stage 5 derives LaTeX (text layer; optional image→LaTeX via `TAGGER_FORMULA_RECOGNIZER=vlm`, subprocess-isolated), `mathml_emitter` converts via `latex2mathml`, and Stage 10's `_embed_mathml_af` attaches `/AF` (`/Supplement`, `application/mathml+xml`) on the `/Formula` element + `/Alt` + catalog `/AF`. veraPDF UA-1 stays 106/106.

### Performance: one Heron pass + shared IO cache

Stage 3 used to run the Heron region detector 3× per native page (via the table, heading, and formula merges). It now computes the regions once and passes them down (3× → 1× inference on the ~40%-of-runtime stage). `tagger/page_cache.py` adds a per-document page-image cache (`render_page`, fitz, `lru_cache(maxsize=8)` bounded for M1 8 GB) shared by Heron / TableFormer / picodet / the formula renderer, and a cached `open_pdf` handle so Stages 1/3/5 parse the PDF once. `pipeline.run()` clears both per document. All behavior-preserving (dp-bench identical).

### Audit + reporting surfaces (`tagger/audit/`)

Read-only, separate from tagging. `act_rules` evaluates the 8 ACT/PDF-UA rules; `matterhorn` re-expresses them as Matterhorn 1.1 failure-condition IDs; `screen_reader` linearizes the struct tree into the AT announcement stream (`smell_test` returns intrinsic reading-experience defects). `scripts/verapdf_gate.py` + `.github/workflows/ci.yml` gate every push on veraPDF UA-1; `scripts/screen_reader_corpus.py` sweeps a directory.

### Correctness methodology (conformance ≠ correctness)

Conformance (valid/present) is self-scored fully (veraPDF/ACT/Matterhorn/intrinsic defects). Correctness (tags actually right) needs ground truth: the 35-doc **PDF-Accessibility-Benchmark** (= our PDF-A-B; `tagger/benchmark/`, `scratch/run_benchmark.py`). Re-tagging all 35 with current code and scoring vs expert labels: structural criteria 90–100% agreement, beating Adobe's checker; **alt-text quality 0%** is the unvalidated content-quality hole. CPU-VLM pilots (SmolVLM 256M/500M) confirmed small vision models hallucinate chart specifics (language-prior dominance) — so the planned semantic *judge* keeps VLMs out of the perception loop: deterministic perception (pdfplumber) → text-only LLM reasoning over physical-layout vs tag-tree.

## 5. What's parked / opt-in

- **Image→LaTeX formula recogniser** (`vlm` mode) — needs an isolated venv (pix2tex/UniMERNet pins conflict with the main venv → subprocess-only). The MathML substrate ships regardless.
- **Semantic-correctness judge** — deterministic perception + text-only quantized LLM (llama.cpp on M1, OpenVINO on x86 prod); the real "automated semantics" engine, not yet built.
- **Type-routed alt-text upgrade** — SigLIP bucket + OCR'd labels + value-safe template for data-bearing figures (small VLMs unreliable here per the pilot).
- **PicoDet layout backend** — A/B-evaluated, not default (lost MHS gate, ~50% slower on CPU); retained for re-eval.
- **Color contrast (WCAG 1.4.3)** — separate repo; integration hook only.
- **Remediation policy** — structure additions always-on; source modifications (fonts/contrast) detect-and-report by default, opt-in/gated only.
</content>
