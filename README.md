# Auto-Tagger PDF Pipeline

An end-to-end Python pipeline for automatically adding semantic structural tags to untagged or poorly tagged PDFs, targeting accessibility standards (PDF/UA-1) and the W3C Accessibility Conformance Testing (ACT) rules.

## 🚀 Where we stand

The pipeline runs in two interchangeable layout backends, selected by `TAGGER_LAYOUT_BACKEND` or `LAYOUT.backend` in `tagger/config.py`:

- **`cpu` (CPU-native, default for local work):** Docling Heron (RT-DETRv2 layout, MIT) for region classification, Docling TableFormer (MIT) for table structure, SigLIP zero-shot for figure-type classification, RapidOCR PP-OCRv4 ONNX for scanned-page OCR. **No MinerU, no AGPL anywhere.**
- **`mineru` (GPU on Modal):** the original MinerU2.5-Pro layout path retained for the throughput / Modal-fleet story.

### Measured headline numbers

**dp-bench (200 docs, native, CPU backend, no GPU):**

| metric | CPU pipeline | GPU pipeline (MinerU + V2 fixes) | Δ vs GPU |
|---|---|---|---|
| overall | **0.823** | 0.802 | **+0.022** |
| NID (reading order) | **0.888** | 0.874 | +0.014 |
| TEDS (tables) | **0.581** | 0.429 | **+0.152** |
| MHS (headings) | **0.720** | 0.716 | +0.005 |

The CPU pipeline now beats the GPU pipeline on every dp-bench metric at ~2.2 s/doc.

**Audit batch (14 real-world tagged PDFs, vs PREP and PDFix):**

| validator | OURS | PREP | PDFix |
|---|---|---|---|
| **veraPDF UA-1 compliant** | **9/14** | 6/14 | 9/14 |
| **W3C ACT pass / fail / N/A** (8 rules × 14 docs) | **84 / 0 / 28** | 84 / **2** / 26 | 77 / **3** / 32 |

We are the only one of the three with **zero** ACT-rule failures across the audit batch. PREP fails 1 heading-skip + 1 empty-heading; PDFix fails 3 empty-headings — all caught by our Stage-8 enforcers.

## 🛠️ Architecture

The pipeline processes documents in an immutable, declarative style where `PageElement` objects flow through ten stages, then a struct-tree writer:

```
Stage 0  page_classifier      Native vs scanned vs mixed vs corrupt
                              (sparse-text-density override catches image-of-text
                              docs where PREP injected a header-only text layer)
Stage 1a native_extractor     pdfplumber char-level extraction (born-digital)
Stage 1b scanned_extractor    RapidOCR (PP-OCRv4 ONNX) on scanned / mixed pages
Stage 2   text_merger          chars → words → line elements
Stage 3   layout_detector      pluggable LayoutModelAdapter:
                              - cpu_layout_detector: Heron + TableFormer + lattice
                                + heading-on-pdfplumber-lines + Heron-additive
                                semantic headings on native pages
                              - MinerU on Modal (legacy GPU path)
Stage 4+5 content_router       Maps PageElements into regions; specialists
          + specialists        produce TaggedElements (Docling TableFormer for
                              borderless tables — beats GPU on TEDS)
Stage 6   consistency_validator Rule engine; converts bad elements to Artifact
Stage 7   cross_page_merger    Merges elements split across page boundaries
Stage 8a  heading_ranker       H1–H6 assignment by font-tier rarity
Stage 8a' heading_hierarchy_enforcer  PDF/UA-1 7.4.2 (no skip), first-H1,
                              no empty/punct-only headings
Stage 8b  toc_detector
Stage 8c  artifact_detector    page numbers, repeated margin furniture, watermarks
Stage 8d  caption_detector
Stage 8e  list_builder         L > LI > Lbl + LBody nesting
Stage 8f  pdfua_structural_enforcer  empty/punct-only body → Artifact, every
                              Figure has /Alt, floating Caption → P
Stage 9   alt_text_generator   mode = "siglip" (default) | "placeholder" | "vlm"
                              — SigLIP zero-shot buckets + McGraw-Hill templates
                              with caption-aware suffix logic
Stage 10  struct_tree_writer   builds StructTreeRoot, injects BDC/EMC marked
                              content, font-aware glyph counting (Type0 = 2-byte)
```

The whole pipeline is **CPU-only by default** and runs locally in ~2.2 s/doc on M1, no MinerU spawned. The full test suite (266 passing + 3 skipped) runs locally in under 30 s.

## 🧪 Conformance audit module (`tagger/audit/`)

A read-only checker layer that scores any tagged PDF (ours, PREP, PDFix, anything) against the eight rules our pipeline cares about:

```
ACT-6cfa84   /  WCAG 1.1.1   Figure has /Alt or /ActualText
ACT-36b590   /  WCAG 1.3.1   Heading is non-empty
ACT-b40fd1   /  WCAG 3.1.1   Catalog /Lang is a valid BCP-47 tag
PDFUA-7.4.2  /  WCAG 1.3.1   No heading-level skips
PDFUA-7.1-10 /  WCAG 2.4.2   /Info /Title set AND /DisplayDocTitle true
PDFUA-7.5.2  /  WCAG 1.3.1   /Caption colocates with /Figure or /Table
PDFUA-7.5.3  /  WCAG 1.3.1   Every /LI is inside /L
PDFUA-7.1-1                  /MarkInfo /Marked is true
```

CLI: `python -m tagger.audit.act_rules <pdf> [...]` (add `--json` for the raw aggregate).

## ✅ What's working

1. **CPU-native layout** — Docling Heron + TableFormer, beats GPU on all four dp-bench metrics, ~2.2 s/doc.
2. **Scanned-PDF support** — RapidOCR (PP-OCRv4 ONNX) closes the one honest scope boundary of the CPU backend.
3. **Borderless tables** — Docling TableFormer beats GPU's `0.429 → 0.581` on TEDS.
4. **Figure alt-text** — SigLIP zero-shot buckets + McGraw-Hill templates with caption awareness.
5. **PDF/UA-1 + ACT enforcement** — heading-hierarchy + structural enforcers prevent the failure modes PREP and PDFix exhibit.
6. **PDF/UA-2 formula MathML** — `/Formula` elements carry MathML as a PDF 2.0 Associated File (`/AF` Supplement) + `/Alt`; LaTeX from the text layer by default, image→LaTeX (`TAGGER_FORMULA_RECOGNIZER=vlm`) optional. veraPDF UA-1 stays 106/106.
7. **Local test suite** — 275 passing under `TAGGER_LAYOUT_BACKEND=cpu`, no MinerU spawned, in under 45 s.

## ⏳ What's parked / deferred

- **Semantic formula MathML at scale** — the MathML substrate ships; richer semantic LaTeX for image-only / garbled-glyph formulas needs the image→LaTeX recogniser (`vlm` mode) activated via an isolated recogniser venv (pix2tex/UniMERNet pins conflict with the main venv, so it runs subprocess-only).
- **Color contrast (WCAG 1.4.3)** — handled in a separate repo per the user's call.
- **Heading-on-image semantics** on heavily scanned docs (Stage 1 OCR strips font signal, so H1/H2 distinction on scan-only pages relies on Heron region categories).

## 📦 Configuration knobs

All thresholds and backend choices live in `tagger/config.py`. The flags users actually set:

| env var | maps to | values |
|---|---|---|
| `TAGGER_LAYOUT_BACKEND` | `LAYOUT.backend` | `cpu` (default) / `picodet` / `mineru` |
| `TAGGER_ALT_TEXT_MODE` | `ALT_TEXT.mode` | `siglip` (default) / `placeholder` / `vlm` |
| `TAGGER_OCR_QUALITY` | `OCR.quality` | `speed` / `balanced` (default) / `quality` |
| `TAGGER_FORMULA_RECOGNIZER` | `FORMULA.recognizer` | `text` (default) / `vlm` |

## 🚀 Quick start

```bash
# Tag a PDF locally on CPU (no MinerU, no GPU)
TAGGER_LAYOUT_BACKEND=cpu python -m tagger.cli tag input.pdf -o output.pdf --report report.json

# Audit a tagged PDF against ACT + PDF/UA-1 rules
python -m tagger.audit.act_rules output.pdf

# Run the test suite (266 passing, CPU backend)
TAGGER_LAYOUT_BACKEND=cpu pytest -q
```

## 📚 More

- `CLAUDE.md` — repo-level guidance for Claude Code work in this repo
- `BLUEPRINT.md` — technical foundation document
- `THROUGHPUT_ARCHITECTURE.md` — throughput + acquisition-readiness argument
</content>
