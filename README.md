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
| overall | **0.839** | 0.802 | **+0.037** |
| NID (reading order) | **0.888** | 0.874 | +0.014 |
| TEDS (tables) | **0.740** | 0.429 | **+0.311** |
| MHS (headings) | **0.726** | 0.716 | +0.010 |

The CPU pipeline beats the GPU pipeline on every dp-bench metric at ~2.2 s/doc. TEDS
jumped **0.581 → 0.740** after the table fixes (nearest-cell text-fill + the Stage-10
cell-drop fix + a Stage-6 fix that stopped valid tables being dropped to /Artifact) —
a neutral cross-dataset shootout (PubTabNet + FinTabNet)
proved the structure model was never the bottleneck; native cell-text-fill was.

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

The whole pipeline is **CPU-only by default** and runs locally on M1, no MinerU spawned. Stage 3 runs **one** Heron inference per native page (shared across the table/heading/formula merges) and a per-document page-image + pdfplumber cache (`tagger/page_cache.py`) removes duplicate rasterization/parsing. The full test suite (303 passing + 3 skipped) runs locally in under 45 s.

## 🧪 Conformance audit module (`tagger/audit/`)

A read-only checker layer that scores any tagged PDF (ours, PREP, PDFix, anything) against the eight rules our pipeline cares about:

```
ACT-6cfa84   /  WCAG 1.1.1   Figure has /Alt or /ActualText
ACT-36b590   /  WCAG 1.3.1   Heading is non-empty
ACT-b40fd1   /  WCAG 3.1.1   Catalog /Lang is a valid BCP-47 tag
PDFUA-7.4.2  /  WCAG 1.3.1   Heading hierarchy: no skip, first=H1, no numbered/unnumbered mix
PDFUA-7.1-10 /  WCAG 2.4.2   /Info /Title set AND /DisplayDocTitle true
PDFUA-7.5.2  /  WCAG 1.3.1   /Caption colocates with /Figure or /Table
PDFUA-7.5.3  /  WCAG 1.3.1   Every /LI is inside /L
PDFUA-7.1-1                  /MarkInfo /Marked is true
```

CLI: `python -m tagger.audit.act_rules <pdf> [...]` (add `--json` for the raw aggregate).

Three reporting surfaces sit on the same checks (no new logic, just re-expression):
- `tagger.audit.act_rules` — the eight ACT/PDF-UA rules above.
- `tagger.audit.matterhorn` — the same results mapped to **Matterhorn Protocol 1.1** failure-condition IDs (e.g. `13-004` figure Alt, `14-003` heading skip, `11-001` Lang, `07-001` DisplayDocTitle), so output speaks the same language as PAC.
- `tagger.audit.screen_reader` — a **deterministic, cross-platform screen-reader linearizer**: walks the struct tree in reading order and emits what NVDA/JAWS/VoiceOver would announce (heading levels, figure Alt, table dims, lists) while silencing artifacts. `linearize(pdf).as_text()` is the transcript; `smell_test(pdf)` returns the issues a reader would hit. `scripts/screen_reader_corpus.py` sweeps a whole directory and gates on issues.

## 🔒 CI conformance gate

`scripts/verapdf_gate.py` tags fixtures through the full pipeline and pipes each output through the **veraPDF** CLI — non-zero exit on any non-compliance. Wired into `.github/workflows/ci.yml` (pytest + the gate on every push). The deterministic line under any "PDF/UA compliant" claim: tagging that *looks* right but fails veraPDF fails the build.

## 📊 Conformance vs. correctness (the honest split)

Two different questions, two different answers — don't conflate them:

- **Conformance** (is the structure *valid/present*? — syntactic, deterministic): we self-score this fully and automatically (veraPDF 106/106, ACT, Matterhorn, intrinsic screen-reader defects). This is our strength.
- **Correctness** (are the tags *actually right*? — semantic): needs ground truth. Measured against the **35-doc expert benchmark** (PDF-Accessibility-Benchmark), our tags agree with human experts **90–100% on structural criteria** (reading order, semantic tagging, table structure, hyperlinks) and **beat Adobe's checker** (e.g. reading order 10 vs 5). The one genuine hole: **alt-text *quality* (0% expert agreement)** — content quality is not yet self-validatable. See `BLUEPRINT.md` for the methodology.

## ✅ What's working

1. **CPU-native layout** — Docling Heron + TableFormer, beats GPU on all four dp-bench metrics, ~2.2 s/doc.
2. **Scanned-PDF support** — RapidOCR (PP-OCRv4 ONNX) closes the one honest scope boundary of the CPU backend.
3. **Borderless tables** — Docling TableFormer beats GPU's `0.429 → 0.581` on TEDS.
4. **Figure alt-text** — SigLIP zero-shot buckets + McGraw-Hill templates with caption awareness.
5. **PDF/UA-1 + ACT enforcement** — heading-hierarchy + structural enforcers prevent the failure modes PREP and PDFix exhibit.
6. **PDF/UA-2 formula MathML** — `/Formula` elements carry MathML as a PDF 2.0 Associated File (`/AF` Supplement) + `/Alt`; LaTeX from the text layer by default, image→LaTeX (`TAGGER_FORMULA_RECOGNIZER=vlm`) optional. veraPDF UA-1 stays 106/106.
7. **Conformance + correctness reporting** — Matterhorn IDs, cross-platform screen-reader linearizer, veraPDF CI gate, and an expert-benchmark correctness harness.
8. **Local test suite** — 303 passing under `TAGGER_LAYOUT_BACKEND=cpu`, no MinerU spawned, in under 45 s.

## ⚠️ Known limitations

- **Alt-text quality is unvalidated** (0% expert agreement) — figures get an `/Alt`, but whether it's *accurate* is the open hole. CPU-VLM pilots (SmolVLM 256M/500M) confirmed small vision models hallucinate chart specifics (language-prior dominance); the shippable CPU answer is type-routed SigLIP + OCR labels + a value-safe template, not a bigger VLM.
- **Semantic correctness isn't self-certifiable on arbitrary docs** — only scored against the 35-doc expert set. The planned fix is a split-pipeline judge: deterministic perception (pdfplumber) + a text-only LLM reasoning over physical-layout vs our tag tree (no VLM in the perception loop → no visual hallucination).
- **Screen-reader check is a simulation** — it linearizes *our own tags*, so it catches missing/broken structure, not whether the structure is *correct*. Real NVDA/JAWS (Windows) / VoiceOver (macOS) remain out-of-process jobs.
- **Table cell structure** trails (TEDS ≈ 0.58) — the other measured soft spot.

## ⏳ Parked / opt-in

- **Image→LaTeX formula recogniser** (`TAGGER_FORMULA_RECOGNIZER=vlm`) — `rapid_latex_ocr` (onnx) in an isolated py3.11 venv (`~/.tagger/latexocr_venv`), batched per doc, opt-in, graceful no-op to text. Recovers real LaTeX for born-digital STEM formulas the text layer flattens to `\text{}` (measured 19%→54%). Default stays `text` (no throughput cost).
- **PicoDet layout backend** (`TAGGER_LAYOUT_BACKEND=picodet`) — A/B-evaluated, NOT default (lost the MHS gate, ~50% slower on CPU); retained for re-eval.
- **Color contrast (WCAG 1.4.3)** — separate repo; integration hook only.
- **Remediation policy:** adding structure (tags/Alt/MathML/reading order) is always-on; modifying the *source* (fonts, contrast) is detect-and-report by default, opt-in/gated only.

## 📦 Configuration knobs

All thresholds and backend choices live in `tagger/config.py`. The flags users actually set:

| env var | maps to | values |
|---|---|---|
| `TAGGER_LAYOUT_BACKEND` | `LAYOUT.backend` | `cpu` (default) / `picodet` / `mineru` |
| `TAGGER_ALT_TEXT_MODE` | `ALT_TEXT.mode` | `siglip` (default) / `placeholder` / `vlm` |
| `TAGGER_OCR_QUALITY` | `OCR.quality` | `speed` / `balanced` (default) / `quality` |
| `TAGGER_FORMULA_RECOGNIZER` | `FORMULA.recognizer` | `text` (default) / `vlm` |
| `TAGGER_TABLE_ENGINE` | `TABLE.engine` | `tableformer` (default) / `ppstructure` / `slanet` |

## 🚀 Quick start

```bash
# Tag a PDF locally on CPU (no MinerU, no GPU)
TAGGER_LAYOUT_BACKEND=cpu python -m tagger.cli tag input.pdf -o output.pdf --report report.json

# Audit a tagged PDF: ACT/PDF-UA rules, Matterhorn IDs, or screen-reader transcript
python -m tagger.audit.act_rules output.pdf
python -m tagger.audit.matterhorn output.pdf
python -m tagger.audit.screen_reader output.pdf          # --issues for just the problems

# Conformance gate (needs veraPDF) + screen-reader corpus sweep
python scripts/verapdf_gate.py
python scripts/screen_reader_corpus.py <dir-of-tagged-pdfs>

# Run the test suite (303 passing, CPU backend)
TAGGER_LAYOUT_BACKEND=cpu pytest -q
```

## 📚 More

- `CLAUDE.md` — repo-level guidance for Claude Code work in this repo
- `BLUEPRINT.md` — technical foundation document
- `THROUGHPUT_ARCHITECTURE.md` — throughput + acquisition-readiness argument
</content>
