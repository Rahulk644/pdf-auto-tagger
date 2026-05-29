# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run full pipeline locally on one PDF
python -m tagger.cli tag input.pdf -o output.pdf --report report.json

# Run only page classification
python -m tagger.cli classify input.pdf

# Run Stages 0-2 (extract + merge, no ML)
python -m tagger.cli extract input.pdf

# Start local Flask API (port 5002)
python -m tagger.cli serve

# Run on Modal GPU (A10G) — processes all 5 corpus PDFs
/Users/rahulkhatri/Library/Python/3.9/bin/modal run run_modal.py

# Run on Modal GPU — Miramar + Summary of Revenues only
/Users/rahulkhatri/Library/Python/3.9/bin/modal run scratch/run_modal_targeted.py

# Run QA evaluation (extraction local, Gemma inference on Modal H100)
/Users/rahulkhatri/Library/Python/3.9/bin/modal run scratch/run_qa_modal.py

# Analyze QA results across documents
python analyze_qa_report.py

# Run tests
pytest
pytest tests/test_stage6.py           # single file
pytest tests/test_stage5.py -v        # verbose
```

**Python environments:**
- `.venv3/` — has `pikepdf`, `pdfplumber`, `pillow`, `modal` (use for local pipeline work)
- `/Users/rahulkhatri/Library/Python/3.9/bin/` — has `modal` CLI
- Modal remote runs Python 3.11 with the `tagger/` directory synced as `/root/tagger`

## Architecture

### 10-Stage Pipeline (`tagger/pipeline.py`)

`AutoTaggerPipeline.run()` calls stages sequentially. Each stage receives `DocumentData` (which holds a `pages` dict of `PageData`) and modifies it in place.

```
Stage 0  page_classifier     — Native vs scanned detection (pdfplumber heuristics)
Stage 1  native_extractor    — Character-level extraction; assigns p{N}_c{idx} IDs
Stage 2  text_merger         — Merges chars → words → line elements (PageElement)
Stage 3  layout_detector     — MinerU detects layout regions (Title, Table, Figure…)
                               + pdfplumber table fallback for tables MinerU misses
Stage 4+5 content_router     — Maps Stage 2 PageElements into Stage 3 regions;
           specialists        — Table/figure/formula specialists produce TaggedElements
Stage 6  consistency_validator — Rule engine; converts bad elements to Artifact
Stage 7  cross_page_merger   — Merges elements split across page boundaries
Stage 8  semantic refinement — Heading level ranking, TOC, artifact, caption, list
Stage 9  alt_text_generator  — Placeholder alt text for Figure elements
Stage 10 struct_tree_writer  — Builds PDF struct tree + injects BDC/EMC markers
```

### Key data flow invariants

**Coordinate spaces** — Two spaces coexist and must never be mixed:
- **Standard (150 DPI, top-left origin):** `PageElement.bbox`, `TaggedElement.bbox`, `LayoutRegion.bbox`. Used for all inter-stage comparisons.
- **Native pdfplumber (72 DPI, top-left origin):** Only used inside Stage 1, Stage 5 table extraction, and Stage 10 BDC injection. Convert with `coord_transformer.py`.

**Element ID scheme** — Stage 1 assigns `element_id = f"p{page_num}_c{char_idx}"` where `char_idx` is the raw `enumerate(page.chars)` index (skipping only blank/zero-size chars). This same ID scheme is used in `merged_from` lists all the way to Stage 10's `inject_bdc_markers` which maps char indices back to MCIDs.

**`merged_from`** — Every `PageElement` and `TaggedElement` carries a `merged_from: list[str]` of Stage-1 char IDs. Stage 10 uses these to decide which content-stream characters belong to each struct tree element. If `merged_from` is empty for a non-table element, Stage 10 cannot inject BDC markers for it.

**Table cells** are NOT `TaggedElement` instances — they live inside `el.specialist_data["cells"]` as dicts with keys `row_idx`, `col_idx`, `text`, `merged_from`, `is_header`, `is_row_header`. Stage 10 reads these directly to build `TR > TH/TD` structure.

### Stage 10 BDC injection (`content_stream_writer.py`)

`inject_bdc_markers` rewrites the page content stream entirely. It strips all existing `BDC`/`BMC`/`EMC` operators from the original stream (to prevent phantom MCID conflicts) then injects new `BDC`/`EMC` around text operators using `char_to_mcid` — a map from Stage-1 char index → MCID built from `merged_from` lists. Characters not in `char_to_mcid` (whitespace skipped by Stage 1) fall outside any BDC block.

### QA evaluation (`tagger/qa/` + `scratch/run_qa_modal.py`)

`scratch/run_qa_modal.py` is the QA runner. It runs locally via `modal run` (so the Modal gRPC client stays alive), does all PDF extraction and rendering locally, and calls `GemmaInference.generate.remote()` on Modal for each chunk:

- **Local:** pdfplumber MCID extraction, pdfminer struct-tree parsing, fitz page rendering
- **Remote:** Gemma 4 31B on H100 (`qa-gemma4-inference` app, up to 9 containers)
- **Chunking:** 60 elements/chunk, sequential within a page, pages processed sequentially
- **Rules:** `tagger/qa/rules_db.py` injects PDF/UA + WCAG 2.2 rules into prompts based on tags present

Results saved to `/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal/` as `qa_<stem>.json`. `analyze_qa_report.py` in the Tagger root prints per-document breakdowns.

**Deploying inference changes:** `modal deploy tagger/qa/modal_inference.py`

**Current production auditor = Gemma 4 E4B via vLLM (replaces the 31B above).** `tagger/qa/modal_gemma_vllm.py` serves `google/gemma-4-E4B-it` on vLLM (H100, thinking ON, temp 0.1) — the small reasoning model beats the old 31B (transformers `.generate()`) at far lower latency, and strips the thinking trace server-side. Deploy: `modal deploy tagger/qa/modal_gemma_vllm.py` (needs the `huggingface-prep` Modal secret for the gated model). HTTP contract: POST `{image_b64, prompt, max_tokens=8000, temperature=0.1, enable_thinking=true}` → `{response (clean JSON array), finish_reason, input/output_tokens, elapsed_s}`. Drive it with the prompt-v2 client `run_corpus_modal.py` (in the PREP-QA-Tool repo): one page image + one chunk of ≤20 elements per request, fan out concurrently (never batch multiple chunks — Gemma 4's heterogeneous head dims make vLLM's Triton backend JIT mid-inference and hang), parse the LAST balanced `[...]`, retry on `finish_reason=="length"` and for any omitted element id. Run: `MODAL_URL=<endpoint> PDFS_DIR=<tagged outputs> PARALLEL=10 python run_corpus_modal.py`.

**Corpus:** Five test PDFs. Tagged outputs land in `output_modal/`.

### Benchmark evaluation (`tagger/benchmark/` + `scratch/run_benchmark.py`)

The third eval axis (alongside veraPDF = ISO conformance and Gemma QA = noisy tag quality): scores tagged output against the **PDF-Accessibility-Benchmark** (125 scholarly PDFs, expert WCAG/PDF-UA labels per criterion). CPU/deterministic — one bounded Modal regen only for the strip+V2 remediation pass.

- `struct_utils.py` — shared struct-tree readers; identifies elements by `/S` presence (NOT the optional `/Type /StructElem`) and resolves `/RoleMap`. The `scratch/layout_*` harness tools import from here.
- `verdicts/` — `base.py` (`Verdict` + `CannotDeriveReason` + `derive_verdict` dispatch) + one module per criterion (verdicts derive from the **expert discriminator**, not veraPDF conformance — they can diverge).
- `loader.py` → `harness.py` (checker vs remediation framing; the hard-assert keeps a `failed` label as an input selector, never an agreement target) → `report.py` (scorecard).
- Run: `PYTHONPATH=. python scratch/run_benchmark.py <benchmark_root> [--remediation-dir DIR]`.

Three axes are correlated but distinct — a doc can be WCAG-accessible yet veraPDF-non-conformant. Don't conflate them.

### Configuration (`tagger/config.py`)

All magic numbers are in frozen dataclasses exported as singletons (`PAGE_CLASSIFIER`, `TEXT_MERGER`, `LAYOUT`, `TABLE`, `VALIDATOR`, `SEMANTIC`, `WRITEBACK`, `PIPELINE`). Stage code imports from `tagger.config` — never hardcode thresholds.

### Running on Modal

`run_modal.py` defines a Modal app with an A10G GPU. The `tagger/` directory is synced via `add_local_dir`. The pipeline runs remotely and returns `(tagged_pdf_bytes, report_bytes)`. The correct modal binary is `/Users/rahulkhatri/Library/Python/3.9/bin/modal`.

## Known architectural constraints

- **MinerU always on Modal** — Never run MinerU (Stage 3) locally; it pegs CPU on M1 8GB. Always use `modal run`.
- **One ML model at a time** — Pipeline is designed for M1 8GB. MinerU (Stage 3) loads, runs, and is GC'd before Stage 9's Qwen VL loads. Don't hold model references across stages.
- **Stage 6 runs before Stage 8** — Any element created or reclassified by Stage 8 (heading ranker, list builder, etc.) bypasses Stage 6 validation.
- **pdfplumber `page.chars` skips** — Stage 1 skips chars where `text.isspace()` or bbox width/height < 0.1. These chars have no `p{N}_c{idx}` ID and are invisible to Stage 10's BDC injection.
- **QA runner needs `modal run`** — `scratch/run_qa_modal.py` must be invoked via `modal run`, not plain `python`. Running it directly causes `ClientClosed` errors when `.generate.remote()` tries to hold a connection without a Modal app context.
