# Auto-Tagger PDF Pipeline

An end-to-end Python pipeline for automatically adding semantic structural tags to untagged or poorly tagged PDFs, targeting accessibility standards (PDF/UA) and semantic QA compliance.

## 🚀 Where We Stand

We have successfully built and integrated a full **10-stage pipeline** capable of reading an untagged PDF, extracting characters, merging them into words and lines, classifying layouts (using MinerU), logically ordering the text, predicting semantic artifacts (Headers/Footers), and generating a fully valid Tagged PDF (`PyMuPDF/fitz`).

The pipeline handles complex spatial heuristics to correctly merge text based on font size and line gaps, overcoming major structural fragmentation issues.

**Current state (verified):** the real pipeline produces **veraPDF PDF/UA-1-conformant output on all 5 PREP corpus docs (5/5)** — ahead of the commercial incumbent PREP (3/5) on the same deterministic standard. Output is measured on three distinct axes: **veraPDF** (ISO/PDF-UA conformance), the **PDF-Accessibility-Benchmark** (125 expert-labelled scholarly docs; we agree 90–100% with experts on the addressable structural criteria and out-agree Adobe on every comparable one), and a **Gemma-4-E4B QA auditor** (tag-quality). QA runs on Modal (vLLM/H100); tagging runs on Modal A10G (MinerU).

## 🛠️ Architecture

The pipeline processes documents in an immutable, declarative style where `PageElement` objects flow through these stages:

- **Stage 0: Page Classifier** (Native PDF vs Image/Scanned detection)
- **Stage 1: Native Extractor** (Character-level extraction using `pdfplumber`)
- **Stage 2: Text Merger** (Multi-pass algorithm merging characters -> words -> lines using dynamic font-based spatial gaps)
- **Stage 3: Layout Detector** (Deep Learning based layout detection using **MinerU** for bounding boxes and reading order)
- **Stage 4+5: Content Router + Specialists** (Maps Stage 2 text lines into Stage 3 regions; table/figure/formula specialists produce `TaggedElement`s)
- **Stage 6: Consistency Validator** (Rule-based safety checks, e.g., overlapping regions, font hierarchy validation)
- **Stage 7: Cross-Page Merger** (Merges elements split across page boundaries)
- **Stage 8: Semantic Refinement** (Heading-level ranking, TOC, caption, list building, and artifact detection — running headers/footers/page-numbers plus repeated vertical-margin watermarks)
- **Stage 9: Alt-Text Generator** (Placeholder `/Alt` by default; optional VLM mode — default backend Gemma-4-E4B, Qwen2.5-VL retained)
- **Stage 10: Struct Tree Writeback** (Builds the `StructTreeRoot` in reading order and injects BDC/EMC marked content; `PyMuPDF`/`pikepdf`)

## ✅ What is Working

1. **Precision Text Extraction**: The baseline `pdfplumber` character extraction successfully maps bounding boxes, font metadata (bold/italic), and color data (resolving early grayscale inversion bugs).
2. **Spatial Merging (Stage 2)**: Dynamic horizontal and vertical gap calculation based on `avg_char_width` works flawlessly. Modifying `word_gap_multiplier = 1.0` and `line_gap_multiplier = 3.0` ensures multi-column financial tables don't accidentally fuse "Row Header" and "$1,000,000" into a single string.
3. **Infinite Loop Protections (Stage 6)**: Optimized `OverlappingRegionRule` down to page-local `O(N_p^2)` to prevent validation hangs.
4. **Output Generation (Stage 10)**: The `PyMuPDF` write-back layer correctly injects the semantic tree (`StructTreeRoot`) and outputs valid, compliant PDFs.

## ❌ What is NOT Working (Or Paused)

1. **Nested Table Generation**: Currently paused (`ON ICE`). Our pipeline wraps entire tables inside a single, monolithic `<Table>` tag. This causes `P -> TD` leakage errors during Semantic QA, as the ground-truth standard expects individual `<TR>`, `<TH>`, and `<TD>` tags.
2. **MinerU Tabular Confusion**: MinerU occasionally classifies dense tabular data as generic `text` blocks. Because Stage 4 relies on MinerU's macro-boxes, perfectly split columns in Stage 2 get concatenated back together if MinerU surrounds them both with a single `text` boundary.
3. **Artifact Bleed (Path C)**: Minor boundary overlap between standard paragraphs and edge-case headers/footers.

## ⚠️ What is Risky

- **Heavy Dependency on Stage 3 (MinerU)**: The biggest architectural risk is treating MinerU bounding boxes as absolute truth in Stage 4. If MinerU hallucinates or merges two adjacent columns into one bounding box, the pipeline has no built-in fallback to override it, directly causing structural corruption in tables.
- **Complex Tree State Management**: Writing deep `struct_id` relationships in `PyMuPDF` is brittle. Any disconnected nodes will result in invisible text or broken screen-reader navigation.

## 🛡️ What is NOT Risky

- **Stage 1 & Stage 2 Extraction/Merging**: These run purely on geometric math and precise font metrics. They do not suffer from ML hallucination.
- **Rule-Based Validation (Stage 6)**: Deterministic, fast, and easily extensible for new QA heuristics.

## Next Steps

1. **Grid topology (Priority 1)**: invert Stage 4 to treat MinerU `LayoutRegion`s as the structural truth and pull raw text into them, producing real `TR`/`TH`/`TD` cell structure instead of over-segmented monolithic tables. Quantified by the layout-accuracy harness (token agreement vs PREP ground truth).
2. **Multi-column / formula reading order**: the one remaining reading-order remediation failure is a dense 2-column STEM paper whose equation/affiliation pages defeat the geometric monotonicity proxy. (The gross column-interleaving and the NIH margin-watermark cases are fixed: reading-order remediation rose 40%→80%.)
3. **Alt-text quality**: when the alt-text stage is tackled, build the quality eval and compare the Gemma-4-E4B vs Qwen2.5-VL-7B backends before locking one in.
