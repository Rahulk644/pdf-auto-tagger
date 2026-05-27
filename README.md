# Auto-Tagger PDF Pipeline

An end-to-end Python pipeline for automatically adding semantic structural tags to untagged or poorly tagged PDFs, targeting accessibility standards (PDF/UA) and semantic QA compliance.

## 🚀 Where We Stand

We have successfully built and integrated a full **10-stage pipeline** capable of reading an untagged PDF, extracting characters, merging them into words and lines, classifying layouts (using MinerU), logically ordering the text, predicting semantic artifacts (Headers/Footers), and generating a fully valid Tagged PDF (`PyMuPDF/fitz`).

The pipeline handles complex spatial heuristics to correctly merge text based on font size and line gaps, overcoming major structural fragmentation issues. We have successfully run QA validation locally using Gemma to compare pipeline outputs against manually-tagged ground truths.

## 🛠️ Architecture

The pipeline processes documents in an immutable, declarative style where `PageElement` objects flow through these stages:

- **Stage 0: Page Classifier** (Native PDF vs Image/Scanned detection)
- **Stage 1: Native Extractor** (Character-level extraction using `pdfplumber`)
- **Stage 2: Text Merger** (Multi-pass algorithm merging characters -> words -> lines using dynamic font-based spatial gaps)
- **Stage 3: Layout Detector** (Deep Learning based layout detection using **MinerU** for bounding boxes and reading order)
- **Stage 4: Content Router** (Maps perfectly-split Stage 2 text lines into Stage 3 macro bounding boxes)
- **Stage 5: Table Extractor** (*Currently ON ICE* - Nested table structure processing)
- **Stage 6: Consistency Validator** (Rule-based safety checks, e.g., overlapping regions, font hierarchy validation)
- **Stage 7: Reading Order** (Topological sorting of layout blocks)
- **Stage 8: Semantic Analyzer** (Heuristic detection of Artifacts like Headers, Footers, and Pagination)
- **Stage 9: Struct Tree Builder** (Logical parent-child tree mapping)
- **Stage 10: PDF Writeback** (Generates standard-compliant Tagged PDF using `PyMuPDF`)

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

1. **Resolve MinerU Bottlenecks**: Implement a fallback check in Stage 4 to refuse merging elements if they possess a massive horizontal gap, overriding MinerU's box.
2. **Tackle Artifact Stubs (Path C)**: Address header/footer leakage.
3. **Unlock Table Extraction (Priority 1)**: Move nested table extraction off the ICE and implement inner `TD`/`TH` generation.
