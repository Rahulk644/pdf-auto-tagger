
"""
Pipeline orchestrator — runs stages 0→10 sequentially.

Manages model lifecycle (load/unload) for M1 8GB memory constraints.
Each stage receives the output of the previous stage and produces
structured data for the next.

Only one ML model is loaded at a time.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

from tagger.config import STANDARD_DPI
from tagger.models.confidence import ConfidenceTracker
from tagger.models.data_types import (
    DocumentData,
    LayoutCategory,
    PageData,
    PageElement,
    TaggedElement,
)

logger = logging.getLogger(__name__)


class AutoTaggerPipeline:
    """
    Sequential pipeline orchestrator.

    Usage:
        pipeline = AutoTaggerPipeline()
        report = pipeline.run("input.pdf", "output.pdf")
    """

    def __init__(self):
        self.tracker = ConfidenceTracker()
        self._timings: dict[str, float] = {}

    def run(
        self,
        input_pdf: str,
        output_pdf: str | None = None,
        report_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full auto-tagging pipeline.

        Args:
            input_pdf: Path to input PDF.
            output_pdf: Path for tagged output PDF (optional for V1).
            report_path: Path for JSON confidence report (optional).

        Returns:
            Pipeline report as a dict.
        """
        start_time = time.time()
        input_path = Path(input_pdf)
        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        logger.info("=" * 60)
        logger.info("AUTO-TAGGER PIPELINE START: %s", input_path.name)
        logger.info("=" * 60)

        doc_data = DocumentData(
            input_path=str(input_path),
            num_pages=0,
        )

        # ------------------------------------------------------------------
        # Stage 0: Classify pages
        # ------------------------------------------------------------------
        classifications = self._timed("stage0", self._stage0_classify, input_pdf)
        doc_data.num_pages = len(classifications)
        for c in classifications:
            doc_data.pages[c.page_num] = PageData(
                page_num=c.page_num,
                classification=c,
            )

        # ------------------------------------------------------------------
        # Stage 1: Extract text + metadata
        # ------------------------------------------------------------------
        extracted = self._timed("stage1", self._stage1_extract, input_pdf, classifications)
        for page_num, elements in extracted.items():
            if page_num in doc_data.pages:
                doc_data.pages[page_num].elements = elements

        # ------------------------------------------------------------------
        # Stage 2: Merge text fragments
        # ------------------------------------------------------------------
        self._timed("stage2", self._stage2_merge, doc_data)

        # ------------------------------------------------------------------
        # Stage 3: Layout detection (loads MinerU, unloads after)
        # ------------------------------------------------------------------
        self._timed("stage3", self._stage3_layout, input_pdf, doc_data)

        # ------------------------------------------------------------------
        # Stage 4+5: Route + specialist extraction
        # ------------------------------------------------------------------
        self._timed("stage4_5", self._stage4_5_route_extract, doc_data)

        # ------------------------------------------------------------------
        # Stage 6: Validate consistency
        # ------------------------------------------------------------------
        self._timed("stage6", self._stage6_validate, doc_data)

        # ------------------------------------------------------------------
        # Stage 7: Cross-page merge
        # ------------------------------------------------------------------
        self._timed("stage7", self._stage7_cross_page, doc_data)

        # ------------------------------------------------------------------
        # Stage 8: Semantic refinement
        # ------------------------------------------------------------------
        self._timed("stage8", self._stage8_refine, doc_data)

        # ------------------------------------------------------------------
        # Stage 9: Alt text for figures
        # ------------------------------------------------------------------
        self._timed("stage9", self._stage9_alttext, input_pdf, doc_data)

        # ------------------------------------------------------------------
        # Stage 10: Write to PDF
        # ------------------------------------------------------------------
        if output_pdf:
            self._timed("stage10", self._stage10_write, input_pdf, output_pdf, doc_data)
        else:
            logger.info("[Stage 10] Writeback — SKIPPED (no output path specified)")

        # ------------------------------------------------------------------
        # Generate report
        # ------------------------------------------------------------------
        total_time = time.time() - start_time
        report = self._generate_report(doc_data, total_time)

        if report_path:
            report_file = Path(report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("Report saved to: %s", report_file)

        # Save confidence report
        if report_path:
            conf_path = Path(report_path).with_suffix(".confidence.json")
            self.tracker.save_report(conf_path)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE in %.1fs", total_time)
        logger.info("=" * 60)

        return report

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage0_classify(self, input_pdf: str):
        """Stage 0: Page classification."""
        from tagger.stage0_classifier.page_classifier import classify_pages

        logger.info("[Stage 0] Classifying pages...")
        classifications = classify_pages(input_pdf)
        for c in classifications:
            logger.info(
                "  Page %d: %s (conf=%.2f, chars=%d, img_cov=%.2f)",
                c.page_num, c.page_type.value, c.confidence,
                c.char_count, c.image_coverage,
            )
        return classifications

    def _stage1_extract(self, input_pdf, classifications):
        """Stage 1: Text extraction."""
        from tagger.stage1_extraction.native_extractor import extract_native_pages
        from tagger.stage1_extraction.scanned_extractor import extract_scanned_pages

        logger.info("[Stage 1] Extracting text...")

        # Native extraction
        native_elements = extract_native_pages(input_pdf, classifications)

        # Scanned extraction (MinerU OCR)
        scanned_elements = extract_scanned_pages(input_pdf, classifications)

        # Merge results (for mixed pages, both paths contribute)
        all_elements: dict[int, list[PageElement]] = {}
        for page_num in set(list(native_elements.keys()) + list(scanned_elements.keys())):
            page_els = []
            page_els.extend(native_elements.get(page_num, []))
            page_els.extend(scanned_elements.get(page_num, []))
            all_elements[page_num] = page_els

        total = sum(len(v) for v in all_elements.values())
        logger.info("  Extracted %d raw elements from %d pages", total, len(all_elements))
        return all_elements

    def _stage2_merge(self, doc_data: DocumentData):
        """Stage 2: Text merger."""
        from tagger.stage2_merger.text_merger import merge_page_elements

        logger.info("[Stage 2] Merging text fragments...")
        total_before = 0
        total_after = 0

        for page_num, page_data in doc_data.pages.items():
            if not page_data.elements:
                continue
            total_before += len(page_data.elements)
            page_data.elements = merge_page_elements(page_data.elements, page_num)
            total_after += len(page_data.elements)

        logger.info("  Merged: %d chars → %d paragraphs", total_before, total_after)

    def _stage3_layout(self, input_pdf: str, doc_data: DocumentData):
        """Stage 3: Layout detection with MinerU."""
        logger.info("[Stage 3] Layout detection...")

        try:
            from tagger.stage3_layout.layout_detector import MinerULayoutDetector

            detector = MinerULayoutDetector()
            detector.load()

            try:
                fitz_doc = fitz.open(input_pdf)
                for page_num, page_data in doc_data.pages.items():
                    page_idx = page_num - 1
                    if page_idx >= len(fitz_doc):
                        continue

                    # Render page to image
                    pix = fitz_doc[page_idx].get_pixmap(dpi=STANDARD_DPI)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                    regions = detector.detect(img, page_num)
                    page_data.layout_regions = regions
                    logger.info("  Page %d: %d regions", page_num, len(regions))

                fitz_doc.close()
            finally:
                detector.unload()
                gc.collect()

            # pdfplumber fallback: inject TABLE regions MinerU missed
            self._pdfplumber_table_fallback(input_pdf, doc_data)

        except (ImportError, RuntimeError) as e:
            logger.warning(
                "  Layout detection unavailable (%s). "
                "Using text-only classification fallback.",
                e,
            )
            # Fallback: classify each merged element as Text
            self._fallback_layout_classification(doc_data)

    def _fallback_layout_classification(self, doc_data: DocumentData):
        """
        Fallback layout classification when MinerU is not available.

        Uses simple font-size heuristics to guess element types.
        """
        from tagger.models.data_types import LayoutCategory, LayoutRegion

        for page_num, page_data in doc_data.pages.items():
            regions = []
            # Collect all font sizes on this page
            font_sizes = [
                el.font_size for el in page_data.elements
                if el.font_size is not None and el.font_size > 0
            ]
            if not font_sizes:
                # Everything is text
                for idx, el in enumerate(page_data.elements):
                    regions.append(LayoutRegion(
                        region_id=f"r{page_num}_{idx}",
                        page_num=page_num,
                        bbox=el.bbox,
                        category=LayoutCategory.TEXT,
                        reading_order=idx,
                        confidence=0.5,
                        matched_elements=[el.element_id],
                    ))
                page_data.layout_regions = regions
                continue

            median_size = sorted(font_sizes)[len(font_sizes) // 2]
            max_size = max(font_sizes)

            for idx, el in enumerate(page_data.elements):
                # Simple heuristic: larger than median → heading
                if el.font_size and el.font_size > median_size * 1.3:
                    if el.font_size >= max_size * 0.9:
                        cat = LayoutCategory.TITLE
                    else:
                        cat = LayoutCategory.SECTION_HEADER
                else:
                    cat = LayoutCategory.TEXT

                regions.append(LayoutRegion(
                    region_id=f"r{page_num}_{idx}",
                    page_num=page_num,
                    bbox=el.bbox,
                    category=cat,
                    reading_order=idx,
                    confidence=0.5,
                    matched_elements=[el.element_id],
                ))

            page_data.layout_regions = regions

    def _pdfplumber_table_fallback(self, input_pdf: str, doc_data: DocumentData) -> None:
        """
        Stage 3 post-processing: inject TABLE LayoutRegions for tables that
        pdfplumber detects but MinerU missed.

        Uses text-column alignment strategy — correct for financial documents
        with no explicit ruling lines (e.g. Miramar CAFR).

        Skip injection if an existing MinerU TABLE region is already
        substantially contained within the pdfplumber bbox:
          intersection / MinerU_table_area > CONTAINMENT_THRESHOLD
        This avoids duplicate TABLE regions when MinerU found a tighter bbox
        for the same table (IoU alone under-counts because pdfplumber's text
        strategy returns oversized bboxes).
        """
        import pdfplumber
        from tagger.models.data_types import LayoutCategory, LayoutRegion

        SCALE = STANDARD_DPI / 72.0  # pdfplumber 72 DPI → standard 150 DPI
        CONTAINMENT_THRESHOLD = 0.7
        MIN_ROWS = 3
        MIN_COLS = 2
        MAX_AVG_CELL_LEN = 60   # cells longer than this are prose, not data
        MIN_NUMERIC_RATIO = 0.20  # applied only when n_cols > 2
        MAX_TOTAL_CELLS = 300   # pdfplumber fragments heading text into huge phantom grids
        # If MinerU already classified the candidate area as prose/list, trust that
        # judgment over pdfplumber's text-grid heuristic and skip injection.
        # Use area COVERAGE (fraction of the candidate covered by the union of
        # prose regions), not per-region IoU: MinerU fragments a page into many
        # small text/list regions, and any single one has low IoU with a large
        # candidate even when collectively they blanket it.
        PROSE_CATEGORIES = (LayoutCategory.TEXT, LayoutCategory.LIST_ITEM)
        PROSE_COVERAGE_THRESHOLD = 0.4
        TABLE_SETTINGS = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        }

        def _is_numeric_cell(text: str) -> bool:
            if not text or not text.strip():
                return False
            cleaned = (
                text.strip()
                .lstrip("$")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "")
                .replace("-", "")
                .strip()
            )
            return bool(cleaned) and cleaned.replace(".", "").isdigit()

        def _rect_union_area(rects):
            """Exact union area of axis-aligned rects via coordinate compression."""
            if not rects:
                return 0.0
            xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
            ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
            total = 0.0
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    cx = (xs[i] + xs[i + 1]) / 2
                    cy = (ys[j] + ys[j + 1]) / 2
                    if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                        total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
            return total

        try:
            with pdfplumber.open(input_pdf) as pdf:
                for page_num, page_data in doc_data.pages.items():
                    page_idx = page_num - 1
                    if page_idx >= len(pdf.pages):
                        continue

                    found = pdf.pages[page_idx].find_tables(TABLE_SETTINGS)
                    if not found:
                        continue

                    existing_tables = [
                        r for r in page_data.layout_regions
                        if r.category == LayoutCategory.TABLE
                    ]
                    prose_regions = [
                        r for r in page_data.layout_regions
                        if r.category in PROSE_CATEGORIES
                    ]

                    injected = 0
                    for table in found:
                        if len(table.rows) < MIN_ROWS:
                            continue

                        rows_data = table.extract()
                        if not rows_data:
                            continue

                        # Check 1: minimum columns
                        n_cols = max((len(row) for row in rows_data), default=0)
                        if n_cols < MIN_COLS:
                            continue

                        non_empty = [c for row in rows_data for c in row if c and c.strip()]

                        # Check 3: average cell text length — prose paragraphs masquerading as cells
                        if non_empty and (sum(len(c) for c in non_empty) / len(non_empty)) > MAX_AVG_CELL_LEN:
                            continue

                        # Check 4: grid size — pdfplumber fragments heading/paragraph text into
                        # huge phantom grids (e.g. 53×10=530 cells); real tables stay under 300
                        if len(table.rows) * n_cols > MAX_TOTAL_CELLS:
                            continue

                        # Check 2: numeric content ratio — only enforced for wide tables (>2 cols)
                        if n_cols > 2 and non_empty:
                            numeric_ratio = sum(1 for c in non_empty if _is_numeric_cell(c)) / len(non_empty)
                            if numeric_ratio < MIN_NUMERIC_RATIO:
                                continue

                        # Convert bbox to 150 DPI standard coords
                        x0, y0, x1, y1 = (v * SCALE for v in table.bbox)

                        # Defer to MinerU: if it classified this area as text/list,
                        # it already judged this is not a table — trust that and skip.
                        cand_area = (x1 - x0) * (y1 - y0)
                        clipped = []
                        for pr in prose_regions:
                            pb = pr.bbox
                            cx0 = max(x0, pb[0]); cy0 = max(y0, pb[1])
                            cx1 = min(x1, pb[2]); cy1 = min(y1, pb[3])
                            if cx1 > cx0 and cy1 > cy0:
                                clipped.append((cx0, cy0, cx1, cy1))
                        coverage = _rect_union_area(clipped) / cand_area if cand_area > 0 else 0.0
                        if coverage > PROSE_COVERAGE_THRESHOLD:
                            logger.info(
                                "  Page %d: pdfplumber fallback deferring to MinerU "
                                "text/list (%.0f%% prose coverage)", page_num, coverage * 100,
                            )
                            continue

                        # Skip if any MinerU TABLE is already substantially
                        # contained within this pdfplumber bbox
                        skip = False
                        for er in existing_tables:
                            mb = er.bbox
                            ix0 = max(x0, mb[0]); iy0 = max(y0, mb[1])
                            ix1 = min(x1, mb[2]); iy1 = min(y1, mb[3])
                            if ix1 > ix0 and iy1 > iy0:
                                inter = (ix1 - ix0) * (iy1 - iy0)
                                mb_area = (mb[2] - mb[0]) * (mb[3] - mb[1])
                                if mb_area > 0 and (inter / mb_area) > CONTAINMENT_THRESHOLD:
                                    skip = True
                                    break
                        if skip:
                            continue

                        synthetic = LayoutRegion(
                            region_id=f"r{page_num}_pb_{injected}",
                            page_num=page_num,
                            bbox=(x0, y0, x1, y1),
                            category=LayoutCategory.TABLE,
                            reading_order=len(page_data.layout_regions) + injected,
                            confidence=0.7,
                        )
                        page_data.layout_regions.append(synthetic)
                        injected += 1
                        logger.info(
                            "  Page %d: pdfplumber fallback injected TABLE "
                            "(rows=%d bbox=%.0f,%.0f,%.0f,%.0f)",
                            page_num, len(table.rows), x0, y0, x1, y1,
                        )
        except Exception as e:
            logger.warning("pdfplumber table fallback failed: %s", e)

    def _stage4_5_route_extract(self, doc_data: DocumentData):
        """Stage 4+5: Route regions and create initial tagged elements."""
        logger.info("[Stage 4+5] Routing and initial tagging...")

        from tagger.stage4_router.content_router import route_page, diagnose_page
        from tagger.stage5_specialists.table_extractor import extract_table_native
        from tagger.models.data_types import LayoutCategory

        total_tagged = 0
        for page_num, page_data in doc_data.pages.items():
            tagged = route_page(
                page_num=page_num,
                mineru_regions=page_data.layout_regions,
                page_elements=page_data.elements,
                containment_threshold=0.5,
            )
            logger.info("Page %d Diagnostics: %s", page_num, diagnose_page(tagged))

            # Stage 5: run table specialist on TABLE regions
            table_regions = [
                r for r in page_data.layout_regions
                if r.category == LayoutCategory.TABLE
            ]
            if table_regions and page_data.classification:
                for region in table_regions:
                    table_struct = extract_table_native(
                        doc_data.input_path, page_num, region, page_data.classification,
                    )
                    if table_struct:
                        for el in tagged:
                            if el.element_id == region.region_id:
                                el.specialist_data = {
                                    "html": table_struct.html,
                                    "num_rows": table_struct.num_rows,
                                    "num_cols": table_struct.num_cols,
                                    "has_header": table_struct.has_header,
                                    "cells": getattr(table_struct, "cells", [])
                                }
                                break

            page_data.tagged_elements = tagged
            total_tagged += len(tagged)

        logger.debug("Created %d initially tagged elements.", total_tagged)

    def _stage6_validate(self, doc_data: DocumentData):
        """Stage 6: Consistency validation."""
        from tagger.stage6_validator.consistency_validator import (
            validate_elements,
            ValidationContext,
        )

        logger.info("[Stage 6] Validating consistency...")

        all_tagged = []
        for page_data in doc_data.pages.values():
            all_tagged.extend(page_data.tagged_elements)

        if all_tagged:
            context = ValidationContext(all_elements=all_tagged)
            validate_elements(all_tagged, self.tracker, context)

    def _stage7_cross_page(self, doc_data: DocumentData):
        """Stage 7: Cross-page merge."""
        from tagger.stage7_cross_page.cross_page_merger import merge_cross_page

        logger.info("[Stage 7] Cross-page merge...")

        all_tagged: list[TaggedElement] = []
        for page_data in doc_data.pages.values():
            all_tagged.extend(page_data.tagged_elements)

        if all_tagged:
            merge_cross_page(all_tagged, doc_data.num_pages)

    def _stage8_refine(self, doc_data: DocumentData):
        """Stage 8: Semantic refinement."""
        from tagger.stage8_semantic.heading_ranker import assign_heading_levels
        from tagger.stage8_semantic.toc_detector import detect_toc_entries
        from tagger.stage8_semantic.artifact_detector import detect_artifacts
        from tagger.stage8_semantic.caption_detector import detect_captions
        from tagger.stage8_semantic.list_builder import build_list_structure

        logger.info("[Stage 8] Semantic refinement...")

        # Collect all tagged elements across pages
        all_tagged: list[TaggedElement] = []
        for page_data in doc_data.pages.values():
            all_tagged.extend(page_data.tagged_elements)

        if not all_tagged:
            return

        # 8a: Heading levels
        assign_heading_levels(all_tagged)

        # 8b: TOC detection
        detect_toc_entries(all_tagged, doc_data.num_pages)

        # 8c: Artifact detection (running headers/footers)
        detect_artifacts(all_tagged, doc_data.num_pages)

        # 8d: Caption detection
        detect_captions(all_tagged)

        # 8e: List structure — may add/remove elements (P→LI promotion merges
        # separated marker+body), so redistribute the result back into each
        # page's tagged_elements (Stage 10 re-collects from page_data).
        all_tagged = build_list_structure(all_tagged)
        from collections import defaultdict
        by_page: dict[int, list[TaggedElement]] = defaultdict(list)
        for el in all_tagged:
            by_page[el.page_num].append(el)
        for page_num, page_data in doc_data.pages.items():
            page_data.tagged_elements = by_page.get(page_num, [])

    def _stage9_alttext(self, input_pdf: str, doc_data: DocumentData):
        """Stage 9: Alt text for figure elements."""
        from tagger.stage9_alttext.alt_text_generator import generate_alt_text_placeholders

        logger.info("[Stage 9] Alt text generation (placeholder mode)...")

        all_tagged: list[TaggedElement] = []
        for page_data in doc_data.pages.values():
            all_tagged.extend(page_data.tagged_elements)

        count = generate_alt_text_placeholders(all_tagged, input_pdf)
        logger.info("  Generated %d placeholder alt texts", count)

    def _stage10_write(self, input_pdf: str, output_pdf: str, doc_data: DocumentData):
        """Stage 10: Struct tree writeback."""
        from tagger.stage10_writeback.struct_tree_writer import (
            retag_existing_pdf,
            tag_untagged_pdf,
        )

        # Collect all tagged elements
        all_tagged: list[TaggedElement] = []
        for page_data in doc_data.pages.values():
            all_tagged.extend(page_data.tagged_elements)

        # Check if PDF has an existing struct tree (V1) vs needs one built (V2).
        # We check the actual PDF structure, not element MCIDs — a stripped PDF
        # still has BDC markers in content streams that produce non-None MCIDs.
        has_existing_tags = self._pdf_has_struct_tree(input_pdf)

        if has_existing_tags:
            logger.info("[Stage 10] Re-tagging existing tagged PDF...")
            stats = retag_existing_pdf(input_pdf, output_pdf, all_tagged)
            logger.info(
                "  Re-tag stats: %d matched, %d changed, %d unmatched",
                stats.get("matched", 0),
                stats.get("changed", 0),
                stats.get("unmatched", 0),
            )
        else:
            logger.info("[Stage 10] Building struct tree for untagged PDF...")
            stats = tag_untagged_pdf(
                input_pdf, output_pdf, all_tagged, doc_data.num_pages,
            )
            logger.info(
                "  Writeback stats: %d elements written across %d pages",
                stats.get("total_elements_written", 0),
                stats.get("pages_modified", 0),
            )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self, doc_data: DocumentData, total_time: float) -> dict:
        """Generate the final pipeline report."""
        all_tagged = []
        for page_data in doc_data.pages.values():
            all_tagged.extend(page_data.tagged_elements)

        # Count tags
        tag_counts: dict[str, int] = {}
        for el in all_tagged:
            tag = el.pdf_tag.value
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Count review items
        needs_review = [el for el in all_tagged if el.needs_review]

        # Page type counts
        page_types = {}
        for page_data in doc_data.pages.values():
            if page_data.classification:
                pt = page_data.classification.page_type.value
                page_types[pt] = page_types.get(pt, 0) + 1

        report = {
            "input_file": doc_data.input_path,
            "total_pages": doc_data.num_pages,
            "page_types": page_types,
            "summary": {
                "total_elements": len(all_tagged),
                "needs_review": len(needs_review),
                "review_rate_percent": round(
                    len(needs_review) / len(all_tagged) * 100, 1
                ) if all_tagged else 0,
                "total_time_seconds": round(total_time, 2),
            },
            "tag_distribution": tag_counts,
            "stage_timings": {k: round(v, 2) for k, v in self._timings.items()},
            "elements": [
                {
                    "element_id": el.element_id,
                    "page_num": el.page_num,
                    "pdf_tag": el.pdf_tag.value,
                    "text": el.text[:200] if el.text else "",
                    "confidence": round(el.confidence, 3),
                    "needs_review": el.needs_review,
                    "review_reason": el.review_reason,
                    "layout_category": el.layout_category,
                    "font_size": el.font_size,
                    "font_weight": el.font_weight,
                }
                for el in all_tagged
            ],
            "confidence_report": self.tracker.generate_report(),
        }

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_has_struct_tree(pdf_path: str) -> bool:
        """Return True if the PDF has an existing StructTreeRoot (V1 path)."""
        try:
            import pikepdf
            with pikepdf.open(pdf_path) as pdf:
                return "/StructTreeRoot" in pdf.Root
        except Exception:
            return False

    def _timed(self, stage_name: str, func, *args, **kwargs):
        """Run a function and record its execution time."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self._timings[stage_name] = elapsed
        logger.info("  [%s] completed in %.1fs", stage_name, elapsed)
        return result
