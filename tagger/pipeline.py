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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

from tagger.config import PIPELINE, STANDARD_DPI
from tagger.models.confidence import ConfidenceTracker
from tagger.models.data_types import (
    DocumentData,
    LayoutCategory,
    PageData,
    PageElement,
    PageType,
    PDFTag,
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
        merged = self._timed("stage2", self._stage2_merge, doc_data)

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
        # Stage 9: Alt text for figures (loads Qwen, unloads after)
        # ------------------------------------------------------------------
        # Deferred — requires Qwen2.5-VL model
        logger.info("[Stage 9] Alt text generation — DEFERRED (model not yet integrated)")

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

    def _stage4_5_route_extract(self, doc_data: DocumentData):
        """Stage 4+5: Route regions and create initial tagged elements."""
        logger.info("[Stage 4+5] Routing and initial tagging...")

        # Map layout categories to initial PDF tags
        category_to_tag: dict[LayoutCategory, PDFTag] = {
            LayoutCategory.TITLE:          PDFTag.H1,
            LayoutCategory.SECTION_HEADER: PDFTag.H2,
            LayoutCategory.TEXT:           PDFTag.P,
            LayoutCategory.LIST_ITEM:      PDFTag.LI,
            LayoutCategory.TABLE:          PDFTag.TABLE,
            LayoutCategory.FORMULA:        PDFTag.FORMULA,
            LayoutCategory.PICTURE:        PDFTag.FIGURE,
            LayoutCategory.CAPTION:        PDFTag.P,  # refined in Stage 8
            LayoutCategory.FOOTNOTE:       PDFTag.NOTE,
            LayoutCategory.PAGE_HEADER:    PDFTag.ARTIFACT,
            LayoutCategory.PAGE_FOOTER:    PDFTag.ARTIFACT,
        }

        total_tagged = 0
        for page_num, page_data in doc_data.pages.items():
            tagged: list[TaggedElement] = []

            # Match elements to regions
            element_map = {el.element_id: el for el in page_data.elements}

            for region in page_data.layout_regions:
                # Find elements within this region
                matched_els = []
                if region.matched_elements:
                    matched_els = [
                        element_map[eid] for eid in region.matched_elements
                        if eid in element_map
                    ]
                else:
                    # Fall back to spatial matching
                    from tagger.stage1_extraction.coord_transformer import bbox_contains
                    matched_els = [
                        el for el in page_data.elements
                        if bbox_contains(region.bbox, el.bbox, tolerance=5.0)
                    ]

                pdf_tag = category_to_tag.get(region.category, PDFTag.P)

                for el in matched_els:
                    tagged_el = TaggedElement(
                        element_id=el.element_id,
                        page_num=page_num,
                        pdf_tag=pdf_tag,
                        text=el.text,
                        bbox=el.bbox,
                        confidence=region.confidence,
                        original_mcid=el.mcid,
                        font_name=el.font_name,
                        font_size=el.font_size,
                        font_weight=el.font_weight,
                        merged_from=el.merged_from,
                        layout_category=region.category.value,
                    )
                    tagged.append(tagged_el)

                # If region has no matched elements (e.g., figures),
                # create a tagged element for the region itself
                if not matched_els and region.category in (
                    LayoutCategory.PICTURE,
                    LayoutCategory.TABLE,
                    LayoutCategory.FORMULA,
                ):
                    tagged.append(TaggedElement(
                        element_id=region.region_id,
                        page_num=page_num,
                        pdf_tag=pdf_tag,
                        text="",
                        bbox=region.bbox,
                        confidence=region.confidence,
                        layout_category=region.category.value,
                    ))

            page_data.tagged_elements = tagged
            total_tagged += len(tagged)

        logger.info("  Created %d initial tagged elements", total_tagged)

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
        logger.info("[Stage 7] Cross-page merge — basic implementation")
        # TODO: Full cross-page table/list continuation in P9

    def _stage8_refine(self, doc_data: DocumentData):
        """Stage 8: Semantic refinement."""
        from tagger.stage8_semantic.heading_ranker import assign_heading_levels
        from tagger.stage8_semantic.toc_detector import detect_toc_entries
        from tagger.stage8_semantic.artifact_detector import detect_artifacts
        from tagger.stage8_semantic.caption_detector import detect_captions

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
        detect_artifacts(all_tagged)

        # 8d: Caption detection
        detect_captions(all_tagged)

    def _stage10_write(self, input_pdf: str, output_pdf: str, doc_data: DocumentData):
        """Stage 10: Struct tree writeback."""
        logger.info("[Stage 10] Struct tree writeback — V1 (re-tag existing)")
        # TODO: Full writeback in P11
        # For now, just copy the input to output
        import shutil
        shutil.copy2(input_pdf, output_pdf)
        logger.info("  Copied input to output (writeback not yet implemented)")

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
    # Timing helper
    # ------------------------------------------------------------------

    def _timed(self, stage_name: str, func, *args, **kwargs):
        """Run a function and record its execution time."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self._timings[stage_name] = elapsed
        logger.info("  [%s] completed in %.1fs", stage_name, elapsed)
        return result
