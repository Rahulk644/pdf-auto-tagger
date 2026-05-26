"""
MinerU2.5 layout detection implementation.

Uses the MinerU 2.5 VLM (1.2B parameters) for two-stage layout analysis:
  1. Global layout pass on downsampled page image
  2. High-res crops for complex regions (tables, formulas)

Outputs bounding boxes with categories and reading order.

Requires: pip install "pdf-auto-tagger[mineru]"
"""

from __future__ import annotations

import gc
import logging

from PIL import Image

from tagger.config import LAYOUT, STANDARD_DPI
from tagger.models.data_types import LayoutCategory, LayoutRegion
from tagger.stage3_layout.model_adapter import LayoutModelAdapter

logger = logging.getLogger(__name__)


# Map MinerU category strings to our LayoutCategory enum
_CATEGORY_MAP: dict[str, LayoutCategory] = {
    "title": LayoutCategory.TITLE,
    "section-header": LayoutCategory.SECTION_HEADER,
    "section_header": LayoutCategory.SECTION_HEADER,
    "text": LayoutCategory.TEXT,
    "plain text": LayoutCategory.TEXT,
    "list-item": LayoutCategory.LIST_ITEM,
    "list_item": LayoutCategory.LIST_ITEM,
    "table": LayoutCategory.TABLE,
    "formula": LayoutCategory.FORMULA,
    "equation": LayoutCategory.FORMULA,
    "picture": LayoutCategory.PICTURE,
    "figure": LayoutCategory.PICTURE,
    "image": LayoutCategory.PICTURE,
    "caption": LayoutCategory.CAPTION,
    "footnote": LayoutCategory.FOOTNOTE,
    "page-header": LayoutCategory.PAGE_HEADER,
    "page_header": LayoutCategory.PAGE_HEADER,
    "header": LayoutCategory.PAGE_HEADER,
    "page-footer": LayoutCategory.PAGE_FOOTER,
    "page_footer": LayoutCategory.PAGE_FOOTER,
    "footer": LayoutCategory.PAGE_FOOTER,
}


class MinerULayoutDetector(LayoutModelAdapter):
    """
    MinerU2.5-based layout detector.

    Load/unload lifecycle for M1 8GB memory management.
    ~1.5GB RAM when loaded.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LAYOUT.model_name
        self._client = None
        self._loaded = False

    def load(self) -> None:
        """Load MinerU2.5 model into memory."""
        if self._loaded:
            return

        try:
            from mineru_vl_utils import MinerUClient
            self._client = MinerUClient(
                backend="transformers",
                model_name=self.model_name,
            )
            self._loaded = True
            logger.info("MinerU2.5 loaded: %s (~%dMB)", self.model_name, self.memory_footprint_mb)
        except ImportError:
            raise RuntimeError(
                "MinerU not installed. Install with: "
                'pip install "pdf-auto-tagger[mineru]"'
            )

    def detect(self, page_image: Image.Image, page_num: int) -> list[LayoutRegion]:
        """Run layout detection on a page image."""
        if not self._loaded or self._client is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        result = self._client.two_step_extract(page_image)
        return self._parse_output(result, page_num, page_image.size)

    def unload(self) -> None:
        """Release model from memory."""
        if self._client is not None:
            del self._client
            self._client = None
        self._loaded = False
        gc.collect()
        logger.info("MinerU2.5 unloaded")

    @property
    def memory_footprint_mb(self) -> int:
        return 1500  # ~1.5GB for 1.2B model

    @property
    def name(self) -> str:
        return f"MinerU2.5 ({self.model_name})"

    def _parse_output(
        self,
        raw_result: dict,
        page_num: int,
        image_size: tuple[int, int],
    ) -> list[LayoutRegion]:
        """
        Parse MinerU's raw output into LayoutRegion objects.

        MinerU output format varies by version — this handles the
        common structure with layout_dets or similar keys.
        """
        regions: list[LayoutRegion] = []
        img_width, img_height = image_size

        # MinerU output may contain 'layout_dets' or similar
        detections = []
        if isinstance(raw_result, dict):
            detections = (
                raw_result.get("layout_dets", [])
                or raw_result.get("detections", [])
                or raw_result.get("blocks", [])
            )
        elif isinstance(raw_result, list):
            detections = raw_result

        for idx, det in enumerate(detections):
            # Extract category
            cat_str = str(det.get("category", det.get("type", "text"))).lower().strip()
            category = _CATEGORY_MAP.get(cat_str, LayoutCategory.TEXT)

            # Extract bbox — may be [x0, y0, x1, y1] or dict with keys
            bbox_raw = det.get("bbox", det.get("poly", [0, 0, 0, 0]))
            if isinstance(bbox_raw, dict):
                bbox = (
                    float(bbox_raw.get("x0", 0)),
                    float(bbox_raw.get("y0", 0)),
                    float(bbox_raw.get("x1", 0)),
                    float(bbox_raw.get("y1", 0)),
                )
            elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
                bbox = tuple(float(v) for v in bbox_raw[:4])
            else:
                continue

            # Confidence
            confidence = float(det.get("score", det.get("confidence", 0.5)))
            if confidence < LAYOUT.min_region_confidence:
                continue

            regions.append(LayoutRegion(
                region_id=f"r{page_num}_{idx}",
                page_num=page_num,
                bbox=bbox,
                category=category,
                reading_order=idx,
                confidence=confidence,
            ))

        # Sort by reading order (top-to-bottom, left-to-right)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        for i, r in enumerate(regions):
            r.reading_order = i

        logger.debug(
            "Page %d: %d layout regions detected",
            page_num, len(regions),
        )
        return regions
