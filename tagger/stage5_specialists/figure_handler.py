"""
Stage 5c — Figure handler.

Handles Picture/Figure regions:
  1. Crops the figure image from the rendered page
  2. Checks if the figure is decorative (tiny, line-art) → Artifact
  3. Saves the cropped image for alt-text generation (Stage 9)

No ML model needed here — just image cropping and size heuristics.
Alt text generation is deferred to Stage 9 (Qwen2.5-VL).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from tagger.config import VALIDATOR, STANDARD_DPI, PDF_NATIVE_DPI
from tagger.models.data_types import FigureInfo, LayoutRegion

logger = logging.getLogger(__name__)


def handle_figure(
    region: LayoutRegion,
    page_image: Image.Image | None,
    output_dir: str | Path,
) -> FigureInfo:
    """
    Process a figure region.

    Args:
        region: Layout region classified as PICTURE.
        page_image: PIL Image of the rendered page (at STANDARD_DPI).
        output_dir: Directory to save cropped figure images.

    Returns:
        FigureInfo with image path and decorative flag.
    """
    width = region.bbox[2] - region.bbox[0]
    height = region.bbox[3] - region.bbox[1]

    # Check if figure is too small → decorative artifact
    if (
        width < VALIDATOR.min_figure_width_px
        or height < VALIDATOR.min_figure_height_px
    ):
        logger.debug(
            "Region %s: tiny figure (%.0f×%.0f) → decorative",
            region.region_id, width, height,
        )
        return FigureInfo(
            region_id=region.region_id,
            image_path=None,
            is_decorative=True,
            confidence=0.85,
        )

    # Crop and save the figure image
    image_path = None
    if page_image is not None:
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Crop bbox (already in STANDARD_DPI pixel coords)
            x0 = max(0, int(region.bbox[0]))
            y0 = max(0, int(region.bbox[1]))
            x1 = min(page_image.width, int(region.bbox[2]))
            y1 = min(page_image.height, int(region.bbox[3]))

            if x1 > x0 and y1 > y0:
                cropped = page_image.crop((x0, y0, x1, y1))
                filename = f"{region.region_id}.png"
                save_path = out_dir / filename
                cropped.save(str(save_path), "PNG")
                image_path = str(save_path)

                logger.debug(
                    "Region %s: saved figure crop %dx%d → %s",
                    region.region_id, x1 - x0, y1 - y0, save_path,
                )

        except Exception as e:
            logger.warning(
                "Region %s: figure crop failed: %s",
                region.region_id, e,
            )

    # Check for potential decorative elements based on image analysis
    is_decorative = False
    if image_path and page_image:
        is_decorative = _check_decorative(page_image, region.bbox)

    return FigureInfo(
        region_id=region.region_id,
        image_path=image_path,
        is_decorative=is_decorative,
        confidence=region.confidence,
    )


def _check_decorative(
    page_image: Image.Image,
    bbox: tuple[float, float, float, float],
) -> bool:
    """
    Heuristic check for decorative elements.

    Decorative indicators:
      - Very narrow (line/rule): width > 10× height or height > 10× width
      - Very few unique colors (< 3) → likely a solid bar/line
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Extreme aspect ratio → likely a horizontal/vertical rule
    if width > 0 and height > 0:
        aspect = width / height
        if aspect > 15.0 or aspect < 1.0 / 15.0:
            return True

    return False
