"""
Stage 9 — Alt text generator for figures.

Two modes:
  1. Placeholder mode (default): Generates a review-required placeholder
     alt text for all figures. This makes the output valid per PDF/UA
     (the /Alt attribute exists) while signalling human review is needed.

  2. VLM mode (optional): Uses Qwen2.5-VL to generate real alt text.
     Requires ~7GB RAM — only enabled when explicitly requested and
     the model is available.

Placeholder mode runs with zero additional dependencies and is always
available. VLM mode loads/unloads the model for M1 8GB safety.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from tagger.config import ALT_TEXT, STANDARD_DPI
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)


def generate_alt_text_placeholders(
    elements: list[TaggedElement],
    input_pdf: str | None = None,
) -> int:
    """
    Generate placeholder alt text for all Figure elements.

    Sets alt_text to a descriptive placeholder and flags the element
    for human review. This makes the PDF technically valid (has /Alt)
    while clearly indicating review is needed.

    Args:
        elements: All tagged elements from the pipeline.
        input_pdf: Path to the PDF (used to extract figure dimensions).

    Returns:
        Number of figures that received placeholder alt text.
    """
    count = 0

    for el in elements:
        if el.pdf_tag != PDFTag.FIGURE:
            continue

        # Skip if already has alt text
        if el.alt_text:
            continue

        # Generate informative placeholder
        width = el.bbox[2] - el.bbox[0]
        height = el.bbox[3] - el.bbox[1]
        aspect = width / height if height > 0 else 1.0

        if aspect > 2.0:
            shape_hint = "wide"
        elif aspect < 0.5:
            shape_hint = "tall"
        else:
            shape_hint = "approximately square"

        el.alt_text = (
            f"[Figure on page {el.page_num}: "
            f"{shape_hint} image, {width:.0f}×{height:.0f}px. "
            f"Alt text requires human review.]"
        )

        # Flag for review
        el.needs_review = True
        el.review_reason = (
            el.review_reason or ""
        ) + " [alt_text_placeholder] Figure needs descriptive alt text."

        count += 1

    if count > 0:
        logger.info("Alt text placeholders: %d figures flagged for review", count)

    return count


def generate_alt_text_vlm(
    elements: list[TaggedElement],
    input_pdf: str,
) -> int:
    """
    Generate alt text using Qwen2.5-VL (7B).

    Loads the VLM, generates alt text for each figure, then unloads.
    Requires ~7GB RAM — not safe on M1 8GB with other processes.

    Args:
        elements: All tagged elements from the pipeline.
        input_pdf: Path to the PDF (for figure image extraction).

    Returns:
        Number of figures that received VLM-generated alt text.
    """
    figures = [el for el in elements if el.pdf_tag == PDFTag.FIGURE and not el.alt_text]
    if not figures:
        return 0

    try:
        import os
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError:
        logger.warning(
            "VLM alt text requires transformers + torch. "
            "Falling back to placeholders."
        )
        return generate_alt_text_placeholders(elements, input_pdf)

    logger.info("Loading Qwen2.5-VL for alt text generation...")

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            ALT_TEXT.model_name,
            torch_dtype="auto",
        )
        if torch.backends.mps.is_available():
            model.to("mps")
        elif torch.cuda.is_available():
            model.to("cuda")

        processor = AutoProcessor.from_pretrained(
            ALT_TEXT.model_name,
            use_fast=True,
        )

        # Open PDF for figure extraction
        doc = fitz.open(input_pdf)
        count = 0

        for el in figures:
            page_idx = el.page_num - 1
            if page_idx >= len(doc):
                continue

            # Crop figure region from page
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=STANDARD_DPI)
            full_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Crop to figure bbox
            x0, y0, x1, y1 = el.bbox
            x0 = max(0, int(x0))
            y0 = max(0, int(y0))
            x1 = min(pix.width, int(x1))
            y1 = min(pix.height, int(y1))

            if x1 <= x0 or y1 <= y0:
                continue

            fig_img = full_img.crop((x0, y0, x1, y1))

            # Generate alt text
            try:
                messages = [
                    {"role": "system", "content": ALT_TEXT.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": fig_img},
                            {"type": "text", "text": "Describe this figure."},
                        ],
                    },
                ]

                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text_input],
                    images=[fig_img],
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=ALT_TEXT.max_output_tokens,
                        temperature=ALT_TEXT.temperature,
                        do_sample=True,
                    )

                generated = processor.batch_decode(
                    output_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )[0].strip()

                if generated:
                    el.alt_text = generated
                    el.needs_review = False
                    count += 1
                    logger.debug(
                        "Alt text for %s: %s",
                        el.element_id, generated[:80],
                    )

            except Exception as e:
                logger.warning("Alt text gen failed for %s: %s", el.element_id, e)

        doc.close()

        logger.info("VLM alt text generated for %d figures", count)

    except Exception as e:
        logger.error("VLM alt text failed: %s", e)
        return generate_alt_text_placeholders(elements, input_pdf)

    finally:
        # Unload model
        try:
            del model
            del processor
        except NameError:
            pass
        gc.collect()
        try:
            import torch
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        logger.info("VLM unloaded")

    # Generate placeholders for any remaining figures
    remaining = generate_alt_text_placeholders(elements, input_pdf)

    return count + remaining
