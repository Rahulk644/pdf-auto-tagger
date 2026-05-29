"""
Stage 9 — Alt text generator for figures.

Modes:
  1. Placeholder mode (default): Generates a review-required placeholder
     alt text for all figures. This makes the output valid per PDF/UA
     (the /Alt attribute exists) while signalling human review is needed.

  2. VLM mode (optional): generates real alt text with a vision-language model.
     Two interchangeable backends, selected by ``ALT_TEXT.vlm_backend``:
       - "gemma_e4b" (default): calls the deployed Gemma-4-E4B vLLM endpoint
         (the same model used by the QA auditor — one VLM for the whole stack).
       - "qwen": loads Qwen2.5-VL-7B in-process (~7GB; kept for the head-to-head
         quality comparison when the alt-text stage is tackled).

Placeholder mode runs with zero additional dependencies and is always available.
The two VLM backends' relative alt-text QUALITY is not yet measured (the alt-text
quality eval is deferred) — compare them when the alt-text stage is reached.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from tagger.config import ALT_TEXT, STANDARD_DPI
from tagger.models.data_types import PDFTag, TaggedElement

logger = logging.getLogger(__name__)


def _crop_figure(doc, el: TaggedElement) -> Image.Image | None:
    """Render the figure's page at STANDARD_DPI and crop to its bbox."""
    page_idx = el.page_num - 1
    if page_idx >= len(doc):
        return None
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=STANDARD_DPI)
    full_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    x0, y0, x1, y1 = el.bbox
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(pix.width, int(x1)); y1 = min(pix.height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return full_img.crop((x0, y0, x1, y1))


def generate_alt_text_vlm(elements: list[TaggedElement], input_pdf: str) -> int:
    """Dispatch to the configured VLM backend (default: Gemma-4-E4B endpoint).

    Falls back to placeholders if the backend is unavailable.
    """
    backend = ALT_TEXT.vlm_backend
    if backend == "qwen":
        return generate_alt_text_qwen(elements, input_pdf)
    if backend == "gemma_e4b":
        return generate_alt_text_e4b(elements, input_pdf)
    logger.warning("Unknown alt-text vlm_backend %r; using placeholders", backend)
    return generate_alt_text_placeholders(elements, input_pdf)


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


def _e4b_caption(endpoint: str, fig_img: Image.Image) -> str | None:
    """POST one figure crop to the Gemma-4-E4B vLLM endpoint, return a caption.

    Reuses the QA auditor's HTTP contract ({image_b64, prompt, ...} -> {response}),
    but with thinking OFF and the accessibility captioning prompt — we want a clean
    1-2 sentence description, not a reasoning trace.
    """
    buf = io.BytesIO()
    fig_img.save(buf, format="PNG")
    payload = json.dumps({
        "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "prompt": ALT_TEXT.system_prompt + "\n\nDescribe this figure.",
        "max_tokens": ALT_TEXT.max_output_tokens,
        "temperature": ALT_TEXT.temperature,
        "enable_thinking": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("response") or "").strip()
    return text or None


def generate_alt_text_e4b(elements: list[TaggedElement], input_pdf: str) -> int:
    """Generate alt text via the deployed Gemma-4-E4B vLLM endpoint.

    Endpoint URL comes from the ``ALT_TEXT.gemma_endpoint_env`` environment
    variable. Unset/unreachable -> placeholders (graceful, like the Qwen path).
    """
    figures = [el for el in elements if el.pdf_tag == PDFTag.FIGURE and not el.alt_text]
    if not figures:
        return 0

    endpoint = os.environ.get(ALT_TEXT.gemma_endpoint_env)
    if not endpoint:
        logger.warning(
            "Gemma E4B alt text needs $%s (endpoint URL); using placeholders.",
            ALT_TEXT.gemma_endpoint_env,
        )
        return generate_alt_text_placeholders(elements, input_pdf)

    logger.info("Generating alt text via Gemma-4-E4B endpoint (%d figures)", len(figures))
    doc = fitz.open(input_pdf)
    count = 0
    try:
        for el in figures:
            fig_img = _crop_figure(doc, el)
            if fig_img is None:
                continue
            try:
                caption = _e4b_caption(endpoint, fig_img)
            except Exception as e:
                logger.warning("E4B alt text failed for %s: %s", el.element_id, e)
                continue
            if caption:
                el.alt_text = caption
                el.needs_review = False
                count += 1
                logger.debug("E4B alt text for %s: %s", el.element_id, caption[:80])
    finally:
        doc.close()

    logger.info("Gemma-4-E4B alt text generated for %d figures", count)
    # Placeholders for any figure that didn't get a caption.
    return count + generate_alt_text_placeholders(elements, input_pdf)


def generate_alt_text_qwen(
    elements: list[TaggedElement],
    input_pdf: str,
) -> int:
    """
    Generate alt text using Qwen2.5-VL (7B), loaded in-process.

    Loads the VLM, generates alt text for each figure, then unloads.
    Requires ~7GB RAM — not safe on M1 8GB with other processes. Retained for the
    head-to-head quality comparison against the Gemma-E4B backend.

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
            fig_img = _crop_figure(doc, el)
            if fig_img is None:
                continue

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
