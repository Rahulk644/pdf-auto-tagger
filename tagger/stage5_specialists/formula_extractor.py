"""
Stage 5d — Formula extractor.

Two modes:
  1. Raw text mode (default): Uses pdfplumber text with heuristic
     LaTeX detection. Low confidence — flags for review.

  2. UniMERNet mode (optional): Subprocess-based image→LaTeX via
     UniMERNet. Requires Python <3.14 venv + unimernet package.

Raw text mode is always available. UniMERNet integration is designed
to work via the same subprocess pattern as MinerU.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from tagger.config import FORMULA
from tagger.models.data_types import FormulaResult, LayoutRegion, PageElement

logger = logging.getLogger(__name__)


def extract_formula(
    region: LayoutRegion,
    matched_elements: list[PageElement],
    page_image: Image.Image | None = None,
    use_vlm: bool = False,
) -> FormulaResult:
    """
    Extract formula content from a region.

    Args:
        region: Layout region classified as FORMULA.
        matched_elements: PageElements that overlap this region.
        page_image: Full page image (needed for UniMERNet mode).
        use_vlm: If True, try UniMERNet for image→LaTeX.

    Returns:
        FormulaResult with LaTeX string.
    """
    # Try UniMERNet if requested and available
    if use_vlm and page_image is not None:
        result = _try_unimernet(region, page_image)
        if result is not None:
            return result

    # Fallback: raw text extraction
    return _extract_raw_text(region, matched_elements)


def _extract_raw_text(
    region: LayoutRegion,
    matched_elements: list[PageElement],
) -> FormulaResult:
    """Extract formula content from raw text (fallback)."""
    raw_text = " ".join(el.text for el in matched_elements if el.text).strip()

    if not raw_text:
        logger.debug("Region %s: empty formula region", region.region_id)
        return FormulaResult(
            region_id=region.region_id,
            latex="",
            is_inline=_is_inline_formula(region),
            confidence=0.3,
        )

    # Check if the text already looks like LaTeX
    latex_chars = {"\\", "^", "_", "{", "}", "\\frac", "\\sum", "\\int"}
    is_latex = any(c in raw_text for c in latex_chars)

    # Simple cleanup for common math font issues
    cleaned = raw_text
    if not is_latex:
        # Wrap in \text{} if it doesn't look like LaTeX
        cleaned = f"\\text{{{raw_text}}}"

    return FormulaResult(
        region_id=region.region_id,
        latex=cleaned if is_latex else f"\\text{{{raw_text}}}",
        is_inline=_is_inline_formula(region),
        confidence=0.4 if is_latex else 0.2,
    )


def _try_unimernet(
    region: LayoutRegion,
    page_image: Image.Image,
) -> FormulaResult | None:
    """
    Try UniMERNet formula extraction via subprocess.

    Returns None if UniMERNet is not available.
    """
    # Check if we have a compatible Python
    py_bin = _find_unimernet_python()
    if py_bin is None:
        return None

    # Crop formula region from page image
    x0, y0, x1, y1 = region.bbox
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(page_image.width, int(x1))
    y1 = min(page_image.height, int(y1))

    if x1 <= x0 or y1 <= y0:
        return None

    crop = page_image.crop((x0, y0, x1, y1))

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        crop.save(f.name)
        img_path = f.name

    try:
        # Run UniMERNet via subprocess
        script = (
            "import sys, json\n"
            "import os\n"
            "os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'\n"
            "from unimernet.common.config import Config\n"
            "from unimernet.processors import load_processor\n"
            "from unimernet.models import load_model\n"
            "from PIL import Image\n"
            f"img = Image.open('{img_path}').convert('RGB')\n"
            f"model_name = '{FORMULA.model_name}'\n"
            "cfg = Config({'model': {'name': model_name}})\n"
            "model = load_model(cfg)\n"
            "processor = load_processor(cfg)\n"
            "inputs = processor(img)\n"
            "result = model.generate(inputs)\n"
            "print(json.dumps({'latex': result, 'status': 'ok'}))\n"
        )

        result = subprocess.run(
            [py_bin, "-c", script],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode == 0:
            output = json.loads(result.stdout.strip())
            latex = output.get("latex", "")
            if latex:
                return FormulaResult(
                    region_id=region.region_id,
                    latex=latex,
                    is_inline=_is_inline_formula(region),
                    confidence=0.85,
                )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.debug("UniMERNet failed for %s: %s", region.region_id, e)

    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)

    return None


def find_latexocr_python() -> str | None:
    """Locate a Python with rapid_latex_ocr (the onnx image→LaTeX recogniser).
    Order: env TAGGER_LATEXOCR_PYTHON, then ~/.tagger/latexocr_venv. rapid_latex_ocr
    caps at Python<3.13 so it can't live in the main py3.14 venv — it runs in an
    isolated venv via subprocess. Returns None (→ graceful text fallback) if absent."""
    candidates = [
        os.environ.get("TAGGER_LATEXOCR_PYTHON"),
        str(Path.home() / ".tagger" / "latexocr_venv" / "bin" / "python"),
    ]
    for py_bin in candidates:
        if py_bin and os.path.exists(py_bin):
            try:
                r = subprocess.run(
                    [py_bin, "-c", "import rapid_latex_ocr; print('ok')"],
                    capture_output=True, text=True, timeout=20,
                )
                if r.returncode == 0 and "ok" in r.stdout:
                    return py_bin
            except Exception:
                pass
    return None


def batch_rapid_latex(image_paths: list[str], py_bin: str | None = None) -> dict[str, str]:
    """Image→LaTeX for many formula crops in ONE subprocess (LaTeXOCR loaded once —
    a doc can have 100+ formulas, so per-crop spawning is infeasible). Returns
    {image_path: latex}; missing/failed crops are omitted (caller keeps text-layer
    LaTeX for those). Pure graceful: empty dict if the recogniser venv is absent."""
    if not image_paths:
        return {}
    py_bin = py_bin or find_latexocr_python()
    if py_bin is None:
        return {}
    manifest = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(image_paths, manifest)
    manifest.close()
    out_path = manifest.name + ".out"
    script = (
        "import sys, json\n"
        "from rapid_latex_ocr import LaTeXOCR\n"
        "ocr = LaTeXOCR()\n"
        "paths = json.load(open(sys.argv[1]))\n"
        "res = {}\n"
        "for p in paths:\n"
        "    try:\n"
        "        with open(p, 'rb') as f: data = f.read()\n"
        "        latex, _ = ocr(data)\n"
        "        if latex and latex.strip(): res[p] = latex.strip()\n"
        "    except Exception: pass\n"
        "json.dump(res, open(sys.argv[2], 'w'))\n"
    )
    try:
        # ~0.1-0.4s/crop on CPU + one-time model load; scale timeout with count.
        subprocess.run([py_bin, "-c", script, manifest.name, out_path],
                       capture_output=True, text=True, timeout=60 + 2 * len(image_paths))
        if os.path.exists(out_path):
            return json.load(open(out_path))
    except Exception as e:
        logger.warning("rapid_latex_ocr batch failed: %s", e)
    finally:
        for p in (manifest.name, out_path):
            if os.path.exists(p):
                os.unlink(p)
    return {}


def _find_unimernet_python() -> str | None:
    """Find a Python with unimernet installed."""
    import sys
    candidates = [sys.executable, str(Path.home() / ".tagger" / "unimernet_venv" / "bin" / "python")]
    for py_bin in candidates:
        if os.path.exists(py_bin):
            result = subprocess.run(
                [py_bin, "-c", "import unimernet; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                return py_bin
    return None


def _is_inline_formula(region: LayoutRegion) -> bool:
    """
    Heuristic: inline formulas are typically short and narrow.

    Display formulas tend to be wider and on their own line.
    """
    height = region.bbox[3] - region.bbox[1]
    width = region.bbox[2] - region.bbox[0]

    # If the formula region is roughly one line tall → inline
    # Standard line height at 150 DPI ≈ 20-25px for 11pt text
    return height < 35 and width < 200
