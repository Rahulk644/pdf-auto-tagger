"""
MinerU2.5 layout detection — subprocess-based.

Since mineru-vl-utils requires Python <3.14, we spawn a Python 3.11
worker subprocess that handles model loading and inference.

The worker communicates via JSON-over-stdin/stdout, making memory
management clean: killing the subprocess fully reclaims GPU/MPS memory.

Falls back to in-process loading if the current Python satisfies
the version constraint.
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

# Path to the worker script
_WORKER_SCRIPT = Path(__file__).parent / "mineru_worker.py"

# Python 3.11 binary candidates
_PY311_CANDIDATES = [
    "/opt/homebrew/bin/python3.11",
    "/usr/local/bin/python3.11",
    "python3.11",
    "/opt/homebrew/bin/python3.12",
    "/usr/local/bin/python3.12",
    "python3.12",
    "/opt/homebrew/bin/python3.13",
    "/usr/local/bin/python3.13",
    "python3.13",
]


def _find_compatible_python() -> str | None:
    """Find a Python binary that satisfies mineru-vl-utils (<3.14)."""
    for candidate in _PY311_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip().replace("Python ", "")
                major, minor = int(version.split(".")[0]), int(version.split(".")[1])
                if 10 <= minor <= 13 and major == 3:
                    return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            continue
    return None


def _find_or_create_mineru_venv(python_bin: str) -> str:
    """
    Find or create a venv with mineru-vl-utils installed.

    Returns the path to the venv's python binary.
    """
    venv_dir = Path.home() / ".tagger" / "mineru_venv"
    venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        # Check if mineru is installed
        result = subprocess.run(
            [str(venv_python), "-c", "import mineru_vl_utils; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return str(venv_python)

    # Create venv
    logger.info("Creating MinerU venv at %s with %s", venv_dir, python_bin)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [python_bin, "-m", "venv", str(venv_dir)],
        check=True, timeout=30,
    )

    # Install mineru-vl-utils
    logger.info("Installing mineru-vl-utils[transformers]...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-q",
         "mineru-vl-utils[transformers]"],
        check=True, timeout=600,
    )

    return str(venv_python)


class MinerULayoutDetector(LayoutModelAdapter):
    """
    MinerU2.5-based layout detector.

    Uses a subprocess worker for Python version compatibility.
    ~700MB–1GB RAM in the worker process.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LAYOUT.model_name
        self._process: subprocess.Popen | None = None
        self._loaded = False
        self._tmp_dir: str | None = None

    def load(self) -> None:
        """Load MinerU2.5 model via subprocess."""
        if self._loaded:
            return

        # First try in-process (if Python version is compatible)
        major, minor = sys.version_info[:2]
        if minor < 14:
            self._load_inprocess()
            return

        # Find compatible Python
        py_bin = _find_compatible_python()
        if py_bin is None:
            raise RuntimeError(
                "MinerU requires Python <3.14. No compatible Python found. "
                "Install Python 3.11-3.13: brew install python@3.13"
            )

        # Find or create the MinerU venv
        venv_python = _find_or_create_mineru_venv(py_bin)

        # Start worker subprocess
        self._tmp_dir = tempfile.mkdtemp(prefix="tagger_mineru_")

        logger.info("Starting MinerU worker subprocess...")
        self._process = subprocess.Popen(
            [venv_python, str(_WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        # Send load command
        response = self._send_command({
            "action": "load",
            "model_name": self.model_name,
        })

        if response.get("status") != "ok":
            error = response.get("error", "Unknown error")
            self._cleanup_process()
            raise RuntimeError(f"MinerU load failed: {error}")

        self._loaded = True
        logger.info("MinerU2.5 loaded via subprocess (~%dMB)", self.memory_footprint_mb)

    def _load_inprocess(self) -> None:
        """Direct in-process loading when Python version is compatible."""
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from mineru_vl_utils import MinerUClient
        except ImportError as e:
            raise RuntimeError(
                f"MinerU not installed ({e}). Install with: "
                'pip install "mineru-vl-utils[transformers]"'
            )

        import os
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        import torch

        logger.info("Loading MinerU2.5 in-process: %s", self.model_name)

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto",
        )
        if torch.backends.mps.is_available():
            model.to("mps")
        elif torch.cuda.is_available():
            model.to("cuda")

        processor = AutoProcessor.from_pretrained(
            self.model_name, use_fast=True,
        )

        self._client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
        )
        self._loaded = True
        self._inprocess = True
        logger.info("MinerU2.5 loaded in-process")

    def detect(self, page_image: Image.Image, page_num: int) -> list[LayoutRegion]:
        """Run layout detection on a page image."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # In-process path
        if hasattr(self, "_inprocess") and self._inprocess:
            result = self._client.two_step_extract(page_image)
            return self._parse_regions(
                self._normalize_raw(result), page_num, page_image.width, page_image.height
            )

        # Subprocess path: save image to temp file
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix="tagger_mineru_")

        img_path = os.path.join(self._tmp_dir, f"page_{page_num}.png")
        page_image.save(img_path)

        response = self._send_command({
            "action": "detect",
            "image_path": img_path,
        })

        if response.get("status") != "ok":
            logger.error("MinerU detection failed: %s", response.get("error"))
            return []

        regions_raw = response.get("regions", [])
        return self._parse_regions(regions_raw, page_num, page_image.width, page_image.height)

    def unload(self) -> None:
        """Release model from memory."""
        if hasattr(self, "_inprocess") and self._inprocess:
            if hasattr(self, "_client"):
                del self._client
            self._inprocess = False
            gc.collect()
            try:
                import torch
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        else:
            self._cleanup_process()

        self._loaded = False

        # Clean up temp dir
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

        logger.info("MinerU2.5 unloaded")

    @property
    def memory_footprint_mb(self) -> int:
        return 1000

    @property
    def name(self) -> str:
        return f"MinerU2.5 ({self.model_name})"

    def _send_command(self, command: dict) -> dict:
        """Send a JSON command to the worker and read the response."""
        if self._process is None or self._process.stdin is None:
            return {"status": "error", "error": "Worker not running"}

        try:
            self._process.stdin.write(json.dumps(command) + "\n")
            self._process.stdin.flush()

            # Read response with timeout
            response_line = self._process.stdout.readline()
            if not response_line:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read()
                return {
                    "status": "error",
                    "error": f"Worker died. stderr: {stderr[:500]}",
                }

            return json.loads(response_line)

        except (BrokenPipeError, json.JSONDecodeError) as e:
            return {"status": "error", "error": str(e)}

    def _cleanup_process(self) -> None:
        """Kill the worker subprocess."""
        if self._process is not None:
            try:
                self._send_command({"action": "quit"})
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @staticmethod
    def _normalize_raw(result) -> list[dict]:
        """Normalize in-process MinerU output."""
        detections = []
        if isinstance(result, dict):
            detections = (
                result.get("layout_dets", [])
                or result.get("detections", [])
                or result.get("blocks", [])
                or result.get("elements", [])
            )
        elif isinstance(result, list):
            detections = result

        regions = []
        for det in detections:
            region = {
                "category": str(det.get("category", det.get("type", "text"))).lower().strip(),
                "score": float(det.get("score", det.get("confidence", 0.5))),
            }
            bbox_raw = det.get("bbox", det.get("poly", [0, 0, 0, 0]))
            if isinstance(bbox_raw, dict):
                region["bbox"] = [
                    float(bbox_raw.get("x0", 0)), float(bbox_raw.get("y0", 0)),
                    float(bbox_raw.get("x1", 0)), float(bbox_raw.get("y1", 0)),
                ]
            elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
                region["bbox"] = [float(v) for v in bbox_raw[:4]]
            else:
                continue
            regions.append(region)
        return regions

    def _parse_regions(
        self, raw_regions: list[dict], page_num: int, img_width: int = 1000, img_height: int = 1000
    ) -> list[LayoutRegion]:
        """Parse normalized region dicts into LayoutRegion objects."""
        regions: list[LayoutRegion] = []

        for idx, det in enumerate(raw_regions):
            from tagger.stage3_layout.layout_detector import _CATEGORY_MAP, LayoutCategory, LayoutRegion
            from tagger.config import LAYOUT
            
            category = _CATEGORY_MAP.get(det.get("category", "text"), LayoutCategory.TEXT)

            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            
            # MinerU / Qwen-VL outputs coordinates normalized to 1000
            # If coordinates are in [0, 1000] space, scale them back to the image DPI
            if max(x0, x1, y0, y1) <= 1000.0:
                x0 = (x0 / 1000.0) * img_width
                x1 = (x1 / 1000.0) * img_width
                y0 = (y0 / 1000.0) * img_height
                y1 = (y1 / 1000.0) * img_height

            bbox_tuple = (x0, y0, x1, y1)

            # Skip degenerate
            if bbox_tuple[2] <= bbox_tuple[0] or bbox_tuple[3] <= bbox_tuple[1]:
                continue

            confidence = det.get("score", 0.5)
            if confidence < LAYOUT.min_region_confidence:
                continue

            regions.append(LayoutRegion(
                region_id=f"r{page_num}_{idx}",
                page_num=page_num,
                bbox=bbox_tuple,
                category=category,
                reading_order=idx,
                confidence=confidence,
            ))

        return regions
