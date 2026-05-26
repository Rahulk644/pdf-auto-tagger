"""
MinerU2.5 subprocess worker.

Runs in a Python 3.11 process (mineru-vl-utils requires <3.14).
Communicates with the main pipeline via JSON over stdin/stdout.

Protocol:
  → {"action": "load", "model_name": "..."}
  ← {"status": "ok"}

  → {"action": "detect", "image_path": "/tmp/page.png"}
  ← {"status": "ok", "regions": [...]}

  → {"action": "unload"}
  ← {"status": "ok"}
"""

import json
import sys
import gc
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("mineru_worker")


def main():
    model = None
    processor = None
    client = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _respond({"status": "error", "error": f"Invalid JSON: {line}"})
            continue

        action = msg.get("action")

        if action == "load":
            model_name = msg.get("model_name", "opendatalab/MinerU2.5-2509-1.2B")
            try:
                import os
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                import torch
                from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
                from mineru_vl_utils import MinerUClient

                logger.info("Loading %s...", model_name)

                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype="auto",
                )
                # Use MPS on Apple Silicon
                if torch.backends.mps.is_available():
                    model.to("mps")
                    logger.info("Using MPS device")
                elif torch.cuda.is_available():
                    model.to("cuda")
                    logger.info("Using CUDA device")
                else:
                    logger.info("Using CPU device")

                processor = AutoProcessor.from_pretrained(
                    model_name, use_fast=True,
                )

                client = MinerUClient(
                    backend="transformers",
                    model=model,
                    processor=processor,
                )

                logger.info("Model loaded successfully")
                _respond({"status": "ok"})

            except Exception as e:
                _respond({"status": "error", "error": str(e)})

        elif action == "detect":
            if client is None:
                _respond({"status": "error", "error": "Model not loaded"})
                continue

            image_path = msg.get("image_path")
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                result = client.two_step_extract(img)

                # Normalize output to a serializable list of dicts
                regions = _normalize_result(result)
                _respond({"status": "ok", "regions": regions})

            except Exception as e:
                logger.error("Detection failed: %s", e)
                _respond({"status": "error", "error": str(e)})

        elif action == "unload":
            if client is not None:
                del client
                client = None
            if model is not None:
                del model
                model = None
            if processor is not None:
                del processor
                processor = None

            gc.collect()
            try:
                import torch
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass

            logger.info("Model unloaded")
            _respond({"status": "ok"})

        elif action == "ping":
            _respond({"status": "ok", "msg": "pong"})

        elif action == "quit":
            _respond({"status": "ok"})
            break

        else:
            _respond({"status": "error", "error": f"Unknown action: {action}"})


def _normalize_result(result) -> list[dict]:
    """Normalize MinerU output into a flat list of region dicts."""
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
    elif hasattr(result, "blocks"):
        detections = getattr(result, "blocks")
    elif hasattr(result, "layout_dets"):
        detections = getattr(result, "layout_dets")
    else:
        try:
            detections = list(result)
        except Exception:
            pass

    regions = []
    for det in detections:
        region = {}

        if hasattr(det, "get"):
            cat = det.get("category", det.get("type", "text"))
            bbox_raw = det.get("bbox", det.get("poly", [0, 0, 0, 0]))
            score = det.get("score", det.get("confidence", 0.5))
        else:
            cat = getattr(det, "category", getattr(det, "type", "text"))
            bbox_raw = getattr(det, "bbox", getattr(det, "poly", [0, 0, 0, 0]))
            score = getattr(det, "score", getattr(det, "confidence", 0.5))

        region["category"] = str(cat).lower().strip()

        if isinstance(bbox_raw, dict) and hasattr(bbox_raw, "get"):
            region["bbox"] = [
                float(bbox_raw.get("x0", 0)),
                float(bbox_raw.get("y0", 0)),
                float(bbox_raw.get("x1", 0)),
                float(bbox_raw.get("y1", 0)),
            ]
        elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
            region["bbox"] = [float(v) for v in bbox_raw[:4]]
        else:
            continue

        try:
            region["score"] = float(score)
        except Exception:
            region["score"] = 0.5

        regions.append(region)

    return regions


def _respond(data: dict):
    """Write JSON response to stdout."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
