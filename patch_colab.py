import os
import sys

# 1. Patch mineru_worker.py
file_path_1 = "tagger/stage3_layout/mineru_worker.py"
with open(file_path_1, "r") as f:
    content_1 = f.read()

old_code_1 = """def _normalize_result(result) -> list[dict]:
    \"\"\"Normalize MinerU output into a flat list of region dicts.\"\"\"
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
        region = {}

        # Category
        region["category"] = str(
            det.get("category", det.get("type", "text"))
        ).lower().strip()

        # Bbox
        bbox_raw = det.get("bbox", det.get("poly", [0, 0, 0, 0]))
        if isinstance(bbox_raw, dict):
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

        # Confidence
        region["score"] = float(det.get("score", det.get("confidence", 0.5)))

        regions.append(region)

    return regions"""

new_code_1 = """def _normalize_result(result) -> list[dict]:
    \"\"\"Normalize MinerU output into a flat list of region dicts.\"\"\"
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

    return regions"""

if old_code_1 in content_1:
    content_1 = content_1.replace(old_code_1, new_code_1)
    with open(file_path_1, "w") as f:
        f.write(content_1)
    print("✅ mineru_worker.py parsed and patched!")
else:
    print("⚠️ Could not find old code in mineru_worker.py.")

# 2. Patch formula_extractor.py
file_path_2 = "tagger/stage5_specialists/formula_extractor.py"
with open(file_path_2, "r") as f:
    content_2 = f.read()

old_code_2 = """def _find_unimernet_python() -> str | None:
    \"\"\"Find a Python with unimernet installed.\"\"\"
    venv_python = Path.home() / ".tagger" / "unimernet_venv" / "bin" / "python"
    if venv_python.exists():
        result = subprocess.run(
            [str(venv_python), "-c", "import unimernet; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return str(venv_python)
    return None"""

new_code_2 = """def _find_unimernet_python() -> str | None:
    \"\"\"Find a Python with unimernet installed.\"\"\"
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
    return None"""

if old_code_2 in content_2:
    content_2 = content_2.replace(old_code_2, new_code_2)
    with open(file_path_2, "w") as f:
        f.write(content_2)
    print("✅ formula_extractor.py parsed and patched!")
else:
    print("⚠️ Could not find old code in formula_extractor.py.")
