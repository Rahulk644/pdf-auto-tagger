"""
Flask API for the PDF auto-tagger.

Endpoints:
  POST /tag          — Upload a PDF, run the pipeline, return report
  POST /classify     — Upload a PDF, run Stage 0 only (page classification)
  POST /extract      — Upload a PDF, run Stages 0-2 (extract + merge)
  GET  /health       — Health check

Port 5002 (avoids conflict with PREP-QA-Tool on 5001).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from tagger.config import PIPELINE
from tagger.pipeline import AutoTaggerPipeline

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Session storage for uploaded files
_SESSIONS: dict[str, dict] = {}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "version": "0.1.0"})


@app.route("/tag", methods=["POST"])
def tag_pdf():
    """
    Upload a PDF and run the full auto-tagging pipeline.

    Expects multipart form data with a 'pdf_file' field.

    Returns JSON report with all tagged elements and confidence scores.
    """
    if "pdf_file" not in request.files:
        return jsonify({"error": "No pdf_file in request"}), 400

    pdf_file = request.files["pdf_file"]
    if not pdf_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Create session
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.mkdtemp(prefix=f"tagger_{session_id[:8]}_"))

    input_path = session_dir / pdf_file.filename
    pdf_file.save(str(input_path))

    output_path = session_dir / f"tagged_{pdf_file.filename}"
    report_path = session_dir / "report.json"

    # Store session
    _SESSIONS[session_id] = {
        "session_dir": str(session_dir),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
    }

    try:
        pipeline = AutoTaggerPipeline()
        report = pipeline.run(
            input_pdf=str(input_path),
            output_pdf=str(output_path),
            report_path=str(report_path),
        )

        return jsonify({
            "session_id": session_id,
            "status": "complete",
            "report": report,
        })

    except Exception as e:
        logger.exception("Pipeline failed for session %s", session_id)
        return jsonify({
            "session_id": session_id,
            "status": "error",
            "error": str(e),
        }), 500


@app.route("/classify", methods=["POST"])
def classify_pages():
    """
    Upload a PDF and run only Stage 0 (page classification).

    Quick endpoint for understanding a document before full processing.
    """
    if "pdf_file" not in request.files:
        return jsonify({"error": "No pdf_file in request"}), 400

    pdf_file = request.files["pdf_file"]
    if not pdf_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    session_dir = Path(tempfile.mkdtemp(prefix="tagger_classify_"))
    input_path = session_dir / pdf_file.filename
    pdf_file.save(str(input_path))

    try:
        from tagger.stage0_classifier.page_classifier import classify_pages as _classify

        classifications = _classify(str(input_path))

        return jsonify({
            "status": "complete",
            "filename": pdf_file.filename,
            "pages": [
                {
                    "page_num": c.page_num,
                    "page_type": c.page_type.value,
                    "char_count": c.char_count,
                    "image_coverage": round(c.image_coverage, 3),
                    "unicode_validity": round(c.unicode_validity, 3),
                    "char_density": round(c.char_density, 5),
                    "confidence": round(c.confidence, 3),
                    "page_width_pt": c.page_width_pt,
                    "page_height_pt": c.page_height_pt,
                }
                for c in classifications
            ],
        })

    except Exception as e:
        logger.exception("Classification failed")
        return jsonify({"status": "error", "error": str(e)}), 500

    finally:
        # Clean up
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)


@app.route("/extract", methods=["POST"])
def extract_elements():
    """
    Upload a PDF and run Stages 0–2 (classify + extract + merge).

    Returns extracted and merged text elements with font metadata.
    """
    if "pdf_file" not in request.files:
        return jsonify({"error": "No pdf_file in request"}), 400

    pdf_file = request.files["pdf_file"]
    if not pdf_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    session_dir = Path(tempfile.mkdtemp(prefix="tagger_extract_"))
    input_path = session_dir / pdf_file.filename
    pdf_file.save(str(input_path))

    try:
        from tagger.stage0_classifier.page_classifier import classify_pages as _classify
        from tagger.stage1_extraction.native_extractor import extract_native_pages
        from tagger.stage2_merger.text_merger import merge_page_elements

        # Stage 0
        classifications = _classify(str(input_path))

        # Stage 1
        raw_elements = extract_native_pages(str(input_path), classifications)

        # Stage 2
        merged_elements: dict[int, list] = {}
        for page_num, chars in raw_elements.items():
            merged = merge_page_elements(chars, page_num)
            merged_elements[page_num] = [
                {
                    "element_id": el.element_id,
                    "text": el.text,
                    "bbox": list(el.bbox),
                    "font_name": el.font_name,
                    "font_size": el.font_size,
                    "font_weight": el.font_weight,
                    "font_color": el.font_color,
                    "is_italic": el.is_italic,
                    "confidence": el.confidence,
                }
                for el in merged
            ]

        total_elements = sum(len(v) for v in merged_elements.values())

        return jsonify({
            "status": "complete",
            "filename": pdf_file.filename,
            "total_pages": len(classifications),
            "total_elements": total_elements,
            "pages": {
                str(page_num): {
                    "classification": classifications[page_num - 1].page_type.value
                    if page_num <= len(classifications) else "unknown",
                    "elements": elements,
                }
                for page_num, elements in merged_elements.items()
            },
        })

    except Exception as e:
        logger.exception("Extraction failed")
        return jsonify({"status": "error", "error": str(e)}), 500

    finally:
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)


@app.route("/download/<session_id>", methods=["GET"])
def download_output(session_id: str):
    """Download the tagged output PDF for a completed session."""
    session = _SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    output_path = session.get("output_path")
    if not output_path or not Path(output_path).exists():
        return jsonify({"error": "Output file not found"}), 404

    return send_file(output_path, as_attachment=True)


@app.route("/report/<session_id>", methods=["GET"])
def get_report(session_id: str):
    """Get the JSON report for a completed session."""
    session = _SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    report_path = session.get("report_path")
    if not report_path or not Path(report_path).exists():
        return jsonify({"error": "Report not found"}), 404

    with open(report_path, "r") as f:
        report = json.load(f)

    return jsonify(report)


@app.route("/cleanup/<session_id>", methods=["POST"])
def cleanup_session(session_id: str):
    """Clean up a session's temporary files."""
    session = _SESSIONS.pop(session_id, None)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    import shutil
    session_dir = session.get("session_dir")
    if session_dir:
        shutil.rmtree(session_dir, ignore_errors=True)

    return jsonify({"status": "cleaned", "session_id": session_id})


def create_app() -> Flask:
    """Application factory for the Flask app."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return app


if __name__ == "__main__":
    create_app()
    app.run(
        host=PIPELINE.flask_host,
        port=PIPELINE.flask_port,
        debug=PIPELINE.flask_debug,
    )
