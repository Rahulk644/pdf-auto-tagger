"""veraPDF PDF/UA-1 conformance gate, surfaced as a test.

Runs the full pipeline on each conformance fixture and asserts the tagged
output is veraPDF UA-1 compliant. Skips when veraPDF isn't installed locally
(CI sets VERAPDF_REQUIRED=1 and installs it, so it can't be skipped there).

This is the deterministic line under the "PDF/UA compliant" claim: tagging that
looks right but fails veraPDF fails here.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from verapdf_gate import find_verapdf, validate_ua1, tag  # noqa: E402

FIXTURES = sorted((REPO / "tests" / "fixtures" / "conformance").glob("*.pdf"))
_VERAPDF = find_verapdf()


@pytest.mark.skipif(_VERAPDF is None, reason="veraPDF CLI not installed locally")
@pytest.mark.skipif(not FIXTURES, reason="no conformance fixtures")
@pytest.mark.parametrize("pdf", FIXTURES, ids=lambda p: p.stem)
def test_fixture_is_ua1_compliant(pdf, tmp_path):
    from tagger.config import LAYOUT
    if LAYOUT.backend not in ("cpu", "picodet"):
        pytest.skip("requires a CPU layout backend (no MinerU locally)")
    tagged = tag(pdf, tmp_path)
    ok, failed, detail = validate_ua1(_VERAPDF, tagged)
    assert ok, f"{pdf.name} not PDF/UA-1 compliant: {detail}"
