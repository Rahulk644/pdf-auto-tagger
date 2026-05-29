"""unicode_mapping verdict (adjacent axis, PDF/UA 7.21.x Unicode recoverability).

Extraction-based: counts glyphs the PDF text layer can't map to real Unicode —
"(cid:N)" / U+FFFD (missing mapping) and Private-Use-Area codepoints (look like
text, inaccessible to assistive tech). Ratio logic is unit-tested here; corpus
behaviour is checked manually.
"""
import pikepdf

from tagger.benchmark.verdicts import unicode_mapping
from tagger.benchmark.verdicts.base import CannotDeriveReason


def test_ratio_clean_text():
    r, u, t = unicode_mapping._unmapped_ratio("Hello world, fully recoverable.")
    assert u == 0 and r == 0.0 and t > 0


def test_ratio_cid_glyphs():
    # 2 unmapped (cid) + 3 mapped (a,b,c) -> 2/5
    r, u, t = unicode_mapping._unmapped_ratio("(cid:5)(cid:6)abc")
    assert u == 2 and t == 5 and abs(r - 0.4) < 1e-9


def test_ratio_replacement_char():
    r, u, t = unicode_mapping._unmapped_ratio("ab�c")  # one U+FFFD
    assert u == 1 and t == 4


def test_ratio_private_use_area():
    # 2 PUA glyphs (U+E000, U+F8FF) + 3 mapped (a,b,c) -> 2/5 — looks like text, isn't
    r, u, t = unicode_mapping._unmapped_ratio("abc")
    assert u == 2 and t == 5 and abs(r - 0.4) < 1e-9


def test_is_pua_ranges():
    assert unicode_mapping._is_pua("") and unicode_mapping._is_pua("")
    assert unicode_mapping._is_pua("\U000f0000") and unicode_mapping._is_pua("\U00100000")
    assert not unicode_mapping._is_pua("a") and not unicode_mapping._is_pua("π")  # pi


def test_ratio_whitespace_ignored():
    r, u, t = unicode_mapping._unmapped_ratio("a b\nc\t")  # 3 glyphs, none unmapped
    assert u == 0 and t == 3


def test_fail_threshold_logic():
    high = "(cid:1)" * 5 + "ok"           # 5 unmapped / 7 -> well above threshold
    low = "(cid:1)" + "x" * 200           # 1 / 201 -> below threshold
    assert unicode_mapping._unmapped_ratio(high)[0] > unicode_mapping._FAIL_RATIO
    assert unicode_mapping._unmapped_ratio(low)[0] <= unicode_mapping._FAIL_RATIO


def test_no_text_cannot_derive(tmp_path):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    p = tmp_path / "blank.pdf"
    pdf.save(str(p))
    pdf.close()
    with pikepdf.open(str(p)) as d:
        v = unicode_mapping.verdict(d, str(p))
    assert v.status == "cannot_derive" and v.reason == CannotDeriveReason.NoElementsOfType
