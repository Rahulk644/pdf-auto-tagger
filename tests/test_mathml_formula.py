"""PDF/UA-2 formula MathML: emitter + Associated-File embedding (no ML)."""
import pikepdf
from pikepdf import Array, Dictionary, Name

from tagger.stage5_specialists.mathml_emitter import latex_to_mathml
from tagger.stage10_writeback.struct_tree_writer import _embed_mathml_af


# ---- emitter -------------------------------------------------------------

def test_emitter_converts_latex():
    m = latex_to_mathml(r"E = mc^2", is_inline=True)
    assert m is not None and "<math" in m and 'display="inline"' in m


def test_emitter_promotes_display_block():
    m = latex_to_mathml(r"\frac{a}{b}", is_inline=False)
    assert 'display="block"' in m and 'display="inline"' not in m


def test_emitter_none_on_empty():
    assert latex_to_mathml("", is_inline=False) is None
    assert latex_to_mathml("   ", is_inline=True) is None


def test_emitter_handles_text_wrapper():
    # The extractor's fallback for non-LaTeX content still yields valid MathML.
    m = latex_to_mathml(r"\text{P \ll Q}", is_inline=True)
    assert m is not None and "<math" in m


# ---- Associated-File embedding ------------------------------------------

def test_embed_mathml_af_builds_valid_filespec():
    pdf = pikepdf.new()
    mathml = '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>'
    fs = _embed_mathml_af(pdf, mathml, 0)
    assert str(fs.get("/Type")) == "/Filespec"
    assert str(fs.get("/AFRelationship")) == "/Supplement"
    ef = fs.get("/EF").get("/F")
    assert str(ef.get("/Type")) == "/EmbeddedFile"
    # MIME subtype application/mathml+xml, slash encoded
    assert str(ef.get("/Subtype")) == "/application#2Fmathml+xml"
    assert bytes(ef.read_bytes()) == mathml.encode("utf-8")
