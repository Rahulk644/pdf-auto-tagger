"""Struct-tree reading utilities — guards the two hand-validation findings:
F1 struct elements identified by /S (NOT the optional /Type /StructElem),
F2 custom /S tags resolved through /RoleMap to standard types.
"""
import pikepdf
from pikepdf import Array, Dictionary, Name

from tagger.benchmark.struct_utils import (
    mcid_tag_map,
    role_resolver,
    strip_tag,
    tag_counts,
)


def _typeless_rolemapped_pdf(path):
    """1-page PDF whose struct elems have NO /Type and use custom RoleMapped tags."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    page = pdf.pages[0]
    sr = pdf.make_indirect(Dictionary({"/Type": Name.StructTreeRoot}))
    sr["/RoleMap"] = Dictionary({"/HEAD_42": Name("/H1"), "/BODY": Name("/P")})
    # struct elements: NO /Type key, custom /S, MCID content refs
    h = pdf.make_indirect(Dictionary({"/S": Name("/HEAD_42"), "/Pg": page.obj, "/K": 0}))
    p = pdf.make_indirect(Dictionary({"/S": Name("/BODY"), "/Pg": page.obj, "/K": 1}))
    doc = pdf.make_indirect(Dictionary({"/S": Name("/Document"), "/K": Array([h, p])}))
    sr["/K"] = doc
    pdf.Root["/StructTreeRoot"] = sr
    pdf.save(str(path))
    pdf.close()


def test_role_resolver_follows_chain():
    sr = Dictionary({"/RoleMap": Dictionary(
        {"/A": Name("/B"), "/B": Name("/P"), "/Title": Name("/H1")})})
    r = role_resolver(sr)
    assert r("/A") == "/P"          # chain A->B->P
    assert r("/Title") == "/H1"
    assert r("/P") == "/P"          # already standard
    assert r("/Unknown") == "/Unknown"


def test_role_resolver_cycle_safe():
    sr = Dictionary({"/RoleMap": Dictionary({"/A": Name("/B"), "/B": Name("/A")})})
    r = role_resolver(sr)
    assert r("/A") in ("/A", "/B")  # terminates, doesn't hang


def test_tag_counts_typeless_and_rolemapped(tmp_path):
    p = tmp_path / "t.pdf"
    _typeless_rolemapped_pdf(p)
    c = tag_counts(p)
    # custom tags resolved to standard; counted despite missing /Type
    assert c["/H1"] == 1 and c["/P"] == 1 and c["/Document"] == 1


def test_mcid_tag_map_typeless_and_rolemapped(tmp_path):
    p = tmp_path / "t.pdf"
    _typeless_rolemapped_pdf(p)
    with pikepdf.open(str(p)) as pdf:
        m = mcid_tag_map(pdf)
    assert m[(0, 0)] == "/H1"       # HEAD_42 -> H1, MCID 0
    assert m[(0, 1)] == "/P"        # BODY -> P, MCID 1


def test_tag_counts_untagged_returns_none(tmp_path):
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 200))
    p = tmp_path / "u.pdf"
    pdf.save(str(p))
    pdf.close()
    assert tag_counts(p) is None


def test_strip_tag():
    assert strip_tag("/H1") == "H1"
    assert strip_tag(None) == "Artifact"
    assert strip_tag("") == "Artifact"
