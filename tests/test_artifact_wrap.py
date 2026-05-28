"""Tests for Stage 10 clause 7.1-3 artifact wrapping (content_stream_writer)."""

import pikepdf
import pytest

from tagger.stage10_writeback.content_stream_writer import (
    _rewrite_stream,
    artifact_wrap_forms,
)

_MC_OPEN = {"BDC", "BMC"}
_PAINT = {"f", "F", "f*", "S", "s", "B", "B*", "b", "b*", "sh", "Tj", "TJ", "'", '"'}


def _parse(data: bytes):
    pdf = pikepdf.new()
    stream = pdf.make_stream(data)
    return list(pikepdf.parse_content_stream(stream))


def _walk(instructions, do_subtype=lambda n: None):
    """Replay instructions, returning depth/nesting facts for assertions."""
    depth = 0
    art_stack = []          # bool per open MC: is it an /Artifact?
    mark_depths = []        # depth at which each mark-producing op executes
    mcid_opens = []         # every /MCID value opened
    barrier_artifact_open = []  # artifact open flag observed at each q/Q/BT/ET
    art_open = lambda: any(art_stack)

    for ins in instructions:
        op = str(ins.operator)
        if op in _MC_OPEN:
            is_art = op == "BMC" and str(ins.operands[0]) == "/Artifact"
            if op == "BDC" and len(ins.operands) >= 2:
                try:
                    if "/MCID" in ins.operands[1]:
                        mcid_opens.append(int(ins.operands[1]["/MCID"]))
                except Exception:
                    pass
            art_stack.append(is_art)
            depth += 1
            continue
        if op == "EMC":
            depth -= 1
            if art_stack:
                art_stack.pop()
            assert depth >= 0, "EMC without matching BMC/BDC"
            continue
        if op in ("q", "Q", "BT", "ET"):
            barrier_artifact_open.append(art_open())
        if op == "Do":
            sub = do_subtype(str(ins.operands[0]) if ins.operands else "")
            if sub == "/Form":
                continue  # form invocation is not a mark
            mark_depths.append(depth)
            continue
        if op in _PAINT or op == "INLINE_IMAGE":
            mark_depths.append(depth)

    return {
        "final_depth": depth,
        "mark_depths": mark_depths,
        "mcid_opens": mcid_opens,
        "barrier_artifact_open": barrier_artifact_open,
    }


def test_every_mark_wrapped_and_nested():
    data = (
        b"BT /F1 12 Tf 100 700 Td (Hello) Tj ( ) Tj (World) Tj ET\n"
        b"q 1 0 0 1 0 0 cm 10 10 100 100 re f Q\n"
        b"0 0 1 RG 5 5 m 50 50 l S\n"
        b"/Im0 Do\n"
        b"q /Fo0 Do Q\n"
        b"BT 200 200 Td (foot) Tj ET\n"
    )
    instrs = _parse(data)

    # "Hello"=5 chars -> idx 0..4; " "=idx5 (unmapped); "World"=idx6..10 -> MCID 0.
    char_to_mcid = {i: 0 for i in list(range(0, 5)) + list(range(6, 11))}
    mcid_to_tag = {0: "P"}
    subs = {"/Im0": "/Image", "/Fo0": "/Form"}
    do_subtype = lambda n: subs.get(n)

    new_cs, injected = _rewrite_stream(instrs, char_to_mcid, mcid_to_tag, do_subtype)
    facts = _walk(new_cs, do_subtype)

    assert injected == {0}
    assert facts["final_depth"] == 0, "marked content not balanced"
    assert all(d >= 1 for d in facts["mark_depths"]), "a mark sits outside marked content"
    # MCID 0 must open exactly once (whitespace stays inside the tag, no reopen).
    assert facts["mcid_opens"] == [0], f"duplicate/unexpected MCID opens: {facts['mcid_opens']}"
    # No artifact sequence straddles a q/Q/BT/ET barrier.
    assert not any(facts["barrier_artifact_open"]), "artifact MC straddles a barrier"


def test_no_content_operators_dropped_or_reordered():
    data = (
        b"BT /F1 12 Tf 100 700 Td (Hello) Tj ( ) Tj (World) Tj ET\n"
        b"q 10 10 100 100 re f Q\n"
        b"5 5 m 50 50 l S\n"
    )
    instrs = _parse(data)
    char_to_mcid = {i: 0 for i in list(range(0, 5)) + list(range(6, 11))}
    new_cs, _ = _rewrite_stream(instrs, char_to_mcid, {0: "P"}, lambda n: None)

    def content_ops(seq):
        return [str(i.operator) for i in seq if str(i.operator) not in {"BDC", "BMC", "EMC"}]

    assert content_ops(new_cs) == content_ops(instrs), "content operators changed"


def test_empty_mapping_artifacts_everything():
    data = b"10 10 100 100 re f\nBT (x) Tj ET\n"
    instrs = _parse(data)
    new_cs, injected = _rewrite_stream(instrs, {}, {}, lambda n: None)
    facts = _walk(new_cs)
    assert injected == set()
    assert facts["final_depth"] == 0
    assert all(d >= 1 for d in facts["mark_depths"])
    assert facts["mcid_opens"] == []  # no tagged content


def test_form_xobject_marks_wrapped():
    pdf = pikepdf.new()
    form = pdf.make_stream(b"10 10 50 50 re f\n")
    form["/Type"] = pikepdf.Name("/XObject")
    form["/Subtype"] = pikepdf.Name("/Form")
    form["/BBox"] = pikepdf.Array([0, 0, 100, 100])
    form_ref = pdf.make_indirect(form)

    page = pdf.add_blank_page(page_size=(200, 200))
    page.obj["/Resources"] = pikepdf.Dictionary(
        {"/XObject": pikepdf.Dictionary({"/Fo0": form_ref})}
    )
    page.obj["/Contents"] = pdf.make_stream(b"q /Fo0 Do Q\n")

    n = artifact_wrap_forms(pdf)
    assert n == 1

    facts = _walk(list(pikepdf.parse_content_stream(form_ref)))
    assert facts["final_depth"] == 0
    assert all(d >= 1 for d in facts["mark_depths"]), "form mark not artifact-wrapped"
