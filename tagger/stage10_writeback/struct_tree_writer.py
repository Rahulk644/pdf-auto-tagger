"""
Stage 10 — Struct tree writeback via pikepdf.

Writes the auto-tagger's output into the PDF's structure tree,
making the document accessible per PDF/UA.

Two modes:
  V1 (re-tag): Modifies existing tagged PDF's struct tree entries
  V2 (full):   Builds struct tree from scratch for untagged PDFs

pikepdf Dictionary() takes string keys with '/' prefix.
Name objects are used only for values (e.g., Name.Document).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pikepdf
from pikepdf import Dictionary, Name, Array, String

from tagger.config import WRITEBACK
from tagger.models.data_types import PDFTag, TaggedElement
from tagger.stage1_extraction.coord_transformer import pdf_to_standard

logger = logging.getLogger(__name__)

def _is_numeric_content(text: str) -> bool:
    """Return True if text is empty or contains only numeric/currency content."""
    if not text:
        return True
    cleaned = text.strip().lstrip("$").replace(",", "").replace("(", "").replace(")", "").replace("%", "").replace("-", "").strip()
    return not cleaned or cleaned.replace(".", "").isdigit()

def retag_existing_pdf(
    input_path: str | Path,
    output_path: str | Path,
    tagged_elements: list[TaggedElement],
) -> dict:
    """
    V1: Re-tag an existing tagged PDF by modifying struct tree entries.

    Walks the existing structure tree, matches elements by MCID,
    and updates the /S (structure type) entries.
    """
    stats = {
        "total_struct_elems": 0,
        "matched": 0,
        "changed": 0,
        "unmatched": 0,
    }

    # Build MCID → tag mapping from our results
    mcid_to_tag: dict[int, tuple[str, int]] = {}
    for el in tagged_elements:
        if el.original_mcid is not None:
            mcid_to_tag[el.original_mcid] = (el.pdf_tag.value, el.page_num)

    if not mcid_to_tag:
        logger.warning("No elements have MCIDs — cannot re-tag. Copying input as-is.")
        import shutil
        shutil.copy2(str(input_path), str(output_path))
        return stats

    try:
        pdf = pikepdf.open(str(input_path))
        root = pdf.Root

        struct_tree_root = root.get("/StructTreeRoot")
        if struct_tree_root is None:
            logger.warning("PDF has no StructTreeRoot — cannot re-tag")
            pdf.save(str(output_path))
            pdf.close()
            return stats

        _walk_and_retag(struct_tree_root, mcid_to_tag, stats)

        pdf.save(str(output_path))
        pdf.close()

        logger.info(
            "Re-tagged: %d struct elements, %d matched, %d changed, %d unmatched",
            stats["total_struct_elems"], stats["matched"],
            stats["changed"], stats["unmatched"],
        )

    except Exception as e:
        logger.error("Re-tag failed: %s", e)
        import shutil
        shutil.copy2(str(input_path), str(output_path))

    return stats


def _mcid_k(mcids: list[int]):
    """Build a struct element's /K value from its page-local MCIDs.

    One MCID -> the integer itself; several (split content) -> an Array of
    integers, all referring to marked content on the element's /Pg.
    """
    if not mcids:
        return None
    if len(mcids) == 1:
        return mcids[0]
    return Array(list(mcids))


def _intersect_area(a, b) -> float:
    """Absolute intersection area of two (x0, y0, x1, y1) boxes in one space."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _append_k_child(elem, child) -> None:
    """Append a child struct element to elem's /K, normalizing /K to an Array.

    /K may legally mix marked-content MCID integers with child struct elements,
    so a bare int (single MCID) is promoted to an Array before appending.
    """
    k = elem.get("/K")
    if k is None:
        elem["/K"] = Array([child])
    elif isinstance(k, Array):
        k.append(child)
    else:
        elem["/K"] = Array([k, child])


def _tag_link_annotations(
    pdf,
    page,
    doc_elem,
    owners: list[tuple[tuple, object]],
    page_height_pt: float,
    parent_map_entries: list,
    next_sp: int,
    stats: dict,
) -> int:
    """Bind each Link annotation on the page to a Link struct element (clause 7.18.5-1).

    For every /Link annotation: create a Link struct elem holding an OBJR back to
    the annotation, nest it under the most-overlapping block (or Document), and
    reassign the annotation's /StructParent to a fresh ParentTree key that resolves
    to the Link. Marked content is left untouched (link text stays under its block).
    """
    annots = page.obj.get("/Annots")
    if annots is None:
        return next_sp

    for a in annots:
        if str(a.get("/Subtype")) != "/Link":
            continue
        rect = a.get("/Rect")
        if rect is None:
            continue

        rect_std = pdf_to_standard(
            tuple(float(x) for x in rect), page_height_pt
        )
        owner = doc_elem
        best = 0.0
        for bbox, se in owners:
            area = _intersect_area(rect_std, bbox)
            if area > best:
                best = area
                owner = se

        link_elem = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Link"),
            "/P": owner,
            "/K": Array([Dictionary({
                "/Type": Name.OBJR,
                "/Obj": a,
                "/Pg": page.obj,
            })]),
        }))
        _append_k_child(owner, link_elem)

        a["/StructParent"] = next_sp
        parent_map_entries.append((next_sp, link_elem))
        next_sp += 1
        stats["links_tagged"] = stats.get("links_tagged", 0) + 1

    return next_sp


def tag_untagged_pdf(
    input_path: str | Path,
    output_path: str | Path,
    tagged_elements: list[TaggedElement],
    total_pages: int,
) -> dict:
    """
    V2: Build a complete struct tree for an untagged PDF.

    Creates /MarkInfo, /StructTreeRoot, Document element,
    StructElem for each tagged element, and /Tabs /S on pages.
    Also injects BDC/EMC operators into the content streams.
    """
    from tagger.stage10_writeback.content_stream_writer import (
        artifact_wrap_forms,
        inject_bdc_markers,
    )

    stats = {
        "total_elements_written": 0,
        "pages_modified": 0,
        "struct_tree_created": False,
        "links_tagged": 0,
    }

    try:
        pdf = pikepdf.open(str(input_path))
        root = pdf.Root

        # MCIDs are assigned page-locally by inject_bdc_markers (the single MCID
        # authority); the struct tree below is built from the maps it returns.

        # 1. Set MarkInfo
        root["/MarkInfo"] = pdf.make_indirect(
            Dictionary({"/Marked": True})
        )

        # 2. Create StructTreeRoot (make indirect so it can be referenced)
        struct_tree_root = pdf.make_indirect(
            Dictionary({"/Type": Name.StructTreeRoot})
        )

        # 3. Create Document element
        doc_elem = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Document"),
            "/P": struct_tree_root,
            "/K": Array([]),
        }))

        # Group elements by page
        by_page: dict[int, list[TaggedElement]] = {}
        for el in tagged_elements:
            by_page.setdefault(el.page_num, []).append(el)

        # 4. Build struct elements for each page
        parent_map_entries: list[tuple[int, object]] = []
        # Single monotonic counter feeding both page /StructParents and annotation
        # /StructParent keys, so the two never collide in the ParentTree number tree.
        next_sp = 0

        for page_num in sorted(by_page.keys()):
            page_elements = by_page[page_num]
            page_idx = page_num - 1

            if page_idx >= len(pdf.pages):
                continue

            page = pdf.pages[page_idx]
            mb = page.mediabox
            page_height_pt = float(mb[3]) - float(mb[1])

            # (bbox, struct_elem) candidates for nesting Link annotations.
            page_struct_owners: list[tuple[tuple, object]] = []

            # Sort elements by reading order
            page_elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))

            # 4.1 Inject BDC markers — this allocates the page-local MCIDs.
            element_to_mcids, mcid_to_element, _mcid_tag = inject_bdc_markers(
                pdf, page, page_num, page_elements
            )

            # MCID -> the struct element that owns that marked content. The
            # page's ParentTree array is built from this at the end of the page
            # so it is indexed by MCID (correct by construction).
            mcid_to_structelem: dict[int, object] = {}

            # Track consecutive LI elements for grouping into L
            i = 0
            while i < len(page_elements):
                el = page_elements[i]

                # Skip artifacts
                if el.pdf_tag == PDFTag.ARTIFACT:
                    i += 1
                    continue

                # Check for list item run → wrap in L
                if el.pdf_tag == PDFTag.LI:
                    # Find all consecutive LI elements
                    li_run = [el]
                    j = i + 1
                    while j < len(page_elements) and page_elements[j].pdf_tag == PDFTag.LI:
                        li_run.append(page_elements[j])
                        j += 1

                    # Create L (list) container
                    li_struct_elems = Array([])
                    for li_el in li_run:
                        li_mcids = element_to_mcids.get(li_el.element_id, [])
                        if not li_mcids:
                            continue
                        li_struct, lbody = _build_list_item_struct(
                            pdf, li_el, doc_elem, page.obj,
                            li_mcids,
                        )
                        li_struct_elems.append(li_struct)
                        # Nest any overlapping Link under LBody (LI itself should
                        # hold only Lbl/LBody), keeping list structure valid.
                        page_struct_owners.append((li_el.bbox, lbody))
                        # The list item's marked content lives under LBody.
                        for m in li_mcids:
                            mcid_to_structelem[m] = lbody
                        stats["total_elements_written"] += 1

                    if li_struct_elems:
                        list_elem = pdf.make_indirect(Dictionary({
                            "/Type": Name.StructElem,
                            "/S": Name("/L"),
                            "/P": doc_elem,
                            "/K": li_struct_elems,
                        }))
                        # Re-parent each LI to the L container. _build_list_item_struct
                        # sets /P to doc_elem (the L doesn't exist yet); veraPDF reads
                        # /P for parentStandardType, so an unfixed /P fails clause 7.2-17.
                        for li_struct in li_struct_elems:
                            li_struct["/P"] = list_elem
                        doc_elem["/K"].append(list_elem)

                    i = j
                    continue

                # Check for TOC item run → wrap in TOC (mirrors LI → L).
                # PDF/UA clause 7.2-26: every TOCI must be a child of a TOC.
                if el.pdf_tag == PDFTag.TOCI:
                    toci_run = [el]
                    j = i + 1
                    while j < len(page_elements) and page_elements[j].pdf_tag == PDFTag.TOCI:
                        toci_run.append(page_elements[j])
                        j += 1

                    toc_elem = pdf.make_indirect(Dictionary({
                        "/Type": Name.StructElem,
                        "/S": Name("/TOC"),
                        "/P": doc_elem,
                        "/K": Array([]),
                    }))

                    for toci_el in toci_run:
                        toci_mcids = element_to_mcids.get(toci_el.element_id, [])
                        if not toci_mcids:
                            continue
                        toci_dict = {
                            "/Type": Name.StructElem,
                            "/S": Name("/TOCI"),
                            "/P": toc_elem,
                            "/K": _mcid_k(toci_mcids),
                            "/Pg": page.obj,
                        }
                        if toci_el.text:
                            toci_dict["/ActualText"] = String(toci_el.text)
                        toci_struct = pdf.make_indirect(Dictionary(toci_dict))
                        toc_elem["/K"].append(toci_struct)
                        page_struct_owners.append((toci_el.bbox, toci_struct))
                        for m in toci_mcids:
                            mcid_to_structelem[m] = toci_struct
                        stats["total_elements_written"] += 1

                    if len(toc_elem["/K"]) > 0:
                        doc_elem["/K"].append(toc_elem)
                        stats["total_elements_written"] += 1

                    i = j
                    continue

                # TABLE nested structure
                if el.pdf_tag == PDFTag.TABLE and el.specialist_data.get("cells"):
                    # Construct nested TR/TH/TD
                    table_struct_elem = pdf.make_indirect(Dictionary({
                        "/Type": Name.StructElem,
                        "/S": Name("/Table"),
                        "/P": doc_elem,
                        "/K": Array([]),
                    }))
                    doc_elem["/K"].append(table_struct_elem)
                    # Table/TR own no marked content, so they are not entered in
                    # the MCID-indexed ParentTree array — only their TH/TD cells.

                    # Group cells by row
                    rows = {}
                    for cell in el.specialist_data["cells"]:
                        rows.setdefault(cell["row_idx"], []).append(cell)

                    for row_idx in sorted(rows.keys()):
                        tr_elem = pdf.make_indirect(Dictionary({
                            "/Type": Name.StructElem,
                            "/S": Name("/TR"),
                            "/P": table_struct_elem,
                            "/K": Array([]),
                        }))
                        table_struct_elem["/K"].append(tr_elem)

                        for cell in sorted(rows[row_idx], key=lambda c: c["col_idx"]):
                            cell_id = f"{el.element_id}_cell_{cell['row_idx']}_{cell['col_idx']}"
                            cell_mcids = element_to_mcids.get(cell_id, [])

                            is_empty = not cell.get("merged_from")
                            if not is_empty and not cell_mcids:
                                continue # Skip non-empty cells that failed BDC injection

                            # Dynamically determine row header status based on text content
                            cell_text = cell.get("text", "")
                            
                            if not cell_text and cell.get("merged_from"):
                                # pdfplumber failed to extract text, but the cell has physical characters.
                                # Assume it is not numeric.
                                is_numeric = False
                            else:
                                is_numeric = _is_numeric_content(cell_text)
                                
                            is_dynamic_row_header = (
                                cell["col_idx"] == 0
                                and not cell.get("is_header")
                                and (bool(cell_text.strip()) or bool(cell.get("merged_from")))
                                and not is_numeric
                            )
                            is_row_header = cell.get("is_row_header") or is_dynamic_row_header

                            td_tag = "TH" if cell.get("is_header") or is_row_header else "TD"

                            td_elem_dict = {
                                "/Type": Name.StructElem,
                                "/S": Name(f"/{td_tag}"),
                                "/P": tr_elem,
                            }
                            
                            if not is_empty:
                                td_elem_dict["/K"] = _mcid_k(cell_mcids)
                                td_elem_dict["/Pg"] = page.obj

                            if td_tag == "TH":
                                if cell.get("is_header"):
                                    td_elem_dict["/A"] = Dictionary({"/O": Name("/Table"), "/Scope": Name("/Column")})
                                elif is_row_header:
                                    td_elem_dict["/A"] = Dictionary({"/O": Name("/Table"), "/Scope": Name("/Row")})

                            if cell.get("text"):
                                td_elem_dict["/ActualText"] = String(cell["text"])

                            td_elem = pdf.make_indirect(Dictionary(td_elem_dict))
                            tr_elem["/K"].append(td_elem)
                            for m in cell_mcids:
                                mcid_to_structelem[m] = td_elem
                            stats["total_elements_written"] += 1

                    stats["total_elements_written"] += 1
                    i += 1
                    continue

                # Regular element
                mcids = element_to_mcids.get(el.element_id, [])
                if not mcids:
                    i += 1
                    continue

                tag_name = el.pdf_tag.value
                if tag_name not in WRITEBACK.tag_role_map:
                    tag_name = "P"

                struct_elem_dict = {
                    "/Type": Name.StructElem,
                    "/S": Name(f"/{tag_name}"),
                    "/P": doc_elem,
                    "/K": _mcid_k(mcids),
                    "/Pg": page.obj,
                }

                if el.text:
                    struct_elem_dict["/ActualText"] = String(el.text)

                if el.pdf_tag == PDFTag.FIGURE and el.alt_text:
                    struct_elem_dict["/Alt"] = String(el.alt_text)

                struct_elem = pdf.make_indirect(Dictionary(struct_elem_dict))

                doc_elem["/K"].append(struct_elem)
                page_struct_owners.append((el.bbox, struct_elem))
                for m in mcids:
                    mcid_to_structelem[m] = struct_elem

                stats["total_elements_written"] += 1
                i += 1

            # Build the page's ParentTree array INDEXED BY MCID: position k holds
            # the struct element that owns page-local MCID k. Every allocated
            # MCID maps to an element that produced a struct element above, so
            # the range 0..n-1 is fully covered.
            n_mcids = len(mcid_to_element)
            missing = [k for k in range(n_mcids) if k not in mcid_to_structelem]
            if missing:
                logger.warning(
                    "Page %d: %d MCIDs without a struct element (%s); using OBJR-free gap.",
                    page_num, len(missing), missing[:10],
                )
            page_struct_parents = Array(
                [mcid_to_structelem.get(k, doc_elem) for k in range(n_mcids)]
            )

            # Set page's StructParents (page-level: array indexed by MCID)
            page_key = next_sp
            next_sp += 1
            page.obj["/StructParents"] = page_key
            parent_map_entries.append((page_key, page_struct_parents))

            # Tag Link annotations on this page (object-level StructParent keys).
            next_sp = _tag_link_annotations(
                pdf, page, doc_elem, page_struct_owners,
                page_height_pt, parent_map_entries, next_sp, stats,
            )

            # Set reading order
            page.obj["/Tabs"] = Name("/S")

            stats["pages_modified"] += 1

        # 5. Build ParentTree number tree
        nums = Array([])
        for idx, parents in parent_map_entries:
            nums.append(idx)
            nums.append(parents)

        parent_tree = pdf.make_indirect(Dictionary({
            "/Nums": nums,
        }))

        # 6. Finalize struct tree
        struct_tree_root["/K"] = doc_elem
        struct_tree_root["/ParentTree"] = parent_tree

        root["/StructTreeRoot"] = struct_tree_root
        stats["struct_tree_created"] = True

        # 7. Set language
        if "/Lang" not in root:
            root["/Lang"] = String("en-US")

        # 8. Artifact-wrap marks inside Form XObjects (their content streams have
        #    their own marked-content scope, untouched by page-level injection).
        artifact_wrap_forms(pdf)

        # Save
        pdf.save(str(output_path))
        pdf.close()

        logger.info(
            "Tagged PDF written: %d elements across %d pages",
            stats["total_elements_written"], stats["pages_modified"],
        )

    except Exception as e:
        logger.error("Struct tree writeback failed: %s", e)
        import traceback
        traceback.print_exc()
        import shutil
        shutil.copy2(str(input_path), str(output_path))

    return stats


def _build_list_item_struct(
    pdf,
    el: TaggedElement,
    parent,
    page_obj,
    mcids: list[int],
):
    """
    Build a StructElem for a list item with Lbl + LBody children.

    PDF/UA structure: LI > [ Lbl, LBody ]. The list item's marked content is
    referenced from LBody (/K = its page-local MCIDs). Returns (li_elem, lbody)
    so the caller can index the ParentTree array by those MCIDs to LBody.
    """
    label = el.specialist_data.get("list_label", "") if hasattr(el, "specialist_data") and el.specialist_data else ""
    body = el.specialist_data.get("list_body", el.text) if hasattr(el, "specialist_data") and el.specialist_data else el.text

    children = Array([])

    if label:
        lbl = pdf.make_indirect(Dictionary({
            "/Type": Name.StructElem,
            "/S": Name("/Lbl"),
            "/ActualText": String(label),
        }))
        children.append(lbl)

    lbody = pdf.make_indirect(Dictionary({
        "/Type": Name.StructElem,
        "/S": Name("/LBody"),
        "/K": _mcid_k(mcids),
        "/Pg": page_obj,
        "/ActualText": String(body or ""),
    }))
    children.append(lbody)

    li_elem = pdf.make_indirect(Dictionary({
        "/Type": Name.StructElem,
        "/S": Name("/LI"),
        "/P": parent,
        "/K": children if len(children) > 1 else children[0],
        "/Pg": page_obj,
        "/ActualText": String(el.text or ""),
    }))

    # Parent links for the children.
    lbody["/P"] = li_elem
    if label:
        children[0]["/P"] = li_elem

    return li_elem, lbody


def _walk_and_retag(
    node,
    mcid_to_tag: dict[int, tuple[str, int]],
    stats: dict,
) -> None:
    """Recursively walk the struct tree and update /S entries."""
    if not isinstance(node, Dictionary):
        return

    node_type = node.get("/Type")
    if node_type == Name.StructElem:
        stats["total_struct_elems"] += 1

        k_val = node.get("/K")
        mcid = _extract_mcid(k_val)

        if mcid is not None and mcid in mcid_to_tag:
            new_tag, page_num = mcid_to_tag[mcid]
            old_tag = str(node.get("/S", ""))

            if old_tag != f"/{new_tag}":
                node["/S"] = Name(f"/{new_tag}")
                stats["changed"] += 1
                logger.debug(
                    "MCID %d: %s → /%s (page %d)",
                    mcid, old_tag, new_tag, page_num,
                )
            stats["matched"] += 1
        else:
            stats["unmatched"] += 1

    # Recurse into /K children
    k_val = node.get("/K")
    if isinstance(k_val, Array):
        for child in k_val:
            if isinstance(child, Dictionary):
                _walk_and_retag(child, mcid_to_tag, stats)
    elif isinstance(k_val, Dictionary):
        _walk_and_retag(k_val, mcid_to_tag, stats)


def _extract_mcid(k_val) -> int | None:
    """Extract MCID from a struct element's /K value."""
    if isinstance(k_val, int):
        return int(k_val)

    if isinstance(k_val, Dictionary):
        mcid = k_val.get("/MCID")
        if mcid is not None:
            return int(mcid)

    if isinstance(k_val, Array):
        for item in k_val:
            result = _extract_mcid(item)
            if result is not None:
                return result

    return None
