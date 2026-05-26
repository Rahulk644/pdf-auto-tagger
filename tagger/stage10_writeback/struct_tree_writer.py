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

logger = logging.getLogger(__name__)


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
    """
    stats = {
        "total_elements_written": 0,
        "pages_modified": 0,
        "struct_tree_created": False,
    }

    try:
        pdf = pikepdf.open(str(input_path))
        root = pdf.Root

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
        parent_map_entries: list[tuple[int, Array]] = []
        mcid_counter = 0

        for page_num in sorted(by_page.keys()):
            page_elements = by_page[page_num]
            page_idx = page_num - 1

            if page_idx >= len(pdf.pages):
                continue

            page = pdf.pages[page_idx]

            # Sort elements by reading order
            page_elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))

            page_struct_parents = Array([])

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
                        li_struct = _build_list_item_struct(
                            pdf, li_el, doc_elem, page.obj,
                            mcid_counter,
                        )
                        li_struct_elems.append(li_struct)
                        page_struct_parents.append(li_struct)
                        mcid_counter += 1
                        stats["total_elements_written"] += 1

                    list_elem = pdf.make_indirect(Dictionary({
                        "/Type": Name.StructElem,
                        "/S": Name("/L"),
                        "/P": doc_elem,
                        "/K": li_struct_elems,
                    }))
                    doc_elem["/K"].append(list_elem)

                    i = j
                    continue

                # Regular element
                tag_name = el.pdf_tag.value
                if tag_name not in WRITEBACK.tag_role_map:
                    tag_name = "P"

                struct_elem_dict = {
                    "/Type": Name.StructElem,
                    "/S": Name(f"/{tag_name}"),
                    "/P": doc_elem,
                    "/K": mcid_counter,
                    "/Pg": page.obj,
                }

                if el.text:
                    struct_elem_dict["/ActualText"] = String(el.text)

                if el.pdf_tag == PDFTag.FIGURE and el.alt_text:
                    struct_elem_dict["/Alt"] = String(el.alt_text)

                struct_elem = pdf.make_indirect(Dictionary(struct_elem_dict))

                doc_elem["/K"].append(struct_elem)
                page_struct_parents.append(struct_elem)

                mcid_counter += 1
                stats["total_elements_written"] += 1
                i += 1

            # Set page's StructParents
            page.obj["/StructParents"] = len(parent_map_entries)
            parent_map_entries.append((len(parent_map_entries), page_struct_parents))

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
    mcid: int,
):
    """
    Build a StructElem for a list item with Lbl + LBody children.

    PDF/UA structure: LI > [ Lbl, LBody ]
    """
    label = el.specialist_data.get("list_label", "")
    body = el.specialist_data.get("list_body", el.text)

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

    return li_elem


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
