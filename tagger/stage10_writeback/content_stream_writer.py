import logging
import pikepdf

logger = logging.getLogger(__name__)

def inject_bdc_markers(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    page_num: int,
    tagged_elements: list,
    element_mcid_map: dict[str, int]
) -> set[int]:
    """
    Inject BDC and EMC markers into the page content stream for the tagged elements.
    Matches text operators to elements using absolute character positions.
    Returns the set of MCIDs that were successfully injected.
    """
    injected_mcids = set()

    # 1. Map absolute character indices to MCIDs based on TaggedElement merged_from
    char_to_mcid = {}
    mcid_to_tag = {}
    for el in tagged_elements:
        if el.page_num != page_num:
            continue
            
        if el.pdf_tag.value == "Table" and el.specialist_data.get("cells"):
            for cell in el.specialist_data["cells"]:
                if not cell.get("merged_from"):
                    continue
                cell_id = f"{el.element_id}_cell_{cell['row_idx']}_{cell['col_idx']}"
                if cell_id not in element_mcid_map:
                    continue
                
                mcid = element_mcid_map[cell_id]
                td_tag = "TH" if cell.get("is_header") or cell.get("is_row_header") else "TD"
                mcid_to_tag[mcid] = td_tag
                
                for source_id in cell["merged_from"]:
                    if source_id.startswith(f"p{page_num}_c"):
                        try:
                            char_idx = int(source_id.split("_c")[1])
                            char_to_mcid[char_idx] = mcid
                        except ValueError:
                            pass
        else:
            if el.element_id not in element_mcid_map:
                continue
                
            mcid = element_mcid_map[el.element_id]
            mcid_to_tag[mcid] = el.pdf_tag.value
            
            for source_id in el.merged_from:
                # Source IDs are like "p1_c5"
                if source_id.startswith(f"p{page_num}_c"):
                    try:
                        char_idx = int(source_id.split("_c")[1])
                        char_to_mcid[char_idx] = mcid
                    except ValueError:
                        pass

    if not char_to_mcid:
        logger.debug(f"Page {page_num}: No characters mapped to MCIDs. Skipping BDC injection.")
        return injected_mcids

    logger.debug("Page %d: char_to_mcid size=%d, unique MCIDs=%s", page_num, len(char_to_mcid), set(char_to_mcid.values()))

    # 2. Parse the content stream
    cs = pikepdf.parse_content_stream(page)
    new_cs = []
    
    current_char_idx = 0
    active_mcid = None
    
    # 3. Iterate and inject
    for instruction in cs:
        op = str(instruction.operator)
        
        if op in ("Tj", "TJ", "\"", "\x27"):
            # Calculate how many characters this operator draws
            op_text_len = 0
            if op == "Tj" or op == "\x27" or op == "\"":
                # Single string operand
                op_text_len = len(str(instruction.operands[0]))
            elif op == "TJ":
                # Array of strings and numbers
                for item in instruction.operands[0]:
                    if isinstance(item, pikepdf.String):
                        op_text_len += len(str(item))
                        
            # Determine which MCID this operator belongs to
            # We check the first character of the operator that maps to an MCID
            target_mcid = None
            for i in range(current_char_idx, current_char_idx + op_text_len):
                if i in char_to_mcid:
                    target_mcid = char_to_mcid[i]
                    break
                    
            if target_mcid != active_mcid:
                # We need to switch or close the active MCID
                if active_mcid is not None:
                    # Close previous
                    new_cs.append(pikepdf.ContentStreamInstruction(pikepdf._core._ObjectList([]), pikepdf.Operator("EMC")))
                
                if target_mcid is not None:
                    # Open new
                    injected_mcids.add(target_mcid)
                    tag_name = pikepdf.Name(f"/{mcid_to_tag[target_mcid]}")
                    props = pikepdf.Dictionary({"/MCID": target_mcid})
                    new_cs.append(pikepdf.ContentStreamInstruction(
                        pikepdf._core._ObjectList([tag_name, props]), 
                        pikepdf.Operator("BDC")
                    ))
                
                active_mcid = target_mcid

            # Add the text operator itself
            new_cs.append(instruction)
            
            # Advance character index
            current_char_idx += op_text_len
            
        else:
            # Non-text operator
            
            # If we exit a text block, we must close any active BDC to respect PDF spec
            if op == "ET" and active_mcid is not None:
                new_cs.append(pikepdf.ContentStreamInstruction(pikepdf._core._ObjectList([]), pikepdf.Operator("EMC")))
                active_mcid = None
                
            new_cs.append(instruction)

    # Close any lingering BDC at the end of the stream
    if active_mcid is not None:
        new_cs.append(pikepdf.ContentStreamInstruction(pikepdf._core._ObjectList([]), pikepdf.Operator("EMC")))

    # 4. Write back the new content stream
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_cs))
    logger.debug(f"Page {page_num}: Injected BDC markers for {len(injected_mcids)} MCIDs.")
    return injected_mcids

