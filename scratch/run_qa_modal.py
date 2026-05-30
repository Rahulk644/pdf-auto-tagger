"""
QA runner: all extraction/rendering runs locally, only Gemma inference goes to Modal.

For each PDF:
  1. Locally: pdfplumber extraction, struct tree parsing, fitz page rendering
  2. All chunks across all pages are spawned in parallel to Modal
  3. Results collected and aggregated per page
  4. Saves QA report JSON
"""
import json
import sys
import io
import base64
import re
from pathlib import Path

import modal

app = modal.App("qa-runner")
_GemmaInference = modal.Cls.from_name("qa-gemma4-inference", "GemmaInference")

CHUNK_SIZE = 60

# ── Helpers ───────────────────────────────────────────────────────────────────

NO_MERGE_TAGS = {'TD', 'TH', 'TR', 'Lbl'}
Y_TOLERANCE = 3
H_GAP_MAX = 5
H_OVERLAP_MAX = -5


def _merge_adjacent_elements(page_elements):
    if not page_elements:
        return page_elements
    items = sorted(
        page_elements.items(),
        key=lambda kv: (-round(kv[1]['bbox'][1], 1), kv[1]['bbox'][0])
    )
    merged = {}
    skip = set()
    for i, (mcid_a, el_a) in enumerate(items):
        if mcid_a in skip:
            continue
        current = {**el_a, 'bbox': list(el_a['bbox'])}
        current.setdefault('merged_mcids', [])
        if el_a['tag'] in NO_MERGE_TAGS:
            merged[mcid_a] = current
            continue
        for j in range(i + 1, len(items)):
            mcid_b, el_b = items[j]
            if mcid_b in skip:
                continue
            if el_b['tag'] != current['tag'] or el_b['tag'] in NO_MERGE_TAGS:
                break
            if abs(el_b['bbox'][1] - current['bbox'][1]) > Y_TOLERANCE:
                break
            x_gap = el_b['bbox'][0] - current['bbox'][2]
            if x_gap > H_GAP_MAX:
                continue
            if x_gap < H_OVERLAP_MAX:
                continue
            sep = '' if x_gap < 0.1 else ' '
            if el_b['text'] not in ('[Empty]', ''):
                if current['text'] in ('[Empty]', ''):
                    current['text'] = el_b['text']
                else:
                    current['text'] = current['text'] + sep + el_b['text']
            current['bbox'] = [
                min(current['bbox'][0], el_b['bbox'][0]),
                min(current['bbox'][1], el_b['bbox'][1]),
                max(current['bbox'][2], el_b['bbox'][2]),
                max(current['bbox'][3], el_b['bbox'][3]),
            ]
            current.setdefault('merged_mcids', []).append(mcid_b)
            skip.add(mcid_b)
        merged[mcid_a] = current
    return merged


def _extract_pdfplumber_data(pdf_bytes):
    import pdfplumber
    actual_by_page = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                mcid_groups = {}
                page_num = page.page_number
                actual_by_page[page_num] = {}
                for element in page.chars + page.images:
                    tag = element.get('tag')
                    mcid = element.get('mcid')
                    if tag and mcid is not None:
                        key = str(mcid)
                        if key not in mcid_groups:
                            mcid_groups[key] = {'tag': tag, 'x0': [], 'top': [], 'x1': [], 'bottom': [], 'chars': []}
                        mcid_groups[key]['x0'].append(element['x0'])
                        mcid_groups[key]['top'].append(element['top'])
                        mcid_groups[key]['x1'].append(element['x1'])
                        mcid_groups[key]['bottom'].append(element['bottom'])
                        if 'text' in element:
                            mcid_groups[key]['chars'].append(element['text'])
                for key, data in mcid_groups.items():
                    min_x0 = min(data['x0'])
                    max_x1 = max(data['x1'])
                    min_top = min(data['top'])
                    max_bot = max(data['bottom'])
                    norm_y0 = page.height - max_bot
                    norm_y1 = page.height - min_top
                    bbox = [min_x0, norm_y0, max_x1, norm_y1]
                    tag_clean = data['tag'].replace('/', '')
                    text = ''.join(data['chars']).strip() or ('[Image]' if tag_clean == 'Figure' else '[Empty]')
                    actual_by_page[page_num][key] = {'mcid': key, 'tag': tag_clean, 'bbox': bbox, 'text': text}
                actual_by_page[page_num] = {
                    k: v for k, v in actual_by_page[page_num].items()
                    if not (v['text'] == '[Empty]' and (v['bbox'][2] - v['bbox'][0]) < 6)
                }
                actual_by_page[page_num] = _merge_adjacent_elements(actual_by_page[page_num])
    except Exception as e:
        print(f"pdfplumber extraction error: {e}")
    return actual_by_page


def _parse_struct_tree(doc, page_obj_to_idx):
    from pdfminer.pdftypes import resolve1
    catalog = resolve1(doc.catalog)
    if 'StructTreeRoot' not in catalog:
        return {}
    struct_tree_root = resolve1(catalog['StructTreeRoot'])
    result = {}

    def walk(elem_ref, pg_id=None, depth=0):
        elem = resolve1(elem_ref)
        if not isinstance(elem, dict):
            return
        tag_name = None
        if 'S' in elem:
            t = resolve1(elem['S'])
            tag_name = t.name if hasattr(t, 'name') else str(t)
        if 'Pg' in elem:
            ref = elem['Pg']
            if hasattr(ref, 'objid'):
                pg_id = ref.objid
        if 'K' in elem:
            kids = resolve1(elem['K'])
            if not isinstance(kids, list):
                kids = [kids]
            for kid_ref in kids:
                kid = resolve1(kid_ref)
                if isinstance(kid, int):
                    if pg_id and tag_name is not None:
                        idx = page_obj_to_idx.get(pg_id)
                        if idx is not None:
                            result.setdefault(idx, {})[kid] = {'tag': tag_name, 'depth': depth}
                elif isinstance(kid, dict):
                    kt = resolve1(kid.get('Type'))
                    kt_name = kt.name if hasattr(kt, 'name') else str(kt) if kt else None
                    if kt_name == 'MCR':
                        mcid = resolve1(kid['MCID'])
                        ref = kid.get('Pg', elem.get('Pg'))
                        kid_pg_id = pg_id
                        if hasattr(ref, 'objid'):
                            kid_pg_id = ref.objid
                        if kid_pg_id and tag_name is not None:
                            idx = page_obj_to_idx.get(kid_pg_id)
                            if idx is not None:
                                result.setdefault(idx, {})[mcid] = {'tag': tag_name, 'depth': depth}
                    else:
                        walk(kid_ref, pg_id, depth + 1)

    if 'K' in struct_tree_root:
        kids = resolve1(struct_tree_root['K'])
        if not isinstance(kids, list):
            kids = [kids]
        for kid in kids:
            walk(kid, depth=0)
    return result


def _extract_struct_tags(pdf_bytes):
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    try:
        stream = io.BytesIO(pdf_bytes)
        parser = PDFParser(stream)
        doc = PDFDocument(parser)
        pages = list(PDFPage.get_pages(io.BytesIO(pdf_bytes)))
        page_obj_to_idx = {p.pageid: i for i, p in enumerate(pages)}
        return _parse_struct_tree(doc, page_obj_to_idx)
    except Exception as e:
        print(f"struct tag extraction error: {e}")
        return {}


def _inject_dynamic_rules(elements_list):
    qa_dir = Path(__file__).parent.parent / "tagger" / "qa"
    sys.path.insert(0, str(qa_dir))
    try:
        from rules_db import MASTER_RULES_DB
    except ImportError:
        return "No special complex structure rules apply."
    active_rules = set()
    for element in elements_list:
        tag = str(element.get('current_tag') or element.get('tag', ''))
        raw_text = str(element.get('text', ''))
        for category, data in MASTER_RULES_DB.items():
            if any(trigger in tag for trigger in data["triggers"]):
                active_rules.add(data["rule"])
            if category == "MATH_AND_FORMULAS":
                math_chars = ['=', '+', '-', '≤', '≥', '∫', '∑']
                if sum(1 for c in raw_text if c in math_chars) >= 2:
                    active_rules.add(data["rule"])
    return "\n\n".join(list(active_rules)) if active_rules else "No special complex structure rules apply."


def _parse_response(response_text, chunk, filename, page_num, chunk_idx, n_chunks):
    try:
        raw = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        fence = re.search(r'```json\s*(.*?)\s*```', raw, flags=re.DOTALL)
        if fence:
            raw = fence.group(1)
        else:
            of = re.search(r'```json\s*', raw)
            if of:
                raw = raw[of.end():]
        s = raw.find('[')
        e_idx = raw.rfind(']')
        if s == -1:
            raise ValueError("No JSON array in response")
        if e_idx == -1 or e_idx < s:
            partial = raw[s:]
            last = partial.rfind('},')
            if last == -1:
                last = partial.rfind('}')
            if last != -1:
                raw = partial[:last + 1] + ']'
            else:
                raise ValueError("Truncated with no recoverable objects")
        else:
            raw = raw[s:e_idx + 1]
        chunk_data = json.loads(raw)
        print(f"[{filename} | Page {page_num} | Chunk {chunk_idx+1}/{n_chunks}] Parsed {len(chunk_data)}/{len(chunk)}", flush=True)
        return chunk_data
    except Exception as ex:
        print(f"[{filename} | Page {page_num} | Chunk {chunk_idx+1}] Parse error: {ex}", flush=True)
        return []


# ── Main audit function (runs locally) ───────────────────────────────────────

def audit_pdf(pdf_bytes: bytes, filename: str) -> dict:
    import fitz
    fitz.TOOLS.mupdf_display_errors(False)

    print(f"[{filename}] Extracting elements...", flush=True)
    actual_by_page = _extract_pdfplumber_data(pdf_bytes)
    struct_tags = _extract_struct_tags(pdf_bytes)

    fitz_doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    total_pages = len(fitz_doc)

    # ── Build all chunks across all pages ─────────────────────────────────────
    page_elements = {}   # page_num -> elements_list
    spawn_items = []     # (page_num, chunk_idx, n_chunks, chunk, img_b64, prompt)

    for page_num in range(1, total_pages + 1):
        page_idx = page_num - 1
        elements_map = actual_by_page.get(page_num, {})
        elements_list = list(elements_map.values())

        struct_page = struct_tags.get(page_idx, {})
        for el in elements_list:
            try:
                mcid_int = int(el['mcid'])
            except Exception:
                mcid_int = el['mcid']
            tag_info = struct_page.get(mcid_int, {})
            el['depth'] = tag_info.get('depth', 0) if isinstance(tag_info, dict) else 0
            if isinstance(tag_info, dict) and 'tag' in tag_info:
                el['current_tag'] = tag_info['tag']
                el['tag'] = tag_info['tag']
            else:
                el['current_tag'] = el.get('tag', 'Unknown')

        page_elements[page_num] = elements_list
        injected_rules = _inject_dynamic_rules(elements_list)

        fitz_page = fitz_doc[page_idx]
        pix = fitz_page.get_pixmap(dpi=150)
        img_b64 = base64.b64encode(pix.tobytes('png')).decode()

        clean_elements = [
            {
                'element_index': i,
                'mcid': el.get('mcid', ''),
                'tag': el.get('current_tag') or el.get('tag', ''),
                'bbox': el.get('bbox', []),
                'text': el.get('text', '')
            }
            for i, el in enumerate(elements_list)
        ]

        chunks = (
            [clean_elements]
            if len(clean_elements) <= CHUNK_SIZE
            else [clean_elements[i:i + CHUNK_SIZE] for i in range(0, len(clean_elements), CHUNK_SIZE)]
        )

        rules_text = (
            f"Apply these PDF/UA and WCAG rules:\n\n{injected_rules}"
            if injected_rules and injected_rules.strip() != "No special complex structure rules apply."
            else ""
        )

        for chunk_idx, chunk in enumerate(chunks):
            prompt = f"""You are a PDF accessibility expert enforcing PDF/UA and WCAG 2.2 standards.

{rules_text}

### PAYLOAD PARTITIONING NOTICE
This page has been partitioned into {len(chunks)} chunk(s).
You are processing chunk {chunk_idx + 1} of {len(chunks)}, containing {len(chunk)} elements.

### HARD CONSTRAINTS
- Return exactly {len(chunk)} objects
- Use element_index EXACTLY as provided
- Do not reorder, skip, or merge elements

### ELEMENTS — Chunk {chunk_idx + 1} of {len(chunks)}
{json.dumps(chunk)}

CRITICAL: Return a JSON array with EXACTLY {len(chunk)} objects.

Schema:
[
  {{
    "element_index": <exact integer from input>,
    "mcid": "the mcid value",
    "current_tag": "the tag as provided",
    "is_correct": true or false,
    "wcag_rule_violated": "specific rule violated or null",
    "suggested_tag": "correct tag or same as current if correct",
    "corrective_reasoning": "brief explanation"
  }}
]"""
            spawn_items.append((page_num, chunk_idx, len(chunks), chunk, img_b64, prompt))

    total_chunks = len(spawn_items)
    print(f"[{filename}] {total_pages} pages, {total_chunks} chunks — spawning all in parallel...", flush=True)

    # ── Spawn all chunks simultaneously ───────────────────────────────────────
    spawned = []
    for page_num, chunk_idx, n_chunks, chunk, img_b64, prompt in spawn_items:
        handle = _GemmaInference().generate.spawn(img_b64, prompt)
        spawned.append((page_num, chunk_idx, n_chunks, chunk, img_b64, prompt, handle))
    print(f"[{filename}] All {total_chunks} chunks spawned.", flush=True)

    # ── Collect results (in spawn order) ──────────────────────────────────────
    results_by_page = {pn: [] for pn in range(1, total_pages + 1)}

    for page_num, chunk_idx, n_chunks, chunk, img_b64, prompt, handle in spawned:
        response_text = None
        try:
            response_text = handle.get(timeout=1800)
        except Exception as e:
            print(f"[{filename} | Page {page_num} | Chunk {chunk_idx+1}] Error: {e} — retrying...", flush=True)
            try:
                response_text = _GemmaInference().generate.spawn(img_b64, prompt).get(timeout=1800)
            except Exception as e2:
                print(f"[{filename} | Page {page_num} | Chunk {chunk_idx+1}] Final failure: {e2}", flush=True)

        if response_text:
            results_by_page[page_num].extend(
                _parse_response(response_text, chunk, filename, page_num, chunk_idx, n_chunks)
            )

    # ── Aggregate per-page stats ───────────────────────────────────────────────
    all_page_results = []
    total_elements = 0
    total_correct = 0

    for page_num in range(1, total_pages + 1):
        elements_list = page_elements[page_num]
        all_json_data = results_by_page[page_num]

        for item in all_json_data:
            if not isinstance(item, dict):
                continue
            idx = item.get('element_index', -1)
            if not (0 <= idx < len(elements_list)):
                continue
            current = elements_list[idx].get('current_tag', '')
            if item.get('is_correct') is True:
                item['suggested_tag'] = current
                item['wcag_rule_violated'] = None
            elif item.get('is_correct') is False:
                if item.get('suggested_tag', '').strip() == current.strip():
                    item['is_correct'] = True
                    item['wcag_rule_violated'] = None

        correct = sum(1 for el in all_json_data if isinstance(el, dict) and el.get('is_correct') is True)
        flagged = sum(1 for el in all_json_data if isinstance(el, dict) and el.get('is_correct') is False)
        total_elements += len(elements_list)
        total_correct += correct
        print(f"[{filename} | Page {page_num}] {len(elements_list)} elements | {correct} correct | {flagged} flagged", flush=True)

        all_page_results.append({
            'page_number': page_num,
            'status': 'success',
            'data': all_json_data,
            'metadata': {
                'element_count': len(elements_list),
                'elements_correct': correct,
                'elements_flagged': flagged,
            }
        })

    accuracy = total_correct / total_elements if total_elements > 0 else 0
    print(f"\n[{filename}] COMPLETE | {total_elements} elements | {total_correct} correct | {accuracy:.1%}", flush=True)

    return {
        'filename': filename,
        'status': 'success',
        'results': all_page_results,
        'total_elements': total_elements,
        'correct': total_correct,
        'accuracy': accuracy,
    }


@app.local_entrypoint()
def main():
    OUTPUT_DIR = Path("/Users/rahulkhatri/Tagger/output_modal")
    RESULTS_DIR = Path("/Users/rahulkhatri/QA Tool/scratch/qa_results_modal")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = [
        OUTPUT_DIR / "miramar_untagged.pdf",
        OUTPUT_DIR / "Summary of Revenues and Expenditures.pdf",
    ]

    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"Skipping {pdf_path.name}", flush=True)
            continue

        print(f"\nRunning QA on {pdf_path.name}...", flush=True)
        pdf_bytes = pdf_path.read_bytes()
        result = audit_pdf(pdf_bytes, pdf_path.name)

        out_file = RESULTS_DIR / f"qa_{pdf_path.stem}.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"Saved: {out_file}", flush=True)
        print(f"Elements: {result['total_elements']}  Accuracy: {result['accuracy']:.1%}", flush=True)
