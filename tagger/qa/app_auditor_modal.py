import io
import os
import math
import json
import base64
import tempfile
import re
import shutil
import pickle
import traceback
import threading
import uuid
import time
import random
import concurrent.futures
from collections import deque
import modal

FITZ_CACHE = {}
PICKLE_CACHE = {}
AUDIT_RESULTS_STORE = {}
FITZ_CACHE_LOCK = threading.Lock()
RATE_LIMIT_RPM = 30


# Maximum elements per AI audit call.
# Math: 500 (prompt overhead) + 225 × N ≤ 16,500
# → N ≤ 71. Conservative ceiling at 60 to account
# for text length variance in snippets.
CHUNK_SIZE = 60

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pdfplumber
import fitz  # PyMuPDF
fitz.TOOLS.mupdf_display_errors(False)  # suppress "missing font descriptor" and similar MuPDF noise

from dotenv import load_dotenv
load_dotenv()

# Modal inference — references the deployed qa-gemma4-inference app by name.
# Deploy once with: modal deploy tagger/qa/modal_inference.py
_GemmaInference = modal.Cls.from_name("qa-gemma4-inference", "GemmaInference")

class _ModalResponse:
    """Thin wrapper so existing chunk_response.text / .usage_metadata calls work unchanged."""
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = None

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdftypes import resolve1

from rules_db import MASTER_RULES_DB

# ─────────────────────────────────────────────────────────────────────────────
# PDFMINER TAGGED AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class TaggedPDFPageAggregator(PDFPageAggregator):
    """Intercepts BDC operators to associate rendered chars/images with their MCID."""
    def __init__(self, rsrcmgr, pageno=1, laparams=None):
        super().__init__(rsrcmgr, pageno=pageno, laparams=laparams)
        self.mcid_stack = []
        self.mcid_bboxes = {}

    def begin_tag(self, tag, props=None):
        mcid = None
        if props and 'MCID' in props:
            val = props['MCID']
            if isinstance(val, int):
                mcid = val
        self.mcid_stack.append(mcid)

    def end_tag(self):
        if self.mcid_stack:
            self.mcid_stack.pop()

    def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate):
        mcid = self.mcid_stack[-1] if self.mcid_stack else None
        adv = super().render_char(matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate)
        if mcid is not None and hasattr(self, 'cur_item') and getattr(self.cur_item, '_objs', None):
            last_item = self.cur_item._objs[-1]
            if hasattr(last_item, 'bbox'):
                self.mcid_bboxes.setdefault(mcid, []).append(last_item.bbox)
        return adv

    def render_image(self, name, stream):
        mcid = self.mcid_stack[-1] if self.mcid_stack else None
        super().render_image(name, stream)
        if mcid is not None and hasattr(self, 'cur_item') and getattr(self.cur_item, '_objs', None):
            last_item = self.cur_item._objs[-1]
            if hasattr(last_item, 'bbox'):
                self.mcid_bboxes.setdefault(mcid, []).append(last_item.bbox)


class TaggedPDFPageInterpreter(PDFPageInterpreter):
    def do_BDC(self, tag, props):
        super().do_BDC(tag, props)
        if hasattr(self.device, 'begin_tag'):
            self.device.begin_tag(tag, props)

    def do_BMC(self, tag):
        super().do_BMC(tag)
        if hasattr(self.device, 'begin_tag'):
            self.device.begin_tag(tag, None)

    def do_EMC(self):
        super().do_EMC()
        if hasattr(self.device, 'end_tag'):
            self.device.end_tag()


# ─────────────────────────────────────────────────────────────────────────────
# STRUCT TREE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_struct_tree(doc, page_obj_to_idx):
    """Returns {page_idx: {mcid: tag_name}} from the StructTreeRoot."""
    catalog = resolve1(doc.catalog)
    if 'StructTreeRoot' not in catalog:
        return {}
    struct_tree_root = resolve1(catalog['StructTreeRoot'])
    page_idx_to_mcid_tags = {}

    def walk_tree(elem_ref, current_pg_id=None, depth=0):
        elem = resolve1(elem_ref)
        if not isinstance(elem, dict):
            return
        tag_name = None
        if 'S' in elem:
            tag = resolve1(elem['S'])
            tag_name = tag.name if hasattr(tag, 'name') else str(tag)
        if 'Pg' in elem:
            pg_ref = elem['Pg']
            if hasattr(pg_ref, 'objid'):
                current_pg_id = pg_ref.objid
        if 'K' in elem:
            kids = resolve1(elem['K'])
            if not isinstance(kids, list):
                kids = [kids]
            for kid_ref in kids:
                kid = resolve1(kid_ref)
                if isinstance(kid, int):
                    if current_pg_id and tag_name is not None:
                        page_idx = page_obj_to_idx.get(current_pg_id)
                        if page_idx is not None:
                            page_idx_to_mcid_tags.setdefault(page_idx, {})[kid] = {'tag': tag_name, 'depth': depth}
                elif isinstance(kid, dict):
                    kid_type = resolve1(kid.get('Type'))
                    kid_type_name = kid_type.name if hasattr(kid_type, 'name') else str(kid_type) if kid_type else None
                    if kid_type_name == 'MCR':
                        mcid = resolve1(kid['MCID'])
                        pg_ref = kid.get('Pg', elem.get('Pg'))
                        pg_id = current_pg_id
                        if hasattr(pg_ref, 'objid'):
                            pg_id = pg_ref.objid
                        if pg_id and tag_name is not None:
                            page_idx = page_obj_to_idx.get(pg_id)
                            if page_idx is not None:
                                page_idx_to_mcid_tags.setdefault(page_idx, {})[mcid] = {'tag': tag_name, 'depth': depth}
                    else:
                        walk_tree(kid_ref, current_pg_id, depth + 1)

    if 'K' in struct_tree_root:
        kids = resolve1(struct_tree_root['K'])
        if not isinstance(kids, list):
            kids = [kids]
        for kid in kids:
            walk_tree(kid, depth=0)
    return page_idx_to_mcid_tags

# ─────────────────────────────────────────────────────────────────────────────
# TAG COLOUR MAP
# ─────────────────────────────────────────────────────────────────────────────

def tag_color(tag_name):
    if tag_name.startswith('H'): return (0.86, 0.20, 0.18)
    elif tag_name == 'P': return (0.13, 0.69, 0.30)
    elif tag_name in ('Table', 'TR', 'TH', 'TD'): return (0.58, 0.20, 0.83)
    elif tag_name == 'Figure': return (0.95, 0.61, 0.07)
    elif tag_name in ('L', 'LI', 'Lbl', 'LBody'): return (0.07, 0.72, 0.90)
    else: return (0.20, 0.47, 0.95)

# ─────────────────────────────────────────────────────────────────────────────
# PDFPLUMBER EXTRACTION
# Note: Verified PyMuPDF `rawdict` does not contain MCID, so retaining this 
# approach for MCID->bbox mapping as it accurately parses the content stream.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# ADJACENT ELEMENT MERGER
#
# WHY THIS EXISTS:
# pdfplumber extracts content by iterating the PDF content stream
# character by character, grouping them into MCIDs based on BDC/EMC
# markers. A single logical element (one sentence, one heading) can
# be split into multiple MCIDs when:
#   1. A hyperlink annotation mid-sentence creates a new MCID boundary
#   2. A soft hyphen or line-continuation splits a word
#   3. Font or style changes within one logical run create new objects
#   4. The PDF authoring tool emits separate runs for adjacent content
#
# CONSEQUENCE OF NOT FIXING:
#   - Inflated element count in reports
#   - Split flagged elements appear as multiple findings for one problem
#   - Write-back will only correct one MCID of a multi-MCID element
#
# MERGE CRITERIA (ALL must be true to merge B into A):
#   1. Same tag
#   2. Tag not in NO_MERGE_TAGS
#   3. Same visual line (Y within Y_TOLERANCE)
#   4. Horizontally adjacent (gap within H_GAP_MAX, no deep overlap)
#
# KNOWN LIMITATIONS — TO BE ADDRESSED IN FUTURE ITERATIONS:
#
#   [LIMIT-1] RTL TEXT (Arabic, Hebrew):
#   The sort assumes left-to-right reading order. For RTL text,
#   elements flow right to left so the X-ascending sort reverses
#   reading order. The x_gap calculation becomes negative for valid
#   RTL adjacent elements, hitting H_OVERLAP_MAX and refusing to merge.
#   FIX WHEN NEEDED: detect RTL via /Lang entry on the element and
#   reverse X sort direction for those elements.
#
#   [LIMIT-2] ROTATED OR VERTICAL TEXT:
#   Rotated text has different bbox geometry — visually adjacent
#   elements may have large X or Y distances in coordinate space.
#   Y_TOLERANCE and H_GAP_MAX checks will reject valid merges.
#   FIX WHEN NEEDED: detect rotation matrix from the content stream
#   and apply rotated adjacency checks.
#
#   [LIMIT-3] MULTI-COLUMN LAYOUTS WITH CLOSE COLUMNS:
#   If two columns are less than H_GAP_MAX points apart, the last
#   element of column 1 and first element of column 2 may incorrectly
#   merge if they share the same Y and tag. H_GAP_MAX is set to 5
#   (conservative) to reduce this risk, but it cannot be eliminated
#   without full page-level column boundary detection.
#   FIX WHEN NEEDED: implement column boundary detection using the
#   distribution of X coordinates across all elements on the page,
#   then refuse merges that cross a detected column boundary.
#
#   [LIMIT-4] SUPERSCRIPT AND SUBSCRIPT REFERENCES:
#   Footnote reference markers (e.g. superscript ¹) have a Y offset
#   from the baseline. Depending on font size, this may exceed
#   Y_TOLERANCE and keep the reference as a separate element.
#   This is a cosmetic issue only — the reference stays readable,
#   just as a separate Span entry.
#   FIX WHEN NEEDED: detect vertical shift relative to baseline
#   using font metrics and apply a larger Y_TOLERANCE for Span tags.
#
#   [LIMIT-5] LBODY SPLIT AROUND LINK ANNOTATION:
#   Pattern: LBody "start" → Link "click here" → LBody "end"
#   The loop breaks on the Link tag mismatch, so the two LBody
#   fragments do not merge. This is actually correct — merging
#   across a Link would destroy the annotation structure. But the
#   report will still show two LBody entries for one visual sentence.
#   FIX WHEN NEEDED: use the struct tree parent relationships to
#   group elements sharing the same LI parent, then present them
#   as one logical unit in the report layer without merging MCIDs.
#
#   [LIMIT-6] PERFORMANCE ON VERY DENSE PAGES:
#   The look-ahead is O(n²) in the number of elements per page.
#   For normal documents (20-60 elements after ghost filtering)
#   this is negligible. For very dense forms (150+ elements),
#   processing time may be noticeable.
#   FIX WHEN NEEDED: add early termination when no merge has
#   occurred in the last N iterations, or switch to a spatial
#   index (interval tree) for adjacency lookup.
#
#   [LIMIT-7] WRITE-BACK MCID LOSS:
#   When two MCIDs merge, the second MCID is discarded from the
#   report. The write-back feature will only correct the first MCID
#   in the original PDF StructTreeRoot, leaving the second unchanged.
#   FIX BEFORE WRITE-BACK: replace skip.add(mcid_b) with:
#     current.setdefault('merged_mcids', []).append(mcid_b)
#   and pass merged_mcids through the corrections payload so the
#   write-back endpoint can update all MCIDs in the source PDF.
# ─────────────────────────────────────────────────────────────────

# Tags that legitimately appear multiple times on the same visual
# line and must NEVER be merged with each other:
#   TD  — table data cells sit side by side in the same row
#   TH  — table header cells sit side by side in the same row
#   TR  — row containers, guarded defensively
#   Lbl — list bullet markers are intentionally separate from LBody
NO_MERGE_TAGS = {'TD', 'TH', 'TR', 'Lbl'}

# Maximum vertical distance in PDF points between two elements'
# Y coordinates to be considered on the same visual line.
# 3 points handles minor baseline shifts without merging
# elements from adjacent lines (typical line spacing: 12-14pt).
Y_TOLERANCE = 3

# Maximum horizontal gap in PDF points between right edge of A
# and left edge of B to be considered adjacent.
# Set to 5 (conservative) to reduce risk of merging across
# close multi-column boundaries. See [LIMIT-3].
H_GAP_MAX = 5

# Maximum leftward overlap allowed between elements.
# Small negative values occur due to floating point bbox
# imprecision in pdfplumber and are acceptable.
# Values more negative than this indicate genuine structural
# overlap (e.g. overlapping columns) — do not merge.
H_OVERLAP_MAX = -5


def merge_adjacent_elements(page_elements):
    """
    Merges MCIDs that represent the same logical element but were
    split by the PDF content stream into multiple adjacent entries.

    Args:
        page_elements: dict of {mcid_str: element_dict} for one page.
                       Each element_dict has: mcid, tag, bbox, text.

    Returns:
        dict of {mcid_str: element_dict} with adjacent same-tag
        elements on the same line merged into single entries.
    """
    if not page_elements:
        return page_elements

    # Sort into visual reading order:
    #   Primary:   Y descending — higher on page processed first
    #   Secondary: X ascending  — left to right within a line
    # Y is rounded to 1 decimal to prevent floating point ordering
    # instability between elements on the same visual line.
    items = sorted(
        page_elements.items(),
        key=lambda kv: (-round(kv[1]['bbox'][1], 1), kv[1]['bbox'][0])
    )

    merged = {}
    skip = set()  # MCIDs already absorbed into a previous element

    for i, (mcid_a, el_a) in enumerate(items):

        if mcid_a in skip:
            continue

        # Build mutable accumulator from element A
        current = {
            'mcid': el_a['mcid'],
            'tag':  el_a['tag'],
            'bbox': list(el_a['bbox']),
            'text': el_a['text'],
        }
        # Preserve any extra fields (depth, current_tag, etc.)
        for k, v in el_a.items():
            if k not in current:
                current[k] = v
        current.setdefault('merged_mcids', [])

        # GUARD: never merge elements whose own tag is in NO_MERGE_TAGS.
        # Same-tag siblings on the same line are intentionally separate
        # for these structural container types.
        if el_a['tag'] in NO_MERGE_TAGS:
            merged[mcid_a] = current
            continue

        # Look ahead at subsequent elements for merge candidates
        for j in range(i + 1, len(items)):
            mcid_b, el_b = items[j]

            if mcid_b in skip:
                continue

            # RULE 1: Tags must match exactly.
            # Merging across tag types destroys semantic structure.
            # Break because tag changes are structural — no further
            # candidates on this line will match.
            if el_b['tag'] != current['tag']:
                break

            # RULE 2: Candidate tag must not be in NO_MERGE_TAGS.
            if el_b['tag'] in NO_MERGE_TAGS:
                break

            # RULE 3: Must be on the same visual line.
            y_diff = abs(el_b['bbox'][1] - current['bbox'][1])
            if y_diff > Y_TOLERANCE:
                # All remaining items are further away vertically
                # (list is sorted by Y descending) — stop scanning.
                break

            # RULE 4: Must be horizontally adjacent.
            # x_gap: distance from right edge of current to left of B
            x_gap = el_b['bbox'][0] - current['bbox'][2]

            if x_gap > H_GAP_MAX:
                # Too far apart — not adjacent.
                # Do NOT break: another element further right could
                # still be within range if content stream is reordered.
                continue

            if x_gap < H_OVERLAP_MAX:
                # Significant overlap — structural, not a text split.
                continue

            # ── All rules passed — merge B into current ──────────────

            # Separator: no space for tight runs (gap < 0.1pt),
            # single space for visible gaps between text runs.
            sep = '' if x_gap < 0.1 else ' '

            # Text combination: guard against appending [Empty]
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
            # See [LIMIT-7]: mcid_b is lost here. Before implementing
            # write-back, change this to:
            #   current.setdefault('merged_mcids', []).append(mcid_b)

        merged[mcid_a] = current

    return merged

def extract_pdfplumber_data(pdf_bytes):
    """Returns actual_elements_by_page using pdfplumber."""
    actual_elements_by_page = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                mcid_groups = {}
                page_num = page.page_number
                actual_elements_by_page[page_num] = {}

                for element in page.chars + page.images:
                    tag = element.get('tag')
                    mcid = element.get('mcid')
                    if tag and mcid is not None:
                        key = str(mcid)
                        if key not in mcid_groups:
                            mcid_groups[key] = {
                                'tag': tag,
                                'x0': [], 'top': [], 'x1': [], 'bottom': [],
                                'chars': []
                            }
                        mcid_groups[key]['x0'].append(element['x0'])
                        mcid_groups[key]['top'].append(element['top'])
                        mcid_groups[key]['x1'].append(element['x1'])
                        mcid_groups[key]['bottom'].append(element['bottom'])
                        if 'text' in element:
                            mcid_groups[key]['chars'].append(element['text'])

                for key, data in mcid_groups.items():
                    min_x0   = min(data['x0'])
                    max_x1   = max(data['x1'])
                    min_top  = min(data['top'])
                    max_bot  = max(data['bottom'])
                    norm_y0  = page.height - max_bot
                    norm_y1  = page.height - min_top
                    bbox     = [min_x0, norm_y0, max_x1, norm_y1]
                    tag_clean = data['tag'].replace('/', '')
                    text      = ''.join(data['chars']).strip() or ('[Image]' if tag_clean == 'Figure' else '[Empty]')

                    actual_elements_by_page[page_num][key] = {
                        'mcid': key, 'tag': tag_clean, 'bbox': bbox, 'text': text
                    }

                actual_elements_by_page[page_num] = {
                    key: val for key, val in actual_elements_by_page[page_num].items()
                    if not (
                        val['text'] == '[Empty]' and
                        (val['bbox'][2] - val['bbox'][0]) < 6
                    )
                }

                # Merge adjacent split elements into single logical units.
                # Must run AFTER ghost filtering — ghost elements must be
                # removed first or they interfere with adjacency detection
                # by creating false same-tag neighbours.
                actual_elements_by_page[page_num] = merge_adjacent_elements(
                    actual_elements_by_page[page_num]
                )
    except Exception:
        pass
    return actual_elements_by_page

def extract_structural_tags(pdf_bytes):
    """Returns page_idx_to_mcid_tags, page_idx_to_mcid_bboxes for the entire PDF."""
    stream = io.BytesIO(pdf_bytes)
    try:
        parser = PDFParser(stream)
        doc    = PDFDocument(parser)

        pages = list(PDFPage.get_pages(stream))
        page_obj_to_idx = {p.pageid: i for i, p in enumerate(pages)}

        page_idx_to_mcid_tags  = parse_struct_tree(doc, page_obj_to_idx)
        page_idx_to_mcid_bboxes = {}

        rsrcmgr = PDFResourceManager()
        for idx, page in enumerate(pages):
            device = TaggedPDFPageAggregator(rsrcmgr, laparams=LAParams())
            interpreter = TaggedPDFPageInterpreter(rsrcmgr, device)
            interpreter.process_page(page)
            page_idx_to_mcid_bboxes[idx] = device.mcid_bboxes

        return page_idx_to_mcid_tags, page_idx_to_mcid_bboxes
    except Exception:
        return {}, {}

def draw_tags_on_page(fitz_page, mcid_tags, mcid_bboxes):
    try:
        page_h = fitz_page.rect.height
        for mcid, tag_info in mcid_tags.items():
            tag_name = tag_info['tag'] if isinstance(tag_info, dict) else tag_info
            if mcid in mcid_bboxes and mcid_bboxes[mcid]:
                bboxes = mcid_bboxes[mcid]
                x0 = min(b[0] for b in bboxes if len(b) == 4)
                y0_pm = min(b[1] for b in bboxes if len(b) == 4)
                x1 = max(b[2] for b in bboxes if len(b) == 4)
                y1_pm = max(b[3] for b in bboxes if len(b) == 4)

                fitz_y0 = page_h - y1_pm
                fitz_y1 = page_h - y0_pm

                color = tag_color(tag_name)
                rect  = fitz.Rect(x0, fitz_y0, x1, fitz_y1)
                fitz_page.draw_rect(rect, color=color, width=1.5, fill=color, fill_opacity=0.15)

                label = f' <{tag_name}> '
                point = fitz.Point(x0, max(fitz_y0 - 2, 8))
                text_rect = fitz.Rect(
                    x0, max(fitz_y0 - 12, 0),
                    x0 + len(label) * 5, max(fitz_y0, 12)
                )
                fitz_page.draw_rect(text_rect, color=color, fill=color, fill_opacity=1.0)
                fitz_page.insert_text(point, label, fontsize=8, color=(1, 1, 1), fontname='helv')
    except Exception as e:
        print(f"Error drawing tags on page: {e}")
    return fitz_page

def inject_dynamic_rules(elements_list):
    active_rules = set()
    for element in elements_list:
        tag = str(element.get('current_tag') or element.get('tag', ''))
        raw_text = str(element.get('text', ''))

        for category, data in MASTER_RULES_DB.items():
            if any(trigger in tag for trigger in data["triggers"]):
                active_rules.add(data["rule"])
            if category == "MATH_AND_FORMULAS":
                math_chars = ['=', '+', '-', '≤', '≥', '∫', '∑']
                found_math_chars = sum(1 for char in raw_text if char in math_chars)
                if found_math_chars >= 2:
                    active_rules.add(data["rule"])

    return "\n\n".join(list(active_rules)) if active_rules else "No special complex structure rules apply."

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:3001"]}})
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB

@app.route('/upload_file', methods=['POST'])
def upload_file_endpoint():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        session_uuid = uuid.uuid4().hex
        session_dir = os.path.join(tempfile.gettempdir(), f"openloader_auditor_{session_uuid}")
        os.makedirs(session_dir, exist_ok=True)
        
        pdf_path = os.path.join(session_dir, 'doc.pdf')
        pdf_file.save(pdf_path)

        with open(pdf_path, 'rb') as f: pdf_bytes = f.read()

        actual_by_page = extract_pdfplumber_data(pdf_bytes)
        tags, bboxes = extract_structural_tags(pdf_bytes)

        cache_data = {
            'actual_by_page': actual_by_page,
            'tags': tags,
            'bboxes': bboxes,
        }
        
        with open(os.path.join(session_dir, 'cache.pkl'), 'wb') as f:
            pickle.dump(cache_data, f)

        fitz_doc = fitz.open(pdf_path)
        total_pages = len(fitz_doc)
        fitz_doc.close()
        
        total_chunks = 0
        total_elements = 0
        chunks_per_page = {}
        for page_num, elements_map in actual_by_page.items():
            elements_count = len(elements_map)
            total_elements += elements_count
            c = math.ceil(elements_count / CHUNK_SIZE) if elements_count > 0 else 1
            total_chunks += c
            chunks_per_page[page_num] = c

        return jsonify({
            'status': 'success',
            'session_id': session_uuid,
            'total_pages': total_pages,
            'total_chunks': total_chunks,
            'total_elements': total_elements,
            'chunks_per_page': chunks_per_page
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_page_data', methods=['GET'])
def get_page_data():
    session_id = request.args.get('session_id')
    page_num = int(request.args.get('page')) # 1-indexed
    
    if not session_id:
        return jsonify({'error': 'No session ID'}), 400
        

    session_dir = os.path.join(tempfile.gettempdir(), f"openloader_auditor_{session_id}")
    if not os.path.exists(session_dir):
        return jsonify({'error': 'Session expired or not found'}), 400

    try:
        pdf_path = os.path.join(session_dir, 'doc.pdf')
        
        with FITZ_CACHE_LOCK:
            if session_id not in PICKLE_CACHE:
                with open(os.path.join(session_dir, 'cache.pkl'), 'rb') as f:
                    PICKLE_CACHE[session_id] = pickle.load(f)
            cache = PICKLE_CACHE[session_id]

        page_idx = page_num - 1

        elements_map = cache['actual_by_page'].get(page_num, {})
        elements_list = list(elements_map.values())
        
        struct_tags = cache['tags'].get(page_idx, {})
        for el in elements_list:
            mcid_val = el['mcid']
            try:
                mcid_int = int(mcid_val)
            except:
                mcid_int = mcid_val
            tag_info = struct_tags.get(mcid_int, {})
            el['depth'] = tag_info.get('depth', 0) if isinstance(tag_info, dict) else 0
            if isinstance(tag_info, dict) and 'tag' in tag_info:
                el['current_tag'] = tag_info['tag']
                el['tag'] = tag_info['tag']
            else:
                el['current_tag'] = el.get('tag', 'Unknown')
                
            if 'bbox' in el and len(el['bbox']) == 4:
                el['bbox'] = [
                    round(el['bbox'][0]),
                    round(el['bbox'][1]),
                    round(el['bbox'][2]),
                    round(el['bbox'][3])
                ]

        injected_rules = inject_dynamic_rules(elements_list)

        with FITZ_CACHE_LOCK:
            if session_id not in FITZ_CACHE:
                FITZ_CACHE[session_id] = fitz.open(pdf_path)
            fitz_doc = FITZ_CACHE[session_id]
        
        page = fitz_doc[page_idx]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Clean image
        pix_clean = page.get_pixmap(dpi=150)
        clean_b64 = base64.b64encode(pix_clean.tobytes('png')).decode('utf-8')

        def render_annotated_page(source_doc, page_idx, mcid_tags, mcid_bboxes):
            tmp = fitz.open()
            tmp.insert_pdf(source_doc, from_page=page_idx, to_page=page_idx)
            draw_tags_on_page(tmp[0], mcid_tags, mcid_bboxes)
            pix = tmp[0].get_pixmap(dpi=150)
            result = base64.b64encode(pix.tobytes('png')).decode('utf-8')
            tmp.close()
            return result

        # Annotated image
        annotated_b64 = render_annotated_page(
            fitz_doc, page_idx,
            cache['tags'].get(page_idx, {}),
            cache['bboxes'].get(page_idx, {})
        )

        payload = {
            'page': page_num,
            'page_width': page_width,
            'page_height': page_height,
            'clean_image_base64': clean_b64,
            'annotated_image_base64': annotated_b64,
            'elements_list': elements_list,
            'injected_rules': injected_rules
        }
        return jsonify(payload), 200
    except Exception as e:
        print(f"Error in get_page_data for page {page_num}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    data = request.json
    session_id = data.get('session_id')
    if session_id:
        session_dir = os.path.join(tempfile.gettempdir(), f"openloader_auditor_{session_id}")
        if os.path.exists(session_dir):
            try:
                with FITZ_CACHE_LOCK:
                    if session_id in FITZ_CACHE:
                        try:
                            FITZ_CACHE[session_id].close()
                        except Exception as close_err:
                            pass
                        del FITZ_CACHE[session_id]
                    if session_id in PICKLE_CACHE:
                        del PICKLE_CACHE[session_id]
                    if session_id in AUDIT_RESULTS_STORE:
                        del AUDIT_RESULTS_STORE[session_id]
                
                audit_path = os.path.join(session_dir, 'audit_results.pkl')
                if os.path.exists(audit_path):
                    os.remove(audit_path)
                
                shutil.rmtree(session_dir)
                return jsonify({'status': 'success'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'status': 'ignored'}), 200

@app.route('/audit_page', methods=['POST'])
def audit_page():
    temp_img_path = None
    data = request.json
    filename = data.get('filename', 'Unknown')
    page_number = data.get('page_number')
    clean_image_base64 = data.get('clean_image_base64')
    injected_rules = data.get('injected_rules')
    elements_list = data.get('elements_list')

    try:
        audit_start_time = time.time()
        print(f"[{filename} | AUDIT START] Page {page_number} | {len(elements_list)} elements")
        
        if not clean_image_base64:
            raise ValueError("clean_image_base64 is missing")
            
        img_data = base64.b64decode(clean_image_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_img.write(img_data)
            temp_img_path = temp_img.name

        clean_elements = []
        for i, el in enumerate(elements_list):
            clean_elements.append({
                'element_index': i,
                'mcid': el.get('mcid', ''),
                'tag': el.get('current_tag') or el.get('tag', ''),
                'bbox': el.get('bbox', []),
                'text': el.get('text', '')
            })

        # ── Chunk if needed ────────────────────────────
        if len(clean_elements) <= CHUNK_SIZE:
            chunks = [clean_elements]
        else:
            chunks = [
                clean_elements[i:i + CHUNK_SIZE]
                for i in range(0, len(clean_elements),
                               CHUNK_SIZE)
            ]
            print(f"[{filename} | CHUNKED] Page {page_number}: {len(chunks)} chunks for {len(clean_elements)} elements")

        # ── Proactive Stagger to Protect Upload API ────────
        time.sleep(random.uniform(0, 2.5))
        
        # ── Encode image ONCE before chunk loop ────────
        with open(temp_img_path, "rb") as _f:
            uploaded_file = base64.b64encode(_f.read()).decode()

        all_json_data = []
        total_tokens_in  = 0
        total_tokens_out = 0
        total_tokens_total = 0
        total_elapsed = 0.0
        max_attempts_used = 1

        def _process_chunk_task(chunk_idx, chunk):
            chunk_prompt = f"""You are an expert accessibility
judge strictly enforcing PDF/UA and WCAG standards.

### PAYLOAD PARTITIONING NOTICE
This page has been partitioned into {len(chunks)}
chunk(s) to prevent token window saturation.
You are processing chunk {chunk_idx + 1} of
{len(chunks)}, containing {len(chunk)} elements.

Process this chunk as a self-contained context.
Do not assume or invent elements outside this payload.
The page image is provided for full visual context —
use it for spatial reasoning even though not all
elements are in this chunk's JSON array.

### HARD CONSTRAINTS
- You MUST return exactly {len(chunk)} objects
- Use element_index EXACTLY as provided in input
  (these are global page indices, not chunk-local)
- Do not reorder, skip, or merge elements

### PDF/UA AND WCAG RULES FOR THIS PAGE
{injected_rules}

### ELEMENTS — Chunk {chunk_idx + 1} of {len(chunks)}
{json.dumps(chunk)}

Return a valid JSON array ONLY.
No markdown fences. No preamble. No explanation.

[{{
  "element_index": <exact integer from input>,
  "mcid": "<mcid from input>",
  "current_tag": "<tag from input>",
  "is_correct": true or false,
  "wcag_rule_violated": "<specific rule text or null>",
  "suggested_tag": "<correct tag, or same as current_tag if correct>",
  "corrective_reasoning": "<one sentence explanation>"
}}]"""

            chunk_start = time.time()
            time.sleep(random.uniform(0, 3) + (chunk_idx * 0.5))
            chunk_attempt = 0
            chunk_response = None
            chunk_last_error = None
            max_retries = 12

            while chunk_attempt < max_retries:
                try:
                    chunk_response = _ModalResponse(
                        _GemmaInference().generate.remote(uploaded_file, chunk_prompt)
                    )
                    break
                except Exception as api_err:
                    chunk_attempt += 1
                    chunk_last_error = api_err
                    print(f"[{filename} | Page {page_number} | CHUNK {chunk_idx+1}/"
                          f"{len(chunks)}] Modal error "
                          f"attempt {chunk_attempt}: "
                          f"{api_err}")
                    if chunk_attempt < max_retries:
                        wait = random.uniform(2, min(45, 4 * (1.5 ** chunk_attempt)))
                        print(f"[{filename} | Page {page_number}] [RETRY] Waiting {wait:.2f}s...")
                        time.sleep(wait)

            if not chunk_response:
                try:
                    print(f"[{filename} | Page {page_number}] [FALLBACK] Chunk {chunk_idx+1} retrying via Modal")
                    chunk_response = _ModalResponse(
                        _GemmaInference().generate.remote(uploaded_file, chunk_prompt)
                    )
                except Exception as fb_err:
                    print(f"[{filename} | Page {page_number}] [FALLBACK ERROR] Chunk "
                          f"{chunk_idx+1}: {fb_err}")
                    raise Exception(json.dumps({
                        "page": page_number,
                        "status": "failed",
                        "error": "Modal inference failed"
                    }))

            local_tokens_in = 0
            local_tokens_out = 0
            local_tokens_total = 0

            local_elapsed = round(
                time.time() - chunk_start, 2)
            local_attempts = chunk_attempt + 1

            # Parse chunk response
            try:
                chunk_raw = (chunk_response.text
                             if chunk_response.text else "")

                chunk_raw = re.sub(
                    r'<think>.*?</think>', '',
                    chunk_raw, flags=re.DOTALL)

                # Strip markdown fences
                fence_match = re.search(
                    r'```json\s*(.*?)\s*```',
                    chunk_raw, flags=re.DOTALL)
                if fence_match:
                    chunk_raw = fence_match.group(1)
                else:
                    open_fence = re.search(
                        r'```json\s*', chunk_raw)
                    if open_fence:
                        chunk_raw = chunk_raw[
                            open_fence.end():]

                # Find array boundaries
                s = chunk_raw.find('[')
                e = chunk_raw.rfind(']')

                if s == -1:
                    raise ValueError(
                        f"No JSON array in chunk "
                        f"{chunk_idx+1} response")

                if e == -1 or e < s:
                    # Truncated — attempt partial recovery
                    print(f"[{filename} | Page {page_number}] [TRUNCATED] Chunk {chunk_idx+1}"
                          f": attempting partial recovery")
                    partial = chunk_raw[s:]
                    last = partial.rfind('},')
                    if last == -1:
                        last = partial.rfind('}')
                    if last != -1:
                        chunk_raw = partial[:last+1] + ']'
                        print(f"[{filename} | Page {page_number}] [TRUNCATED] Chunk "
                              f"{chunk_idx+1}: recovered "
                              f"partial array")
                    else:
                        raise ValueError(
                            f"Chunk {chunk_idx+1} truncated"
                            f" with no recoverable objects")
                else:
                    chunk_raw = chunk_raw[s:e+1]

                chunk_data = json.loads(chunk_raw)
                print(f"[{filename} | Page {page_number}] [CHUNK {chunk_idx+1}/{len(chunks)}]"
                      f" Parsed {len(chunk_data)}"
                      f"/{len(chunk)} elements")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[{filename} | Page {page_number}] [CHUNK PARSE ERROR] "
                      f"Chunk {chunk_idx+1}: {e}")
                raise Exception(json.dumps({
                    'status': 'error',
                    'message': f'Response parsing failed on page '
                               f'{page_number} Chunk {chunk_idx+1}. '
                               f'Error: {str(e)}',
                }))
            
            return {
                'data': chunk_data,
                'tokens_in': local_tokens_in,
                'tokens_out': local_tokens_out,
                'tokens_total': local_tokens_total,
                'elapsed': local_elapsed,
                'attempts': local_attempts
            }

        for chunk_idx, chunk in enumerate(chunks):
            try:
                res = _process_chunk_task(chunk_idx, chunk)
                all_json_data.extend(res['data'])
                total_tokens_in += res['tokens_in']
                total_tokens_out += res['tokens_out']
                total_tokens_total += res['tokens_total']
                total_elapsed += res['elapsed']
                max_attempts_used = max(max_attempts_used, res['attempts'])
            except Exception as e:
                try:
                    err_payload = json.loads(str(e))
                except:
                    raise e
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                return jsonify(err_payload), 422 if err_payload.get('status') in ('error', 'failed') else 206
        # ── Delete image ONCE after all chunks ─────────
        # Do NOT delete inside the loop
        try:
            pass  # no remote file to delete
        except Exception:
            pass
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        # ── Assign combined results ─────────────────────
        json_data = all_json_data
        token_count = total_tokens_out
        elapsed_seconds = round(time.time() - audit_start_time, 2)
        tokens_input = total_tokens_in
        tokens_total = total_tokens_total

        # Sanitize: enforce is_correct consistency
        for item in json_data:
            if not isinstance(item, dict):
                continue
            idx = item.get('element_index', -1)
            if not (0 <= idx < len(elements_list)):
                continue
            current = elements_list[idx].get('current_tag', '')
            if item.get('is_correct') == True:
                item['suggested_tag'] = current
                item['wcag_rule_violated'] = None
            elif item.get('is_correct') == False:
                if item.get('suggested_tag', '').strip() == current.strip():
                    item['is_correct'] = True
                    item['wcag_rule_violated'] = None

        elements_flagged = sum(1 for el in json_data if isinstance(el, dict) and el.get('is_correct') == False)
        elements_correct = sum(1 for el in json_data if isinstance(el, dict) and el.get('is_correct') == True)
        
        chunk_msg = f" | Chunks: {len(chunks)}" if len(chunks) > 1 else ""
        print(f"[{filename} | AUDIT DONE] Page {page_number} | "
              f"{elapsed_seconds}s | "
              f"Tokens in:{tokens_input} out:{token_count} | "
              f"Flagged: {elements_flagged}/{len(elements_list)}{chunk_msg}")
              
        return jsonify({
            'status': 'success', 
            'data': json_data,
            'metadata': {
                'tokens_output': token_count,
                'tokens_input': tokens_input,
                'tokens_total': tokens_total,
                'time_seconds': elapsed_seconds,
                'model': 'gemma-4-31b-it',
                'page_number': page_number,
                'element_count': len(elements_list),
                'elements_flagged': elements_flagged,
                'elements_correct': elements_correct,
                'attempts_used': max_attempts_used,
                'prompt_preview': "Chunked processing — prompt not captured."
            }
        }), 200

    except Exception as e:
        if 'uploaded_file' in locals():
            try:
                pass  # no remote file to delete
            except Exception:
                pass
        if temp_img_path and os.path.exists(temp_img_path): os.remove(temp_img_path)
        return jsonify({'status': 'error', 'message': str(e), 'prompt_preview': "Chunked processing — prompt not captured."}), 500

def _audit_single_page(task):
    """
    Runs the full audit pipeline for one page.
    Called by /audit_page_batch in parallel threads.
    
    Args:
        task: dict with keys:
          page_number, clean_image_base64,
          injected_rules, elements_list, filename
    
    Returns:
        dict with keys:
          page_number, status, data, metadata, error
    """
    import time
    import base64
    import tempfile
    import os
    import json
    import re

    page_number = task['page_number']
    clean_image_base64 = task.get('clean_image_base64')
    injected_rules = task.get('injected_rules', '')
    elements_list = task.get('elements_list', [])
    filename = task.get('filename', 'Unknown')
    doc_id = task.get('docId')
    temp_img_path = None

    try:
        if not clean_image_base64:
            raise ValueError("clean_image_base64 missing")

        img_data = base64.b64decode(clean_image_base64)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".png"
        ) as tmp:
            tmp.write(img_data)
            temp_img_path = tmp.name

        rules_text = f"Apply these PDF/UA and WCAG rules relevant to elements on this page:\n\n{injected_rules}" if injected_rules and injected_rules.strip() else ""

        clean_elements = []
        for i, el in enumerate(elements_list):
            clean_elements.append({
                'element_index': i,
                'mcid': el.get('mcid', ''),
                'tag': el.get('current_tag') or el.get('tag', ''),
                'bbox': el.get('bbox', []),
                'text': el.get('text', '')
            })

        # ── Chunk if needed ────────────────────────────
        if len(clean_elements) <= CHUNK_SIZE:
            chunks = [clean_elements]
        else:
            chunks = [
                clean_elements[i:i + CHUNK_SIZE]
                for i in range(0, len(clean_elements),
                               CHUNK_SIZE)
            ]
            print(f"[CHUNKED] Batch Page {page_number}: "
                  f"{len(chunks)} chunks for "
                  f"{len(clean_elements)} elements")

        # ── Proactive Stagger to Protect Upload API ────────
        time.sleep(random.uniform(0, 2.5))
        
        # ── Encode image ONCE before chunk loop ────────
        with open(temp_img_path, "rb") as _f:
            uploaded_file = base64.b64encode(_f.read()).decode()

        all_json_data = []
        total_tokens_in  = 0
        total_tokens_out = 0
        total_tokens_total = 0
        total_elapsed = 0.0
        max_attempts_used = 1

        def _process_chunk_task(chunk_idx, chunk):
            chunk_prompt = f"""You are a PDF accessibility expert enforcing PDF/UA and WCAG 2.2 standards.

{rules_text}

### PAYLOAD PARTITIONING NOTICE
This page has been partitioned into {len(chunks)} chunk(s) to prevent token window saturation.
You are processing chunk {chunk_idx + 1} of {len(chunks)}, containing {len(chunk)} elements.

Process this chunk as a self-contained context. Do not assume or invent elements outside this payload.
The page image is provided for full visual context — use it for spatial reasoning even though not all elements are in this chunk's JSON array.

### HARD CONSTRAINTS
- You MUST return exactly {len(chunk)} objects
- Use element_index EXACTLY as provided in input (these are global page indices, not chunk-local)
- Do not reorder, skip, or merge elements

### ELEMENTS — Chunk {chunk_idx + 1} of {len(chunks)}
{json.dumps(chunk)}

CRITICAL: Return a JSON array with EXACTLY {len(chunk)} objects, one per element, in the same order. Do NOT skip, summarize, or stop early.

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

            chunk_start = time.time()
            time.sleep(random.uniform(0, 3) + (chunk_idx * 0.5))
            chunk_attempt = 0
            chunk_response = None
            chunk_last_error = None
            max_retries = 12

            while chunk_attempt < max_retries:
                try:
                    chunk_response = _ModalResponse(
                        _GemmaInference().generate.remote(uploaded_file, chunk_prompt)
                    )
                    break
                except Exception as api_err:
                    chunk_attempt += 1
                    chunk_last_error = api_err
                    print(f"[BATCH CHUNK {chunk_idx+1}/"
                          f"{len(chunks)}] Modal error "
                          f"attempt {chunk_attempt}: "
                          f"{api_err}")
                    if chunk_attempt < max_retries:
                        wait = random.uniform(2, min(45, 4 * (1.5 ** chunk_attempt)))
                        print(f"[BATCH CHUNK {chunk_idx+1}] [RETRY] Waiting {wait:.2f}s...")
                        time.sleep(wait)

            if not chunk_response:
                try:
                    print(f"[FALLBACK] Chunk {chunk_idx+1} retrying via Modal")
                    chunk_response = _ModalResponse(
                        _GemmaInference().generate.remote(uploaded_file, chunk_prompt)
                    )
                except Exception as fb_err:
                    print(f"[FALLBACK ERROR] Chunk "
                          f"{chunk_idx+1}: {fb_err}")
                    raise Exception(
                        f"Chunk {chunk_idx+1}/{len(chunks)}"
                        f" failed after all retries and "
                        f"fallback: {chunk_last_error}"
                    )

            local_tokens_in = 0
            local_tokens_out = 0
            local_tokens_total = 0

            local_elapsed = round(
                time.time() - chunk_start, 2)
            local_attempts = chunk_attempt + 1

            # Parse chunk response
            try:
                chunk_raw = (chunk_response.text
                             if chunk_response.text else "")

                chunk_raw = re.sub(
                    r'<think>.*?</think>', '',
                    chunk_raw, flags=re.DOTALL)

                # Strip markdown fences
                fence_match = re.search(
                    r'```json\s*(.*?)\s*```',
                    chunk_raw, flags=re.DOTALL)
                if fence_match:
                    chunk_raw = fence_match.group(1)
                else:
                    open_fence = re.search(
                        r'```json\s*', chunk_raw)
                    if open_fence:
                        chunk_raw = chunk_raw[
                            open_fence.end():]

                # Find array boundaries
                s = chunk_raw.find('[')
                e = chunk_raw.rfind(']')

                if s == -1:
                    raise ValueError(
                        f"No JSON array in chunk "
                        f"{chunk_idx+1} response")

                if e == -1 or e < s:
                    # Truncated — attempt partial recovery
                    print(f"[TRUNCATED] Chunk {chunk_idx+1}"
                          f": attempting partial recovery")
                    partial = chunk_raw[s:]
                    last = partial.rfind('},')
                    if last == -1:
                        last = partial.rfind('}')
                    if last != -1:
                        chunk_raw = partial[:last+1] + ']'
                        print(f"[TRUNCATED] Chunk "
                              f"{chunk_idx+1}: recovered "
                              f"partial array")
                    else:
                        raise ValueError(
                            f"Chunk {chunk_idx+1} truncated"
                            f" with no recoverable objects")
                else:
                    chunk_raw = chunk_raw[s:e+1]

                chunk_data = json.loads(chunk_raw)
                print(f"[CHUNK {chunk_idx+1}/{len(chunks)}]"
                      f" Parsed {len(chunk_data)}"
                      f"/{len(chunk)} elements")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[CHUNK PARSE ERROR] "
                      f"Chunk {chunk_idx+1}: {e}")
                raise Exception(
                    f"Chunk {chunk_idx+1}/{len(chunks)} "
                    f"parse failed: {e}"
                )
            
            return {
                'data': chunk_data,
                'tokens_in': local_tokens_in,
                'tokens_out': local_tokens_out,
                'tokens_total': local_tokens_total,
                'elapsed': local_elapsed,
                'attempts': local_attempts
            }

        for chunk_idx, chunk in enumerate(chunks):
            res = _process_chunk_task(chunk_idx, chunk)
            all_json_data.extend(res['data'])
            total_tokens_in += res['tokens_in']
            total_tokens_out += res['tokens_out']
            total_tokens_total += res['tokens_total']
            total_elapsed += res['elapsed']
            max_attempts_used = max(max_attempts_used, res['attempts'])
        # ── Delete image ONCE after all chunks ─────────
        try:
            pass  # no remote file to delete
        except Exception:
            pass
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        json_data = all_json_data
        token_count = total_tokens_out
        elapsed = total_elapsed
        tokens_input = total_tokens_in
        tokens_total = total_tokens_total

        # Sanitize: enforce is_correct consistency
        for item in json_data:
            if not isinstance(item, dict):
                continue
            idx = item.get('element_index', -1)
            if not (0 <= idx < len(elements_list)):
                continue
            current = elements_list[idx].get('current_tag','')
            if item.get('is_correct') == True:
                item['suggested_tag'] = current
                item['wcag_rule_violated'] = None
            elif item.get('is_correct') == False:
                if item.get('suggested_tag','').strip() \
                        == current.strip():
                    item['is_correct'] = True
                    item['wcag_rule_violated'] = None

        elements_flagged = sum(
            1 for el in json_data
            if isinstance(el, dict)
            and el.get('is_correct') == False
        )
        elements_correct = sum(
            1 for el in json_data
            if isinstance(el, dict)
            and el.get('is_correct') == True
        )

        print(f"[BATCH DONE] Page {page_number} | "
              f"{elapsed}s | "
              f"Tokens in:{tokens_input} out:{token_count} | "
              f"Flagged:{elements_flagged}/{len(elements_list)}")

        return {
            'page_number': page_number,
            'status': 'success',
            'data': json_data,
            'docId': doc_id,
            'metadata': {
                'tokens_output': token_count,
                'tokens_input': tokens_input,
                'tokens_total': tokens_total,
                'time_seconds': elapsed,
                'model': 'gemma-4-31b-it',
                'page_number': page_number,
                'element_count': len(elements_list),
                'elements_flagged': elements_flagged,
                'elements_correct': elements_correct,
                'attempts_used': max_attempts_used
            }
        }

    except Exception as e:
        if 'uploaded_file' in locals() and uploaded_file:
            try:
                pass  # no remote file to delete
            except Exception:
                pass
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        print(f"[BATCH ERROR] Page {page_number}: {e}")
        return {
            'page_number': page_number,
            'status': 'error',
            'error': str(e),
            'data': [],
            'docId': doc_id,
            'metadata': {}
        }

@app.route('/audit_page_batch', methods=['POST'])
def audit_page_batch():
    """
    Audits multiple pages in parallel using ThreadPoolExecutor.
    Respects the 15 RPM rate limit by batching into groups
    of RATE_LIMIT_RPM with a 60 second wait between batches.
    
    Request body:
      {
        "filename": "doc.pdf",
        "pages": [
          {
            "page_number": 1,
            "clean_image_base64": "...",
            "injected_rules": "...",
            "elements_list": [...]
          },
          ...
        ]
      }
    
    Response:
      {
        "status": "success",
        "results": [
          { "page_number": 1, "status": "success",
            "data": [...], "metadata": {...} },
          ...
        ],
        "total_time_seconds": 123.4
      }
    """
    import time
    data = request.json
    filename = data.get('filename', 'Unknown')
    pages = data.get('pages', [])

    if not pages:
        return jsonify({'error': 'No pages provided'}), 400

    # Inject filename into each task, preserving per-page filename when set
    for p in pages:
        p['filename'] = p.get('filename') or filename

    print(f"[BATCH START] {filename} | "
          f"{len(pages)} pages | "
          f"Max parallel: {RATE_LIMIT_RPM}")

    batch_start = time.time()
    all_results = []

    # Split pages into RPM-sized batches
    # Each batch fires in parallel, then waits 60s
    # before the next batch if more pages remain
    page_batches = [
        pages[i:i + RATE_LIMIT_RPM]
        for i in range(0, len(pages), RATE_LIMIT_RPM)
    ]

    for batch_idx, batch in enumerate(page_batches):
        if batch_idx > 0:
            # Wait for rate limit window to reset
            print(f"[BATCH] Rate limit pause 60s "
                  f"before batch {batch_idx + 1} "
                  f"of {len(page_batches)}...")
            time.sleep(60)

        print(f"[BATCH] Firing batch {batch_idx + 1} "
              f"({len(batch)} pages in parallel)...")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch)
        ) as executor:
            future_to_page = {
                executor.submit(_audit_single_page, task): task
                for task in batch
            }
            for future in concurrent.futures.as_completed(
                future_to_page
            ):
                result = future.result()
                all_results.append(result)

    # Sort results by page_number for consistent ordering
    all_results.sort(key=lambda r: r['page_number'])

    total_time = round(time.time() - batch_start, 2)
    print(f"[BATCH COMPLETE] {filename} | "
          f"{len(pages)} pages | "
          f"{total_time}s total")

    return jsonify({
        'status': 'success',
        'results': all_results,
        'total_time_seconds': total_time
    }), 200

@app.route('/audit_page_stream', methods=['POST'])
def audit_page_stream():
    data = request.json
    pages = data.get('pages', [])
    filename = data.get('filename', 'Unknown')
    for p in pages:
        p['filename'] = p.get('filename') or filename

    def generate():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(pages), 100)
        ) as page_executor:
            futures = {
                page_executor.submit(_audit_single_page, task): task
                for task in pages
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    task = futures[future]
                    result = {
                        'page_number': task.get('page_number'),
                        'docId': task.get('docId'),
                        'filename': task.get('filename'),
                        'status': 'error',
                        'error': str(e),
                    }
                yield f"data: {json.dumps(result)}\n\n"
        yield "data: {\"done\": true}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )

@app.route('/store_audit_results', methods=['POST'])
def store_audit_results():
    body = request.json
    session_id = body.get('session_id')
    page_number = body.get('page_number')
    if not session_id or not page_number:
        return jsonify({'error': 'Missing fields'}), 400
    
    if session_id not in AUDIT_RESULTS_STORE:
        AUDIT_RESULTS_STORE[session_id] = {}
    
    AUDIT_RESULTS_STORE[session_id][page_number] = {
        'data': body.get('data', []),
        'metadata': body.get('metadata', {})
    }

    # Write to disk
    session_dir = os.path.join(tempfile.gettempdir(), f"openloader_auditor_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    audit_path = os.path.join(session_dir, 'audit_results.pkl')
    with open(audit_path, 'wb') as f:
        pickle.dump(AUDIT_RESULTS_STORE[session_id], f)

    return jsonify({'status': 'stored'}), 200

@app.route('/get_audit_results', methods=['GET'])
def get_audit_results():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    
    if session_id not in AUDIT_RESULTS_STORE:
        session_dir = os.path.join(tempfile.gettempdir(), f"openloader_auditor_{session_id}")
        audit_path = os.path.join(session_dir, 'audit_results.pkl')
        if os.path.exists(audit_path):
            with open(audit_path, 'rb') as f:
                AUDIT_RESULTS_STORE[session_id] = pickle.load(f)
        else:
            return jsonify({'status': 'success', 'results': {}}), 200
            
    results = AUDIT_RESULTS_STORE.get(session_id, {})
    return jsonify({
        'status': 'success',
        'results': results
    }), 200

if __name__ == '__main__':
    app.run(port=5001, debug=True, threaded=True)
