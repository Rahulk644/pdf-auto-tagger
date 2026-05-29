"""
Core data types shared across all pipeline stages.

Every stage consumes and produces instances of these dataclasses.
This is the single source of truth for the data contract between stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Stage 0 output
# ---------------------------------------------------------------------------

class PageType(str, Enum):
    """Classification of a PDF page by content extraction method needed."""

    NATIVE = "native"
    """Text is directly extractable from the content stream."""

    SCANNED = "scanned"
    """Page is a raster image — requires OCR."""

    MIXED = "mixed"
    """Page has both extractable text and significant image content."""

    CORRUPT = "corrupt"
    """Page content is garbled, unreadable, or structurally broken."""


@dataclass
class PageClassification:
    """Result of Stage 0 page-level classification."""

    page_num: int
    """1-indexed page number."""

    page_type: PageType

    char_count: int
    """Number of extractable text characters found by pdfplumber."""

    image_coverage: float
    """Fraction of page area covered by image XObjects (0.0–1.0)."""

    unicode_validity: float
    """Fraction of extracted characters that are valid Unicode (0.0–1.0)."""

    char_density: float
    """Characters per square inch of page area."""

    confidence: float
    """How confident we are in this classification (0.0–1.0)."""

    page_width_pt: float
    """Page width in PDF points (1/72 inch)."""

    page_height_pt: float
    """Page height in PDF points (1/72 inch)."""


# ---------------------------------------------------------------------------
# Stage 1 + 2 output
# ---------------------------------------------------------------------------

@dataclass
class PageElement:
    """
    A single extractable element on a page.

    Output of Stage 1 (extraction) refined by Stage 2 (merging).
    Represents a word, line, or paragraph depending on merge stage.
    """

    element_id: str
    """Unique ID across the entire document (e.g., 'p3_e17')."""

    page_num: int
    """1-indexed page number."""

    text: str
    """Extracted text content."""

    bbox: tuple[float, float, float, float]
    """(x0, y0, x1, y1) in standardized 150-DPI coords, origin top-left."""

    font_name: str | None = None
    """Primary font name (e.g., 'TimesNewRomanPSMT')."""

    font_size: float | None = None
    """Font size in points as declared in the PDF."""

    font_weight: str | None = None
    """'bold' or 'normal' — inferred from font name or style flags."""

    font_color: str | None = None
    """Hex color string (e.g., '#000000')."""

    is_italic: bool = False
    """Whether the text is italic."""

    upright: bool = True
    """False for rotated/vertical glyphs (pdfplumber `upright`). Stage 2 clusters
    rotated text separately so it never interleaves with horizontal lines."""

    source: Literal["pdfplumber", "mineru_ocr", "rapidocr"] = "pdfplumber"
    """Which extraction path produced this element. `rapidocr` is the CPU-native
    OCR path (PP-OCRv4 via onnxruntime) used by Stage 1 for scanned pages."""

    confidence: float = 1.0
    """Extraction confidence (0.0–1.0). OCR text is typically < 1.0."""

    mcid: int | None = None
    """Original MCID from the PDF struct tree, if the PDF was already tagged."""

    merged_from: list[str] = field(default_factory=list)
    """Element IDs that were merged to form this element (Stage 2)."""

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


# ---------------------------------------------------------------------------
# Stage 3 output
# ---------------------------------------------------------------------------

class LayoutCategory(str, Enum):
    """Categories output by the layout detection model."""

    TITLE = "Title"
    SECTION_HEADER = "Section-header"
    TEXT = "Text"
    LIST_ITEM = "List-item"
    TABLE = "Table"
    FORMULA = "Formula"
    PICTURE = "Picture"
    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    PAGE_HEADER = "Page-header"
    PAGE_FOOTER = "Page-footer"


@dataclass
class LayoutRegion:
    """
    A detected layout region on a page (output of Stage 3).

    Each region has a category, bounding box, reading order, and links
    to the PageElements it contains.
    """

    region_id: str
    """Unique ID across the document (e.g., 'r3_7')."""

    page_num: int
    """1-indexed page number."""

    bbox: tuple[float, float, float, float]
    """(x0, y0, x1, y1) in standardized 150-DPI coords."""

    category: LayoutCategory
    """What kind of content this region contains."""

    reading_order: int
    """Position in the reading sequence on this page (0-indexed)."""

    confidence: float
    """Model confidence for this detection (0.0–1.0)."""

    matched_elements: list[str] = field(default_factory=list)
    """element_ids from Stage 2 that fall within this region."""

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


# ---------------------------------------------------------------------------
# Stage 5 output — specialist extraction results
# ---------------------------------------------------------------------------

@dataclass
class TableStructure:
    """Extracted table structure with rows, headers, and cells."""

    region_id: str
    html: str
    """Full HTML representation: <table><tr><th>...</th></tr>...</table>"""

    num_rows: int
    num_cols: int
    has_header: bool
    confidence: float


@dataclass
class FormulaResult:
    """Extracted formula content."""

    region_id: str
    latex: str
    """LaTeX string representation of the formula."""

    is_inline: bool
    """Whether this is an inline formula (within text) or display formula."""

    confidence: float


@dataclass
class FigureInfo:
    """Information about an extracted figure."""

    region_id: str
    image_path: str | None
    """Path to the cropped figure image file."""

    alt_text: str | None = None
    """Generated alt text (filled in Stage 9)."""

    is_decorative: bool = False
    """If True, this is a decorative element → Artifact, not Figure."""

    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Final output — tagged elements
# ---------------------------------------------------------------------------

class PDFTag(str, Enum):
    """Standard PDF structure tags per PDF/UA."""

    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"
    H5 = "H5"
    H6 = "H6"
    P = "P"
    SPAN = "Span"
    L = "L"
    LI = "LI"
    LBL = "Lbl"
    LBODY = "LBody"
    TABLE = "Table"
    TR = "TR"
    TH = "TH"
    TD = "TD"
    THEAD = "THead"
    TBODY = "TBody"
    TFOOT = "TFoot"
    FIGURE = "Figure"
    FORMULA = "Formula"
    CAPTION = "Caption"
    BLOCKQUOTE = "BlockQuote"
    QUOTE = "Quote"
    NOTE = "Note"
    REFERENCE = "Reference"
    CODE = "Code"
    TOC = "TOC"
    TOCI = "TOCI"
    ARTIFACT = "Artifact"
    DOCUMENT = "Document"
    PART = "Part"
    SECT = "Sect"


@dataclass
class TaggedElement:
    """
    Final output of the pipeline — an element with its assigned PDF tag.

    Ready for struct tree writeback (Stage 10) and JSON report generation.
    """

    element_id: str
    page_num: int
    pdf_tag: PDFTag
    text: str
    bbox: tuple[float, float, float, float]

    alt_text: str | None = None
    """Alt text for figures — only populated for Figure elements."""

    confidence: float = 1.0
    """Overall tagging confidence (0.0–1.0)."""

    needs_review: bool = False
    """If True, this element should be reviewed by a human or the Gemma validator."""

    review_reason: str | None = None
    """Why this element was flagged for review."""

    # Provenance metadata
    original_mcid: int | None = None
    """MCID from the source PDF, if it was already tagged."""

    original_tag: str | None = None
    """Tag from the source PDF, if it was already tagged."""

    font_name: str | None = None
    font_size: float | None = None
    font_weight: str | None = None

    cross_page: bool = False
    """Whether this element spans a page boundary."""

    merged_from: list[str] = field(default_factory=list)
    """Element IDs that were merged to form this element."""

    layout_category: str | None = None
    """LayoutCategory that this element was classified as in Stage 3."""

    specialist_data: dict = field(default_factory=dict)
    """Extra data from specialist extraction (table HTML, LaTeX, etc.)."""


# ---------------------------------------------------------------------------
# Pipeline-level containers
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    """All data for a single page, accumulated through the pipeline."""

    page_num: int
    classification: PageClassification | None = None
    elements: list[PageElement] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    tagged_elements: list[TaggedElement] = field(default_factory=list)


@dataclass
class DocumentData:
    """All data for the entire document."""

    input_path: str
    num_pages: int
    pages: dict[int, PageData] = field(default_factory=dict)
    """page_num → PageData"""

    metadata: dict = field(default_factory=dict)
    """Document-level metadata (title, author, language, etc.)."""
