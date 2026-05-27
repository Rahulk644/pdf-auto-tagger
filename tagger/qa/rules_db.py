# rules_db.py
# Standards covered:
#   - PDF/UA-1 (ISO 14289-1:2012/2014)
#   - PDF/UA-2 (ISO 14289-2:2024) — current gold standard
#   - WCAG 2.2 Level AA (W3C, October 2023)
#   - Tagged PDF Best Practice Guide (PDF Association)

MASTER_RULES_DB = {

    "HEADINGS": {
        "triggers": ["H1", "H2", "H3", "H4", "H5", "H6", "H"],
        "rule": (
            "PDF/UA + WCAG 2.2 HEADING RULES:\n"
            "1. Heading levels must reflect true visual and semantic "
            "hierarchy. Do NOT skip levels (e.g. H1 → H3 is a "
            "violation; H1 → H2 → H3 is correct).\n"
            "2. A document should have exactly ONE <H1> representing "
            "the main document title or top-level heading. Multiple "
            "<H1> tags are a violation unless the document uses "
            "strongly-structured (unnumbered <H>) tagging.\n"
            "3. Heading tags (<H1>–<H6>) are reserved strictly for "
            "section/document headings. Form field labels, table "
            "captions, figure captions, and UI labels must NEVER use "
            "heading tags — use <P> instead.\n"
            "4. Heading structure elements must not be empty. An "
            "empty <H2> or similar is semantically inappropriate.\n"
            "5. No block-level structures (Table, L, Figure) should "
            "be nested directly inside a heading element.\n"
            "6. Subheadings and subtitles have no dedicated PDF "
            "structure type — use <P> or <Span> with visual styling "
            "instead.\n"
            "PDF/UA-2 addition: Heading nesting must be verifiable "
            "through the structure tree, not just visual appearance."
        )
    },

    "PARAGRAPHS_AND_INLINE": {
        "triggers": ["P", "Span", "Quote", "BlockQuote", "Note",
                     "Reference", "BibEntry", "Code", "Sub", "Sup"],
        "rule": (
            "PDF/UA + WCAG 2.2 PARAGRAPH AND INLINE RULES:\n"
            "1. <P> is the correct tag for all body text, form field "
            "labels, instructional text, helper text, placeholder "
            "descriptions, and any text that is not a heading, list "
            "item, table cell, or caption.\n"
            "2. <Span> is for inline text within a block element "
            "requiring differentiation (language change, emphasis). "
            "<Span> must not be used as a substitute for block-level "
            "tags like <P>.\n"
            "3. <Quote> is inline (inside <P>). <BlockQuote> is "
            "block-level for extended quotations. Do not swap them.\n"
            "4. Abbreviations requiring expansion must use the /E "
            "entry on the <Span> element.\n"
            "5. Language changes within a document must be marked "
            "using the /Lang entry on a <Span> or block element. "
            "WCAG 2.2 SC 3.1.2 requires this.\n"
            "6. Superscript (<Sub>) and subscript (<Sub>) inline "
            "elements must be used for mathematical notation inline "
            "within text. Do not use separate <Figure> tags for "
            "simple inline super/subscripts.\n"
            "7. <Code> should be used for inline code references "
            "within a paragraph, not bare <Span>."
        )
    },

    "LISTS": {
        "triggers": ["L", "LI", "Lbl", "LBody"],
        "rule": (
            "PDF/UA + WCAG 2.2 LIST RULES:\n"
            "1. All visual lists (bulleted, numbered, definition) "
            "must use the full list structure: <L> → <LI> → <Lbl> "
            "+ <LBody>.\n"
            "2. List markers (bullets, numbers, letters) must be "
            "placed inside <Lbl> ONLY. They must never appear inside "
            "<LBody>.\n"
            "3. The text content of each list item belongs in "
            "<LBody>. Direct text as an immediate child of <LI> "
            "without <LBody> is technically processable but "
            "non-conformant with PDF/UA.\n"
            "4. Nested lists must be structured as a new <L> element "
            "inside the parent <LBody>, not as a sibling of <LI>.\n"
            "5. <L> must contain only <LI> elements as direct "
            "children. No <P> or other block tags directly inside "
            "<L>.\n"
            "6. An <LI> containing only a <Lbl> with no <LBody> is "
            "acceptable only for a list item with no content beyond "
            "the marker (rare). An empty <LBody> is inappropriate.\n"
            "PDF/UA-2 addition: List structure must be resolvable "
            "through the structure tree with correct parent-child "
            "relationships."
        )
    },

    "TABLES": {
        "triggers": ["Table", "TR", "TH", "TD", "THead", "TBody",
                     "TFoot"],
        "rule": (
            "PDF/UA + WCAG 2.2 TABLE RULES:\n"
            "1. All tabular data visually presented in a grid must "
            "use <Table> → <TR> → <TH>/<TD> structure. Do not use "
            "spaced <P> tags to simulate table layout.\n"
            "2. <Table> must contain <TR> directly, or within "
            "optional <THead>, <TBody>, <TFoot> grouping elements.\n"
            "3. Every <TR> must contain only <TH> or <TD> children. "
            "No <P> or <Span> directly inside <TR>.\n"
            "4. Header cells must use <TH>, not <TD>. <TH> must have "
            "a Scope attribute (Row, Col, Both, or None) to indicate "
            "what it headers. Missing Scope on <TH> is a PDF/UA "
            "violation.\n"
            "5. Empty cells must always be <TD> — empty <TD> is "
            "explicitly permitted by PDF/UA. An empty <TH> is a "
            "violation. Never flag an empty <TD> as incorrect.\n"
            "6. Cells spanning multiple rows or columns require "
            "ColSpan and/or RowSpan attributes on the <TH> or <TD>.\n"
            "7. Complex tables (irregular headers, nested structure) "
            "require explicit header/ID associations via the Headers "
            "attribute on <TD>.\n"
            "8. Do not use tables for visual layout of non-tabular "
            "content. Layout tables are a PDF/UA violation.\n"
            "9. Nested tables (a <Table> inside a <TD>) are "
            "permitted but require full independent structure.\n"
            "WCAG 2.2 SC 1.3.1: Structure must be programmatically "
            "determinable. SC 4.1.2: Table headers must be "
            "identifiable by assistive technology."
        )
    },

    "IMAGES_AND_FIGURES": {
        "triggers": ["Figure", "Artifact"],
        "rule": (
            "PDF/UA + WCAG 2.2 IMAGE AND ARTIFACT RULES:\n"
            "1. All meaningful images, charts, diagrams, icons, and "
            "graphics that convey information must be tagged as "
            "<Figure> with a non-empty Alt attribute.\n"
            "2. Alt text must describe the PURPOSE or INFORMATION of "
            "the image, not just its appearance. 'Chart' or 'Image' "
            "alone is not acceptable alt text.\n"
            "3. Decorative images, background graphics, dividers, "
            "and visual-only elements that convey no information must "
            "be tagged as Artifact, not <Figure>. Screen readers "
            "must ignore them.\n"
            "4. Page numbers, headers, footers, and running elements "
            "must be marked as Artifact with the Pagination "
            "subtype.\n"
            "5. An empty <Figure> element (no Alt and no content) is "
            "semantically inappropriate and a PDF/UA violation.\n"
            "6. <Figure> for a chart or diagram that cannot be fully "
            "described in short alt text should additionally have a "
            "long description via the /ActualText or associated "
            "<Caption> element.\n"
            "WCAG 2.2 SC 1.1.1: All non-text content must have a "
            "text alternative. SC 1.4.5: Images of text must be "
            "avoided unless essential."
        )
    },

    "MATH_AND_FORMULAS": {
        "triggers": ["Formula", "Math"],
        "rule": (
            "PDF/UA + WCAG 2.2 MATH AND FORMULA RULES:\n"
            "1. Mathematical expressions — whether rendered as text, "
            "raster images, or vector graphics — must be tagged as "
            "<Formula> and REQUIRE a non-empty Alt attribute that "
            "provides a readable text equivalent of the expression.\n"
            "2. Fragmented formula components (superscripts, "
            "subscripts, fractions, radicals split across multiple "
            "content streams) that form a single logical formula must "
            "be unified under one <Formula> container. Do NOT split "
            "them into separate <Figure> or bare <Span> elements.\n"
            "3. Simple inline superscripts and subscripts within "
            "running text may use <Sub>/<Sup> inline elements inside "
            "<P> rather than <Formula>, provided they are not "
            "standalone mathematical expressions.\n"
            "4. MathML content embedded in PDF 2.0 should be "
            "associated with the <Formula> structure element per "
            "PDF/UA-2 requirements.\n"
            "WCAG 2.2 SC 1.1.1 applies: mathematical content must "
            "have a text alternative accessible to screen readers."
        )
    },

    "FORMS_AND_LINKS": {
        "triggers": ["Form", "Widget", "Link", "Annot"],
        "rule": (
            "PDF/UA + WCAG 2.2 FORM AND LINK RULES:\n"
            "FORMS:\n"
            "1. Form field LABELS (e.g. 'Full Name', 'Email address') "
            "must be tagged <P>, never <H1>–<H6>. Heading tags are "
            "reserved for document structure, not UI labels.\n"
            "2. Instructional/helper text (e.g. 'Example: "
            "name@example.com', 'Limit 1000 characters') must be "
            "tagged <P> and must appear BEFORE the <Form> element in "
            "the tag tree to preserve logical reading order.\n"
            "3. Each <Form> structure element must enclose exactly "
            "ONE widget annotation (OBJR). Multiple widgets inside "
            "one <Form> is a violation.\n"
            "4. Every form field's Tooltip property must be "
            "non-empty. Screen readers announce the Tooltip during "
            "Tab-key navigation. Best practice: Tooltip should "
            "include both the label and instruction, e.g. 'Email "
            "address — Example: name@example.com'.\n"
            "5. Form fields collecting personal data must have "
            "descriptive internal field names (e.g. 'Email_Address', "
            "'Full_Name') not generic auto-generated names like "
            "'TextField1' or 'Field_3'.\n"
            "6. Checkboxes and radio buttons must have both a group "
            "label (<P> before the group) and individual item labels "
            "via Tooltip on each widget.\n"
            "LINKS:\n"
            "7. Every <Link> structure element must contain one or "
            "more OBJR entries referencing the link annotation.\n"
            "8. Link text must be descriptive of the destination. "
            "'Click here' and bare URLs as link text are WCAG 2.2 "
            "SC 2.4.4 violations.\n"
            "9. Links that open new windows or trigger downloads must "
            "announce this in their accessible name or adjacent text.\n"
            "WCAG 2.2 SC 1.3.5: Input purpose must be "
            "programmatically determinable for personal data fields. "
            "SC 2.4.4: Link purpose must be determinable from text "
            "alone or context. SC 4.1.2: Name, role, value must be "
            "programmatically determinable for all UI components."
        )
    },

    "CAPTIONS_AND_FOOTNOTES": {
        "triggers": ["Caption", "Note", "Reference", "Annot",
                     "BibEntry"],
        "rule": (
            "PDF/UA CAPTION AND FOOTNOTE RULES:\n"
            "1. <Caption> for a <Figure> must appear as an immediate "
            "SIBLING of <Figure>, placed after it in the structure "
            "tree. <Caption> must NOT be nested inside <Figure>.\n"
            "2. <Caption> for a <Table> must appear as the first or "
            "last child of <Table>, not outside it.\n"
            "3. <Note> must only be used for explicitly referenced "
            "footnotes or endnotes. The <Lbl> inside <Note> must "
            "exactly match the <Lbl> inside the corresponding "
            "<Reference> that points to it.\n"
            "4. Footnote reference markers in body text must use "
            "<Reference> containing a <Lbl> that matches the "
            "footnote's own <Lbl>.\n"
            "5. Bibliographic references use <BibEntry>. These "
            "should be structured inside a <BibEntry> container "
            "rather than bare <P> elements."
        )
    },

    "TOC_AND_NAVIGATION": {
        "triggers": ["TOC", "TOCI", "Index", "Sect", "Part",
                     "Article", "Div"],
        "rule": (
            "PDF/UA NAVIGATION AND STRUCTURE RULES:\n"
            "1. <TOC> must contain only <TOCI> children. Each "
            "<TOCI> should contain a <Reference> element pointing to "
            "the relevant heading.\n"
            "2. Dot-leaders and decorative spacing in TOC entries "
            "must be marked as Artifact, not wrapped in text tags.\n"
            "3. <Index> elements are typically structured internally "
            "as nested <L>/<LI> lists.\n"
            "4. <Sect> is for major document sections. <Part> is for "
            "top-level divisions of a book-like document. <Article> "
            "is for self-contained compositions. <Div> is a generic "
            "grouping element with no inherent semantic.\n"
            "5. PDF/UA requires the document to have a defined "
            "document title in XMP metadata (dc:title) and the "
            "viewer must be set to display the title instead of the "
            "filename.\n"
            "6. The document language must be declared in XMP "
            "metadata. Language changes within content must be "
            "marked with /Lang on the relevant element.\n"
            "7. Bookmarks are recommended (not strictly required by "
            "PDF/UA-1) for documents longer than a few pages. "
            "PDF/UA-2 strengthens this recommendation. Bookmarks "
            "must reflect the heading hierarchy.\n"
            "8. Page Tab Order must be set to 'Use Document "
            "Structure' in Page Properties to ensure keyboard "
            "navigation follows the tag tree order."
        )
    },

    "READING_ORDER_AND_ARTIFACTS": {
        "triggers": ["Artifact", "Span", "Div", "P"],
        "rule": (
            "PDF/UA READING ORDER AND ARTIFACT RULES:\n"
            "1. The logical reading order defined by the structure "
            "tree must match the visual reading order a sighted user "
            "would follow. Multi-column layouts, sidebars, callouts, "
            "and floating elements commonly break this.\n"
            "2. Content that is purely decorative, presentational, "
            "or not part of the logical document (borders, "
            "backgrounds, watermarks, decorative rules, drop caps "
            "duplicated in the tag tree) must be marked as Artifact.\n"
            "3. Running headers and footers that repeat on every "
            "page must be Artifact unless they contain unique "
            "navigational content.\n"
            "4. All real content (non-Artifact) must have Unicode "
            "mappings so that text selection and screen reader "
            "pronunciation is correct. Missing or incorrect Unicode "
            "mappings are a PDF/UA violation.\n"
            "5. Content spanning page breaks must remain logically "
            "one element in the structure tree — do not split a "
            "paragraph into two separate <P> tags solely because of "
            "a page boundary.\n"
            "6. No structure element should be completely empty "
            "except where explicitly permitted: <TD>, <LI>, <Span>, "
            "<Div>, and <Document> may be empty. <Figure>, <Formula>, "
            "and heading elements must not be empty.\n"
            "PDF/UA-2 addition: Structure element attributes "
            "(BBox, Placement, WritingMode) must be used consistently "
            "to support reflow and content extraction."
        )
    },

    "DOCUMENT_METADATA": {
        "triggers": ["Document", "Part"],
        "rule": (
            "PDF/UA DOCUMENT-LEVEL REQUIREMENTS:\n"
            "1. The PDF must contain a StructTreeRoot — without it "
            "the document is completely inaccessible to assistive "
            "technology.\n"
            "2. XMP metadata must include: document title (dc:title), "
            "document language (dc:language), and the PDF/UA "
            "identifier (pdfuaid:part = 1 for PDF/UA-1, = 2 for "
            "PDF/UA-2).\n"
            "3. The document's Initial View must be set to display "
            "the document title, not the filename.\n"
            "4. All fonts must be embedded. Non-embedded fonts "
            "cannot guarantee correct Unicode extraction.\n"
            "5. The document must not be encrypted in a way that "
            "prevents assistive technology access (screen reader "
            "access must remain permitted in security settings).\n"
            "6. PDF/UA-2 (ISO 14289-2:2024) additionally requires: "
            "comprehensive structure element attributes, proper "
            "handling of associated files, and full conformance with "
            "PDF 2.0 (ISO 32000-2) tagging mechanisms."
        )
    }
}