# Clean corpus — batch 1 (2026-05-28)

Genuinely-untagged, public-domain PDFs sourced from the public internet, for testing
the tagger's generalization on documents **PREP never touched** (removing the
"PREP-derived inputs" asterisk on the 5/5-vs-3/5 result).

Selection rules applied to every file:
- **Public domain** (US federal work, or pre-1929 work in the US public domain).
- **Genuinely untagged** — verified `/StructTreeRoot` absent (no struct tree, no MarkInfo).
- **Native text** (not scanned) — the tagger's Stage-1 native extraction applies; scanned/OCR
  docs were deliberately excluded (MinerU text extraction for scans isn't implemented).
- **1–2 representative pages** excerpted from a larger source to isolate a complexity.

| File | Source work | Pages | Complexity profile |
|---|---|---|---|
| `math_recreations_toc.pdf` | *Mathematical Recreations and Essays*, W. W. Rouse Ball — Project Gutenberg #26839 | 12–13 | Multi-level **table of contents** with dot-leaders + nested sub-entries + page numbers; chapter headings |
| `nasa_tech_report.pdf` | NASA Technical Report, NTRS ID 20040121077 | 2–3 | Single-column **technical report**: section headings + **numbered lists** + body prose + references |
| `economic_science_charts.pdf` | *The Alphabet of Economic Science*, P. H. Wicksteed — Project Gutenberg #32497 | 21–22 | **Figures/charts** (Cartesian axes, gridlines, plotted curves) + figure **captions** + running header + page number |
| `pencil_of_nature_plate.pdf` | *The Pencil of Nature*, W. H. Fox Talbot — Project Gutenberg #33447 | 23–24 | Embedded raster **photographic plate** (figure) + figure **caption** + heading + body prose |
| `matter_ether_motion.pdf` | *Matter, Ether, and Motion*, A. E. Dolbear — Project Gutenberg #31428 | 75–76 | Single-column prose + inline fractions + centered **display equations** + running header + page number |

## Source URLs
- Gutenberg #26839: https://www.gutenberg.org/files/26839/26839-pdf.pdf
- NASA NTRS 20040121077: https://ntrs.nasa.gov/api/citations/20040121077/downloads/20040121077.pdf
- Gutenberg #32497: https://www.gutenberg.org/files/32497/32497-pdf.pdf
- Gutenberg #33447: https://www.gutenberg.org/files/33447/33447-pdf.pdf
- Gutenberg #31428: https://www.gutenberg.org/files/31428/31428-pdf.pdf

## Licensing
- NASA NTRS: US Government work → public domain (17 U.S.C. § 105).
- Gutenberg works: underlying texts are pre-1929 → US public domain. Sourced via Project Gutenberg.

## Coverage & known gaps (to address in batch 2)
Covered across the batch: TOC/heading hierarchy, ordered lists, vector charts + captions,
raster figure + caption, mathematical equations, running headers / page numbers (artifact
candidates).

Gaps — deliberately or unavoidably absent this batch:
- **No dense data-grid table.** Modern table-rich gov PDFs (Census, Treasury Bulletin, GAO)
  are all 508-**tagged**; older equivalents are **scanned**. A genuinely-untagged native data
  table needs a different source (e.g., FRASER historical financial docs, older native NTRS).
- **No multi-column body flow** (all 5 are single-column) and **no fillable form**.
- **Source skew:** 4/5 are Project Gutenberg. Diversify with untagged-native government
  sources in batch 2.

Next step: run the batch through the pipeline (Modal) and grade with veraPDF UA1 to measure
generalization beyond the PREP-derived corpus.
