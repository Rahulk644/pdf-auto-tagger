# Accessibility Capability Ledger

**Purpose:** the single source of truth for *what our auto-tagger does today, how well it
does it, what we're building next, what's planned, what's deferred, and what's
deliberately skipped* — keyed to the PDF/UA tag taxonomy and the syntax→structure→semantics
check stack. Update this whenever capability or a measured number changes.

**Last updated:** 2026-05-31
**North-star:** minimize manual remediation — measured against an **incumbent commercial
remediation baseline (anonymized)**: a 1,033-doc managed-services corpus (774 with both
source PDF + the incumbent's tags; ~1.08M tagged elements).

> **Measurement honesty note.** The incumbent baseline's CSV bbox coordinates are
> unreliable (off-page under every fixed interpretation; per-doc scale variance), so
> *spatial* tag-by-tag matching is unsound. We compare on **convention-free per-document
> type counts** (coverage) + our own benchmarks (quality).

---

## 1. The five requirements of an accessible PDF — our standing

| Requirement | Standing | Evidence |
|---|---|---|
| **Semantic & structural tagging** | Strong on high-value tags; fine semantic inline tags flattened to `<P>` | dp-bench overall **0.839**; suite 308 green |
| **Artifacting** (hide decorative) | Headers/footers/page-#s/margin-watermarks artifacted | Stage 8c `artifact_detector` |
| **Logical reading order** | AI regions ordered by XY-cut; multi-column/floating still imperfect | NID **~0.83–0.888** |
| **Alternative text** | Presence solved; **guidelines-VLM produces real, grounded descriptions** (opt-in/GPU), validated non-hallucinating; rubric strengthened to reject placeholders | VLM 100% rubric-compliant vs placeholder 0%; **0/20 ungrounded numbers** on data figures |
| **Document metadata** (Lang, Title) | Set + enforced | veraPDF UA-1 clean, Matterhorn 11-001/07-001 mapped |

---

## 2. The check stack — where we pass, where we choke

- **Level 1 — Syntax** (well-formed tree, MarkInfo, ParentTree, MCID consistency, /AF, valid refs): **SOLID.** veraPDF UA-1 compliant, CI-gated (`scripts/verapdf_gate.py`). Not a choke point.
- **Level 2 — Structure** (no heading skips, first=H1, L>LI>Lbl+LBody, TR>TH/TD, no empty/punct headings, every Figure has /Alt): **SOLID.** Stage 8a′ + 8f enforcers; auditor (`tagger/audit/act_rules.py`) validated against the independent veraPDF-corpus.
- **Level 3 — Semantics** (correct tag for meaning, logical order, meaningful alt, figure-vs-text judgment): **THE FRONTIER — where every remaining gap lives.** Machine-uncheckable; this is why the incumbent still uses human remediators.

---

## 3. Tag taxonomy coverage (what we actually emit)

✅ emit · ◐ partial/rare · ❌ not emitted · ⊘ deliberately skipped

| Category | Emitted | Skipped / not emitted |
|---|---|---|
| **Grouping (12)** | ✅ Document, Caption, TOC, TOCI · ◐ BlockQuote, Note | ⊘ Art, Sect, Part, Div, Index, NonStruct, Private *(flat Document is valid)* |
| **Block (8)** | ✅ P, H1–H6 | ⊘ generic H *(we use specific levels — better)* |
| **List (4)** | ✅ L, LI, Lbl, LBody | — |
| **Table (7)** | ✅ Table, TR, TH, TD | ⊘ THead, TBody, TFoot *(rows direct under Table — valid)* |
| **Inline (15)** | ✅ Span, Link, Annot | ⊘ Quote, Reference, BibEntry, Code *(flatten to P — valid survival tactic)* · ⊘ Ruby/RB/RT/RP/Warichu/WT/WP *(Asian — out of scope)* |
| **Illustration/Interactive (3)** | ✅ Figure, Formula, **Form** *(new 2026-05-30)* | — |
| **Artifact** | ✅ (property, not a tag) | — |

---

## 4. How good — quality scorecard

| Metric | Value | Source |
|---|---|---|
| veraPDF UA-1 | **compliant** (CI-gated) | `scripts/verapdf_gate.py` |
| Test suite | **308 passed**, 3 skipped | `pytest` (cpu backend) |
| dp-bench overall | **0.839** | dp-bench |
| Reading order (NID) | **0.83–0.888** | dp-bench / corpus |
| Tables (TEDS) | **0.740** | dp-bench |
| Headings (MHS) | **0.726** | arXiv \section scoreboard |
| Short alt-text vs rubric | **93% compliant** | Alt4Blind |
| Long-description alt-text | **0%** (the hole) | rubric |

**Coverage vs incumbent (untagged/from-scratch population) — 80-doc run, mean per-doc recall:**

| Type | recall | docs covered | read |
|---|---|---|---|
| **Headings** | **0.72** | 48/51 | healthy — matches independent MHS 0.726; the corrected harness is trustworthy |
| **Figure** | 0.52 *(vs incumbent)* | 33/52 | **investigated → not a real gap**: our count tracks actual source images; incumbent over-counts IMG (repeated logos/inline) + tags phantom/vector. Surplus → Artifact correctly |
| **Link** | **0.06 → addressed** | 2/18 → fix shipped | was #1 gap (we tagged existing annots only); **auto-detection now synthesizes /Link for bare URL/email text** |
| **Form** | metric artifact | — | scorecard counts *source* widgets not our output; `/Form` producer verified directly (217/217) |
| P | granularity-noisy | — | we merge lines→blocks; content captured (3,586 P + 24,402 table-cells vs 67,223 incumbent line-Ps), not a coverage gap |

> The earlier "1% heading recall" was a **measurement artifact** (spatial matcher over unreliable
> incumbent bboxes + table cells invisible to an element-level harness). Convention-free counts
> landing at 0.72 — matching the independent arXiv scoreboard — is the validation that the corrected
> harness is sound.

---

## 5. Corpus reality (informs every priority)

- **68% of real-world docs are already tagged** (527/774 have a StructTreeRoot) → our `retag_existing_pdf` path. Verified **neutral-to-improving** (audit fails down, zero struct orphans) — we do **not** corrupt them.
- **32% (247) untagged** → our `tag_untagged_pdf` from-scratch path = where "auto-tagging" is actually tested.
- These are **business / process / form** docs → prioritize **links, forms, tables**, not poetic typography. Priorities are ranked by **measured corpus prevalence**, not edge-case notoriety.

---

## 6. Roadmap by state

### NOW (shipped, in production)
P, H1–H6, L/LI/Lbl/LBody, Table/TR/TH/TD, Figure, Formula (+MathML /AF, opt-in image→LaTeX), Caption, Link (existing annots, OBJR), **Form (widgets → /Form + OBJR + /TU-from-fieldname)**, Artifact (headers/footers/page-#s/margin-watermarks), Lang + Title metadata. Pre-tagged docs retagged safely.

### SHIPPED THIS SESSION (committed: 5c15fa6, f1f5b76)
- ✅ `/Form` producer for widget annotations (`_tag_widget_annotations`) — `/Form`+OBJR+`/TU`-from-fieldname; 217/217 on test doc 389, audit 3→0, integrity clean, unit-tested.
- ✅ **Link auto-detection** (`_autodetect_link_annotations`) — whole-token regex finds bare URL/email text with no covering annotation → synthesizes a functional `/Link` `Annot` (`/A /URI`), which existing `_tag_link_annotations` wraps into `/Link` struct + OBJR + `/Contents`. Verified: 1396 → 29 links, 1114 → 21 (incl `mailto:`), 1058 → 14, audit 3→0, integrity clean, unit-tested. **Lift re-measured honestly:** closes the *bare-URL/email* subset only; the rest of the incumbent's link gap is **non-URL cross-reference links** (bill numbers, "click here", TOC phrases) whose target can't be derived from text — inherently non-deterministic, **deferred**. Known wart: line-wrapped URLs truncate. Suite 303→**305**.
- ✅ Convention-free coverage scorecard harness (`scratch/prep_baseline/`) — 80-doc run complete.
- ✅ **Digital-signature safety guard** (`pdf_is_signed` + Stage 10 gate) — signed PDFs (49/774 in corpus) are detected and emitted **unmodified** (rewriting invalidates the signature); flagged `signed_unmodified`. Unit-tested end-to-end.
- ✅ **Adjacent-label `/TU` for cryptic form fields** — 37% of corpus field names are generated ids (`Text12`, `X3`); for those, `/TU` is now derived from the visible label (text to the left / above the widget) instead of the useless name. `_clean_label` rejects non-word labels (numbers, markers) → safe fallback to the field name; never a misleading `/TU`. Controlled test passes (`Text1` + "Employee Name" → `/TU` "Employee Name"). Low applicability on this corpus's *grid* forms (they fall back safely), real win for structured forms.
- ✅ **Alt-text quality: guidelines-VLM + strengthened rubric.** Rewrote the alt-text prompt (`config.py`) to the McGowan guidelines (convey meaning/data, no appearance prefix, anti-hallucination clause); deployed the Gemma-4-E4B vision endpoint on Modal. Scoreboard (50 figures): the VLM produces **real, grounded, meaningful** descriptions ("a boy in a wheelchair smiles in a classroom, illustrating special education") where the baseline emitted **placeholders** ("Figure (description needed)"). **Exposed that the rubric was too weak** — it passed placeholders at 100%; added a `placeholder` check → baseline now correctly 0%, VLM 100% (eyeballed pairs). Unit-tested. ✅ **Chart-hallucination validated**: across 20 data-bearing figures (numbers cross-checked vs RapidOCR of the crop), **0 ungrounded VLM numbers** — the VLM cites only numbers present in the chart (8/20) or describes qualitatively without inventing values (12/20). Anti-hallucination clause holds. Lowered `max_output_tokens` 150→44 to keep short alt under the 150-char cap. **Guidelines-VLM is production-ready as the opt-in (GPU-gated) figure-alt path.**
- ✅ **Cross-page true-merge INVESTIGATED → defer.** Prevalence probe: only **2/20 docs (10%)** have a table/list continuation. High complexity (cross-page MCID/`/Pg`, cell reconciliation) for modest benefit (two adjacent valid tables is not a conformance failure) → deferred with the number.
- ✅ **Figure under-detection INVESTIGATED → no fix.** Diagnostic (30 docs): 20 ok/over, 6 "undercount" that are really **incumbent over-count** (we capture the actual source images; surplus correctly → Artifact per PDF4/H67), 3 incumbent-only (phantom/vector), 1 true detection-miss. The "recall 0.52 vs incumbent" is ~90% an incumbent IMG-counting artifact, not our miss. Chasing it would invent figures.

- ✅ **Redaction guard INVESTIGATED → do not build.** Prevalence probe (150 docs): the naive "dark filled rect over dark text" heuristic flagged 43%, but inspection showed **all false positives** — heading underlines (1px-tall rules) and light background/content panels that pdfplumber misreads as `fill=0` (covering 256–660 chars of *readable* text). True redaction prevalence in this corpus is ~0, and the heuristic can't distinguish a redaction box from a background panel via extraction alone → a guard would **hide real content**. Deferred (would need rendered-pixel occlusion analysis for a near-zero-prevalence case).

### NEXT (immediate · deterministic · measured)
1. **Validate alt-text VLM on data charts** — chart-heavy sample to confirm it conveys values WITHOUT hallucinating (the unproven high-risk case).
2. **Alt-text quality** (the genuine semantic hole) — scoreboard current corpus alt-text vs the McGowan rubric, then the document-context VLM upgrade on Modal.

### PLANNED (will do · needs GPU or larger work · scoreboard-gated)
- **Alt-text quality** → document-context VLM (Gemma-E4B on Modal) scored vs the McGowan rubric — turns "blue bars" into "Q3 revenue $2M→$5M". *(the genuine semantic hole; Modal earns its spend here)*
- **Table topology** → colspan/rowspan, hierarchical TH, multi-line wrapped-cell false-row-break (the TEDS 0.74 bucket).
- **Reading-order model** for multi-column/floating regions (LayoutReader-style over Heron regions) — gated on a reading-order scoreboard (prior layout-model swaps lost end-to-end).
- **Cross-page TRUE merge** → single logical `/Table`//`/L` (today Stage 7 *detects + flags* artifact-aware, but doesn't structurally stitch).
- **Form quality** → radio-button grouping (adjacent-visual-label `/TU` shipped).

### DEFERRED (real but low corpus prevalence / genuinely hard)
**Vector-graphic figures** (logos/diagrams drawn as paths, not raster — pdfplumber doesn't see them as images; ~3–4/30 docs in the figure diagnostic); flowchart logical (non-geometric) reading order + connector arrows; sub-total indentation hierarchy in tables; fake/printed checkboxes (Wingdings/vector squares, non-AcroForm); TOC dot-leader artifacting; QR/barcode payload decode → alt-text; **compound (multi-line) link merge** (line-wrapped URLs truncate today); cross-page hyphenation `/ActualText`; inline `/Lang` language-switch spans; drop caps.

### SKIPPED (out of scope — with reason)
- **Asian ruby/warichu tags** (Ruby/RB/RT/RP/Warichu/WT/WP) — not in corpus language scope.
- **Optional grouping tags** (Art/Sect/Part/Div/Index/NonStruct/Private) — flat `<Document>` is fully PDF/UA-valid; screen readers navigate by heading hierarchy, not hidden wrappers.
- **THead/TBody/TFoot grouping** — TR direct under Table is conformant.
- **Generic `<H>`** — we emit specific H1–H6.
- **Fine semantic inline tags** (Code/Quote/Reference/BibEntry → P) — industry-standard automation survival tactic; passes the checker, preserves the text payload.

---

## 7. Measurement infrastructure
- `scratch/prep_baseline/index_csv.py` — shard the incumbent CSV → per-doc JSON.
- `scratch/prep_baseline/count_scorecard.py` — convention-free coverage (the trustworthy metric).
- `scratch/prep_baseline/annot_census.py` — link/widget blast-radius census.
- `scratch/prep_baseline/retag_quality.py` — source-vs-output audit-delta on pre-tagged docs.
- Existing: dp-bench (`scratch/run_dpbench.py`), veraPDF gate, arXiv heading scoreboard, `tagger/audit/` (act_rules, matterhorn, screen_reader).
