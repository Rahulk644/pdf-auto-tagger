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
| **Semantic & structural tagging** | Strong on high-value tags; fine semantic inline tags flattened to `<P>` | dp-bench overall **0.839**; suite 305 green |
| **Artifacting** (hide decorative) | Headers/footers/page-#s/margin-watermarks artifacted | Stage 8c `artifact_detector` |
| **Logical reading order** | AI regions ordered by XY-cut; multi-column/floating still imperfect | NID **~0.83–0.888** |
| **Alternative text** | *Presence* solved; *quality* is the hole | short-alt 93% guideline-compliant; long-description **0%** |
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
| Test suite | **305 passed**, 3 skipped | `pytest` (cpu backend) |
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
| **Figure** | **0.52** | 33/52 | **#2 gap** — we miss figures entirely on ~19 docs + undercount |
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

### SHIPPED THIS SESSION (uncommitted on disk)
- ✅ `/Form` producer for widget annotations (`_tag_widget_annotations`) — `/Form`+OBJR+`/TU`-from-fieldname; 217/217 on test doc 389, audit 3→0, integrity clean, unit-tested.
- ✅ **Link auto-detection** (`_autodetect_link_annotations`) — whole-token regex finds bare URL/email text with no covering annotation → synthesizes a functional `/Link` `Annot` (`/A /URI`), which existing `_tag_link_annotations` wraps into `/Link` struct + OBJR + `/Contents`. Verified: doc 1396 → 29 links, doc 1114 → 21 (incl `mailto:`), audit 3→0, integrity clean, unit-tested. **Known wart:** line-wrapped URLs truncate (→ deferred compound-link merge). Suite 303→**305**.
- ✅ Convention-free coverage scorecard harness (`scratch/prep_baseline/`) — 80-doc run complete.

### NEXT (immediate · deterministic · measured)
1. **Figure under-detection** (recall 0.52, ~19/52 docs missed) — diagnose: dropped inline images vs over-artifacting. The new measured #2 gap.
2. **Redaction & digital-signature safety guards** — detect opaque rect over live text (don't surface redacted text) and signed docs (don't alter the byte-stream). Do-no-harm, high stakes for legal docs.
3. Re-run the coverage scorecard counting links/forms from OUR OUTPUT (not source) to quantify the link-fix lift.

### PLANNED (will do · needs GPU or larger work · scoreboard-gated)
- **Alt-text quality** → document-context VLM (Gemma-E4B on Modal) scored vs the McGowan rubric — turns "blue bars" into "Q3 revenue $2M→$5M". *(the genuine semantic hole; Modal earns its spend here)*
- **Table topology** → colspan/rowspan, hierarchical TH, multi-line wrapped-cell false-row-break (the TEDS 0.74 bucket).
- **Reading-order model** for multi-column/floating regions (LayoutReader-style over Heron regions) — gated on a reading-order scoreboard (prior layout-model swaps lost end-to-end).
- **Cross-page TRUE merge** → single logical `/Table`//`/L` (today Stage 7 *detects + flags* artifact-aware, but doesn't structurally stitch).
- **Form quality** → adjacent-visual-label `/TU` (not just field name), radio-button grouping.

### DEFERRED (real but low corpus prevalence / genuinely hard)
Flowchart logical (non-geometric) reading order + connector arrows; sub-total indentation hierarchy in tables; fake/printed checkboxes (Wingdings/vector squares, non-AcroForm); TOC dot-leader artifacting; QR/barcode payload decode → alt-text; compound (multi-line) link merge; cross-page hyphenation `/ActualText`; inline `/Lang` language-switch spans; drop caps.

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
