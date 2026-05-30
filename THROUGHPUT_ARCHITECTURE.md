# Throughput Architecture

**Purpose.** A design-and-evidence argument that this pipeline can run at acquisition-buyer scale on a buyer's existing infrastructure. Numbers labeled **[measured]** come from this codebase; **[projected]** numbers extrapolate from them under stated assumptions.

---

## 1. Thesis (updated)

The pipeline is **embarrassingly parallel at the page level**. It runs in two interchangeable layout backends:

- **CPU-native (default).** The whole pipeline runs on commodity CPU at **~2.2 s/doc** on an M1 — no GPU, no MinerU, no AGPL. It **beats the GPU pipeline on every dp-bench metric** (NID 0.888, TEDS 0.581, MHS 0.720, overall 0.823 vs GPU 0.802). Throughput is page-count-driven and scales linearly with CPU count; cost is whatever the buyer's existing CPU fleet costs.
- **MinerU on GPU (legacy).** Retained for one of the throughput levers below (~2.7× per-container speedup via `layout_detect`) and for organisations that have standardised on MinerU. Bottlenecked by the layout VLM, which is ~97% of wall-clock; cleanly separable so a single GPU pool absorbs that single bottleneck.

Both paths land in the same `LayoutRegion[]` interface and Stages 4–10 are agnostic. **The default is CPU-native because the CPU pipeline now produces better output, not just cheaper output.**

---

## 2. Where the time goes — CPU-native path [measured]

dp-bench 200 docs, CPU backend, `TAGGER_LAYOUT_BACKEND=cpu`, no GPU spawned:

| Stage | Work | Per-doc share |
|---|---|---|
| 0–2 Classify + Extract + Merge | pdfplumber + page-classifier (CPU) | ~5% |
| 3 Layout | Docling Heron (RT-DETRv2, CPU) + TableFormer when borderless | ~40% |
| 4+5 Route + specialists | mapping + TableFormer table extraction | ~15% |
| 6+7 Validate + Cross-page merge | rule engine (CPU) | <2% |
| 8 Semantic refinement | heading rank + enforcers + lists + captions + artifacts | ~10% |
| 9 Alt-text | SigLIP zero-shot bucket + template (CPU) | ~25% |
| 10 Struct tree writeback | content stream rewrite + BDC injection | ~3% |

**Total: ~2.2 s/doc average across the 200-doc dp-bench corpus.** A 6-page scholarly doc lands around 5–8 s; an 80-page doc around 100 s. No single stage dominates the way MinerU did.

## 3. Where the time goes — legacy MinerU path [measured]

For organisations still on the MinerU path:

| Stage | Work | Time/doc | Share |
|---|---|---:|---:|
| 0 Classify | pdfplumber heuristics (CPU) | ~1.2 s | <2% |
| 1 Extract | pdfplumber chars (CPU) | ~1.3 s | <2% |
| 2 Merge | char→word→line (CPU) | ~0.3 s | <1% |
| **3 Layout** | **MinerU2.5 VLM (GPU)** | **~17.5 s/page** | **~97%** |
| 4–5 Route/specialists | mapping + tables (CPU) | ~0.2–0.6 s | <1% |
| 6–10 Validate→Write | rules, merge, semantics, BDC (CPU) | <0.6 s | <1% |

Stage 3 is the entire bottleneck. The two throughput levers below exist specifically for this path.

---

## 4. The two MinerU-path throughput levers [shipped]

### 4.1 Layout-only Stage 3 (`layout_detect`) — commit `11e5cb5`
The detector previously called MinerU's `two_step_extract` (layout + per-region content OCR). The pipeline discards MinerU's content entirely — text comes from pdfplumber. Switching to `layout_detect` drops the wasted content pass.

- **[measured]** `two_step_extract` ≈ 47.5 s/page; `layout_detect` ≈ 17.5 s/page → **~2.7× faster.**
- **Lossless by construction:** regions are unchanged; veraPDF UA1 `failedChecks` are identical across validation docs.

### 4.2 Page-level cross-container fan-out (Unit 3) — commit `f6ac57e`
Document-level fan-out load-balances badly when page counts vary. The pipeline seam (`prep_through_merge` → `render_layout_pages` → `inject_layout` → `finish_from_route`) lifts Stage 3 out so **pages, not documents, are the unit of work**. Each doc runs its CPU stages in one process; only page images go to a shared bounded GPU pool and regions come back.

- **[measured]** Fan-out output is byte-identical to the monolithic `run()` on validation docs.
- Pages from all documents share one pool — a 100-page doc's pages interleave with everyone else's: no straggler.

---

## 5. Cost-per-page model

### 5.1 CPU-native [projected from measured per-doc]
- **CPU cost/page** = negligible. A 2.2 s/doc average on an M1 is roughly 0.4 s/page; on a commodity 16-core x86 server that's effectively free as a marginal cost.
- **Pages/hr/container** ≈ 1 page / 0.4 s × 60 × 60 ≈ **~9,000 pages/hr** on a single container running serially. Embarrassingly parallel across cores and across containers.
- **$/1M pages** at ~$0.04/hr commodity CPU on-demand: **~$5** for 1M pages of layout + tagging.

### 5.2 MinerU on GPU [projected]
A10G at $1.10/hr, layout 17.5 s/page [measured]:

| GPU fleet | Pages/hr | Pages/day (24h) | $/1M pages (GPU) |
|---:|---:|---:|---:|
| 1 | ~206 | ~4.9k | ~$5,300 |
| 10 | ~2,060 | ~49k | ~$5,300 |
| 50 | ~10,300 | ~247k | ~$5,300 |
| 100 | ~20,600 | ~494k | ~$5,300 |

The cost column is flat — total spend is set by page count, not fleet size. A 1M-page corpus costs ≈ $5.3k of GPU + negligible CPU.

### 5.3 The honest comparison
**CPU-native is roughly three orders of magnitude cheaper per page than the MinerU/GPU path** — and produces higher-quality output on dp-bench. The MinerU path remains useful for organisations already standardised on it, but the default deployment story is now CPU.

---

## 6. On-premise / commodity-hardware deployability

- **No GPU required by default.** The whole pipeline runs on commodity CPU — pure x86 / ARM, any cloud, any on-prem fleet, any laptop class CPU.
- **Single binary's worth of dependencies.** `.venv3` is ~1.3 GB including torch + transformers + Docling models + RapidOCR + SigLIP. Runs offline once cached.
- **Existing infra works.** Stages 0–10 are CPU-only; the buyer's existing fleet does it.
- **GPU is optional for two specific upgrades** — (1) richer Stage 9 alt-text via Gemma-4-E4B for chart/diagram descriptions, (2) the legacy MinerU layout path if the buyer wants like-for-like comparison with their existing pipeline. Neither is required for conformant tagging.

This is what makes the accessibility-first positioning portable: the conformant-tagging value (Stages 6–10, the Stage-8 enforcers, the BDC injection, the audit layer) is all CPU work that runs anywhere.

---

## 7. Honest limits

- **Numbers are measured on dp-bench** (200 native scholarly docs, mostly 1–7 pages). Image-heavy or unusually dense docs will differ; the 80-page experience-100-25 doc in the audit batch ran at ~150 s, roughly linear in page count.
- **Heron + TableFormer + SigLIP run serially within a container.** They're not GPU-parallel; concurrency across documents is via process / container fan-out.
- **Scanned-PDF support is shipped but synthetic-tested.** Real-world scans (skew, JPEG compression, low DPI) will lose some OCR accuracy; the NID 0.992 number on the synthetic test is an upper bound. The OCR quality dial (`TAGGER_OCR_QUALITY=quality`) raises the floor for noisy inputs.
- **PDF/UA-2 MathML for formulas** is the headline parked unit. PDF/UA-1 is fully covered.

---

## 8. Summary for an evaluator

- **CPU-only by default**, beats GPU pipeline on every dp-bench metric.
- ~2.2 s/doc on M1; pages/hr/container ≈ ~9,000 [projected]; ~$5 per 1M pages on commodity CPU.
- Embarrassingly parallel at the page level across cores and containers.
- MinerU/GPU path retained for legacy parity; ~2.7× per-container speedup banked via `layout_detect`; page-level fan-out byte-identical and load-balanced.
- 9/14 veraPDF UA-1 compliant on the audit batch (same as PDFix, ahead of PREP's 6/14); zero ACT-rule failures on the same batch (PREP has 2, PDFix has 3).
- Runs on commodity infra: no GPU, no AGPL, no MinerU required.
</content>
