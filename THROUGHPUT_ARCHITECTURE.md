# Throughput Architecture

**Purpose.** A design-and-evidence argument that this pipeline can run at acquisition-buyer
scale on a buyer's existing infrastructure. This is a *thinking-first* document: the
parallelization is real and the per-stage numbers are measured, but it is not a
production deployment guide. Numbers labeled **[measured]** come from this codebase on
Modal A10G; **[projected]** numbers extrapolate from them under stated assumptions.

---

## 1. Thesis

The pipeline is **embarrassingly parallel at the page level** and bottlenecked by a
single stage (layout detection). Once that stage is isolated and distributed — which the
current architecture does — throughput scales linearly with the GPU fleet, and the rest
of the pipeline runs on commodity CPU. Cost is **page-count-driven, not fleet-driven**:
adding workers buys wall-clock, not total spend.

---

## 2. Where the time goes (measured)

Full-pipeline stage timings, MinerU2.5-Pro + `layout_detect`, three ~6-page scholarly
PDFs on a single A10G **[measured, this session]**:

| Stage | Work | Time/doc | Share |
|-------|------|---------:|------:|
| 0 Classify | pdfplumber heuristics (CPU) | ~1.2 s | <2% |
| 1 Extract | pdfplumber chars (CPU) | ~1.3 s | <2% |
| 2 Merge | char→word→line (CPU) | ~0.3 s | <1% |
| **3 Layout** | **MinerU2.5 VLM (GPU)** | **~98–106 s** | **~97%** |
| 4–5 Route/specialists | mapping + tables (CPU) | ~0.2–0.6 s | <1% |
| 6–10 Validate→Write | rules, merge, semantics, BDC (CPU) | <0.6 s total | <1% |

**Stage 3 is ~97% of wall-clock; everything else is sub-second CPU work.** This is the
single most important fact for scaling: there is exactly one thing to parallelize, and it
is cleanly separable from the rest.

---

## 3. The two throughput levers already shipped

### 3.1 Layout-only Stage 3 (`layout_detect`) — commit `11e5cb5`
The detector previously called MinerU's `two_step_extract` (layout **+** per-region
content OCR). The pipeline discards MinerU's content entirely — text comes from
pdfplumber; MinerU supplies only region bboxes/categories. Switching to `layout_detect`
drops the wasted content pass.

- **[measured]** `two_step_extract` ≈ **47.5 s/page**; `layout_detect` ≈ **17.5 s/page** → **~2.7× faster.**
- **Lossless by construction**: `two_step_extract` runs the identical layout step
  internally, so regions are unchanged; veraPDF UA1 `failedChecks` were identical
  across validation docs.

### 3.2 Page-level cross-container fan-out (Unit 3) — commit `f6ac57e`
Document-level fan-out load-balances badly when page counts vary (1 vs 100 pages): a big
doc pins one worker while small docs idle the rest. The pipeline seam
(`prep_through_merge` → `render_layout_pages` → `inject_layout` → `finish_from_route`)
lifts Stage 3 out so **pages, not documents, are the unit of work**. Each doc runs its
CPU stages in one process (so `DocumentData` never crosses the wire); only page images go
out to a shared bounded GPU pool and regions come back.

- **[measured]** Fan-out output is **byte-identical** to the monolithic `run()` on the
  validation docs — the decoupling is provably lossless.
- Pages from all documents share one pool, so a 100-page doc's pages interleave with
  everyone else's: no straggler, near-perfect utilization.

---

## 4. Cost-per-page model

**Assumptions.** A10G at **$1.10/hr** (cloud on-demand; on-prem amortized is lower).
Layout **17.5 s/page [measured]**. CPU stages run on commodity CPU at negligible cost
(<$0.0002/doc). Per-page is currently serial *within* a container (see §6); parallelism
comes from fan-out *across* containers.

- **GPU cost/page** = 17.5 s × ($1.10 / 3600 s) ≈ **$0.0053/page** (~half a cent). **[projected]**
- **Throughput/container** = 3600 / 17.5 ≈ **206 pages/hr.** **[projected]**

| GPU fleet | Pages/hr | Pages/day (24h) | $/1M pages (GPU) |
|----------:|---------:|----------------:|-----------------:|
| 1 | ~206 | ~4.9k | ~$5,300 |
| 10 | ~2,060 | ~49k | ~$5,300 |
| 50 | ~10,300 | ~247k | ~$5,300 |
| 100 | ~20,600 | ~494k | ~$5,300 |

The cost column is flat: **total spend is set by page count, not fleet size** — workers
trade wall-clock, not money. A 1M-page corpus costs ≈ **$5.3k of GPU** plus negligible
CPU, finishable in ~2 days on 100 containers or ~3 weeks on 10.

---

## 5. On-premise / commodity-hardware deployability

The decoupled architecture maps directly onto a buyer's existing infra:

- **CPU fleet does most of the pipeline.** Stages 0–2 and 4–10 (~3% of time) are pure
  CPU and run on whatever the buyer already has.
- **Only layout needs a GPU.** A single on-prem A10G/L4/3090-class card does ~200
  pages/hr; a modest 8-GPU node ≈ **1,650 pages/hr [projected]**. No exotic hardware.
- **GPU-free path exists.** MinerU's **Pipeline Backend** orchestrates lightweight
  CPU-native specialists (DocLayout-YOLO layout + specialized OCR), ~86.2 OmniDocBench
  v1.5 vs Pro's 95.69. This trades quality for *zero GPU requirement* — and enables a
  **hybrid mode**: CPU layout by default, GPU only for hard pages. *(Quality and CPU
  speed of this path are not yet measured — backlog Item B; "CPU-native" must be
  throughput-verified, not assumed.)*

This is what makes the accessibility-first positioning portable: the conformant-tagging
value (Stages 6–10, repair-gating, PDF/UA struct tree) is all cheap CPU work that runs
anywhere; only the upstream layout perception wants acceleration.

---

## 6. Honest limits

- **Numbers are A10G-measured on small samples** (3–7 pages, scholarly docs). Dense or
  image-heavy pages will differ; a real-corpus calibration run is needed before quoting
  these externally.
- **No working cross-page batching engine yet.** vLLM produces degenerate single-region
  layout for MinerU2.5, and LMDeploy is blocked by a transformers dependency conflict
  (both parked as backlog investigations). So within a container, pages are processed
  serially; all current parallelism is *across* containers via fan-out. A working batched
  engine would cut per-container time further and is the main remaining throughput lever.
- **17.5 s/page is the current floor** for the Pro model on A10G — the layout model emits
  a long box list autoregressively. A lighter layout model (DocLayout-YOLO via the
  Pipeline Backend) or true batching would lower it.
- **Pipeline Backend CPU speed is unverified** — the GPU-free story in §5 is architecturally
  sound but needs the Item-B measurement to be quotable.

---

## 7. Summary for an evaluator

- One bottleneck (layout, ~97%), cleanly isolated and page-parallel.
- ~2.7× already banked losslessly (`layout_detect`); page fan-out scales linearly,
  byte-identically.
- ~$0.005/page GPU, flat in fleet size; ~206 pages/hr/container.
- Runs on commodity infra: CPU for ~all of the pipeline, one accessible GPU class for
  layout, and a credible GPU-free path pending one measurement.
- Open lever: a working batched inference engine would compound the per-container number.
