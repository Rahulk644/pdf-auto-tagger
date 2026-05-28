"""PDF-Accessibility-Benchmark evaluation substrate.

Derives per-criterion accessibility verdicts from tagged PDFs (struct tree +
veraPDF + structural predicates) and scores them against the benchmark's
expert WCAG/PDF-UA labels. CPU/deterministic; one bounded Modal regen only for
the strip+V2 remediation pass.

Three eval axes (correlated but distinct): veraPDF = ISO conformance,
PDF-Accessibility-Benchmark = WCAG/expert accessibility, DocLayNet = tag accuracy.
"""
