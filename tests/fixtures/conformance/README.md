# Conformance-gate fixtures

Inputs for `scripts/verapdf_gate.py` — chosen so the ONLY PDF/UA-1 variable is
our tagging. They are born-digital, fully font-embedded source PDFs (from the
dp-bench corpus) that our pipeline tags to veraPDF UA-1 compliance.

Do NOT add inputs with source-level defects we deliberately don't repair
(unembedded fonts → clause 7.21.4.1, encryption, etc.) — those can never pass
regardless of tagging quality and would make the gate untrustworthy.

- `native_scholarly.pdf` — multi-section scholarly page (headings, body).
- `native_with_formulas.pdf` — display formulas (exercises the /Formula MathML
  Associated File path).
