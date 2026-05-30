"""Read-only conformance audit layer.

Separate from the tagging pipeline. Given a tagged PDF (ours, the incumbent, PDFix, or
any other) it walks the struct tree and the catalog and reports per-rule
pass/fail counts for the W3C ACT and PDF/UA-1 rules we explicitly support.

Callable surfaces:
  - audit.act_rules.audit_pdf(path) -> AuditReport   — ACT/PDF-UA rule results
  - audit.matterhorn.to_matterhorn(report)           — same results in Matterhorn
                                                       1.1 failure-condition IDs
  - audit.screen_reader.linearize(path) -> Transcript — deterministic AT read-out
  - python -m tagger.audit.act_rules | matterhorn | screen_reader <pdf> [...]
"""
from tagger.audit.act_rules import AuditReport, audit_pdf
from tagger.audit.alt_text_quality import check_alt_quality
from tagger.audit.matterhorn import to_matterhorn
from tagger.audit.screen_reader import linearize, smell_test

__all__ = ["AuditReport", "audit_pdf", "to_matterhorn", "linearize", "smell_test",
           "check_alt_quality"]
# semantic_judge is intentionally NOT eagerly imported (its judge() pulls
# google-genai + needs an API key); import it explicitly where used.
