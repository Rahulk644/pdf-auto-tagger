"""Read-only conformance audit layer.

Separate from the tagging pipeline. Given a tagged PDF (ours, PREP, PDFix, or
any other) it walks the struct tree and the catalog and reports per-rule
pass/fail counts for the W3C ACT and PDF/UA-1 rules we explicitly support.

Two callable surfaces:
  - audit.act_rules.audit_pdf(path) -> AuditReport — programmatic
  - python -m tagger.audit.act_rules <pdf> [...]    — CLI summary
"""
from tagger.audit.act_rules import AuditReport, audit_pdf

__all__ = ["AuditReport", "audit_pdf"]
