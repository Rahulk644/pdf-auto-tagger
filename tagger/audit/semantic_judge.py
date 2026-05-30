"""Semantic-correctness judge: deterministic perception vs our tags, reasoned by
a text-only LLM.

The correctness gap (is a tag actually RIGHT?) needs an independent witness. The
screen-reader linearizer can't be it — it reads our own tags. So we use TWO
independent text views of the same document and have an LLM find disagreements:

  View A — PHYSICAL layout (independent): pdfplumber lines with font size, bold,
           and position. This is raw perception, with NO semantic interpretation
           — the tagger never gets to influence it.
  View B — our TAGS: the struct-tree elements in reading order (role + text).

The LLM reasons over the two TEXT structures only — no pixels — so there is no
visual hallucination (the small-VLM failure mode). It flags e.g. "a 1.8x-body
bold centred line you tagged /P is physically heading-shaped", reading-order
mismatches, and mislabeled blocks.

This is the PILOT wiring: the reasoner is the Gemini API (smallest model) for
cheap iteration; production swaps in a local quantized text-LLM (llama.cpp on
M1 / OpenVINO on x86) behind the same `judge()` seam. The LLM is fallible too,
so output is FLAGGED DISAGREEMENTS for review + a confidence, never a
certification — calibrated against the 35-doc expert benchmark.

NOTE: this is the deterministic-perception approach, deliberately NOT the
screen-reader approach (a separate future experiment).
"""
from __future__ import annotations

import json
import os
import statistics

import pdfplumber
import pikepdf
from pikepdf import Array, Dictionary

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


def physical_layout(pdf_path: str, max_lines: int = 120) -> list[dict]:
    """View A — independent physical signal per text line (no tag influence)."""
    out: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages, 1):
            lines = page.extract_text_lines()
            all_sizes = [c.get("size") for ln in lines for c in ln.get("chars", []) if c.get("size")]
            body = statistics.median(all_sizes) if all_sizes else 0.0
            pw = page.width or 1.0
            for ln in lines:
                chars = ln.get("chars", [])
                szs = [c.get("size") for c in chars if c.get("size")]
                sz = statistics.median(szs) if szs else 0.0
                fonts = [(c.get("fontname", "") or "").lower() for c in chars]
                bold = bool(fonts) and sum(
                    1 for f in fonts if any(k in f for k in ("bold", "black", "heavy", "semibold"))
                ) >= 0.6 * len(fonts)
                xc = ((ln["x0"] + ln["x1"]) / 2) / pw
                pos = "center" if 0.33 < xc < 0.67 else ("left" if xc <= 0.33 else "right")
                out.append({
                    "page": pi, "text": (ln["text"] or "").strip()[:160],
                    "rel_size": round(sz / body, 2) if body else 1.0,
                    "bold": bold, "pos": pos, "top": round(ln["top"]),
                })
                if len(out) >= max_lines:
                    return out
    return out


def tag_view(pdf_path: str, max_items: int = 200) -> list[dict]:
    """View B — our tags in reading order (role + text)."""
    out: list[dict] = []
    pdf = pikepdf.open(pdf_path)
    try:
        sr = pdf.Root.get("/StructTreeRoot")
        if sr is None:
            return out

        def txt(n):
            for k in ("/ActualText", "/Alt"):
                v = n.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()[:160]
            return ""

        def walk(n):
            if not isinstance(n, Dictionary):
                return
            s = n.get("/S")
            if s is not None and str(s) not in ("/Document", "/Sect", "/Part", "/Div"):
                role = str(s).lstrip("/")
                t = txt(n)
                if t or role in ("Figure", "Table", "Formula"):
                    out.append({"role": role, "text": t})
                    if len(out) >= max_items:
                        return
            k = n.get("/K")
            for c in (k if isinstance(k, Array) else [k] if k is not None else []):
                if isinstance(c, Dictionary):
                    walk(c)
        walk(sr)
    finally:
        pdf.close()
    return out


_PROMPT = """You are auditing the SEMANTIC correctness of a tagged PDF for accessibility.
You are given two INDEPENDENT text views of the same document:

A = PHYSICAL layout from the raw page (font size relative to body text, bold, position). This is what the page visually IS. It has NO tags.
B = the accessibility TAGS we assigned, in reading order (role + text).

Compare them and flag disagreements where our TAGS likely misrepresent the document. Look for:
- text that is physically heading-shaped (rel_size >= ~1.3, or bold+short, often centered) but tagged as a paragraph "P" (missed heading), or vice-versa;
- reading-order mismatches (B's order doesn't follow A's top-to-bottom / column order);
- a block tagged as the wrong role given its physical form.

Reason ONLY over these two text structures. Do NOT invent content you cannot see in A or B.
Return STRICT JSON: {"findings":[{"issue":"...","evidence":"<quote from A and B>","severity":"high|medium|low"}], "agreement":"high|medium|low"}.
If they agree well, return an empty findings list and agreement "high".

A (physical layout):
%s

B (our tags):
%s
"""


def judge(pdf_path: str, model: str | None = None) -> dict:
    """Run the LLM judge. Requires GEMINI_API_KEY (or GOOGLE_API_KEY). Returns the
    parsed JSON findings (or {'raw': text} if the model didn't return clean JSON)."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY to run the semantic judge")
    from google import genai

    a = physical_layout(pdf_path)
    b = tag_view(pdf_path)
    prompt = _PROMPT % (json.dumps(a, ensure_ascii=False), json.dumps(b, ensure_ascii=False))
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model or DEFAULT_MODEL, contents=prompt)
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def main() -> None:
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("usage: GEMINI_API_KEY=... python -m tagger.audit.semantic_judge <pdf> [...]")
        sys.exit(2)
    if "--views" in sys.argv:  # inspect the two views without calling the LLM
        for p in args:
            print(f"\n== {p} ==\nA physical (first 15):")
            for r in physical_layout(p)[:15]:
                print("  ", r)
            print("B tags (first 15):")
            for r in tag_view(p)[:15]:
                print("  ", r)
        return
    for p in args:
        print(f"\n== {p} ==")
        print(json.dumps(judge(p), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
