"""Tests for Stage 2 — Text merger."""

import pytest
from tagger.models.data_types import PageElement
from tagger.stage2_merger.text_merger import (
    merge_chars_to_words,
    merge_words_to_lines,
    merge_page_elements,
)


def _make_char(text: str, x0: float, y0: float, x1: float, y1: float,
               idx: int = 0, page: int = 1, font_size: float = 11.0) -> PageElement:
    """Helper to create a character-level PageElement."""
    return PageElement(
        element_id=f"p{page}_c{idx}",
        page_num=page,
        text=text,
        bbox=(x0, y0, x1, y1),
        font_size=font_size,
    )


class TestMergeCharsToWords:
    """Tests for Pass 1: chars → words."""

    def test_single_word(self):
        """Adjacent chars should merge into one word."""
        chars = [
            _make_char("H", 0, 0, 6, 12, idx=0),
            _make_char("e", 6, 0, 12, 12, idx=1),
            _make_char("l", 12, 0, 18, 12, idx=2),
            _make_char("l", 18, 0, 24, 12, idx=3),
            _make_char("o", 24, 0, 30, 12, idx=4),
        ]
        words = merge_chars_to_words(chars, page_num=1)
        assert len(words) == 1
        assert words[0].text == "Hello"

    def test_two_words_with_gap(self):
        """Chars with a large gap should produce two words."""
        chars = [
            _make_char("H", 0, 0, 6, 12, idx=0),
            _make_char("i", 6, 0, 12, 12, idx=1),
            # Large gap (50 pixels)
            _make_char("W", 62, 0, 68, 12, idx=2),
            _make_char("o", 68, 0, 74, 12, idx=3),
        ]
        words = merge_chars_to_words(chars, page_num=1)
        assert len(words) == 2

    def test_empty_input(self):
        """Empty input should return empty list."""
        assert merge_chars_to_words([], page_num=1) == []

    def test_space_insertion(self):
        """Chars with word-boundary gaps should have spaces in merged text."""
        # Simulate: "Hello World" — chars with a space-width gap
        h_chars = [
            _make_char("H", 0, 0, 6, 12, idx=0),
            _make_char("e", 6, 0, 12, 12, idx=1),
            _make_char("l", 12, 0, 18, 12, idx=2),
            _make_char("l", 18, 0, 24, 12, idx=3),
            _make_char("o", 24, 0, 30, 12, idx=4),
            # Gap of 4 pixels (> 0.3 * avg_char_width=6 → space)
            _make_char("W", 34, 0, 40, 12, idx=5),
            _make_char("o", 40, 0, 46, 12, idx=6),
            _make_char("r", 46, 0, 52, 12, idx=7),
            _make_char("l", 52, 0, 58, 12, idx=8),
            _make_char("d", 58, 0, 64, 12, idx=9),
        ]
        words = merge_chars_to_words(h_chars, page_num=1)
        # Should be 1 word group with space, or 2 separate words
        combined_text = " ".join(w.text for w in words)
        assert "Hello" in combined_text
        assert "World" in combined_text


class TestMergeWordsToLines:
    """Tests for Pass 2: words → lines."""

    def test_same_line(self):
        """Words on same Y-band should merge into one line."""
        words = [
            _make_char("Hello", 0, 0, 30, 12, idx=0),
            _make_char("World", 35, 0, 65, 12, idx=1),
        ]
        lines = merge_words_to_lines(words, page_num=1)
        assert len(lines) == 1
        assert "Hello" in lines[0].text
        assert "World" in lines[0].text

    def test_different_lines(self):
        """Words on different Y positions should be separate lines."""
        words = [
            _make_char("Line1", 0, 0, 30, 12, idx=0),
            _make_char("Line2", 0, 30, 30, 42, idx=1),
        ]
        lines = merge_words_to_lines(words, page_num=1)
        assert len(lines) == 2


class TestBaselineClustering:
    """Pass-1 baseline line clustering (the multi-column char-interleaving fix)."""

    @staticmethod
    def _ch(text, x0, top, x1, bottom, size=10.0, upright=True, idx=0):
        return PageElement(
            element_id=f"p1_c{idx}", page_num=1, text=text,
            bbox=(x0, top, x1, bottom), font_size=size, upright=upright,
        )

    def test_stacked_rows_not_interleaved(self):
        """Two rows at the same x but different baselines must NOT merge+interleave
        (the 'Care'+'Oxford' -> 'Coaf rOex' bug)."""
        from tagger.stage2_merger.text_merger import merge_chars_to_words
        chars = [
            self._ch("A", 0, 0, 8, 12, idx=0), self._ch("B", 8, 0, 16, 12, idx=1),
            self._ch("C", 0, 14, 8, 26, idx=2), self._ch("D", 8, 14, 16, 26, idx=3),
        ]
        words = merge_chars_to_words(chars, 1)
        texts = {w.text for w in words}
        assert texts == {"AB", "CD"}, texts          # two clean rows, not "ACBD"

    def test_superscript_attaches_to_line(self):
        """A small raised glyph whose baseline shift exceeds tol still attaches to
        the line it x-continues (sub/superscript handling)."""
        from tagger.stage2_merger.text_merger import _cluster_lines
        chars = [
            self._ch("X", 0, 0, 8, 12, size=10, idx=0),
            self._ch("y", 8, 2, 14, 12, size=10, idx=1),
            self._ch("2", 14, 0, 18, 7, size=7, idx=2),   # raised, baseline ~7
        ]
        lines = _cluster_lines(chars)
        assert len(lines) == 1 and len(lines[0]) == 3

    def test_rotated_separated_from_upright(self):
        """upright=False glyphs cluster separately and never join horizontal text."""
        from tagger.stage2_merger.text_merger import _cluster_lines
        chars = [
            self._ch("H", 0, 0, 8, 12, idx=0), self._ch("i", 8, 0, 14, 12, idx=1),
            self._ch("V", 0, 0, 8, 12, upright=False, idx=2),  # rotated, same bbox band
        ]
        lines = _cluster_lines(chars)
        # the rotated char is in its own line, not merged into "Hi"
        joined = ["".join(c.text for c in ln) for ln in lines]
        assert "Hi" in joined and any(ln == ["V"] or "".join(c.text for c in ln) == "V" for ln in lines)

    def test_char_count_invariant(self):
        """No char dropped or duplicated across clustering."""
        from tagger.stage2_merger.text_merger import _cluster_lines
        chars = [self._ch(c, (i % 5) * 8, (i // 5) * 14, (i % 5) * 8 + 8, (i // 5) * 14 + 12, idx=i)
                 for i, c in enumerate("abcdefghijklmno")]
        lines = _cluster_lines(chars)
        assert sum(len(ln) for ln in lines) == len(chars)

    def test_single_column_one_line(self):
        """Same-baseline chars stay one line (scope guard — simple prose untouched)."""
        from tagger.stage2_merger.text_merger import _cluster_lines
        chars = [self._ch(c, i * 8, 0, i * 8 + 8, 12, idx=i) for i, c in enumerate("Hello")]
        lines = _cluster_lines(chars)
        assert len(lines) == 1 and len(lines[0]) == 5


class TestFullMerge:
    """Integration test for the full merge pipeline."""

    def test_sample_pdf_integration(self):
        """Run full merge on sample PDF extracted data."""
        from tagger.stage0_classifier.page_classifier import classify_pages
        from tagger.stage1_extraction.native_extractor import extract_native_pages

        classifications = classify_pages("tests/fixtures/sample.pdf")
        raw_elements = extract_native_pages("tests/fixtures/sample.pdf", classifications)

        # Page 1 should have elements
        assert 1 in raw_elements
        page1_chars = raw_elements[1]
        assert len(page1_chars) > 100  # Lots of characters

        # Merge
        merged = merge_page_elements(page1_chars, page_num=1)

        # Should produce much fewer elements than raw chars
        assert len(merged) < len(page1_chars)
        assert len(merged) > 0

        # Text should contain readable words
        all_text = " ".join(el.text for el in merged)
        assert "Auto-Tagger" in all_text
        assert "Document" in all_text
        assert "Chapter" in all_text
