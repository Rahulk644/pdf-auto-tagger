"""Parity oracle: run opendataloader-bench's OWN golden test files (copied verbatim,
MIT) against OUR replicated metric modules.

The copied test files do bare imports like ``from evaluator_table import
evaluate_table``. We alias those original module names to our replicated modules
in sys.modules BEFORE the test modules are imported, so their exact assertions
exercise our code. Passing == our metrics match ODL's exactly == comparable.
"""
import sys

from tagger.benchmark.dpbench import converter, reading_order, heading, table

sys.modules.setdefault("converter_markdown_table", converter)
sys.modules.setdefault("evaluator_reading_order", reading_order)
sys.modules.setdefault("evaluator_heading_level", heading)
sys.modules.setdefault("evaluator_table", table)
