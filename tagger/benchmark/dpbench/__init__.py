"""Local replication of the opendataloader-bench (dp-bench) markdown evaluator.

These modules replicate — faithfully, for exact score comparability — the metric
computations from opendataloader-project/opendataloader-bench (MIT). We run them
in our OWN harness against ODL's published per-engine numbers; nothing is pushed
to their repo (see project memory project-dpbench-initiative).

Metrics (all markdown-in): NID (reading order), MHS (heading structure), TEDS
(table). The table metric derives from PubTabNet (Zhong et al., 1911.10683).
Parity is enforced by ODL's own golden test cases (tests/test_dpbench_metrics.py).
"""
