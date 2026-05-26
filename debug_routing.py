from tagger.pipeline import AutoTaggerPipeline
from tagger.models.data_types import DocumentData
import tagger.stage3_layout.layout_detector as ld

class MockPipeline(AutoTaggerPipeline):
    def _stage3_layout(self, input_pdf, doc_data):
        doc_data.pages[1].layout_regions = ld.MinerULayoutDetector()._parse_regions([
            {"category": "title", "bbox": [113, 100, 616, 134], "score": 0.5},
            {"category": "text", "bbox": [113, 161, 461, 182], "score": 0.5},
            {"category": "text", "bbox": [112, 239, 1000, 258], "score": 0.5},
            {"category": "title", "bbox": [113, 283, 419, 309], "score": 0.5},
            {"category": "text", "bbox": [113, 328, 778, 344], "score": 0.5},
            {"category": "text", "bbox": [113, 366, 307, 382], "score": 0.5},
            {"category": "text", "bbox": [113, 391, 333, 407], "score": 0.5},
            {"category": "text", "bbox": [113, 416, 382, 432], "score": 0.5},
            {"category": "title", "bbox": [113, 461, 377, 485], "score": 0.5},
            {"category": "text", "bbox": [113, 505, 531, 522], "score": 0.5},
            {"category": "page_number", "bbox": [489, 935, 499, 947], "score": 0.5}
        ], 1, 1239, 1754)

pipeline = MockPipeline()
report = pipeline.run("tests/fixtures/sample.pdf", "debug_report.json")
for el in report["elements"]:
    print(el["pdf_tag"], el["text"])
