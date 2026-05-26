from tagger.stage3_layout.mineru_worker import _normalize_result
result = [
    {"category": "title", "bbox": [113, 100, 616, 134], "score": 0.9},
]
print(_normalize_result(result))
