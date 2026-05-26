import tagger.stage3_layout.layout_detector as ld

mock_regions = [
    {"category": "title", "bbox": [113, 100, 616, 134], "score": 0.5},
    {"category": "text", "bbox": [113, 161, 461, 182], "score": 0.5}
]

detector = ld.MinerULayoutDetector()
parsed = detector._parse_regions(mock_regions, 1, 1239, 1754)
for p in parsed:
    print(p)
