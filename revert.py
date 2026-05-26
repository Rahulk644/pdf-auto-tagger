import os
import tagger.stage3_layout.layout_detector as ld

file_path = ld.__file__
with open(file_path, "r") as f:
    content = f.read()

# REVERT parsing signature
content = content.replace(
    "def _parse_regions(\n        self, raw_regions: list[dict], page_num: int, img_width: int = 1000, img_height: int = 1000\n    ) -> list[LayoutRegion]:",
    "def _parse_regions(\n        self, raw_regions: list[dict], page_num: int,\n    ) -> list[LayoutRegion]:"
)

# REVERT method calls
content = content.replace(
    "return self._parse_regions(\n                self._normalize_raw(result), page_num, page_image.width, page_image.height\n            )",
    "return self._parse_regions(\n                self._normalize_raw(result), page_num,\n            )"
)
content = content.replace(
    "return self._parse_regions(regions_raw, page_num, page_image.width, page_image.height)",
    "return self._parse_regions(regions_raw, page_num)"
)

# REVERT scaling logic
old_coord = """            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            
            # MinerU / Qwen-VL outputs coordinates normalized to 1000
            # If coordinates are in [0, 1000] space, scale them back to the image DPI
            if max(x0, x1, y0, y1) <= 1000.0:
                x0 = (x0 / 1000.0) * img_width
                x1 = (x1 / 1000.0) * img_width
                y0 = (y0 / 1000.0) * img_height
                y1 = (y1 / 1000.0) * img_height

            bbox_tuple = (x0, y0, x1, y1)

            # Skip degenerate"""

new_coord = """            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            bbox_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

            # Skip degenerate"""

content = content.replace(old_coord, new_coord)

with open(file_path, "w") as f:
    f.write(content)

print("✅ layout_detector.py reverted successfully!")
