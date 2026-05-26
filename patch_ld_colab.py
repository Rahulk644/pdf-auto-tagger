import tagger.stage3_layout.layout_detector as ld
file_path = ld.__file__

with open(file_path, "r") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if "def _parse_regions(" in line:
        start_idx = i

if start_idx != -1:
    new_code = """    def _parse_regions(
        self, raw_regions: list[dict], page_num: int, img_width: int = 1000, img_height: int = 1000
    ) -> list[LayoutRegion]:
        \"\"\"Parse normalized region dicts into LayoutRegion objects.\"\"\"
        regions: list[LayoutRegion] = []

        for idx, det in enumerate(raw_regions):
            from tagger.stage3_layout.layout_detector import _CATEGORY_MAP, LayoutCategory, LayoutRegion
            from tagger.config import LAYOUT
            
            category = _CATEGORY_MAP.get(det.get("category", "text"), LayoutCategory.TEXT)

            bbox = det.get("bbox", [0, 0, 0, 0])
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

            # Skip degenerate
            if bbox_tuple[2] <= bbox_tuple[0] or bbox_tuple[3] <= bbox_tuple[1]:
                continue

            confidence = det.get("score", 0.5)
            if confidence < LAYOUT.min_region_confidence:
                continue

            regions.append(LayoutRegion(
                region_id=f"r{page_num}_{idx}",
                page_num=page_num,
                bbox=bbox_tuple,
                category=category,
                reading_order=idx,
                confidence=confidence,
            ))

        return regions
"""
    lines[start_idx:] = [new_code]
    with open(file_path, "w") as f:
        f.writelines(lines)
    print("✅ layout_detector.py parsed!")
else:
    print(f"Failed. {start_idx}")

with open(file_path, "r") as f:
    content = f.read()
content = content.replace("self._normalize_raw(result), page_num,", "self._normalize_raw(result), page_num, page_image.width, page_image.height")
content = content.replace("self._parse_regions(regions_raw, page_num)", "self._parse_regions(regions_raw, page_num, page_image.width, page_image.height)")
with open(file_path, "w") as f:
    f.write(content)
print("✅ layout_detector.py calls updated!")
