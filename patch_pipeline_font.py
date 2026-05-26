import os

file_path = "tagger/pipeline.py"
with open(file_path, "r") as f:
    content = f.read()

old_code = """                tagged_el = TaggedElement(
                    element_id=el.element_id,
                    page_num=page_num,
                    pdf_tag=pdf_tag,
                    text=el.text,
                    bbox=el.bbox,
                    confidence=conf,
                    original_mcid=el.mcid,
                )"""

new_code = """                tagged_el = TaggedElement(
                    element_id=el.element_id,
                    page_num=page_num,
                    pdf_tag=pdf_tag,
                    text=el.text,
                    bbox=el.bbox,
                    confidence=conf,
                    original_mcid=el.mcid,
                    font_name=el.font_name,
                    font_size=el.font_size,
                    font_weight=el.font_weight,
                    layout_category=cat.value if hasattr(cat, "value") else str(cat),
                )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, "w") as f:
        f.write(content)
    print("✅ pipeline.py successfully patched to include font properties!")
else:
    print("⚠️ Could not find old code in pipeline.py.")
