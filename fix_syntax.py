import os
file_path = os.path.abspath("tagger/stage3_layout/layout_detector.py")

with open(file_path, "r") as f:
    content = f.read()

content = content.replace("page_image.width, page_image.height page_image.width, page_image.height", "page_image.width, page_image.height")
content = content.replace("page_image.width, page_image.height, page_image.width, page_image.height", "page_image.width, page_image.height")

with open(file_path, "w") as f:
    f.write(content)

print("✅ Syntax fixed!")
