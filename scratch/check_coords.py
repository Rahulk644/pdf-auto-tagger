import pdfplumber
from tagger.config import STANDARD_DPI, PDF_NATIVE_DPI

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    page_width_pt = float(page.width)
    page_height_pt = float(page.height)

scale = STANDARD_DPI / PDF_NATIVE_DPI

# Real 150 DPI bbox (scaled up from the known 72 DPI table box)
x0_150 = 71.99869431764546 * scale
y0_150 = 39.432196588488864 * scale
x1_150 = 529.1919688496017 * scale
y1_150 = 695.6589125814853 * scale

pad = 5.0

print(f"Page size (72 DPI): {page_width_pt} x {page_height_pt}")
print(f"MinerU table bbox (150 DPI): x0={x0_150:.2f}, y0={y0_150:.2f}, x1={x1_150:.2f}, y1={y1_150:.2f}")

crop_box = (
    max(0, x0_150 / scale - pad),
    max(0, y0_150 / scale - pad),
    min(page_width_pt, x1_150 / scale + pad),
    min(page_height_pt, y1_150 / scale + pad),
)

print(f"pdfplumber crop box (72 DPI, top-left): {crop_box[0]:.2f}, {crop_box[1]:.2f}, {crop_box[2]:.2f}, {crop_box[3]:.2f}")
