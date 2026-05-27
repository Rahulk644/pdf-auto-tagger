import pdfplumber

with pdfplumber.open("miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    words = page.extract_words()
    for w in words:
        if "Current" in w["text"]:
            print(f"Current: {w}")
        if "Capital" in w["text"]:
            print(f"Capital: {w}")
