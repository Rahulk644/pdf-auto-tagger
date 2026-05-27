import pdfplumber
with pdfplumber.open("miramar_untagged.pdf") as pdf:
    page = pdf.pages[0]
    tables = page.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
    t = tables[0]
    print("t has rows?", hasattr(t, "rows"))
    print("dir(t):", dir(t))
