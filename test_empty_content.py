import pikepdf
pdf = pikepdf.open("/Users/rahulkhatri/Tagger/output_modal/miramar_untagged.pdf")
doc = pdf.Root.get("/StructTreeRoot").get("/K")

def walk(node):
    if not isinstance(node, pikepdf.Dictionary):
        return
    if node.get("/Type") == pikepdf.Name.StructElem:
        mcid = node.get("/K")
        if isinstance(mcid, int):
            print(f"Tag: {node.get('/S')} MCID: {mcid}")
    k = node.get("/K")
    if isinstance(k, pikepdf.Array):
        for item in k:
            walk(item)
    elif isinstance(k, pikepdf.Dictionary):
        walk(k)

walk(doc)
