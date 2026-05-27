import pikepdf
pdf = pikepdf.open("output_modal/miramar_untagged.pdf")
cs = pikepdf.parse_content_stream(pdf.pages[0])

in_bdc = False
current_mcid = None
current_tag = None
mcid_texts = {}

for op in cs:
    op_str = str(op.operator)
    if op_str == "BDC":
        in_bdc = True
        current_tag = str(op.operands[0])
        if len(op.operands) > 1 and isinstance(op.operands[1], pikepdf.Dictionary):
            current_mcid = int(op.operands[1].get("/MCID", -1))
        else:
            current_mcid = -1
    elif op_str == "EMC":
        in_bdc = False
        current_mcid = None
        current_tag = None
    elif op_str == "Tj" and in_bdc:
        text = str(op.operands[0])
        if current_mcid not in mcid_texts:
            mcid_texts[current_mcid] = {"tag": current_tag, "text": ""}
        mcid_texts[current_mcid]["text"] += text
    elif op_str == "TJ" and in_bdc:
        text = ""
        for item in op.operands[0]:
            if isinstance(item, pikepdf.String):
                text += str(item)
        if current_mcid not in mcid_texts:
            mcid_texts[current_mcid] = {"tag": current_tag, "text": ""}
        mcid_texts[current_mcid]["text"] += text

for mcid, data in sorted(mcid_texts.items()):
    if data["tag"] in ("/TH", "/TD"):
        print(f"MCID {mcid} ({data['tag']}): {repr(data['text'][:50])}")

