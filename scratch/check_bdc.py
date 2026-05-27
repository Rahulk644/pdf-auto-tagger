import pikepdf
pdf = pikepdf.open("output_modal/miramar_untagged.pdf")
cs = pikepdf.parse_content_stream(pdf.pages[0])
bdcs = [(str(op.operands[0]), op.operands[1]) 
        for op in cs if str(op.operator) == "BDC"]
th_bdcs = [b for b in bdcs if b[0] == "/TH"]
td_bdcs = [b for b in bdcs if b[0] == "/TD"]
print(f"TH BDC markers: {len(th_bdcs)}")
print(f"TD BDC markers: {len(td_bdcs)}")
table_bdcs = [b for b in bdcs if b[0] == "/Table"]
print(f"Table BDC markers: {len(table_bdcs)}")
