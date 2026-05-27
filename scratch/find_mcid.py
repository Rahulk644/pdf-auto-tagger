import pikepdf
import sys

pdf = pikepdf.open("output_modal/miramar_untagged.pdf")

def find_mcid(node, target, path=""):
    if isinstance(node, pikepdf.Dictionary):
        s = str(node.get("/S", "Unknown"))
        k = node.get("/K")
        new_path = path + "/" + s
        
        if isinstance(k, pikepdf.Array):
            for child in k:
                find_mcid(child, target, new_path)
        elif isinstance(k, pikepdf.Dictionary):
            find_mcid(k, target, new_path)
        elif isinstance(k, int):
            if k == target:
                print(f"Found MCID {target} at {new_path}")
    elif isinstance(node, pikepdf.Array):
        for child in node:
            find_mcid(child, target, path)

find_mcid(pdf.Root.StructTreeRoot, 88)
