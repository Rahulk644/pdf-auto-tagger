import pikepdf

pdf = pikepdf.open("output_modal/miramar_untagged.pdf")
root = pdf.Root.StructTreeRoot
doc = root.K

def print_tree(node, depth=0):
    if isinstance(node, pikepdf.Dictionary):
        s = node.get("/S")
        s_val = str(s) if s else "Unknown"
        k = node.get("/K")
        scope = node.get("/A", {}).get("/Scope") if isinstance(node.get("/A"), pikepdf.Dictionary) else None
        scope_str = f" [Scope={str(scope)}]" if scope else ""
        print("  " * depth + f"- {s_val}{scope_str}")
        
        if isinstance(k, pikepdf.Array):
            for child in k:
                print_tree(child, depth + 1)
        elif isinstance(k, pikepdf.Dictionary):
            print_tree(k, depth + 1)
        else:
            print("  " * (depth+1) + f"- MCID {k}")
    elif isinstance(node, pikepdf.Array):
        for child in node:
            print_tree(child, depth)

print("Tree structure:")
print_tree(doc)
