import pikepdf
import pdfplumber

def check_pdf(filename):
    print(f"--- Checking {filename} ---")
    
    # 1. Check BDC / EMC balance
    pdf = pikepdf.Pdf.open(filename)
    for i, page in enumerate(pdf.pages):
        cs = pikepdf.parse_content_stream(page)
        bdc_count = sum(1 for op in cs if str(op.operator) == "BDC")
        emc_count = sum(1 for op in cs if str(op.operator) == "EMC")
        print(f"Page {i+1} | BDC: {bdc_count}, EMC: {emc_count}")
        assert bdc_count == emc_count, f"Page {i+1} Unbalanced BDC/EMC!"

    # 2. Check Visual text
    with pdfplumber.open(filename) as pb:
        for i, page in enumerate(pb.pages):
            text = page.extract_text()
            print(f"Page {i+1} char count: {len(text) if text else 0}")
            if text:
                print(f"Sample: {text[:100].replace(chr(10), ' ')}")

if __name__ == "__main__":
    check_pdf("test_clean.pdf")
    check_pdf("output_modal/test_clean.pdf")
