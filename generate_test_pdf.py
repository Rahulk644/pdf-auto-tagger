from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Page 1: Heading and Text
    Story.append(Paragraph("Annual Financial Report 2026", styles["Title"]))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Executive Summary", styles["Heading1"]))
    Story.append(Spacer(1, 12))
    text1 = "This is a clean, generated PDF to serve as a baseline for the structure tagger. It contains no existing structural tags or BDC markers in the content stream."
    Story.append(Paragraph(text1, styles["Normal"]))
    Story.append(Spacer(1, 24))
    
    # Page 2: Table
    Story.append(Paragraph("Revenue Breakdown", styles["Heading2"]))
    Story.append(Spacer(1, 12))
    data = [
        ["Department", "Q1 Revenue", "Q2 Revenue", "Total"],
        ["Sanitation", "$15,000", "$16,500", "$31,500"],
        ["Parks", "$8,200", "$9,100", "$17,300"],
        ["Public Works", "$42,000", "$45,000", "$87,000"]
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    Story.append(t)
    Story.append(Spacer(1, 24))
    
    # Page 3: Conclusion
    Story.append(Paragraph("Conclusion", styles["Heading1"]))
    Story.append(Spacer(1, 12))
    text2 = "In conclusion, the fiscal year has performed exactly as modeled in the projections. All departments are operating within their allocated budgetary constraints."
    Story.append(Paragraph(text2, styles["Normal"]))

    doc.build(Story)

if __name__ == "__main__":
    create_pdf("test_clean.pdf")
    print("Created test_clean.pdf")
