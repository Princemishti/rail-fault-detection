import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(inspection_data: dict, output_dir: str = "reports") -> str:
    """
    Generates a high-quality, professional PDF Railway Inspection Report using ReportLab.
    Returns the file path of the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    inspection_id = inspection_data.get("inspection_id", f"INSP-{int(datetime.utcnow().timestamp())}")
    filename = f"Inspection_Report_{inspection_id}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY_COLOR = colors.HexColor("#1e293b")   # Dark slate
    SECONDARY_COLOR = colors.HexColor("#0f766e") # Teal accent
    CRITICAL_COLOR = colors.HexColor("#dc3545")  # Red
    HIGH_COLOR = colors.HexColor("#fd7e14")      # Orange
    MEDIUM_COLOR = colors.HexColor("#d97706")    # Yellow/Amber
    LOW_COLOR = colors.HexColor("#16a34a")       # Green

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY_COLOR,
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b")
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    cell_bold = ParagraphStyle(
        'CellBold',
        parent=cell_style,
        fontName='Helvetica-Bold'
    )

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>NATIONAL RAILWAY AUTHORITY</b><br/><font size=16 color='#1e293b'><b>AI TRACK INSPECTION REPORT</b></font>", title_style),
            Paragraph(f"<b>Report ID:</b> {inspection_id}<br/><b>Date:</b> {inspection_data.get('timestamp', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))}<br/><b>Status:</b> {inspection_data.get('overall_status', 'N/A')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.0 * inch, 3.2 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY_COLOR, spaceAfter=15))

    # 2. Executive Summary Metrics Table
    overall_status = inspection_data.get("overall_status", "HEALTHY")
    max_severity = inspection_data.get("max_severity", "Low")
    status_bg = CRITICAL_COLOR if overall_status == "DEFECTIVE" else LOW_COLOR
    
    summary_data = [
        [
            Paragraph("<b>Overall Status</b>", cell_bold),
            Paragraph(f"<font color='{status_bg.hexval()}'><b>{overall_status}</b></font>", cell_bold),
            Paragraph("<b>Max Severity</b>", cell_bold),
            Paragraph(f"<b>{max_severity}</b>", cell_bold)
        ],
        [
            Paragraph("<b>Avg Confidence</b>", cell_bold),
            Paragraph(f"{inspection_data.get('confidence', 0)}%", cell_style),
            Paragraph("<b>Inference Time</b>", cell_bold),
            Paragraph(f"{inspection_data.get('inference_time_ms', 0)} ms", cell_style)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # 3. Image Comparison Section (Original vs Visual Detection Bounding Boxes)
    story.append(Paragraph("Visual Inspection & Defect Localization", section_heading))
    
    orig_path = inspection_data.get("original_image_path", "").lstrip("/")
    annot_path = inspection_data.get("annotated_image_path", "").lstrip("/")

    # Resolve absolute paths if relative
    if not os.path.isabs(orig_path):
        orig_path = os.path.join(os.getcwd(), orig_path)
    if not os.path.isabs(annot_path):
        annot_path = os.path.join(os.getcwd(), annot_path)

    img_cells = []
    if os.path.exists(orig_path):
        try:
            img_cells.append([RLImage(orig_path, width=3.4*inch, height=2.4*inch)])
        except Exception:
            img_cells.append([Paragraph("Original Image File Not Found", cell_style)])
    else:
        img_cells.append([Paragraph("Original Image", cell_style)])

    if os.path.exists(annot_path):
        try:
            img_cells.append([RLImage(annot_path, width=3.4*inch, height=2.4*inch)])
        except Exception:
            img_cells.append([Paragraph("Annotated Image File Not Found", cell_style)])
    else:
        img_cells.append([Paragraph("Annotated Image", cell_style)])

    img_table = Table([
        [Paragraph("<b>Original Rail Track Image</b>", cell_bold), Paragraph("<b>YOLOv8 Bounding Box Detection</b>", cell_bold)],
        [img_cells[0][0], img_cells[1][0]]
    ], colWidths=[3.6 * inch, 3.6 * inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    # 4. Detailed Defects & Recommendations Table
    story.append(Paragraph("Detected Defects & Maintenance Recommendations", section_heading))
    
    defects = inspection_data.get("defects", [])
    defect_table_data = [
        [
            Paragraph("<b>Defect Type</b>", cell_bold),
            Paragraph("<b>Confidence</b>", cell_bold),
            Paragraph("<b>Severity</b>", cell_bold),
            Paragraph("<b>Bounding Box (x1,y1,x2,y2)</b>", cell_bold),
            Paragraph("<b>Maintenance Recommendation</b>", cell_bold)
        ]
    ]

    if not defects:
        defect_table_data.append([
            Paragraph("No Track Defects Detected", cell_style),
            Paragraph("N/A", cell_style),
            Paragraph("<font color='green'><b>Low</b></font>", cell_style),
            Paragraph("N/A", cell_style),
            Paragraph("Track in healthy condition. Normal routine monitoring.", cell_style)
        ])
    else:
        for d in defects:
            sev = d.get("severity", "Medium")
            sev_color = "#dc3545" if sev == "Critical" else "#fd7e14" if sev == "High" else "#d97706" if sev == "Medium" else "#16a34a"
            bbox_str = f"[{', '.join(map(str, d.get('bbox', [])))}]"

            defect_table_data.append([
                Paragraph(f"<b>{d.get('class', 'Defect')}</b>", cell_bold),
                Paragraph(f"{d.get('confidence', 0)}%", cell_style),
                Paragraph(f"<font color='{sev_color}'><b>{sev}</b></font>", cell_bold),
                Paragraph(f"<font size=7>{bbox_str}</font>", cell_style),
                Paragraph(d.get("recommendation", "Inspect track segment."), cell_style)
            ])

    defect_table = Table(defect_table_data, colWidths=[1.3*inch, 0.9*inch, 0.9*inch, 1.7*inch, 2.4*inch])
    defect_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f766e")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    # Adjust header text color for white background
    for col in range(len(defect_table_data[0])):
        defect_table_data[0][col].style.textColor = colors.white

    story.append(defect_table)
    story.append(Spacer(1, 20))

    # 5. Footer Sign-off
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
    footer_text = Paragraph(
        "<i>This automated report was generated by the AI-Powered Smart Railway Inspection Platform using YOLOv8 Computer Vision. "
        "Verified by Authorized Railway Engineer. Confidential Document.</i>", subtitle_style
    )
    story.append(footer_text)

    doc.build(story)
    return pdf_path
