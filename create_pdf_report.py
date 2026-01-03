"""
2025 Market Performance Report - PDF Generator
==============================================
Creates a professional PDF summary of the 2025 market performance analysis.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from pathlib import Path
import io

OUTPUT_DIR = Path(r'C:\Users\davet\Documents\GitHub\Industry-analysis\Data\stock_scores')

# Colors
GREEN = HexColor('#00ff88')
RED = HexColor('#ff4444')
BLUE = HexColor('#00d4ff')
DARK_BG = HexColor('#1a1a2e')
LIGHT_TEXT = HexColor('#e0e0e0')
GOLD = HexColor('#ffaa00')

def create_pdf_report():
    """Generate the PDF report."""
    
    output_path = OUTPUT_DIR / '2025_Market_Performance_Report.pdf'
    
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=28,
        textColor=HexColor('#1a5f7a'),
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=HexColor('#666666'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=HexColor('#1a5f7a'),
        spaceBefore=20,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=HexColor('#2d8659'),
        spaceBefore=15,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=black,
        spaceAfter=8,
        leading=14
    )
    
    insight_style = ParagraphStyle(
        'Insight',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#1a5f7a'),
        spaceAfter=10,
        leftIndent=20,
        borderPadding=10,
        leading=14
    )
    
    # Build content
    story = []
    
    # Title Page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("📊 2025 Market Performance Report", title_style))
    story.append(Paragraph("Comprehensive Sector & Industry Analysis", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Elwood's Trading Lab", ParagraphStyle('Author', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER, textColor=HexColor('#2d8659'))))
    story.append(Paragraph("Data through December 31, 2025", ParagraphStyle('Date', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, textColor=HexColor('#888888'))))
    
    story.append(Spacer(1, 1*inch))
    
    # Key Stats Box
    stats_data = [
        ['6,308', '146', '11', '1.4M'],
        ['Stocks Analyzed', 'Industries', 'Sectors', 'Data Points']
    ]
    
    stats_table = Table(stats_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    stats_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 20),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 1), (-1, 1), HexColor('#888888')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
    ]))
    story.append(stats_table)
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This report analyzes market performance across 6,308 stocks, 146 industries, and 11 sectors "
        "throughout 2025. Unlike simple price return metrics, this analysis provides daily granularity "
        "on breadth, momentum, rotation, and sector leadership patterns.",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Key Findings
    story.append(Paragraph("🔑 Key Findings", subheading_style))
    
    findings = [
        "<b>Financial Services</b> dominated 2025, leading the market for 21 weeks and never once being the laggard.",
        "<b>Utilities</b> led for 24 weeks (most of any sector) while maintaining positive scores year-round.",
        "<b>April was catastrophic</b> - 10 of 11 sectors had their worst month in April, with Real Estate and Energy both hitting -35.3.",
        "<b>Metals surged</b> - Copper (+101.4 YTD improvement), Aluminum (+92.1), and Silver (+75.5) showed massive turnarounds.",
        "<b>Technology underwhelmed</b> - Despite price gains, Tech never led a single week and had negative average net scores."
    ]
    
    for finding in findings:
        story.append(Paragraph(f"• {finding}", body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Sector Rankings
    story.append(Paragraph("🏆 Sector Rankings 2025", heading_style))
    
    sector_data = [
        ['Rank', 'Sector', 'Net Score', 'Breadth', 'YTD Δ', 'Weeks Leading'],
        ['1', 'Financial Services', '+15.93', '60.3%', '+14.89', '21'],
        ['2', 'Utilities', '+14.82', '58.8%', '-1.72', '24'],
        ['3', 'Basic Materials', '+6.54', '53.0%', '+34.86', '3'],
        ['4', 'Energy', '+2.04', '49.7%', '-13.50', '5'],
        ['5', 'Industrials', '+0.00', '47.5%', '+10.22', '0'],
        ['6', 'Technology', '-0.11', '46.6%', '-6.95', '0'],
        ['7', 'Communication Services', '-0.49', '45.8%', '+0.52', '0'],
        ['8', 'Real Estate', '-3.42', '45.5%', '+15.80', '0'],
        ['9', 'Healthcare', '-5.06', '43.2%', '+13.03', '0'],
        ['10', 'Consumer Cyclical', '-5.06', '43.7%', '+9.00', '0'],
        ['11', 'Consumer Defensive', '-6.34', '42.1%', '+0.88', '0'],
    ]
    
    sector_table = Table(sector_data, colWidths=[0.5*inch, 1.7*inch, 0.8*inch, 0.7*inch, 0.7*inch, 1*inch])
    sector_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f8f8'), white]),
    ]))
    story.append(sector_table)
    
    story.append(PageBreak())
    
    # Top Industries
    story.append(Paragraph("🚀 Top 15 Industries of 2025", heading_style))
    
    top_industry_data = [
        ['Rank', 'Industry', 'Sector', 'Net', 'YTD Δ'],
        ['1', 'Banks - Diversified', 'Financial Services', '+43.7', '+31.1'],
        ['2', 'Gold', 'Basic Materials', '+35.3', '+55.4'],
        ['3', 'Silver', 'Basic Materials', '+34.0', '+75.5'],
        ['4', 'Other Precious Metals', 'Basic Materials', '+31.3', '+74.8'],
        ['5', 'Utilities - Ind. Power', 'Utilities', '+25.6', '-47.1'],
        ['6', 'Utilities - Reg. Electric', 'Utilities', '+22.4', '-12.5'],
        ['7', 'Asset Management', 'Financial Services', '+20.4', '+4.8'],
        ['8', 'Utilities - Reg. Gas', 'Utilities', '+20.1', '+3.6'],
        ['9', 'Aluminum', 'Basic Materials', '+19.0', '+92.1'],
        ['10', 'Department Stores', 'Consumer Cyclical', '+18.7', '+49.3'],
        ['11', 'REIT - Healthcare', 'Real Estate', '+16.9', '+22.7'],
        ['12', 'Copper', 'Basic Materials', '+16.6', '+101.4'],
        ['13', 'Insurance - Life', 'Financial Services', '+15.1', '+12.5'],
        ['14', 'Insurance - P&C', 'Financial Services', '+13.9', '+30.1'],
        ['15', 'Banks - Regional', 'Financial Services', '+12.1', '+45.4'],
    ]
    
    top_table = Table(top_industry_data, colWidths=[0.5*inch, 1.6*inch, 1.4*inch, 0.7*inch, 0.7*inch])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d8659')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 1), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f0fff0'), white]),
    ]))
    story.append(top_table)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Insight Box
    story.append(Paragraph("📈 Biggest Turnarounds", subheading_style))
    story.append(Paragraph(
        "<b>Copper: +101.4</b> YTD improvement - From struggling to absolutely ripping! "
        "<b>Aluminum: +92.1</b> - Industrial metals party. "
        "<b>Silver: +75.5</b> - Precious metals catching bid. "
        "<b>Regional Banks: +45.4</b> - Massive recovery from 2024 lows.",
        insight_style
    ))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Bottom Industries
    story.append(Paragraph("📉 Bottom 10 Industries of 2025", heading_style))
    
    bottom_industry_data = [
        ['Rank', 'Industry', 'Sector', 'Net', 'YTD Δ'],
        ['1', 'Lumber & Wood Production', 'Basic Materials', '-26.9', '+3.9'],
        ['2', 'Textile Manufacturing', 'Consumer Cyclical', '-25.7', '-36.9'],
        ['3', 'Beverages - Wineries', 'Consumer Defensive', '-24.5', '-10.8'],
        ['4', 'Trucking', 'Industrials', '-21.9', '+30.1'],
        ['5', 'Business Equipment', 'Industrials', '-21.6', '+39.2'],
        ['6', 'Paper & Paper Products', 'Basic Materials', '-21.6', '-12.2'],
        ['7', 'Staffing & Employment', 'Industrials', '-19.6', '+2.5'],
        ['8', 'Household Products', 'Consumer Defensive', '-16.8', '-4.9'],
        ['9', 'Chemicals', 'Basic Materials', '-16.2', '+10.4'],
        ['10', 'Packaged Foods', 'Consumer Defensive', '-15.3', '+4.9'],
    ]
    
    bottom_table = Table(bottom_industry_data, colWidths=[0.5*inch, 1.6*inch, 1.4*inch, 0.7*inch, 0.7*inch])
    bottom_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#c0392b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 1), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#fff0f0'), white]),
    ]))
    story.append(bottom_table)
    
    story.append(PageBreak())
    
    # Current Momentum
    story.append(Paragraph("🔥 Current Momentum (Heading into 2026)", heading_style))
    story.append(Paragraph(
        "Analysis of the last 30 days shows which industries have the strongest momentum entering the new year.",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Hottest Industries Right Now:", subheading_style))
    hot_data = [
        ['#', 'Industry', 'Sector', 'Net Score'],
        ['1', 'Copper', 'Basic Materials', '+71.8'],
        ['2', 'Banks - Diversified', 'Financial Services', '+71.3'],
        ['3', 'Aluminum', 'Basic Materials', '+70.6'],
        ['4', 'Precious Metals & Mining', 'Basic Materials', '+69.2'],
        ['5', 'Silver', 'Basic Materials', '+66.8'],
        ['6', 'Gold', 'Basic Materials', '+59.7'],
        ['7', 'Real Estate - Diversified', 'Real Estate', '+53.7'],
        ['8', 'Banks - Regional', 'Financial Services', '+48.7'],
    ]
    
    hot_table = Table(hot_data, colWidths=[0.4*inch, 1.7*inch, 1.4*inch, 0.8*inch])
    hot_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 1), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ]))
    story.append(hot_table)
    
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Coldest Industries Right Now:", subheading_style))
    cold_data = [
        ['#', 'Industry', 'Sector', 'Net Score'],
        ['1', 'Infrastructure Operations', 'Industrials', '-42.7'],
        ['2', 'Wineries & Distilleries', 'Consumer Defensive', '-23.6'],
        ['3', 'Textile Manufacturing', 'Consumer Cyclical', '-23.6'],
        ['4', 'Household Products', 'Consumer Defensive', '-19.8'],
        ['5', 'Grocery Stores', 'Consumer Defensive', '-18.8'],
    ]
    
    cold_table = Table(cold_data, colWidths=[0.4*inch, 1.7*inch, 1.4*inch, 0.8*inch])
    cold_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 1), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ]))
    story.append(cold_table)
    
    story.append(PageBreak())
    
    # Technology Deep Dive
    story.append(Paragraph("💻 Technology Sector Deep Dive", heading_style))
    story.append(Paragraph(
        "Despite 'Tech +24.6%' headlines, the reality shows significant divergence within the sector:",
        body_style
    ))
    
    tech_data = [
        ['Industry', 'Net Score', 'Breadth', 'Verdict'],
        ['Electronic Components', '+10.6', '55.2%', 'Strong'],
        ['Communication Equipment', '+10.5', '53.5%', 'Strong'],
        ['Semiconductors', '+5.3', '52.6%', 'Bullish'],
        ['Semi Equipment', '+3.9', '53.6%', 'Bullish'],
        ['Software - Infrastructure', '-1.2', '45.6%', 'Weak'],
        ['Solar', '-2.5', '44.6%', 'Weak'],
        ['Software - Application', '-4.3', '42.5%', 'Bearish'],
        ['Consumer Electronics', '-5.9', '41.8%', 'Bearish'],
    ]
    
    tech_table = Table(tech_data, colWidths=[1.8*inch, 0.9*inch, 0.8*inch, 0.8*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
    ]))
    story.append(tech_table)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "<b>Key Insight:</b> Hardware & Semiconductors carried the sector. Software had negative breadth - "
        "meaning more stocks bearish than bullish. The 'Tech rally' was really just mega-caps masking broader weakness.",
        insight_style
    ))
    
    story.append(Spacer(1, 0.4*inch))
    
    # April Analysis
    story.append(Paragraph("📅 The April Bloodbath", heading_style))
    story.append(Paragraph(
        "April 2025 was the worst month for 10 out of 11 sectors. Here are the damage levels:",
        body_style
    ))
    
    april_data = [
        ['Sector', 'April Score', 'Recovery?'],
        ['Real Estate', '-35.3', 'Yes - now +5.0'],
        ['Energy', '-35.3', 'Yes - now +6.2'],
        ['Consumer Cyclical', '-34.5', 'Yes - now +5.2'],
        ['Industrials', '-31.0', 'Yes - now +10.2'],
        ['Healthcare', '-30.8', 'Yes - now +5.3'],
        ['Technology', '-29.5', 'Partial - now +0.5'],
        ['Communication Services', '-24.6', 'Partial - now -3.8'],
        ['Financial Services', '-20.4', 'Strong - now +29.2'],
        ['Utilities (ONLY POSITIVE)', '+7.0', 'Stable at +2.8'],
    ]
    
    april_table = Table(april_data, colWidths=[1.8*inch, 1*inch, 1.4*inch])
    april_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e67e22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('BACKGROUND', (0, -1), (-1, -1), HexColor('#d5f4e6')),
    ]))
    story.append(april_table)
    
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("📊 Methodology", heading_style))
    story.append(Paragraph(
        "<b>Data Collection:</b> Daily bullish and bearish scores collected for 6,308 stocks across all major exchanges.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Net Score:</b> Bullish Score - Bearish Score. Positive = bullish sentiment, Negative = bearish sentiment.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Breadth:</b> Percentage of stocks with bullish score > bearish score. Higher breadth = broader participation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Leadership:</b> Sector with highest net score each week is designated the 'leader'. Lowest is the 'laggard'.",
        body_style
    ))
    story.append(Paragraph(
        "<b>YTD Change:</b> Difference between December average and January average net scores.",
        body_style
    ))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    story.append(Paragraph(
        "─" * 60,
        ParagraphStyle('Line', parent=styles['Normal'], alignment=TA_CENTER, textColor=HexColor('#cccccc'))
    ))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "Elwood's Trading Lab | 2025 Market Performance Report",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=HexColor('#888888'))
    ))
    story.append(Paragraph(
        "Analysis based on 1,396,178 daily observations | Generated January 2026",
        ParagraphStyle('FooterSmall', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=HexColor('#aaaaaa'))
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_pdf_report()