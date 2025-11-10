#!/usr/bin/env python3

import os
from fpdf import FPDF

def create_pdf_from_text(text_file, output_file, title):
    """Create a PDF from a text file"""
    
    # Read the text content
    with open(text_file, 'r') as f:
        content = f.read()
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    
    # Title
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    
    # Content
    pdf.set_font('helvetica', '', 12)
    
    # Split content into lines and add to PDF
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            # Check if line is a header (contains ':' and is short)
            if ':' in line and len(line) < 50 and not line.startswith('-'):
                pdf.set_font('helvetica', 'B', 12)
                pdf.cell(0, 6, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                pdf.set_font('helvetica', '', 12)
            else:
                # Wrap long lines
                if len(line) > 80:
                    words = line.split(' ')
                    current_line = ''
                    for word in words:
                        if len(current_line + word) < 80:
                            current_line += word + ' '
                        else:
                            if current_line:
                                pdf.cell(0, 5, current_line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                            current_line = word + ' '
                    if current_line:
                        pdf.cell(0, 5, current_line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                else:
                    pdf.cell(0, 5, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
        else:
            pdf.ln(2)
    
    # Save PDF
    pdf.output(output_file)
    print(f"Sample medical PDF created: {output_file}")

def create_sample_medical_pdfs():
    """Create multiple sample medical PDF reports for testing"""
    
    reports = [
        ('sample_medical_report.txt', 'sample_medical_report.pdf', 'MEDICAL REPORT'),
        ('sample_emergency_report.txt', 'sample_emergency_report.pdf', 'EMERGENCY DEPARTMENT REPORT'),
        ('sample_cardiac_report.txt', 'sample_cardiac_report.pdf', 'CARDIOLOGY CONSULTATION REPORT')
    ]
    
    for text_file, pdf_file, title in reports:
        if os.path.exists(text_file):
            create_pdf_from_text(text_file, pdf_file, title)
        else:
            print(f"Warning: {text_file} not found")

if __name__ == "__main__":
    create_sample_medical_pdfs()
