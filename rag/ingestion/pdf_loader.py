import os
from pypdf import PdfReader

def load_pdf(filepath: str) -> PdfReader:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")
    return PdfReader(filepath)

def extract_pdf_text(page) -> str:
    text = page.extract_text()
    if text:
        text = " ".join(text.split())
    return text

def ingest_pdf(filepath: str):
    pdf = load_pdf(filepath)
    pdf_name = os.path.basename(filepath)

    data = []
    for i in range(len(pdf.pages)):
        text = extract_pdf_text(pdf.pages[i])
        data.append({
            "text": text,
            "metadata": {
                "source": pdf_name,
                "page": i + 1,
                "type": "pdf"
            }
        })

    return data
