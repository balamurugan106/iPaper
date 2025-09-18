# nlp/nlp_helpers.py
import os
import re
import pdfplumber
import docx
from typing import Optional

# Optional OCR (scanned PDFs) - requires pytesseract + pillow + tesseract installed on system
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

def extract_text_from_pdf(path: str) -> str:
    text_pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_pages.append(t)
    except Exception as e:
        # try OCR as fallback if installed
        if OCR_AVAILABLE:
            text_pages.append(_ocr_pdf(path))
        else:
            raise
    combined = "\n".join(text_pages)
    return combined

def _ocr_pdf(path: str) -> str:
    # simple OCR: convert pages to images if needed (requires pdf2image)
    from pdf2image import convert_from_path
    images = convert_from_path(path)
    texts = []
    for im in images:
        texts.append(pytesseract.image_to_string(im))
    return "\n".join(texts)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)
