import pymupdf
import re


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_extracted_text(text):
    """
    Fix common PDF text-extraction issues.

    Example:
        I25,000 -> ₹25,000
        I75,000 -> ₹75,000
    """

    # Fix Indian Rupee symbol being extracted as capital I
    text = re.sub(r"\bI(?=\d)", "₹", text)

    return text


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pages_from_pdf(pdf_path):
    """
    Extract text from each page of a PDF.

    Returns:
        List of dictionaries containing:
        - text
        - page number
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        # Extract text from page
        text = page.get_text()

        # Normalize extracted text
        text = normalize_extracted_text(text)

        # Only keep pages containing text
        if text.strip():
            pages.append({
                "text": text,
                "page": page_number + 1
            })

    document.close()

    return pages