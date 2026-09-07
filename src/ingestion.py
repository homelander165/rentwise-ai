import fitz


def extract_pages_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number + 1
            })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/agreements/sample_agreement.pdf"

    pages = extract_pages_from_pdf(pdf_path)

    print(f"Total pages extracted: {len(pages)}")

    for page in pages:
        print("\n" + "=" * 50)
        print(f"PAGE {page['page']}")
        print("=" * 50)
        print(page["text"][:500])