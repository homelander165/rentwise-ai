from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion import extract_pages_from_pdf


pdf_path = "data/agreements/sample_agreement.pdf"


pages = extract_pages_from_pdf(pdf_path)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = []

for page in pages:

    page_chunks = text_splitter.split_text(page["text"])

    for chunk in page_chunks:
        chunks.append({
            "text": chunk,
            "page": page["page"]
        })


print(f"Total chunks created: {len(chunks)}")


for i, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 50)
    print(f"CHUNK {i + 1}")
    print(f"PAGE: {chunk['page']}")
    print("=" * 50)

    print(chunk["text"])