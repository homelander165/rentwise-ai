from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import extract_text_from_pdf


def main():
    pdf_path = "data/agreements/sample_agreement.pdf"

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Split the extracted text into chunks
    chunks = text_splitter.split_text(text)

    # Display the results
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print("\n" + "=" * 50)
        print(f"CHUNK {i + 1}")
        print("=" * 50)
        print(chunk)


if __name__ == "__main__":
    main()