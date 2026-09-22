import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion import extract_pages_from_pdf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

CHROMA_DIRECTORY = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rentwise_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(pdf_path):
    """
    Extract PDF pages, split them into chunks, embed them,
    and store them in ChromaDB.

    Existing documents are completely removed before inserting
    the new documents.
    """

    print("\n" + "=" * 60)
    print("BUILDING VECTOR STORE")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    print(f"\nPDF: {pdf_path}")

    # --------------------------------------------------------
    # Extract PDF pages
    # --------------------------------------------------------

    print("\nExtracting pages from PDF...")

    pages = extract_pages_from_pdf(str(pdf_path))

    if not pages:
        raise ValueError(
            "No pages were extracted from the PDF."
        )

    print(f"Pages extracted: {len(pages)}")

    # --------------------------------------------------------
    # Create text splitter
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    documents = []
    metadatas = []

    print("\nCreating chunks...")

    for page_data in pages:

        # Support dictionary-style page data
        if isinstance(page_data, dict):

            page_number = (
                page_data.get("page")
                or page_data.get("page_number")
                or page_data.get("page_num")
            )

            text = (
                page_data.get("text")
                or page_data.get("content")
                or ""
            )

        # Support tuple/list-style page data
        elif isinstance(page_data, (tuple, list)):

            if len(page_data) >= 2:
                page_number = page_data[0]
                text = page_data[1]
            else:
                continue

        # Fallback
        else:
            page_number = None
            text = str(page_data)

        if not text or not text.strip():
            continue

        text = text.strip()

        chunks = text_splitter.split_text(text)

        for chunk in chunks:

            if not chunk.strip():
                continue

            documents.append(chunk.strip())

            metadatas.append(
                {
                    "source": pdf_path.name,
                    "page": page_number
                }
            )

    print(f"Total chunks created: {len(documents)}")

    if not documents:
        raise ValueError(
            "No text chunks were created from the PDF."
        )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded successfully.")

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIRECTORY
    )

    print("Connected to ChromaDB.")

    # --------------------------------------------------------
    # REMOVE OLD DOCUMENTS
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CLEARING OLD DOCUMENTS")
    print("-" * 60)

    try:

        existing_count = vector_store._collection.count()

        print(f"Existing chunks: {existing_count}")

        if existing_count > 0:

            print("Removing previous documents...")

            existing_data = vector_store._collection.get()

            existing_ids = existing_data.get("ids", [])

            if existing_ids:

                vector_store._collection.delete(
                    ids=existing_ids
                )

                print(
                    f"Removed {len(existing_ids)} previous chunks."
                )

            else:
                print("No document IDs found.")

        print("Previous documents removed successfully.")

    except Exception as error:

        raise RuntimeError(
            "Failed to clear existing ChromaDB documents.\n"
            f"Original error: {error}"
        ) from error

    # --------------------------------------------------------
    # ADD NEW DOCUMENTS
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("ADDING NEW DOCUMENTS")
    print("-" * 60)

    try:

        vector_store.add_texts(
            texts=documents,
            metadatas=metadatas
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to add documents to ChromaDB.\n"
            f"Original error: {error}"
        ) from error

    # --------------------------------------------------------
    # Verify final count
    # --------------------------------------------------------

    final_count = vector_store._collection.count()

    print("\n" + "=" * 60)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 60)

    print(f"PDF: {pdf_path.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(documents)}")
    print(f"Chunks stored in ChromaDB: {final_count}")

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if final_count != len(documents):

        raise RuntimeError(
            f"Chunk count mismatch!\n"
            f"Expected: {len(documents)}\n"
            f"Stored: {final_count}"
        )

    print("\n[OK] ChromaDB contains only the new documents.")
    print("[OK] No old chunks remain.")
    print("[OK] Vector store validation passed.")

    return vector_store


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
            "\npython src/vector_store.py "
            "data/agreements/sample_agreement.pdf"
        )

        sys.exit(1)

    pdf_file = sys.argv[1]

    try:

        build_vector_store(pdf_file)

    except Exception as error:

        print("\n" + "=" * 60)
        print("VECTOR STORE BUILD FAILED")
        print("=" * 60)

        print(f"\nError: {error}")

        sys.exit(1)