import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "rentwise_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PERSIST_DIRECTORY = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# CHECK CHROMADB
# ============================================================

def check_chroma():
    """
    Connect to ChromaDB and display stored document chunks,
    including their source and page metadata.
    """

    print("\nLoading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # CONNECT TO CHROMADB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    print("Connected.")

    # --------------------------------------------------------
    # ACCESS COLLECTION
    # --------------------------------------------------------

    collection = vector_store._collection

    count = collection.count()

    # --------------------------------------------------------
    # DISPLAY COLLECTION INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CHROMADB COLLECTION INFORMATION")
    print("=" * 60)

    print(f"\nCollection name: {COLLECTION_NAME}")
    print(f"Total chunks stored: {count}")

    if count == 0:
        print("\nNo chunks are currently stored in ChromaDB.")
        return

    # --------------------------------------------------------
    # GET STORED DOCUMENTS
    # --------------------------------------------------------

    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    # --------------------------------------------------------
    # DISPLAY STORED CHUNKS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STORED CHUNKS")
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = (
            metadatas[index - 1]
            if index - 1 < len(metadatas)
            else {}
        )

        print(f"\n--- CHUNK {index} ---")

        print(
            f"Source: {metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: {metadata.get('page', 'Unknown')}"
        )

        print("\nText:")

        print(document[:1000])

        print("\n" + "-" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        check_chroma()

    except Exception as error:
        print("\n" + "=" * 60)
        print("CHROMADB CHECK FAILED")
        print("=" * 60)

        print(f"\nError: {error}")