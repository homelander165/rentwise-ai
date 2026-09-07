from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion import extract_pages_from_pdf


# ---------------------------------------
# 1. Load PDF
# ---------------------------------------

pdf_path = "data/agreements/sample_agreement.pdf"

pages = extract_pages_from_pdf(pdf_path)


# ---------------------------------------
# 2. Split text into chunks
# ---------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []
metadatas = []

for page in pages:

    page_chunks = text_splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append(chunk)

        metadatas.append({
            "source": "sample_agreement.pdf",
            "page": page["page"]
        })


print(f"Total chunks: {len(chunks)}")


# ---------------------------------------
# 3. Load embedding model
# ---------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------
# 4. Create vector database
# ---------------------------------------

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    metadatas=metadatas,
    collection_name="rentwise_documents",
    persist_directory="chroma_db"
)


print("Vector database created successfully!")


# ---------------------------------------
# 5. Test retrieval
# ---------------------------------------

query = "Who pays the electricity bill?"

results = vector_store.similarity_search(
    query,
    k=3
)


print("\nSEARCH RESULTS")


for i, result in enumerate(results):

    print("\n" + "=" * 50)
    print(f"RESULT {i + 1}")
    print("=" * 50)

    print("Source:", result.metadata.get("source", "Unknown"))
    print("Page:", result.metadata.get("page", "Unknown"))

    print("\nText:")
    print(result.page_content)