from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


documents = [
    "The monthly rent is ₹25,000.",
    "The security deposit is ₹75,000.",
    "The tenant must provide two months written notice before termination.",
    "The tenant is responsible for electricity charges.",
]


vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model,
    collection_name="rentwise_documents",
    persist_directory="chroma_db"
)


print("Vector database created successfully!")


query = "How much is the monthly rent?"

results = vector_store.similarity_search(
    query,
    k=2
)


print("\nSearch Results:")

for result in results:
    print("--------------------")
    print(result.page_content)