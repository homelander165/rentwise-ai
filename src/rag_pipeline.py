import os

from dotenv import load_dotenv
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to existing ChromaDB
vector_store = Chroma(
    collection_name="rentwise_documents",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)


# Retrieve relevant documents
def retrieve_documents(query, k=3):

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


# Generate answer using Gemini
def generate_answer(query, documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are RentWise AI, an assistant that helps users
understand their rental agreements.

Answer the user's question using ONLY the provided
agreement context.

Rules:
1. Do not invent information.
2. Do not assume terms that are not present.
3. If the answer cannot be found in the context,
   clearly say that the information was not found
   in the agreement.
4. Keep the answer clear and concise.

AGREEMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# Main program
if __name__ == "__main__":

    query = "What is the notice period?"

    documents = retrieve_documents(query)

    answer = generate_answer(
        query,
        documents
    )

    print("\n" + "=" * 60)
    print("RENTWISE AI")
    print("=" * 60)

    print("\nQuestion:")
    print(query)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for document in documents:

        print(
            f"- {document.metadata.get('source')} "
            f"(Page {document.metadata.get('page')})"
        )