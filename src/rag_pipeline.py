import os

from dotenv import load_dotenv
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CHROMADB CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "rentwise_documents"


vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR
)


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(query, k=3, distance_threshold=1.5):
    """
    Retrieve relevant document chunks from ChromaDB.

    Args:
        query: User's question.
        k: Maximum number of documents to retrieve.
        distance_threshold: Maximum accepted ChromaDB distance.
            Lower distance indicates a better match.

    Returns:
        List of relevant documents.
    """

    try:
        results_with_scores = (
            vector_store.similarity_search_with_score(
                query,
                k=k
            )
        )

        if not results_with_scores:
            return []

        filtered_results = []

        for document, distance in results_with_scores:

            if distance <= distance_threshold:
                filtered_results.append(document)

        return filtered_results

    except Exception:
        return []


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query, results):
    """
    Generate an answer using retrieved agreement context
    and Gemini.

    Args:
        query: User's question.
        results: Retrieved document chunks.

    Returns:
        Generated answer or a user-friendly error message.
    """

    if not results:
        return (
            "I could not find relevant information about this "
            "question in the rental agreement."
        )


    # --------------------------------------------------------
    # BUILD AGREEMENT CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for i, document in enumerate(results, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        content = document.page_content

        context_parts.append(
            f"""
SOURCE {i}

File: {source}

Page: {page}

CONTENT:

{content}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

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

4. Do not use outside knowledge.

5. Keep the answer clear and concise.

6. When giving an answer, include the relevant source
   citation in this exact format:

   [Source: filename.pdf, Page X]

7. Only cite sources that actually support the answer.

8. If multiple sources/pages support the answer, include
   multiple citations.

9. Do not create or guess page numbers.

AGREEMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""


    # --------------------------------------------------------
    # GENERATE RESPONSE
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if not response or not response.text:
            return (
                "⚠️ I was unable to generate an answer right now. "
                "Please try again."
            )

        return response.text


    # --------------------------------------------------------
    # API ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        error_message = str(e)


        # ----------------------------------------------------
        # 503 - SERVICE UNAVAILABLE
        # ----------------------------------------------------

        if (
            "503" in error_message
            or "UNAVAILABLE" in error_message
            or "high demand" in error_message.lower()
        ):
            return (
                "⚠️ The AI service is temporarily unavailable. "
                "Please try again in a few moments."
            )


        # ----------------------------------------------------
        # 429 - RATE LIMIT / QUOTA
        # ----------------------------------------------------

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
            or "rate limit" in error_message.lower()
            or "quota" in error_message.lower()
        ):
            return (
                "⚠️ Gemini API quota has been exceeded. "
                "Please wait until the quota resets and try again."
            )


        # ----------------------------------------------------
        # 401 / 403 - AUTHENTICATION / PERMISSION
        # ----------------------------------------------------

        if (
            "401" in error_message
            or "403" in error_message
            or "PERMISSION_DENIED" in error_message
            or "UNAUTHENTICATED" in error_message
        ):
            return (
                "⚠️ The AI service could not be accessed. "
                "Please check the Gemini API configuration."
            )


        # ----------------------------------------------------
        # 400 - BAD REQUEST
        # ----------------------------------------------------

        if (
            "400" in error_message
            or "INVALID_ARGUMENT" in error_message
        ):
            return (
                "⚠️ The Gemini request could not be processed. "
                "Please try asking the question again."
            )


        # ----------------------------------------------------
        # GENERIC API ERROR
        # ----------------------------------------------------

        return (
            "⚠️ I was unable to generate an answer right now. "
            "Please try again later."
        )