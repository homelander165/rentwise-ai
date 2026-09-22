import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_COLLECTION_NAME = "rentwise_documents"
CHROMA_PERSIST_DIRECTORY = BASE_DIR / "chroma_db"

PDF_PATH = BASE_DIR / "data" / "agreements" / "sample_agreement.pdf"

OUTPUT_DIRECTORY = BASE_DIR / "data" / "extracted"
OUTPUT_FILE = OUTPUT_DIRECTORY / "agreement_analysis.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini model
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# GLOBAL RESOURCES
# ============================================================

client = None
vector_store = None


# ============================================================
# VALIDATE PROJECT PATHS
# ============================================================

def validate_paths():
    """
    Validate all important project paths before
    running the extraction pipeline.
    """

    print("\n" + "=" * 60)
    print("VALIDATING PROJECT PATHS")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate rental agreement PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Rental agreement PDF not found:\n"
            f"{PDF_PATH.resolve()}"
        )

    if not PDF_PATH.is_file():
        raise ValueError(
            f"PDF path is not a file:\n"
            f"{PDF_PATH.resolve()}"
        )

    if PDF_PATH.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, but found:\n"
            f"{PDF_PATH.name}"
        )

    print(f"[OK] Agreement PDF: {PDF_PATH}")

    # --------------------------------------------------------
    # Validate ChromaDB directory
    # --------------------------------------------------------

    if not CHROMA_PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found:\n"
            f"{CHROMA_PERSIST_DIRECTORY.resolve()}\n\n"
            f"Run vector_store.py first."
        )

    if not CHROMA_PERSIST_DIRECTORY.is_dir():
        raise ValueError(
            f"ChromaDB path is not a directory:\n"
            f"{CHROMA_PERSIST_DIRECTORY.resolve()}"
        )

    print(
        f"[OK] ChromaDB directory: "
        f"{CHROMA_PERSIST_DIRECTORY}"
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"[OK] Output directory: "
        f"{OUTPUT_DIRECTORY}"
    )

    print(
        "\nProject paths validated successfully."
    )


# ============================================================
# VALIDATE GEMINI MODEL
# ============================================================

def validate_gemini_model():
    """
    Verify that the configured Gemini model is available
    with the current API key.
    """

    print("\n" + "=" * 60)
    print("VALIDATING GEMINI MODEL")
    print("=" * 60)

    if not GEMINI_MODEL:
        raise ValueError(
            "GEMINI_MODEL is empty."
        )

    if client is None:
        raise RuntimeError(
            "Gemini client has not been initialized."
        )

    try:
        available_models = list(
            client.models.list()
        )

        available_model_names = set()

        for model in available_models:
            model_name = getattr(
                model,
                "name",
                None
            )

            if model_name:
                available_model_names.add(
                    model_name.removeprefix(
                        "models/"
                    )
                )

        if GEMINI_MODEL not in available_model_names:

            print(
                "\nConfigured model was not found."
            )

            print(
                f"Configured model: {GEMINI_MODEL}"
            )

            print(
                "\nAvailable Gemini models:"
            )

            for model_name in sorted(
                available_model_names
            ):
                print(
                    f"  - {model_name}"
                )

            raise ValueError(
                f"\nGemini model "
                f"'{GEMINI_MODEL}' is not available "
                f"for this API key."
            )

        print(
            f"[OK] Gemini model available: "
            f"{GEMINI_MODEL}"
        )

    except ValueError:
        raise

    except Exception as error:
        raise RuntimeError(
            "Could not validate Gemini model.\n"
            f"Original error: {error}"
        ) from error


# ============================================================
# INITIALIZE RESOURCES
# ============================================================

def initialize_resources():
    """
    Load environment variables, initialize the Gemini client,
    embedding model, and ChromaDB connection.
    """

    global client
    global vector_store

    # --------------------------------------------------------
    # Load environment variables
    # --------------------------------------------------------

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file."
        )

    # --------------------------------------------------------
    # Initialize Gemini client
    # --------------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Validate project paths
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # Validate Gemini model
    # --------------------------------------------------------

    validate_gemini_model()

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Connect to existing ChromaDB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(
            CHROMA_PERSIST_DIRECTORY
        )
    )

    print("Connected to ChromaDB.")

    # --------------------------------------------------------
    # Validate ChromaDB collection
    # --------------------------------------------------------

    validate_chroma_collection()


# ============================================================
# VALIDATE CHROMADB COLLECTION
# ============================================================

def validate_chroma_collection():
    """
    Verify that the ChromaDB collection exists
    and contains document chunks.
    """

    print("\nValidating ChromaDB collection...")

    if vector_store is None:
        raise RuntimeError(
            "ChromaDB has not been initialized."
        )

    try:
        count = vector_store._collection.count()

    except Exception as error:
        raise RuntimeError(
            "Unable to access the ChromaDB collection.\n"
            f"Original error: {error}"
        ) from error

    if count == 0:
        raise ValueError(
            f"ChromaDB collection "
            f"'{CHROMA_COLLECTION_NAME}' "
            f"contains no documents.\n\n"
            f"Run vector_store.py first."
        )

    print(
        f"[OK] ChromaDB collection: "
        f"{CHROMA_COLLECTION_NAME}"
    )

    print(
        f"[OK] Stored document chunks: {count}"
    )


# ============================================================
# RETRIEVAL QUERIES
# ============================================================

EXTRACTION_QUERIES = {

    "financial": [
        "monthly rent and rental payment amount",
        "security deposit amount",
        "maintenance charges",
        "rent increase rent escalation",
        "late payment penalty charges"
    ],

    "agreement": [
        "lease agreement start date",
        "lease agreement end date",
        "lease duration period",
        "renewal of lease agreement"
    ],

    "termination": [
        "notice period required for termination",
        "early termination of lease",
        "termination conditions"
    ],

    "responsibilities": [
        "maintenance and repair responsibility",
        "electricity bill responsibility",
        "water bill responsibility"
    ]
}


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query, k=3):
    """
    Retrieve the most relevant document chunks from ChromaDB.
    """

    if vector_store is None:
        raise RuntimeError(
            "ChromaDB has not been initialized."
        )

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


# ============================================================
# RETRIEVE ALL RELEVANT AGREEMENT CONTEXT
# ============================================================

def retrieve_agreement_context(k=3):
    """
    Run multiple targeted retrieval queries and combine
    the retrieved chunks.

    Duplicate chunks are removed.
    """

    print("\nRetrieving agreement information...")

    all_documents = []

    # --------------------------------------------------------
    # Run each retrieval query
    # --------------------------------------------------------

    for category, queries in EXTRACTION_QUERIES.items():

        print(
            f"\n[{category.upper()}]"
        )

        for query in queries:

            print(
                f"  -> {query}"
            )

            documents = retrieve_documents(
                query,
                k=k
            )

            print(
                f"    Retrieved "
                f"{len(documents)} chunks"
            )

            all_documents.extend(
                documents
            )

    # --------------------------------------------------------
    # Remove duplicate chunks
    # --------------------------------------------------------

    unique_documents = []

    seen_content = set()

    for document in all_documents:

        content = document.page_content.strip()

        if not content:
            continue

        if content not in seen_content:

            unique_documents.append(
                document
            )

            seen_content.add(
                content
            )

    print(
        f"\nTotal retrieved chunks: "
        f"{len(all_documents)}"
    )

    print(
        f"Total unique chunks: "
        f"{len(unique_documents)}"
    )

    # --------------------------------------------------------
    # Build context for Gemini
    # --------------------------------------------------------

    context_parts = []

    for index, document in enumerate(
        unique_documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
--- DOCUMENT CHUNK {index} ---
Source: {source}
Page: {page}

{document.page_content}
"""
        )

    context = "\n".join(
        context_parts
    )

    return context, unique_documents


# ============================================================
# BUILD GEMINI EXTRACTION PROMPT
# ============================================================

def build_extraction_prompt(context):
    """
    Build the prompt used to extract structured information
    from the retrieved rental agreement context.
    """

    prompt = f"""
You are RentWise AI, an AI assistant specialized in
analyzing rental and lease agreements.

Your task is to extract factual information from the
provided rental agreement context.

IMPORTANT RULES:

1. Use ONLY information explicitly present in the
   provided context.

2. NEVER use outside knowledge.

3. NEVER guess or infer missing information.

4. If information is not explicitly present, return null.

5. Preserve the meaning and wording of the agreement
   as accurately as possible.

6. Do not combine unrelated clauses.

7. If different parts of the context contain conflicting
   information, describe the conflict rather than choosing
   a value yourself.

8. Return ONLY valid JSON.

9. Do not use Markdown.

10. Do not put ```json or ``` around the response.

11. Dates should be returned exactly as stated in the
    agreement whenever possible.

12. Monetary values should preserve the currency and
    amount stated in the agreement.

13. For electricity_responsibility, return ONLY the
    responsible party, such as "Tenant" or "Landlord".

14. For water_responsibility, return ONLY the
    responsible party, such as "Tenant" or "Landlord".

15. For maintenance_responsibility, clearly distinguish
    routine maintenance from major structural repairs
    when both are specified.

16. late_payment_penalty should contain ONLY an actual
    monetary fee or penalty.

17. If the agreement does NOT specify a monetary
    late-payment penalty, return null for
    late_payment_penalty.

18. late_payment_condition should describe the condition
    that triggers late-payment action.

19. late_payment_action should describe what the landlord
    or tenant may do after that condition occurs.

20. Do not convert a late-payment condition into a
    monetary penalty.

21. If a field is not explicitly stated, return null.

22. Return exactly the JSON structure provided below.


EXTRACT THE FOLLOWING INFORMATION:

FINANCIAL INFORMATION:

- monthly_rent
- security_deposit
- maintenance_charges
- rent_increase
- late_payment_penalty
- late_payment_condition
- late_payment_action


AGREEMENT INFORMATION:

- lease_start_date
- lease_end_date
- lease_duration
- renewal_terms


TERMINATION INFORMATION:

- notice_period
- early_termination


RESPONSIBILITIES:

- maintenance_responsibility
- electricity_responsibility
- water_responsibility


RETURN EXACTLY THIS JSON STRUCTURE:

{{
    "monthly_rent": null,
    "security_deposit": null,
    "maintenance_charges": null,
    "rent_increase": null,

    "late_payment_penalty": null,
    "late_payment_condition": null,
    "late_payment_action": null,

    "lease_start_date": null,
    "lease_end_date": null,
    "lease_duration": null,
    "renewal_terms": null,

    "notice_period": null,
    "early_termination": null,

    "maintenance_responsibility": null,
    "electricity_responsibility": null,
    "water_responsibility": null
}}


RENTAL AGREEMENT CONTEXT:

{context}
"""

    return prompt


# ============================================================
# SEND CONTEXT TO GEMINI
# ============================================================

def analyze_with_gemini(context):
    """
    Send the retrieved agreement context to Gemini
    and return the raw response text.
    """

    if client is None:
        raise RuntimeError(
            "Gemini client has not been initialized."
        )

    prompt = build_extraction_prompt(
        context
    )

    print(
        "\nSending agreement context to Gemini..."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


# ============================================================
# PARSE GEMINI JSON
# ============================================================

def parse_gemini_json(response_text):
    """
    Convert Gemini's response into a Python dictionary.

    Handles accidental Markdown code fences as a safety measure.
    """

    response_text = response_text.strip()

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    if response_text.startswith("```json"):

        response_text = response_text[
            len("```json"):
        ]

    elif response_text.startswith("```"):

        response_text = response_text[
            len("```"):
        ]

    if response_text.endswith("```"):

        response_text = response_text[
            :-len("```")
        ]

    response_text = response_text.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            response_text
        )

    except json.JSONDecodeError as error:

        print(
            "\n" + "=" * 60
        )

        print(
            "ERROR: INVALID JSON FROM GEMINI"
        )

        print(
            "=" * 60
        )

        print(
            "\nRaw Gemini response:"
        )

        print(
            response_text
        )

        raise ValueError(
            "Gemini returned invalid JSON."
        ) from error

    # --------------------------------------------------------
    # Validate JSON type
    # --------------------------------------------------------

    if not isinstance(data, dict):

        raise ValueError(
            "Gemini JSON response is not an object."
        )

    return data


# ============================================================
# EXPECTED FIELDS
# ============================================================

EXPECTED_FIELDS = [

    "monthly_rent",
    "security_deposit",
    "maintenance_charges",
    "rent_increase",

    "late_payment_penalty",
    "late_payment_condition",
    "late_payment_action",

    "lease_start_date",
    "lease_end_date",
    "lease_duration",
    "renewal_terms",

    "notice_period",
    "early_termination",

    "maintenance_responsibility",
    "electricity_responsibility",
    "water_responsibility"

]


# ============================================================
# VALIDATE EXTRACTED DATA
# ============================================================

def validate_extracted_data(data):
    """
    Make sure all expected fields exist.

    If Gemini accidentally omits a field, add it with null.
    """

    validated_data = {}

    for field in EXPECTED_FIELDS:

        if field in data:

            validated_data[field] = data[field]

        else:

            validated_data[field] = None

    return validated_data


# ============================================================
# SAVE EXTRACTED INFORMATION
# ============================================================

def save_extracted_information(data):
    """
    Save the structured agreement analysis as JSON.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    return OUTPUT_FILE


# ============================================================
# DISPLAY EXTRACTED INFORMATION
# ============================================================

def display_extracted_information(data):
    """
    Print extracted information in a readable format.
    """

    print("\n")

    print(
        "=" * 60
    )

    print(
        "EXTRACTED AGREEMENT INFORMATION"
    )

    print(
        "=" * 60
    )

    for field, value in data.items():

        readable_field = field.replace(
            "_",
            " "
        ).title()

        print(
            f"\n{readable_field}:"
        )

        if value is None:

            print(
                "  Not specified"
            )

        else:

            print(
                f"  {value}"
            )

    print(
        "\n" + "=" * 60
    )


# ============================================================
# MAIN EXTRACTION PIPELINE
# ============================================================

def extract_agreement_information():
    """
    Run the complete agreement extraction pipeline.
    """

    # --------------------------------------------------------
    # STEP 0: Initialize resources
    # --------------------------------------------------------

    initialize_resources()

    # --------------------------------------------------------
    # STEP 1: Retrieve relevant chunks
    # --------------------------------------------------------

    context, documents = retrieve_agreement_context(
        k=3
    )

    if not context.strip():

        raise ValueError(
            "No relevant agreement information "
            "was retrieved from ChromaDB."
        )

    # --------------------------------------------------------
    # STEP 2: Analyze with Gemini
    # --------------------------------------------------------

    raw_response = analyze_with_gemini(
        context
    )

    # --------------------------------------------------------
    # STEP 3: Convert Gemini response to JSON
    # --------------------------------------------------------

    extracted_data = parse_gemini_json(
        raw_response
    )

    # --------------------------------------------------------
    # STEP 4: Validate fields
    # --------------------------------------------------------

    extracted_data = validate_extracted_data(
        extracted_data
    )

    # --------------------------------------------------------
    # STEP 5: Save JSON
    # --------------------------------------------------------

    output_path = save_extracted_information(
        extracted_data
    )

    # --------------------------------------------------------
    # Return extraction results
    # --------------------------------------------------------

    return {
        "data": extracted_data,
        "documents": documents,
        "raw_response": raw_response,
        "output_path": output_path
    }


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "RENTWISE AI - AGREEMENT ANALYSIS"
    )

    print(
        "=" * 60
    )

    try:

        result = extract_agreement_information()

        # ----------------------------------------------------
        # Display extracted information
        # ----------------------------------------------------

        display_extracted_information(
            result["data"]
        )

        # ----------------------------------------------------
        # Display saved file
        # ----------------------------------------------------

        print(
            "\nAnalysis saved to:"
        )

        print(
            result["output_path"]
        )

        # ----------------------------------------------------
        # Display JSON
        # ----------------------------------------------------

        print(
            "\nJSON OUTPUT:"
        )

        print(
            json.dumps(
                result["data"],
                indent=4,
                ensure_ascii=False
            )
        )

    except Exception as error:

        print(
            "\n" + "=" * 60
        )

        print(
            "ERROR"
        )

        print(
            "=" * 60
        )

        print(
            f"\n{type(error).__name__}: {error}"
        )