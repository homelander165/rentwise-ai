import os
import json
import sys

# Force UTF-8 output on Windows to prevent UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

from dotenv import load_dotenv
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CONNECT TO EXISTING CHROMADB
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHROMA_DIRECTORY = os.path.join(
    BASE_DIR,
    "chroma_db"
)

vector_store = Chroma(
    collection_name="rentwise_documents",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIRECTORY
)


# ============================================================
# CLAUSE CATEGORIES
# ============================================================

CLAUSE_CATEGORIES = [
    "Security Deposit",
    "Rent",
    "Late Payment",
    "Lock-in Period",
    "Termination",
    "Notice Period",
    "Maintenance",
    "Repairs",
    "Utilities",
    "Rent Increase",
    "Renewal",
    "Subletting",
    "Guests",
    "Pets",
    "Property Access",
    "Restrictions",
    "Penalties",
    "Forfeiture",
    "Damage Liability",
    "Other"
]


# ============================================================
# SEARCH QUERIES
# ============================================================

REVIEW_QUERIES = [
    "security deposit refund deduction forfeiture",
    "rent payment late payment penalty",
    "lock in period termination early termination",
    "notice period vacating termination",
    "rent increase escalation revision",
    "maintenance repairs tenant landlord responsibility",
    "utilities electricity water gas charges",
    "renewal extension rental agreement",
    "subletting assignment transfer tenancy",
    "guests occupants restrictions",
    "pets animals restrictions",
    "landlord property inspection access",
    "damage liability repairs deductions",
    "penalties charges fees forfeiture",
    "tenant restrictions obligations",
]


# ============================================================
# RETRIEVE REVIEW CLAUSES
# ============================================================

def retrieve_review_clauses(k_per_query=2):

    all_results = []

    print("\n" + "=" * 60)
    print("RETRIEVING CLAUSES FOR REVIEW")
    print("=" * 60)

    for query in REVIEW_QUERIES:

        results = vector_store.similarity_search(
            query,
            k=k_per_query
        )

        all_results.extend(results)

    # --------------------------------------------------------
    # REMOVE DUPLICATE CHUNKS
    # --------------------------------------------------------

    unique_results = []
    seen = set()

    for document in all_results:

        content = document.page_content.strip()

        if content not in seen:

            seen.add(content)
            unique_results.append(document)

    print(f"\nRetrieved unique clauses: {len(unique_results)}")

    return unique_results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        content = document.page_content.strip()

        context_parts.append(
            f"""
CLAUSE ID: {i}

SOURCE FILE: {source}
PAGE: {page}

CLAUSE TEXT:
{content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ANALYZE CLAUSES
# ============================================================

def analyze_clauses(documents):

    if not documents:
        return []

    context = build_context(documents)

    categories = ", ".join(CLAUSE_CATEGORIES)

    prompt = f"""
You are RentWise AI, a rental agreement analysis assistant.

Your task is to identify clauses that a TENANT should carefully
review.

You MUST use ONLY the agreement context provided below.

Do not use outside knowledge.

Do not invent facts, clauses, amounts, dates, page numbers,
deadlines, rights, or obligations.

------------------------------------------------------------
IMPORTANT DISTINCTION
------------------------------------------------------------

Risk level describes how important the clause is for the tenant
to review.

It does NOT mean that the clause is illegal, invalid, unfair,
or unenforceable.

Never make a legal conclusion.

------------------------------------------------------------
RISK LEVELS
------------------------------------------------------------

HIGH:

The clause creates a potentially significant financial,
contractual, or practical consequence for the tenant.

MEDIUM:

The clause creates an important obligation, restriction,
condition, or ambiguity that the tenant should understand.

LOW:

The clause is relatively straightforward but may still be
useful for the tenant to review.

------------------------------------------------------------
CATEGORIES
------------------------------------------------------------

{categories}

------------------------------------------------------------
ACCURACY RULES
------------------------------------------------------------

1. Every finding MUST be directly supported by the provided text.

2. Quote the relevant clause accurately.

3. Do not change currency symbols or amounts.

4. Do not convert or reinterpret amounts.

5. Do not add information that is not present.

6. Do not assume missing information.

7. If a deadline is missing, say that the deadline is not
   specified. Do NOT invent a deadline.

8. If a clause gives the landlord a right, describe exactly
   what the clause says. Do not expand that right.

9. Do not say "legal action" unless the agreement explicitly
   says legal action.

10. Do not say a clause is "illegal", "invalid", or
    "unenforceable".

11. Recommendations must be practical and directly related
    to the identified clause.

12. Do not duplicate the same clause.

13. The source file and page MUST come from the supplied
    CLAUSE ID information.

14. If a clause is not important enough to review, do not
    include it.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return ONLY valid JSON.

The JSON must contain an object with this structure:

{{
    "clauses": [
        {{
            "clause_title": "string",
            "category": "string",
            "risk_level": "HIGH | MEDIUM | LOW",
            "clause_text": "string",
            "why_review": "string",
            "potential_concern": "string",
            "recommendation": "string",
            "source_file": "string",
            "page": "number"
        }}
    ]
}}

------------------------------------------------------------
AGREEMENT CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
RETURN JSON ONLY
------------------------------------------------------------
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    raw_text = response.text.strip()

    # --------------------------------------------------------
    # REMOVE MARKDOWN CODE FENCES IF GEMINI ADDS THEM
    # --------------------------------------------------------

    if raw_text.startswith("```json"):

        raw_text = raw_text[7:]

    elif raw_text.startswith("```"):

        raw_text = raw_text[3:]

    if raw_text.endswith("```"):

        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:

        data = json.loads(raw_text)

    except json.JSONDecodeError as error:

        print("\nERROR: Gemini returned invalid JSON.")
        print("\nRaw response:")
        print(raw_text)

        print("\nJSON error:")
        print(error)

        return []

    # --------------------------------------------------------
    # VALIDATE STRUCTURE
    # --------------------------------------------------------

    if not isinstance(data, dict):

        print("\nERROR: Gemini response is not a JSON object.")

        return []

    clauses = data.get("clauses", [])

    if not isinstance(clauses, list):

        print("\nERROR: 'clauses' is not a list.")

        return []

    return clauses


# ============================================================
# VALIDATE CLAUSE RESULTS
# ============================================================

def validate_clauses(clauses):

    valid_clauses = []

    required_fields = [
        "clause_title",
        "category",
        "risk_level",
        "clause_text",
        "why_review",
        "potential_concern",
        "recommendation",
        "source_file",
        "page"
    ]

    for index, clause in enumerate(clauses, start=1):

        if not isinstance(clause, dict):

            print(
                f"WARNING: Clause {index} is not an object."
            )

            continue

        missing_fields = [
            field
            for field in required_fields
            if field not in clause
        ]

        if missing_fields:

            print(
                f"WARNING: Clause {index} is missing: "
                f"{', '.join(missing_fields)}"
            )

            continue

        # ----------------------------------------------------
        # VALIDATE RISK LEVEL
        # ----------------------------------------------------

        risk = str(
            clause["risk_level"]
        ).upper().strip()

        if risk not in ["HIGH", "MEDIUM", "LOW"]:

            print(
                f"WARNING: Invalid risk level for clause "
                f"{index}: {risk}"
            )

            continue

        clause["risk_level"] = risk

        # ----------------------------------------------------
        # VALIDATE CATEGORY
        # ----------------------------------------------------

        category = clause["category"]

        if category not in CLAUSE_CATEGORIES:

            print(
                f"WARNING: Unknown category for clause "
                f"{index}: {category}"
            )

            clause["category"] = "Other"

        # ----------------------------------------------------
        # VALIDATE PAGE
        # ----------------------------------------------------

        try:

            clause["page"] = int(
                clause["page"]
            )

        except (ValueError, TypeError):

            print(
                f"WARNING: Invalid page for clause "
                f"{index}: {clause['page']}"
            )

            continue

        valid_clauses.append(clause)

    return valid_clauses


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(clauses):

    print("\n" + "=" * 60)
    print("CLAUSES TO REVIEW")
    print("=" * 60)

    if not clauses:

        print(
            "\nNo clauses requiring special review were identified."
        )

        return

    for index, clause in enumerate(clauses, start=1):

        risk = clause["risk_level"]

        # ASCII labels are used instead of Unicode emojis
        # to avoid Windows cp1252 encoding errors.
        if risk == "HIGH":
            icon = "[HIGH]"

        elif risk == "MEDIUM":
            icon = "[MEDIUM]"

        else:
            icon = "[LOW]"

        print(
            f"\n{index}. {clause['clause_title']}"
        )

        print(
            f"   Category: {clause['category']}"
        )

        print(
            f"   Risk Level: {icon} {risk}"
        )

        print(
            "\n   Clause:"
        )

        print(
            f"   \"{clause['clause_text']}\""
        )

        print(
            "\n   Why Review:"
        )

        print(
            f"   {clause['why_review']}"
        )

        print(
            "\n   Potential Concern:"
        )

        print(
            f"   {clause['potential_concern']}"
        )

        print(
            "\n   Recommendation:"
        )

        print(
            f"   {clause['recommendation']}"
        )

        print(
            "\n   Source: "
            f"{clause['source_file']}, "
            f"Page {clause['page']}"
        )

        print("-" * 60)


# ============================================================
# SUMMARY
# ============================================================

def display_summary(clauses):

    high = sum(
        1 for clause in clauses
        if clause["risk_level"] == "HIGH"
    )

    medium = sum(
        1 for clause in clauses
        if clause["risk_level"] == "MEDIUM"
    )

    low = sum(
        1 for clause in clauses
        if clause["risk_level"] == "LOW"
    )

    print("\n" + "=" * 60)
    print("REVIEW SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal Clauses: {len(clauses)}"
    )

    # ASCII labels are used instead of Unicode emojis
    # to avoid Windows cp1252 encoding errors.
    print(
        f"[HIGH RISK] {high}"
    )

    print(
        f"[MEDIUM RISK] {medium}"
    )

    print(
        f"[LOW RISK] {low}"
    )


# ============================================================
# SAVE JSON RESULT
# ============================================================

def save_results(clauses):

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    output_directory = os.path.join(
        base_dir,
        "data",
        "extracted"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_file = os.path.join(
        output_directory,
        "clauses_to_review.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "clauses": clauses
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nStructured results saved to:"
        f"\n{output_file}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RENTWISE AI - STAGE 12.1")
    print("STRUCTURED CLAUSE ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    documents = retrieve_review_clauses(
        k_per_query=2
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    clauses = analyze_clauses(
        documents
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    clauses = validate_clauses(
        clauses
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    display_summary(
        clauses
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    display_results(
        clauses
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    save_results(
        clauses
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 12.1 COMPLETE")
    print("=" * 60)