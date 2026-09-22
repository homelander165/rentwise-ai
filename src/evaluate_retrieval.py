from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "rentwise_documents"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Number of candidates retrieved for each query
K = 3

# Same distance threshold currently used in rag_pipeline.py
DISTANCE_THRESHOLD = 1.5


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "query": "What is the monthly rent?",
        "expected_pages": [1],
    },
    {
        "query": "What is the security deposit?",
        "expected_pages": [1],
    },
    {
        "query": "When does the agreement expire?",
        "expected_pages": [1],
    },
    {
        "query": "What is the notice period?",
        "expected_pages": [2],
    },
    {
        "query": "What are the tenant's responsibilities?",
        "expected_pages": [1],
    },
    {
        "query": "What happens if rent is paid late?",
        "expected_pages": [2],
    },
]


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():
    """
    Load the existing ChromaDB vector store.

    Returns:
        Chroma: Configured ChromaDB vector store.
    """

    print("=" * 70)
    print("LOADING VECTOR STORE")
    print("=" * 70)

    print(
        f"\nChromaDB directory: {CHROMA_DIR}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # Validate ChromaDB directory
    # --------------------------------------------------------

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            "ChromaDB directory not found:\n"
            f"{CHROMA_DIR.resolve()}\n\n"
            "Run vector_store.py first."
        )

    if not CHROMA_DIR.is_dir():

        raise ValueError(
            "ChromaDB path is not a directory:\n"
            f"{CHROMA_DIR.resolve()}"
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded successfully."
    )

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    print(
        "Connected to ChromaDB."
    )

    # --------------------------------------------------------
    # Validate collection
    # --------------------------------------------------------

    count = vector_store._collection.count()

    print(
        f"Documents in ChromaDB: {count}"
    )

    if count == 0:

        raise ValueError(
            "ChromaDB collection is empty."
        )

    print(
        "\n[OK] Vector store loaded successfully."
    )

    return vector_store


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(vector_store, query):
    """
    Retrieve relevant documents for a query and apply
    the configured distance threshold.

    Args:
        vector_store: ChromaDB vector store.
        query: User/test query.

    Returns:
        list: Filtered document-distance pairs.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=K
    )

    filtered_results = []

    for document, distance in results:

        if distance <= DISTANCE_THRESHOLD:

            filtered_results.append(
                (document, distance)
            )

    return filtered_results


# ============================================================
# GET PAGE NUMBER
# ============================================================

def get_page_number(document):
    """
    Extract and normalize the page number from
    document metadata.

    Args:
        document: Retrieved LangChain document.

    Returns:
        int or None: Page number if available.
    """

    metadata = document.metadata

    page = metadata.get("page")

    if page is None:
        return None

    try:

        return int(page)

    except (ValueError, TypeError):

        return None


# ============================================================
# EVALUATE ONE QUERY
# ============================================================

def evaluate_query(
    vector_store,
    test_case,
    test_number
):
    """
    Evaluate one retrieval test case.

    A test passes when at least one retrieved document
    comes from one of the expected pages.

    Args:
        vector_store: ChromaDB vector store.
        test_case: Test case containing query and expected pages.
        test_number: Sequential test number.

    Returns:
        dict: Evaluation result.
    """

    query = test_case["query"]

    expected_pages = test_case[
        "expected_pages"
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        f"TEST {test_number}"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuery:"
    )

    print(
        query
    )

    print(
        f"\nExpected page(s): {expected_pages}"
    )

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    results = retrieve_documents(
        vector_store,
        query
    )

    # --------------------------------------------------------
    # Handle no results
    # --------------------------------------------------------

    if not results:

        print(
            "\nRetrieved documents: 0"
        )

        print(
            "RESULT: FAIL"
        )

        print(
            "Reason: No relevant document was retrieved."
        )

        return {
            "query": query,
            "expected": expected_pages,
            "retrieved": [],
            "passed": False,
        }

    # --------------------------------------------------------
    # Process retrieved documents
    # --------------------------------------------------------

    retrieved_pages = []

    print(
        f"\nRetrieved documents: "
        f"{len(results)}"
    )

    for index, (
        document,
        distance
    ) in enumerate(
        results,
        start=1
    ):

        page = get_page_number(
            document
        )

        if page is not None:

            retrieved_pages.append(
                page
            )

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        print(
            f"\nResult {index}"
        )

        print(
            "-" * 50
        )

        print(
            f"Distance : {distance:.4f}"
        )

        print(
            f"Source   : {source}"
        )

        print(
            f"Page     : {page}"
        )

        content = (
            document.page_content
            .replace("\n", " ")
            .strip()
        )

        print(
            f"Content  : {content[:250]}..."
        )

    # --------------------------------------------------------
    # Determine PASS / FAIL
    # --------------------------------------------------------

    # A test passes if at least one retrieved document
    # comes from an expected page.
    passed = any(
        page in expected_pages
        for page in retrieved_pages
    )

    if passed:

        print(
            "\nRESULT: PASS"
        )

    else:

        print(
            "\nRESULT: FAIL"
        )

        print(
            f"Expected page(s): {expected_pages}, "
            f"but retrieved: {retrieved_pages}"
        )

    return {
        "query": query,
        "expected": expected_pages,
        "retrieved": retrieved_pages,
        "passed": passed,
    }


# ============================================================
# VALIDATE TEST CASES
# ============================================================

def validate_test_cases():
    """
    Validate the configured retrieval test cases
    before running the evaluation.
    """

    if not TEST_CASES:

        raise ValueError(
            "No retrieval test cases configured."
        )

    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        if not isinstance(
            test_case,
            dict
        ):

            raise ValueError(
                f"Test case {index} is not a dictionary."
            )

        if not test_case.get("query"):

            raise ValueError(
                f"Test case {index} has no query."
            )

        if not test_case.get(
            "expected_pages"
        ):

            raise ValueError(
                f"Test case {index} has no expected pages."
            )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(results):
    """
    Display the final retrieval evaluation summary.
    """

    passed = sum(
        result["passed"]
        for result in results
    )

    total = len(results)

    failed = total - passed

    accuracy = (
        (passed / total) * 100
        if total > 0
        else 0
    )

    print(
        "\n\n"
    )

    print(
        "=" * 70
    )

    print(
        "RETRIEVAL EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"{'Test':<6}"
        f"{'Query':<42}"
        f"{'Expected':<12}"
        f"{'Retrieved':<15}"
        f"Result"
    )

    print(
        "-" * 100
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        query = result["query"]

        if len(query) > 39:

            query = (
                query[:36]
                + "..."
            )

        expected = str(
            result["expected"]
        )

        retrieved = str(
            result["retrieved"]
        )

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{index:<6}"
            f"{query:<42}"
            f"{expected:<12}"
            f"{retrieved:<15}"
            f"{status}"
        )

    print(
        "-" * 100
    )

    print(
        f"\nTotal tests : {total}"
    )

    print(
        f"Passed      : {passed}"
    )

    print(
        f"Failed      : {failed}"
    )

    print(
        f"Accuracy    : {accuracy:.2f}%"
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
    }


# ============================================================
# DISPLAY FINAL STATUS
# ============================================================

def display_final_status(summary):
    """
    Display the overall retrieval evaluation status.
    """

    passed = summary["passed"]

    total = summary["total"]

    accuracy = summary["accuracy"]

    print(
        "\n" + "=" * 70
    )

    if passed == total:

        print(
            "RETRIEVAL EVALUATION: PASS"
        )

        print(
            "All test queries retrieved "
            "the expected page."
        )

    elif accuracy >= 80:

        print(
            "RETRIEVAL EVALUATION: PARTIAL PASS"
        )

        print(
            "Most queries retrieved the expected page, "
            "but some queries need improvement."
        )

    else:

        print(
            "RETRIEVAL EVALUATION: FAIL"
        )

        print(
            "Retrieval accuracy needs improvement."
        )

    print(
        "=" * 70
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    """
    Run the complete Stage 14 retrieval evaluation.
    """

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    validate_test_cases()

    # --------------------------------------------------------
    # Load vector store
    # --------------------------------------------------------

    vector_store = load_vector_store()

    results = []

    # --------------------------------------------------------
    # Evaluation configuration
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTesting {len(TEST_CASES)} queries..."
    )

    print(
        f"Retrieval candidates (k): {K}"
    )

    print(
        f"Distance threshold: "
        f"{DISTANCE_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        result = evaluate_query(
            vector_store,
            test_case,
            test_number
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    summary = display_summary(
        results
    )

    # --------------------------------------------------------
    # Display final status
    # --------------------------------------------------------

    display_final_status(
        summary
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            "\n" + "=" * 70
        )

        print(
            "RETRIEVAL EVALUATION FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"\n{type(error).__name__}: {error}"
        )