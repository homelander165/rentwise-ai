import os
import sys
import json
import subprocess
import time

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AGREEMENT_DIR = os.path.join(DATA_DIR, "agreements")
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")

CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

CLAUSES_FILE = os.path.join(
    EXTRACTED_DIR,
    "clauses_to_review.json"
)

ISSUES_FILE = os.path.join(
    EXTRACTED_DIR,
    "issue_detection.json"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RentWise AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    AGREEMENT_DIR,
    exist_ok=True
)

os.makedirs(
    EXTRACTED_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 RentWise AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered rental agreement analysis and Q&A'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Agreement")

    uploaded_file = st.file_uploader(
        "Upload rental agreement",
        type=["pdf"],
        help="Upload a rental agreement in PDF format."
    )

    analyze_button = st.button(
        "🔍 Analyze Agreement",
        use_container_width=True,
        type="primary"
    )

    st.divider()

    st.caption(
        "RentWise AI analyzes your rental agreement "
        "and identifies clauses and potential issues "
        "that may require attention."
    )


# ============================================================
# HELPER: RUN PYTHON SCRIPT
# ============================================================

def run_script(script_name, arguments=None):

    if arguments is None:
        arguments = []

    script_path = os.path.join(
        SRC_DIR,
        script_name
    )

    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Script not found:\n{script_path}"
        )

    command = [
        sys.executable,
        script_path
    ]

    command.extend(arguments)

    # Make sure Python can find modules inside src/
    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        ""
    )

    if existing_pythonpath:
        environment["PYTHONPATH"] = (
            SRC_DIR
            + os.pathsep
            + existing_pythonpath
        )
    else:
        environment["PYTHONPATH"] = SRC_DIR

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment
    )

    return result


# ============================================================
# CLEAR OLD ANALYSIS
# ============================================================

def clear_old_analysis():

    for file_path in [
        CLAUSES_FILE,
        ISSUES_FILE
    ]:

        if os.path.exists(file_path):

            try:
                os.remove(file_path)

            except OSError as error:

                raise RuntimeError(
                    f"Could not remove old analysis file:\n"
                    f"{file_path}\n\n"
                    f"{error}"
                )


# ============================================================
# DISPLAY SCRIPT ERROR
# ============================================================

def show_script_error(result):

    output = ""

    if result.stdout:
        output += (
            "========== STANDARD OUTPUT ==========\n\n"
            + result.stdout
            + "\n\n"
        )

    if result.stderr:
        output += (
            "========== ERROR OUTPUT ==========\n\n"
            + result.stderr
        )

    if not output:
        output = (
            "The script failed, but returned no error output."
        )

    st.code(
        output,
        language="text"
    )


# ============================================================
# ANALYZE UPLOADED AGREEMENT
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.error(
            "Please upload a rental agreement PDF first."
        )

    else:

        agreement_path = os.path.join(
            AGREEMENT_DIR,
            uploaded_file.name
        )

        try:

            # ====================================================
            # STEP 1 — SAVE PDF
            # ====================================================

            with st.status(
                "Processing agreement...",
                expanded=True
            ) as status:

                st.write(
                    "📥 Saving uploaded PDF..."
                )

                with open(
                    agreement_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                st.write(
                    f"✅ Saved: {uploaded_file.name}"
                )


                # =================================================
                # STEP 2 — REMOVE OLD JSON RESULTS
                # =================================================

                st.write(
                    "🧹 Clearing previous analysis..."
                )

                clear_old_analysis()

                st.write(
                    "✅ Previous analysis cleared"
                )


                # =================================================
                # STEP 3 — BUILD CHROMADB
                # =================================================

                st.write(
                    "📄 Extracting PDF text and building ChromaDB..."
                )

                vector_result = run_script(
                    "vector_store.py",
                    [
                        agreement_path
                    ]
                )

                if vector_result.returncode != 0:

                    status.update(
                        label="❌ Vector database creation failed",
                        state="error",
                        expanded=True
                    )

                    st.error(
                        "The vector database could not be created."
                    )

                    show_script_error(
                        vector_result
                    )

                    st.stop()

                st.write(
                    "✅ PDF extracted and indexed"
                )


                # =================================================
                # STEP 4 — CLAUSE ANALYSIS
                # =================================================

                st.write(
                    "🔍 Analyzing rental agreement clauses..."
                )

                clause_result = run_script(
                    "clause_analysis.py"
                )

                if clause_result.returncode != 0:

                    status.update(
                        label="❌ Clause analysis failed",
                        state="error",
                        expanded=True
                    )

                    st.error(
                        "Clause analysis failed."
                    )

                    show_script_error(
                        clause_result
                    )

                    st.stop()

                st.write(
                    "✅ Clauses identified"
                )


                # =================================================
                # STEP 5 — ISSUE DETECTION
                # =================================================

                st.write(
                    "🚨 Detecting potential issues..."
                )

                issue_result = run_script(
                    "issue_detection.py"
                )

                if issue_result.returncode != 0:

                    status.update(
                        label="❌ Issue detection failed",
                        state="error",
                        expanded=True
                    )

                    st.error(
                        "Issue detection failed."
                    )

                    show_script_error(
                        issue_result
                    )

                    st.stop()

                st.write(
                    "✅ Potential issues detected"
                )


                # =================================================
                # STEP 6 — COMPLETE
                # =================================================

                status.update(
                    label="✅ Agreement analysis complete",
                    state="complete",
                    expanded=False
                )


            # ====================================================
            # STORE CURRENT AGREEMENT
            # ====================================================

            st.session_state[
                "current_agreement"
            ] = uploaded_file.name

            st.session_state[
                "analysis_complete"
            ] = True


            # ====================================================
            # RERUN TO DISPLAY RESULTS
            # ====================================================

            st.rerun()


        except Exception as error:

            st.error(
                "❌ Agreement processing failed."
            )

            st.exception(
                error
            )


# ============================================================
# LOAD CLAUSES
# ============================================================

clauses = []

if os.path.exists(CLAUSES_FILE):

    try:

        with open(
            CLAUSES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            clauses_data = json.load(file)

        clauses = clauses_data.get(
            "clauses",
            []
        )

    except Exception as error:

        st.warning(
            f"Could not read clause analysis: {error}"
        )


# ============================================================
# LOAD ISSUES
# ============================================================

issues = []

if os.path.exists(ISSUES_FILE):

    try:

        with open(
            ISSUES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            issues_data = json.load(file)

        issues = issues_data.get(
            "issues",
            []
        )

    except Exception as error:

        st.warning(
            f"Could not read issue analysis: {error}"
        )


# ============================================================
# CURRENT AGREEMENT
# ============================================================

current_agreement = st.session_state.get(
    "current_agreement",
    None
)


# ============================================================
# CURRENT AGREEMENT DISPLAY
# ============================================================

if current_agreement:

    st.success(
        f"📄 Currently analyzed: **{current_agreement}**"
    )


# ============================================================
# AGREEMENT REVIEW DASHBOARD
# ============================================================

if clauses:

    st.divider()

    st.header(
        "📊 Agreement Review"
    )


    # ========================================================
    # RISK COUNTS
    # ========================================================

    high_clauses = sum(
        1
        for clause in clauses
        if str(
            clause.get(
                "risk_level",
                ""
            )
        ).upper() == "HIGH"
    )

    medium_clauses = sum(
        1
        for clause in clauses
        if str(
            clause.get(
                "risk_level",
                ""
            )
        ).upper() == "MEDIUM"
    )

    low_clauses = sum(
        1
        for clause in clauses
        if str(
            clause.get(
                "risk_level",
                ""
            )
        ).upper() == "LOW"
    )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Clauses to Review",
            len(clauses)
        )

    with col2:

        st.metric(
            "🔴 High",
            high_clauses
        )

    with col3:

        st.metric(
            "🟠 Medium",
            medium_clauses
        )

    with col4:

        st.metric(
            "🟢 Low",
            low_clauses
        )


    # ========================================================
    # CLAUSES TO REVIEW
    # ========================================================

    st.subheader(
        "⚠️ Clauses to Review"
    )

    for index, clause in enumerate(
        clauses,
        start=1
    ):

        risk = str(
            clause.get(
                "risk_level",
                "LOW"
            )
        ).upper()

        title = clause.get(
            "clause_title",
            f"Clause {index}"
        )

        category = clause.get(
            "category",
            "Other"
        )

        if risk == "HIGH":

            icon = "🔴"

        elif risk == "MEDIUM":

            icon = "🟠"

        else:

            icon = "🟢"


        with st.expander(
            f"{icon} {title} — {risk}"
        ):

            st.write(
                f"**Category:** {category}"
            )

            st.markdown(
                "**Clause:**"
            )

            st.info(
                clause.get(
                    "clause_text",
                    "Not available."
                )
            )

            st.markdown(
                "**Why Review:**"
            )

            st.write(
                clause.get(
                    "why_review",
                    "Not available."
                )
            )

            st.markdown(
                "**Potential Concern:**"
            )

            st.write(
                clause.get(
                    "potential_concern",
                    "Not available."
                )
            )

            st.markdown(
                "**Recommendation:**"
            )

            st.write(
                clause.get(
                    "recommendation",
                    "Not available."
                )
            )

            st.caption(
                f"📄 Source: "
                f"{clause.get('source_file', 'Unknown')}, "
                f"Page {clause.get('page', 'Unknown')}"
            )


# ============================================================
# DETECTED ISSUES
# ============================================================

if issues:

    st.divider()

    st.subheader(
        "🚨 Detected Issues"
    )

    for index, issue in enumerate(
        issues,
        start=1
    ):

        severity = str(
            issue.get(
                "severity",
                "LOW"
            )
        ).upper()

        title = issue.get(
            "issue_title",
            f"Issue {index}"
        )

        if severity == "HIGH":

            icon = "🔴"

        elif severity == "MEDIUM":

            icon = "🟠"

        else:

            icon = "🟢"


        with st.expander(
            f"{icon} {title} — {severity}"
        ):

            st.write(
                f"**Type:** "
                f"{issue.get('issue_type', 'Other')}"
            )

            st.markdown(
                "**Description:**"
            )

            st.write(
                issue.get(
                    "description",
                    "Not available."
                )
            )

            st.markdown(
                "**Potential Impact:**"
            )

            st.write(
                issue.get(
                    "potential_impact",
                    "Not available."
                )
            )

            st.markdown(
                "**What to Verify:**"
            )

            st.write(
                issue.get(
                    "what_to_verify",
                    "Not available."
                )
            )

            st.markdown(
                "**Recommended Action:**"
            )

            st.write(
                issue.get(
                    "recommended_action",
                    "Not available."
                )
            )

            st.markdown(
                "**Evidence:**"
            )

            st.info(
                issue.get(
                    "evidence",
                    "Not available."
                )
            )

            st.caption(
                f"📄 Source: "
                f"{issue.get('source_file', 'Unknown')}, "
                f"Page {issue.get('page', 'Unknown')}"
            )


# ============================================================
# ASK RENTWISE AI
# ============================================================

st.divider()

st.header(
    "💬 Ask RentWise AI"
)

st.write(
    "Ask questions about the currently analyzed "
    "rental agreement."
)

query = st.text_input(
    "Your question",
    placeholder="Example: What is the security deposit?"
)

ask_button = st.button(
    "Ask Question",
    type="primary"
)


if ask_button:

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    elif not clauses:

        st.warning(
            "Please upload and analyze a rental agreement first."
        )

    else:

        try:

            # Add src directory so rag_pipeline can be imported
            if SRC_DIR not in sys.path:
                sys.path.insert(
                    0,
                    SRC_DIR
                )

            from rag_pipeline import (
                retrieve_documents,
                generate_answer
            )

            with st.spinner(
                "Searching the agreement..."
            ):

                results = retrieve_documents(
                    query,
                    k=3
                )

                answer = generate_answer(
                    query,
                    results
                )

            st.subheader(
                "💡 Answer"
            )

            st.write(
                answer
            )

            if results:

                st.subheader(
                    "📚 Retrieved Sources"
                )

                seen_sources = set()

                for document in results:

                    source = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = document.metadata.get(
                        "page",
                        "Unknown"
                    )

                    source_key = (
                        source,
                        page
                    )

                    if source_key not in seen_sources:

                        seen_sources.add(
                            source_key
                        )

                        st.caption(
                            f"📄 {source} — Page {page}"
                        )

        except Exception as error:

            st.error(
                "Unable to answer the question."
            )

            st.exception(
                error
            )


# ============================================================
# EMPTY STATE
# ============================================================

if not clauses and not current_agreement:

    st.info(
        "👈 Upload a rental agreement from the "
        "sidebar to begin your analysis."
    )