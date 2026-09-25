# RentWise AI

RentWise AI is a Retrieval-Augmented Generation (RAG) application designed to analyze residential rental agreements.

The system processes rental agreement PDFs, extracts and chunks document content, creates semantic embeddings, stores the document chunks in ChromaDB, retrieves relevant clauses, performs structured clause analysis and issue detection, and provides an interactive Streamlit interface for asking questions about the agreement.

---

## Features

- PDF rental agreement ingestion
- Page-aware document extraction
- Text chunking using LangChain
- Semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage and retrieval using ChromaDB
- Rental agreement information extraction
- Structured clause analysis
- Risk classification of agreement clauses
- Potential issue and risk detection
- Retrieval evaluation using predefined test cases
- Source and page-aware retrieval
- Interactive Streamlit web interface
- Gemini-based answer generation using the Google GenAI SDK
- Error handling for Gemini API availability, quota, authentication, and request errors

---

## Architecture

```text
Rental Agreement PDF
        |
        v
PDF Ingestion
        |
        v
Text Extraction
        |
        v
Chunking
        |
        v
Sentence Transformer Embeddings
        |
        v
ChromaDB Vector Store
        |
        v
Semantic Retrieval
        |
        +-------------------------+
        |                         |
        v                         v
Information Extraction     Structured Clause Analysis
                                  |
                                  v
                           Issue Detection
        |                         |
        +-------------+-----------+
                      |
                      v
              Gemini Generation
                      |
                      v
             Source/Page Citation
                      |
                      v
             Streamlit Application
