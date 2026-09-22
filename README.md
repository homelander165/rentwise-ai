# RentWise AI

RentWise AI is a Retrieval-Augmented Generation (RAG) application designed to analyze residential rental agreements. It extracts information from PDF agreements, stores document chunks in ChromaDB, retrieves relevant clauses, performs structured clause analysis and issue detection, and provides an interactive Streamlit interface.

## Features

- PDF rental agreement ingestion
- Page-aware document extraction
- Text chunking using LangChain
- Semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage and retrieval with ChromaDB
- Rental agreement information extraction
- Structured clause analysis
- Potential issue and risk detection
- Retrieval evaluation with predefined test cases
- Interactive Streamlit web interface
- Gemini-based answer generation through the Google GenAI SDK

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
        +----------------------+
        |                      |
        v                      v
Clause Analysis        Issue Detection
        |
        v
Gemini-based Generation
        |
        v
Streamlit Application