# 🧠 Enterprise Knowledge Assistant (RAG with Citations)

## 📌 Overview

The **Enterprise Knowledge Assistant** is a Retrieval-Augmented Generation (RAG) application designed to answer questions using internal company documents such as HR policies, SOPs, product manuals, and internal knowledge resources.

The system retrieves relevant document sections, ranks them, and generates grounded answers using Gemini. It provides citations for supported answers and refuses to answer when the available documents do not contain sufficient information.

The project focuses on reliable document ingestion, hybrid retrieval, grounded generation, citation validation, and secure handling of document content.

---

## ✨ Features

- 📄 **Multi-Format Document Ingestion** — Supports PDF, DOCX, Markdown, and HTML documents.
- 🔍 **OCR Support** — Extracts text from scanned PDFs using Tesseract OCR.
- ✂️ **Document Chunking** — Splits extracted content into smaller chunks for effective retrieval.
- 🧠 **Semantic Search** — Uses embeddings and ChromaDB for vector-based retrieval.
- 🔎 **Keyword Search** — Uses BM25 to retrieve documents based on keyword relevance.
- 🔗 **Hybrid Retrieval** — Combines semantic and keyword search results.
- 🎯 **Reranking** — Reorders retrieved chunks to improve relevance.
- 🤖 **LLM-Powered Answers** — Uses Gemini to generate answers from retrieved company documents.
- 📚 **Citation Support** — Provides document references and chunk information for supported answers.
- 🛡️ **Grounded Responses** — Refuses to answer questions when the available evidence is insufficient.
- 🔐 **Prompt Injection Protection** — Treats retrieved document content as untrusted data rather than instructions.
- 🔄 **Incremental Document Management** — Supports adding, updating, and deleting documents from the knowledge base.
- 🔁 **Retry Handling** — Retries eligible Gemini requests with a bounded number of attempts.
- 🚀 **FastAPI Integration** — Exposes the assistant through API endpoints.
- ✅ **Structured Responses** — Uses Pydantic schemas to validate response structures.

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │   Company Documents     │
                 │ PDF / DOCX / MD / HTML  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Document Parsing & OCR  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Text Chunking         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Embeddings & ChromaDB   │
                 └─────────────────────────┘

 User Query ────────────────┐
                            ▼
                 ┌─────────────────────────┐
                 │   Hybrid Retrieval      │
                 │ Semantic Search + BM25  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       Reranking         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Grounded Answer         │
                 │ Generation with Gemini │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Citation & Safety       │
                 │ Validation              │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Structured Response   │
                 │       through API       │
                 └─────────────────────────┘
```

---

## 📂 Project Structure

The project is organized into modules for document ingestion, retrieval, generation, API handling, and application orchestration.

```text
enterprise-knowledge-assistant/
│
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   │
│   ├── api/
│   │   ├── schemas.py
│   │   ├── ingestion.py
│   │   └── document_management.py
│   │
│   ├── ingestion/
│   │   ├── parsers.py
│   │   ├── chunker.py
│   │   └── schemas.py
│   │
│   └── retrieval/
│       ├── embeddings.py
│       ├── vector_store.py
│       ├── bm25_search.py
│       ├── hybrid_search.py
│       └── reranker.py
│
├── generation/
│   ├── answer_generator.py
│   └── schemas.py
│
├── data/
│   └── original/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure above represents the main project modules. Additional configuration, registry, utility, or local data files may exist in your actual project.*

---

## ⚙️ Processing Pipeline

### 1. Document Ingestion

The application accepts supported company documents in PDF, DOCX, Markdown, and HTML formats.

### 2. Text Extraction

- Digital PDFs are processed using PyMuPDF.
- Scanned PDFs can use OCR through Tesseract and Pytesseract.
- DOCX files are processed using `python-docx`.
- HTML documents are parsed using Beautiful Soup.
- Markdown documents are processed using the Markdown parser.

### 3. Chunking

Extracted text is divided into smaller chunks so that relevant information can be retrieved without passing entire documents to the language model.

Chunk metadata helps retain document references and source information.

### 4. Embedding and Indexing

Document chunks are converted into embeddings and stored in ChromaDB for semantic retrieval. The keyword retrieval component uses BM25.

### 5. Hybrid Retrieval and Reranking

The system combines semantic search and BM25 keyword search to retrieve relevant information. A reranking component then reorders retrieved chunks based on relevance to the query.

### 6. Answer Generation

The retrieved evidence is supplied to Gemini to generate an answer grounded in the available documents.

### 7. Citation and Safety Validation

The generated response is checked against the application's response schema and citation requirements. If the evidence is insufficient, the system returns an unsupported-answer response rather than inventing an answer.

---

## 🔎 Retrieval Approach

The application combines two retrieval methods:

| Method | Purpose |
|---|---|
| Semantic Search | Finds content based on meaning and contextual similarity. |
| BM25 Keyword Search | Finds content based on keyword relevance. |
| Hybrid Retrieval | Combines semantic and keyword retrieval. |
| Reranking | Reorders retrieved chunks to prioritize relevant evidence. |

This approach helps the assistant retrieve information when users phrase questions differently from the wording used in the original documents.

---

## 📚 Grounded Answers and Citations

The assistant is designed to answer questions using retrieved company documentation rather than relying on unsupported assumptions.

Supported answers include citation information associated with the source document and retrieved chunk.

The response schema supports structured answers and answerability information. Questions that cannot be sufficiently answered from the available evidence are refused.

The application also includes safeguards against malicious instructions embedded in retrieved documents.

---

## 🔄 Document Management

The application supports document lifecycle operations:

- **Add:** Ingest a new document into the knowledge base.
- **Update:** Refresh indexed content when a document changes.
- **Delete:** Remove a document and its associated indexed content.

These operations help keep the knowledge base aligned with the available source documents.

---

## 🚀 API Usage

The application uses FastAPI to expose the assistant's functionality.

### Run the Application

Activate your existing virtual environment and run the following command from the project root:

```bash
uvicorn app.main:app --reload
```

### Access the API Documentation

Open the following URL in your browser after the application starts:

```text
http://127.0.0.1:8000/docs
```

The interactive Swagger UI displays the API routes registered by the application, including the available query and document-management operations.

### Main Application Routes

| Route | Purpose |
|---|---|
| `/` | Root endpoint |
| `/health` | Health-check endpoint |
| `/query` | Submit a question to the knowledge assistant |
| `/docs` | Interactive API documentation |

Additional ingestion and document-management routes are available through the registered API routers. Refer to `/docs` for their exact paths and request schemas.

---

## 🔐 Environment Configuration

The application uses environment variables to manage configuration and API credentials.

Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your actual API key.

The `.env` file should remain local and must not be committed to version control.

A `.env.example` file can be shared as a configuration template without exposing real credentials.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | REST API framework |
| Pydantic | Data validation and structured responses |
| PyMuPDF | PDF text extraction |
| Tesseract OCR | Text extraction from scanned documents |
| Pytesseract | Python integration for Tesseract |
| python-docx | DOCX document parsing |
| Beautiful Soup | HTML parsing |
| Markdown | Markdown document processing |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector storage and semantic retrieval |
| Rank BM25 | Keyword-based retrieval |
| Gemini API | Grounded answer generation |
| python-dotenv | Environment variable management |
| Uvicorn | ASGI server |

**Prerequisite:** Tesseract OCR is a separate system dependency and must be installed and accessible to the application when scanned-document OCR is required.

---

## 🛡️ Security Considerations

- API keys are stored in environment variables rather than hardcoded in source files.
- The `.env` file is excluded from Git tracking through `.gitignore`.
- Retrieved document content is treated as untrusted input.
- Unsupported questions are handled without intentionally fabricating answers.
- Structured response validation helps enforce expected response fields.
- Citation checks help maintain traceability between answers and retrieved evidence.

---

## ⚠️ Limitations

- Answer quality depends on the quality and relevance of the ingested documents.
- OCR output can contain errors, particularly for low-quality scans.
- Retrieval may miss relevant information when documents are poorly structured or use unfamiliar terminology.
- Generated answers may still require review for business-critical decisions.
- The application requires a valid Gemini API key for Gemini-based answer generation.
- Tesseract must be installed separately for OCR functionality.

---

## 🔮 Future Improvements

Potential future enhancements include:

- Improving table extraction and preservation of document structure.
- Adding document-level access controls and user authentication.
- Supporting additional document formats and knowledge sources.
- Improving retrieval relevance for complex questions.
- Adding richer citation presentation and source previews.
- Supporting configurable embedding and language-model providers.

---

## 📌 Project Status

The core Enterprise Knowledge Assistant implementation is complete, including document ingestion, OCR support, hybrid retrieval, reranking, grounded answer generation, citations, document management, and API integration.

The application has been run and its existing functional and security checks have been completed successfully.

---

## 👨‍💻 Author

**Zeel Kanudawala**

**Project:** Enterprise Knowledge Assistant (RAG with Citations)
