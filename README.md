# 🛠️ AI Debugging Assistant

A production-level Retrieval-Augmented Generation (RAG) system designed to help developers debug, document, and refactor entire codebases. You can upload a ZIP of your codebase and ask natural language questions about it.

## 🚀 Features
- **Code-Aware Chunking**: Uses language-specific processing (AST-like logic via Langchain's TextSplitters) to break down code logically without breaking functions.
- **Source Code References**: Returns exact file names and accurate line range metadata when explaining bugs.
- **Session Memory**: Uses LLM-based query rewriting on chat history for intelligent follow-up questions.
- **Hybrid Retrieval & Reranking**: Uses FAISS (vector similarity) + BM25 (keyword matching) combined via Reciprocal Rank Fusion, followed by a Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`) to pull only the exact relevant snippets.
- **Observability**: Logs queries, rewritten queries, latencies, and chunks retrieved in a local SQLite database, viewable via the `/stats` API.
- **Clean UI**: Built with Streamlit for a chatty, intuitive experience, with transparent context inspection and rerank scores.

---

## 🏗️ Architecture

## 🏗️ Architecture

```mermaid
graph TD
    A[User Uploads ZIP] --> B(FastAPI Endpoint)
    B --> C{LangChain Document Loaders}
    C --> D[Language-Specific Text Splitter + Line Number Calculation]
    D --> E[HuggingFace Embeddings]
    E --> F[(FAISS Vector Database)]
    D --> G[(BM25 Keyword Index)]

    H[User Asks Question] --> I{Rewrite using Conversation History}
    I --> J[Embed Search Query]
    J --> K{Hybrid Search}
    K -->|Weights: 0.6| F
    K -->|Weights: 0.4| G
    F --> L[Retrieve Top 15 Candidates]
    G --> L
    L --> M[Cross-Encoder Reranking]
    M --> N[Top 6 Relevant Chunks]
    N --> O[Generate Answer with LLM + Grounding Prompt]
    O --> P(Streamlit Frontend & SQlite Logging)
```

---

## 📂 Folder Structure

```
AI_Debugging_Assistant/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── ingestion.py         # Code extraction, chunking, embedding logic
│   ├── retrieval.py         # Search & LLM chain logic
│   ├── requirements.txt     # Backend dependencies
│   ├── uploads/             # Extracted code from ZIP (Gitignored)
│   └── vector_store/        # Local FAISS database (Gitignored)
├── frontend/
│   ├── app.py               # Streamlit interface
│   └── requirements.txt     # Frontend dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Setup & Installation
### Using Docker (Recommended)
1. Ensure Docker and Docker Compose are installed and running.
2. Clone the repository and navigate to its root directory.
3. Build and start the services:
   ```bash
   docker-compose up --build
   ```
4. Access the Streamlit interface at `http://localhost:8501`.

### Manual Setup (Backend)
1. Open a terminal and navigate to the `backend/` directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. If you are using local Ollama, create a `.env` file in the `backend/` directory and set the endpoint URL only if you need a non-default host or port:
   ```ini
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   ```
   If Ollama is running on the default local port, this file is optional.

   Start your Ollama server first before starting the backend:
   ```bash
   ollama serve
   ```
5. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### 2. Frontend Setup
1. Open a new terminal and navigate to the `frontend/` directory.
2. Install dependencies (you can use the same venv or a new one):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Evaluation

We evaluate the retrieval system using a standalone harness. See the before-and-after numbers comparing Vector-only vs Hybrid vs Hybrid+Reranking in our [Results Documentation](backend/eval/results.md).

---

## 🌍 Deployment Steps (Free Tier)

### 1. Vector Database
- Since FAISS stores data on disk as a folder (`vector_store/`), in a cloud environment where storage is ephemeral it will reset on deployment. For production, you can swap FAISS with **Pinecone** (free tier supports up to 100k vectors) or a persistent disk volume on Railway.

### 2. Backend (Render / Railway)
- **Render**: Create a "Web Service", connect your GitHub repo.
  - Root Directory: `backend`
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
  - Set your `GOOGLE_API_KEY` in Render Environment Variables.

### 3. Frontend (Streamlit Cloud / Vercel)
- **Streamlit Community Cloud**:
  - Point it to your repo.
  - Main file path: `frontend/app.py`
  - Ensure your `backend_url` in `app.py` points to your newly deployed Render URL instead of `localhost:8000`.

---
