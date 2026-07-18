import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import time
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

def ensemble_retrieve(retrievers, weights, query):
    docs0 = retrievers[0].invoke(query)
    docs1 = retrievers[1].invoke(query)
    scores = {}
    docs_map = {}
    for docs, weight in zip([docs0, docs1], weights):
        for rank, doc in enumerate(docs):
            if doc.page_content not in scores:
                scores[doc.page_content] = 0
                docs_map[doc.page_content] = doc
            scores[doc.page_content] += weight * (1.0 / (rank + 60))
            
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_map[c] for c, _ in sorted_docs][:15]

from logging_db import log_query

load_dotenv()

VECTOR_STORE_DIR = "vector_store"
# Weights for hybrid search: 0.6 vector, 0.4 BM25 (exact function/variable names matter in code)
HYBRID_WEIGHTS = [0.6, 0.4]

# Load reranker model at module level so it doesn't reload on every query
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_answer(user_query: str, chat_history: list[dict] | None = None) -> dict:
    if not os.path.exists(VECTOR_STORE_DIR):
        return {"answer": "No codebase uploaded yet. Please upload a codebase first.", "sources": [], "context": []}

    # Initialize Local Free LLM (Ollama) early for both rewriting and final generation
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        llm = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.2, base_url=ollama_base_url)
    except Exception as e:
        raise RuntimeError(f"Could not initialize Ollama: {e}")

    search_query = user_query
    if chat_history:
        try:
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
            rewrite_prompt = PromptTemplate.from_template("""Given the conversation history, rewrite the latest user question into a standalone, fully self-contained question suitable for a code search engine. Return only the rewritten question.
            
Conversation History:
{history_text}

Latest Question: {user_query}
""")
            rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            search_query = rewrite_chain.invoke({
                "history_text": history_text,
                "user_query": user_query
            }).strip()
        except Exception:
            pass # Fallback to original query if rewriting fails

    start_time = time.perf_counter()
    error_msg = None
    sources = []
    num_chunks = 0
    
    try:
        # Load embeddings and vector database
        # allow_dangerous_deserialization is required for local FAISS loads in newer LangChain versions
        vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
        
        docs_pkl_path = os.path.join(VECTOR_STORE_DIR, "docs.pkl")
        if not os.path.exists(docs_pkl_path):
            raise FileNotFoundError("No BM25 index found. Please re-upload your codebase (and delete vector_store/ if necessary) to enable hybrid search.")
            
        with open(docs_pkl_path, "rb") as f:
            docs = pickle.load(f)
            
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 15
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
        retrieved_docs = ensemble_retrieve(
            retrievers=[vector_retriever, bm25_retriever], 
            weights=HYBRID_WEIGHTS,
            query=search_query
        )
        
        if retrieved_docs:
            pairs = [[search_query, doc.page_content] for doc in retrieved_docs]
            scores = reranker_model.predict(pairs)
            
            # Zip docs with scores, sort descending by score, keep top 6
            scored_docs = list(zip(retrieved_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_scored_docs = scored_docs[:6]
            
            retrieved_docs = []
            for doc, score in top_scored_docs:
                doc.metadata['rerank_score'] = float(score)
                retrieved_docs.append(doc)
        
        num_chunks = len(retrieved_docs)
        
        # Format retrieved chunks
        context_chunks = []
        
        for idx, doc in enumerate(retrieved_docs):
            source_file = doc.metadata.get("source_file", "Unknown file")
            # Support cross-file reasoning by providing multiple files in context
            if source_file not in sources:
                sources.append(source_file)
                
            start_line = doc.metadata.get("start_line", "?")
            end_line = doc.metadata.get("end_line", "?")
            context_chunks.append(f"--- Chunk {idx + 1} from '{source_file}' (lines {start_line}-{end_line}) ---\n{doc.page_content}\n")
            
        retrieved_chunks_text = "\n".join(context_chunks)

        history_str = ""
        if chat_history:
            history_str = "Conversation History:\n" + "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]) + "\n\n"
        
        prompt = PromptTemplate.from_template("""You are an expert software engineer.

{history_str}Given the following code context:
{retrieved_chunks}

Answer the question:
{user_query}

Rules:
* Be precise
* Point out exact issues
* Suggest fixes with code
* Each chunk includes its file and line range — cite them exactly when pointing out issues.
* Only use the provided code context to answer. If the context does not contain enough information to answer confidently, say so explicitly instead of guessing.""")

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "history_str": history_str,
            "retrieved_chunks": retrieved_chunks_text,
            "user_query": user_query
        })
    except Exception as e:
        error_msg = str(e)
        if "10061" in error_msg or "Connection refused" in error_msg or "Failed to establish a new connection" in error_msg:
            raise RuntimeError(
                f"Could not connect to Ollama at {ollama_base_url}. "
                "Ensure the local Ollama server is running and reachable, or set OLLAMA_BASE_URL to the correct URL."
            ) from e
        raise
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        try:
            log_query(user_query, search_query, sources, latency_ms, num_chunks, error_msg)
        except Exception as log_e:
            print(f"Logging fail: {log_e}")

    
    # Prepare serializable context for frontend highlighting
    context_dicts = [
        {
            "source_file": doc.metadata.get("source_file"),
            "content": doc.page_content,
            "start_line": doc.metadata.get("start_line"),
            "end_line": doc.metadata.get("end_line"),
            "rerank_score": doc.metadata.get("rerank_score")
        }
        for doc in retrieved_docs
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "context": context_dicts
    }
