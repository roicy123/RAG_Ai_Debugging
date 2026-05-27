import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

VECTOR_STORE_DIR = "vector_store"

def get_answer(user_query: str) -> dict:
    if not os.path.exists(VECTOR_STORE_DIR):
        return {"answer": "No codebase uploaded yet. Please upload a codebase first.", "sources": [], "context": []}

    # Load embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # allow_dangerous_deserialization is required for local FAISS loads in newer LangChain versions
    vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve top-k relevant chunks (similarity search)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke(user_query)
    
    # Format retrieved chunks
    context_chunks = []
    sources = []
    
    for idx, doc in enumerate(retrieved_docs):
        source_file = doc.metadata.get("source_file", "Unknown file")
        # Support cross-file reasoning by providing multiple files in context
        if source_file not in sources:
            sources.append(source_file)
            
        context_chunks.append(f"--- Chunk {idx + 1} from '{source_file}' ---\n{doc.page_content}\n")
        
    retrieved_chunks_text = "\n".join(context_chunks)

    # Initialize Local Free LLM (Ollama)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    try:
        llm = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.2, base_url=ollama_base_url)

        prompt = PromptTemplate.from_template("""You are an expert software engineer.

Given the following code context:
{retrieved_chunks}

Answer the question:
{user_query}

Rules:
* Be precise
* Point out exact issues
* Suggest fixes with code
* Mention file names and line numbers if possible""")

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "retrieved_chunks": retrieved_chunks_text,
            "user_query": user_query
        })
    except Exception as e:
        error_message = str(e)
        if "10061" in error_message or "Connection refused" in error_message or "Failed to establish a new connection" in error_message:
            raise RuntimeError(
                f"Could not connect to Ollama at {ollama_base_url}. "
                "Ensure the local Ollama server is running and reachable, or set OLLAMA_BASE_URL to the correct URL."
            ) from e
        raise
    
    # Prepare serializable context for frontend highlighting
    context_dicts = [{"source_file": doc.metadata.get("source_file"), "content": doc.page_content} for doc in retrieved_docs]
    
    return {
        "answer": answer,
        "sources": sources,
        "context": context_dicts
    }
