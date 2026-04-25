import os
import zipfile
import shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"

def extract_zip(zip_path: str, extract_to: str) -> None:
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_text_splitter_for_ext(ext: str):
    """Returns a code-aware text splitter based on the file extension."""
    mapping = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".cpp": Language.CPP,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".html": Language.HTML,
    }
    lang = mapping.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=800, chunk_overlap=150
        )
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

def ingest_codebase(zip_path: str) -> int:
    """Extracts ZIP, chunks code files, and stores embeddings in FAISS."""
    extract_zip(zip_path, UPLOAD_DIR)
    
    docs = []
    
    # Walk through the extracted codebase
    for root, dirs, files in os.walk(UPLOAD_DIR):
        # Ignore things like .git, node_modules, etc. if present
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__']]
        
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # Skip non-text or extremely large files
            if ext in ['.png', '.jpg', '.jpeg', '.zip', '.exe', '.dll', '.so']:
                continue
                
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
                
                # Split using code-aware chunking
                splitter = get_text_splitter_for_ext(ext)
                split_docs = splitter.split_documents(loaded_docs)
                
                # Enhance metadata with precise source path
                for d in split_docs:
                    rel_path = os.path.relpath(file_path, UPLOAD_DIR)
                    d.metadata['source_file'] = rel_path
                
                docs.extend(split_docs)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
                
    if not docs:
        raise ValueError("No understandable text/code files found in the ZIP.")
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Save the chunked vectors to a local FAISS store
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_DIR)
    
    return len(docs)
