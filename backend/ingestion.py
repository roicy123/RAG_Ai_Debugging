import os
import zipfile
import shutil
from pathlib import Path
import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_zip(zip_path: str, extract_to: str) -> None:
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    
    extract_to_abs = os.path.abspath(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            # mitigation for zip-slip vulnerability
            if os.path.isabs(name) or name.startswith(".."):
                raise ValueError(f"Malicious zip file containing absolute or traversal path: {name}")
            target_path = os.path.normpath(os.path.join(extract_to_abs, name))
            if os.path.commonpath([target_path, extract_to_abs]) != extract_to_abs:
                raise ValueError(f"Malicious zip file containing path traversal: {name}")
        zip_ref.extractall(extract_to)

def assign_line_numbers(split_docs: list, file_content: str) -> None:
    current_idx = 0
    for d in split_docs:
        chunk_text = d.page_content
        start_idx = file_content.find(chunk_text, current_idx)
        if start_idx == -1:
            start_idx = file_content.find(chunk_text)
            if start_idx == -1:
                start_idx = 0
                
        start_line = file_content.count('\n', 0, start_idx) + 1
        end_line = start_line + chunk_text.count('\n')
        
        d.metadata['start_line'] = start_line
        d.metadata['end_line'] = end_line
        
        if start_idx != -1:
            current_idx = start_idx + len(chunk_text)

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
    
    file_count = 0
    # Walk through the extracted codebase
    for root, dirs, files in os.walk(UPLOAD_DIR):
        # Ignore things like .git, node_modules, etc. if present
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__']]
        
        for file in files:
            file_count += 1
            if file_count > 5000:
                shutil.rmtree(UPLOAD_DIR)
                raise ValueError("Too many files extracted (>5000). To prevent pathological hangs, upload is rejected.")
                
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
                
                if len(loaded_docs) > 0:
                    file_content = loaded_docs[0].page_content
                else:
                    file_content = ""
                
                # Enhance metadata with precise source path
                for d in split_docs:
                    rel_path = os.path.relpath(file_path, UPLOAD_DIR)
                    d.metadata['source_file'] = rel_path
                    
                # Enhance metadata with line numbers
                if file_content:
                    assign_line_numbers(split_docs, file_content)
                else:
                    for d in split_docs:
                        d.metadata['start_line'] = 1
                        d.metadata['end_line'] = 1
                
                docs.extend(split_docs)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
                
    if not docs:
        raise ValueError("No understandable text/code files found in the ZIP.")
        
    # Save the chunked vectors to a local FAISS store
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_STORE_DIR)
    
    # Save the raw docs for BM25 retrieval
    docs_pkl_path = os.path.join(VECTOR_STORE_DIR, "docs.pkl")
    with open(docs_pkl_path, "wb") as f:
        pickle.dump(docs, f)
    
    return len(docs)
