import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from ingestion import ingest_codebase
from retrieval import get_answer

app = FastAPI(title="AI Debugging Assistant API")

# Allow Streamlit or any frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload-code")
async def upload_code(file: UploadFile = File(...)):
    """Accepts a ZIP file of the codebase, extracts it, and creates vector embeddings."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")
        
    temp_zip_path = f"temp_{file.filename}"
    
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Ingest the codebase
        docs_processed = ingest_codebase(temp_zip_path)
    except Exception as e:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    # Clean up the zip file
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)
    
    return {"message": "Codebase successfully uploaded and processed.", "chunks_created": docs_processed}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Answers user queries based on the vectorized codebase using RAG."""
    try:
        result = get_answer(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-files")
async def list_files():
    """Returns a list of all parsed and extracted files."""
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        return {"files": []}
        
    all_files = []
    for root, _, files in os.walk(upload_dir):
        # Ignore pycache or hidden dirs
        if "__pycache__" in root or "/." in root.replace("\\", "/"):
            continue
            
        for f in files:
            if not f.startswith('.'):
                file_path = os.path.relpath(os.path.join(root, f), upload_dir)
                all_files.append(file_path)
            
    return {"files": all_files}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
