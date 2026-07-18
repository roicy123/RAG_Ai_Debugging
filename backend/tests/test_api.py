import os
import sys
import zipfile
import tempfile
import pytest
import requests
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def is_ollama_reachable():
    try:
        url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        requests.get(url, timeout=2)
        return True
    except:
        return False

def test_upload_api():
    # Make sure to run in a controlled working directory if possible
    # We will upload a simple working zip
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_zip = os.path.join(tmpdir, "sample.zip")
        with zipfile.ZipFile(sample_zip, 'w') as zf:
            zf.writestr("test_file.py", "def test():\n    print('Hello World')\n")
            
        with open(sample_zip, "rb") as f:
            response = client.post("/upload-code", files={"file": ("sample.zip", f, "application/zip")})
            
        assert response.status_code == 200
        data = response.json()
        assert "chunks_created" in data
        assert data["chunks_created"] > 0

@pytest.mark.skipif(not is_ollama_reachable(), reason="Ollama allows skipping dependent tests")        
def test_ask_api():
    payload = {"query": "What does test do?", "history": []}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_stats_api():
    response = client.get("/stats")
    assert response.status_code == 200
    assert "total_queries" in response.json()
