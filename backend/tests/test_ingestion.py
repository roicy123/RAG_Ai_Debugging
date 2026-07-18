import os
import zipfile
import tempfile
import pytest
import sys

# Ensure backend imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion import extract_zip, get_text_splitter_for_ext, assign_line_numbers
from langchain_community.document_loaders import TextLoader

def test_zip_slip_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        malicious_zip = os.path.join(tmpdir, "evil.zip")
        with zipfile.ZipFile(malicious_zip, 'w') as zf:
            zf.writestr("../../evil.txt", "evil content")
            
        with pytest.raises(ValueError, match="traversal"):
            extract_zip(malicious_zip, os.path.join(tmpdir, "extract"))

def test_chunking_line_numbers():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "sample.py")
        code = "def parse():\n    pass\n\n\n\ndef main():\n    parse()\n"
        with open(test_file, 'w') as f:
            f.write(code)
            
        loader = TextLoader(test_file)
        docs = loader.load()
        splitter = get_text_splitter_for_ext(".py")
        split_docs = splitter.split_documents(docs)
        
        file_content = docs[0].page_content
        assign_line_numbers(split_docs, file_content)
            
        assert len(split_docs) > 0
        assert split_docs[0].metadata['start_line'] == 1
        assert split_docs[-1].metadata['end_line'] == 7
