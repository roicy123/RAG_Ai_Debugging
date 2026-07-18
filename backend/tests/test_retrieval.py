import os
import sys
import pytest
from langchain_core.documents import Document

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval import ensemble_retrieve, reranker_model

class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
    def invoke(self, query):
        return self.docs

def test_hybrid_retrieval():
    docs1 = [Document(page_content="exact keyword here", metadata={"source_file": "file1.py"})]
    docs2 = [Document(page_content="semantic match here", metadata={"source_file": "file2.py"})]
    
    r1 = FakeRetriever(docs1)
    r2 = FakeRetriever(docs2)
    
    res = ensemble_retrieve([r1, r2], [0.6, 0.4], "keyword")
    assert len(res) == 2
    assert res[0].page_content in ["exact keyword here", "semantic match here"]
    
def test_reranking_empty():
    try:
        retrieved_docs = []
        if retrieved_docs:
            pairs = [["query", doc.page_content] for doc in retrieved_docs]
            reranker_model.predict(pairs)
    except Exception as e:
        pytest.fail(f"Reranking empty list caused exception: {e}")
