import os
import sys
import json
import pickle
import time

# Evaluation Corpus:
# Using the `psf/requests` repository (specifically the `src/requests` folder) as the sample repo.
# Version: Fetched via git clone --depth 1 from https://github.com/psf/requests.git (as of eval date).
# This provides a realistic 20-file Python codebase with diverse concepts.

sys.path.append(os.path.abspath("backend"))

from ingestion import ingest_codebase
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

def evaluate_retriever(retriever_func, questions):
    """Evaluates a retriever function returning recall@6 and MRR."""
    recall_hits = 0
    mrr_sum = 0
    k = 6
    
    for q in questions:
        query = q["question"]
        expected_files = q["expected_files"]
        
        docs = retriever_func(query)
        top_k = docs[:k]
        
        hit = False
        rank = -1
        for i, doc in enumerate(top_k):
            # Normalizing path styles
            source_file = doc.metadata.get("source_file", "").replace("\\", "/")
            if any(ef in source_file for ef in expected_files):
                hit = True
                rank = i + 1
                break
        
        if hit:
            recall_hits += 1
            mrr_sum += 1.0 / rank

    recall = recall_hits / len(questions)
    mrr = mrr_sum / len(questions)
    return recall, mrr

def run_eval():
    print("Ingesting sample_repo.zip...")
    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
    if os.path.exists("backend/vector_store"):
        import shutil
        shutil.rmtree("backend/vector_store")
    
    # Needs to be called from the directory where VECTOR_STORE_DIR is expected, or change dir.
    # For ingestion.py to work correctly, let's change CWD to backend.
    orig_dir = os.getcwd()
    os.chdir("backend")
    ingest_codebase("eval/sample_repo.zip")
    
    # Load index artifacts
    VECTOR_STORE_DIR = "vector_store"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    
    with open(os.path.join(VECTOR_STORE_DIR, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
        
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 15
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Load questions
    with open("eval/test_questions.json", "r") as f:
        questions = json.load(f)

    # 1. Vector-only
    print("Running Vector-only eval...")
    def vector_only(q):
        return vectorstore.as_retriever(search_kwargs={"k": 6}).invoke(q)
    v_recall, v_mrr = evaluate_retriever(vector_only, questions)
    
    # 2. Hybrid-only
    print("Running Hybrid eval...")
    def hybrid_only(q):
        r = ensemble_retrieve([vector_retriever, bm25_retriever], [0.6, 0.4], q)
        return r[:6]
    h_recall, h_mrr = evaluate_retriever(hybrid_only, questions)
    
    # 3. Hybrid + Reranking
    print("Running Hybrid+Reranking eval...")
    def hybrid_rerank(q):
        r = ensemble_retrieve([vector_retriever, bm25_retriever], [0.6, 0.4], q)
        pairs = [[q, doc.page_content] for doc in r]
        scores = reranker_model.predict(pairs)
        scored = list(zip(r, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:6]]
        
    hr_recall, hr_mrr = evaluate_retriever(hybrid_rerank, questions)
    
    os.chdir(orig_dir)
    
    # Write results
    results_md = f"""# Retrieval Evaluation Results

Tested on {len(questions)} queries against `sample_repo.zip`.

| Configuration | Recall@6 | MRR |
|---------------|----------|-----|
| 1. Vector Only | {v_recall*100:.1f}% | {v_mrr:.2f} |
| 2. Hybrid (Vector + BM25) | {h_recall*100:.1f}% | {h_mrr:.2f} |
| 3. Hybrid + Reranking | {hr_recall*100:.1f}% | {hr_mrr:.2f} |

"""
    with open("backend/eval/results.md", "w") as f:
        f.write(results_md)
    print("Evaluation completed. Written to backend/eval/results.md")

if __name__ == "__main__":
    run_eval()
