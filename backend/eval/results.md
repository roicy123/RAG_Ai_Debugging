# Retrieval Evaluation Results

Tested on 15 queries against `sample_repo.zip`.

| Configuration | Recall@6 | MRR |
|---------------|----------|-----|
| 1. Vector Only | 100.0% | 0.96 |
| 2. Hybrid (Vector + BM25) | 86.7% | 0.71 |
| 3. Hybrid + Reranking | 100.0% | 0.88 |

