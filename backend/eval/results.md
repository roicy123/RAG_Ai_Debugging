# Retrieval Evaluation Results

Tested on 21 queries against `sample_repo.zip`.

| Configuration | Recall@6 | MRR |
|---------------|----------|-----|
| 1. Vector Only | 100.0% | 0.77 |
| 2. Hybrid (Vector + BM25) | 95.2% | 0.64 |
| 3. Hybrid + Reranking | 100.0% | 0.74 |

### Analysis

In this run using the 20-file `psf/requests` project snippet, **Hybrid+Reranking did not improve over vector-only search**.

* **Vector-only baseline** already achieved a perfect **100.0% Recall@6** and the highest **MRR (0.77)** across our 21 diverse query questions.
* **Hybrid (Vector + BM25)** actually slightly reduced Recall (95.2%) and MRR (0.64). The likely reason is that `all-MiniLM-L6-v2` was sufficient to perfectly retrieve files in this ~20 file context universe, meaning injecting BM25 lexical results primarily introduced noise (especially on parsed conceptual queries not sharing vocabulary) thereby crowding out good context hits.
* **Cross-Encoder Reranking** salvaged the noise introduced by BM25 by successfully bumping Recall back up to 100% and recovering MRR to 0.74, but it still fell marginally short of the pure vector baseline MRR.

Given the small sample size and constrained repo footprint, this signals that a heavier hybrid+rerank pipeline isn't universally "better", particularly on small to moderate codebases where dense embeddings alone can already achieve 100% recall.
