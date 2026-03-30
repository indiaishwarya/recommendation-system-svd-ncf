# Recommendation System: SVD → Neural Collaborative Filtering

**Author:** Aishwarya Indi | MS Software Engineering, SJSU  
**Dataset:** MovieLens 100K | **Framework:** PyTorch | **Run on:** Google Colab (no setup needed)

---

## Project Overview

This project implements a full recommendation pipeline progressing from classical matrix factorization to neural collaborative filtering — the same family of algorithms that powers LinkedIn's job recommendations, People You May Know, and content feed.

The domain is deliberately framed as a **job/skill recommendation problem**:
- Users → LinkedIn members
- Movies → Job postings / Skills
- Ratings → Engagement signals (views, clicks, applications, saves)

---

## Why This Problem?

LinkedIn's job feed and PYMK features are fundamentally **ranking problems over sparse interaction data**. A member has viewed a tiny fraction of all available jobs — the interaction matrix is >99% empty. Classical nearest-neighbor methods fail here because they require many overlapping interactions to compute meaningful similarity. Latent factor models and neural embeddings solve this by learning dense representations from sparse signals.

---

## Architecture Progression

### Layer 1 — SVD Matrix Factorization (Classical Baseline)

Decomposes the user-item matrix **R** into latent factor matrices:

```
R ≈ U × Σ × Vᵀ
```

- **U**: User latent factors — "what kind of content this member engages with"
- **Vᵀ**: Item latent factors — "what kind of members this job attracts"
- **k=50** latent dimensions — tuned via validation RMSE

**Tradeoff**: Assumes linear interaction between user and item factors. Fast, interpretable, strong baseline — but limited expressiveness for complex behavioral patterns.

### Layer 2 — Neural Collaborative Filtering (He et al., NeurIPS 2017)

Replaces the dot product with a Multi-Layer Perceptron:

```
User ID ──→ Embedding(64d) ──┐
                               ├──→ Concat(128d) ──→ MLP [128→64→32] ──→ Sigmoid
Item ID ──→ Embedding(64d) ──┘
```

**Key design decisions:**
- Embedding dim=64: balances expressiveness vs overfitting on sparse data
- Dropout=0.2 + BatchNorm: regularization for sparse interaction matrices
- Adam optimizer with weight decay: adapts learning rates per-parameter, converges faster on sparse gradients than SGD
- Xavier initialization: prevents vanishing gradients in deep layers

**Improvement over SVD**: Captures non-linear user-item interactions — a LinkedIn member's job preference depends on non-linear combinations of skills, seniority, location, and career trajectory that a dot product cannot model.

### Layer 3 — Ranking Evaluation with Production Metrics

RMSE measures prediction accuracy. Production recommenders optimize for **ranking quality**.

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Precision@K** | % of top-K recommendations that are relevant | Directly measures feed quality |
| **NDCG@K** | Are the best items ranked highest within K? (position-weighted) | Penalizes relevant items buried at position 8 |
| **MAP** | Average precision across all users | System-wide ranking quality |

> A model that predicts ratings perfectly but puts the best job at position 8 has failed the actual product goal.

---

## Results

| Model | RMSE | NDCG@10 | Precision@10 |
|-------|------|---------|--------------|
| SVD (k=50) | ~0.94 | baseline | baseline |
| NCF (emb=64, MLP 128→64→32) | ~0.91 | +improvement | +improvement |

*Exact numbers generated at runtime — results vary slightly by random seed.*

---

## The Cold Start Problem

Both SVD and NCF fail for **new users with no interaction history**. This is the central unsolved challenge in production recommenders.

**LinkedIn's approach (hybrid):**
1. **Content-based initialization**: Member profile (skills, education, location) → initial embedding before any interactions
2. **Popularity fallback**: Show trending jobs in the inferred field for zero-history users
3. **Progressive personalization**: Update member embeddings after 5–10 interactions
4. **Graph signals**: Connection network provides indirect signal ("members like you also viewed...")

**Next step for this project**: Incorporate item metadata (genre → job category) into a hybrid model to handle cold start gracefully.

---

## Scalability: From 100K to LinkedIn Scale

| This Project | LinkedIn Production |
|---|---|
| 943 users, 1,682 items | 1B+ members, millions of jobs |
| In-memory SVD (scipy) | Distributed training on Spark/Hadoop |
| Full item scoring at inference | Approximate Nearest Neighbor (FAISS/ScaNN) |
| Single model | Two-stage: candidate generation → ranking |
| Offline metrics | Offline metrics → A/B test → online metrics |

---

## How to Run

**Google Colab (recommended):**
1. Upload `recommender_system.ipynb` to Colab
2. Runtime → Change runtime type → GPU (optional, speeds up NCF training)
3. Run All

**Local:**
```bash
pip install torch pandas numpy scikit-learn scipy matplotlib seaborn
jupyter notebook recommender_system.ipynb
```

All data downloads automatically inside the notebook.

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `eda_analysis.png` | Rating distribution, long-tail user/item frequency |
| `training_curves.png` | NCF train vs validation loss across epochs |
| `model_comparison.png` | SVD vs NCF across Precision@K, NDCG@K, MAP@K |
| `embedding_space.png` | PCA visualization of learned item embeddings |

---

## Key Technical Concepts Demonstrated

- **Collaborative filtering** — learning from collective user behavior without item content
- **Matrix factorization** — dimensionality reduction for sparse interaction data
- **Embedding layers** — dense vector representations for discrete entities (users/items)
- **Non-linear interaction modeling** — MLP over concatenated embeddings
- **Ranking evaluation** — NDCG, MAP, Precision@K vs regression metrics
- **Cold start problem** — and hybrid mitigation strategies
- **Regularization** — dropout, weight decay, batch normalization in recommendation context
- **Scalability considerations** — two-stage retrieval, ANN search, distributed training

---

## References

- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW 2017.
- Koren, Y. et al. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- Harper & Konstan (2015). *The MovieLens Datasets*. ACM TIIS.
