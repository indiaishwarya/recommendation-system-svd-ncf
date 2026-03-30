# Recommendation System: SVD → Neural Collaborative Filtering

**Author:** Aishwarya Indi | MS Software Engineering, SJSU  
**Dataset:** MovieLens 100K | **Framework:** PyTorch | **Run on:** Google Colab (no setup needed)

---

## Project Overview

This project implements a full recommendation pipeline progressing from classical matrix factorization to neural collaborative filtering — the same family of algorithms that powers personalized feeds, job recommendations, and content discovery at large-scale platforms.

The problem is framed as a **user-item relevance ranking problem**, generalizable across domains:

| Generic | Example Applications |
|---------|---------------------|
| Users | Platform members, customers, learners |
| Items | Jobs, products, courses, content, connections |
| Ratings | Explicit stars or implicit signals (clicks, views, saves, applies) |

The core challenge is consistent across all these domains: interaction data is extremely sparse, users have seen a tiny fraction of available items, and the system must learn meaningful preferences from that sparse signal to surface the most relevant items at the top of a personalized feed.

---

## Why This Problem?

Personalized recommendation is one of the most impactful problems in applied ML. Every major platform — job boards, streaming services, e-commerce, social networks, learning platforms — depends on ranking models to decide what a user sees next.

The challenge is fundamentally about **sparse data and ranking**, not just prediction:
- A user has interacted with a tiny fraction of all available items — the matrix is >95% empty
- Classical nearest-neighbor methods fail at this sparsity because they require many overlapping interactions
- Latent factor models and neural embeddings solve this by learning dense representations even from sparse signals
- The end goal is not predicting an exact rating but **ranking the right items at the top**

---

## Architecture Progression

### Layer 1 — SVD Matrix Factorization (Classical Baseline)

Decomposes the sparse user-item matrix **R** into latent factor matrices:

```
R ≈ U × Σ × Vᵀ
```

- **U**: User latent factors — captures each user's underlying taste profile
- **Vᵀ**: Item latent factors — captures each item's underlying characteristics
- **k=50** latent dimensions — tuned via validation RMSE

**Why it works**: Users with similar taste profiles get similar recommendations even if they have never rated the same items directly — the latent space bridges them.

**Tradeoff**: Assumes linear interaction between user and item factors. Fast, interpretable, strong baseline — but limited expressiveness for complex behavioral patterns.

### Layer 2 — Neural Collaborative Filtering (He et al., WWW 2017)

Replaces the dot product with a Multi-Layer Perceptron to model non-linear interactions:

```
User ID ──→ Embedding(64d) ──┐
                               ├──→ Concat(128d) ──→ MLP [128→64→32] ──→ Sigmoid
Item ID ──→ Embedding(64d) ──┘
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| Embedding dim=64 | Balances expressiveness vs overfitting on sparse data |
| Dropout=0.2 + BatchNorm | Regularization — sparse matrices are prone to overfitting |
| Adam + weight decay | Adapts per-parameter learning rates; converges faster than SGD on sparse gradients |
| Xavier initialization | Prevents vanishing/exploding gradients in deep embedding layers |
| Sigmoid output | Maps score to (0,1) — compatible with normalized engagement signals |

**Improvement over SVD**: Real user preferences are non-linear — a user's affinity for an item depends on complex combinations of multiple latent factors simultaneously, not just their dot product.

### Layer 3 — Ranking Evaluation with Production Metrics

RMSE measures prediction accuracy. Production recommenders optimize for **ranking quality** — whether the best items actually appear at the top of the list.

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Precision@K** | Of the top K shown, how many are actually relevant? | Directly measures feed quality the user experiences |
| **NDCG@K** | Are the most relevant items ranked highest within K? | Position-weighted — penalizes burying a great item at rank 8 |
| **MAP** | Average precision across all users | System-wide ranking quality, not just per-user |

> A model that perfectly predicts star ratings but places the most relevant item at position 8 has failed the actual product goal.

---

## Results

| Model | RMSE | NDCG@10 | Precision@10 |
|-------|------|---------|--------------|
| SVD (k=50) | ~0.94 | baseline | baseline |
| NCF (emb=64, MLP 128→64→32) | ~0.91 | +improvement | +improvement |

*Exact numbers generated at runtime and saved to `results/`. Values vary slightly by random seed.*

---

## The Cold Start Problem

Both SVD and NCF fail for **new users with no interaction history** — this is the central unsolved challenge in production recommenders.

**Common mitigation strategies:**
1. **Content-based initialization**: Use user profile attributes to generate an initial embedding before any interactions occur
2. **Popularity-based fallback**: Surface trending or highly-rated items within an inferred category for zero-history users
3. **Progressive personalization**: Begin updating personalized embeddings after a small number of early interactions
4. **Graph/social signals**: Leverage network connections as indirect preference signal

**Next step for this project**: Incorporate item metadata (genre features) into a hybrid model that combines collaborative and content signals to handle cold start gracefully.

---

## Scalability Considerations

| This Project | Production Scale |
|---|---|
| 943 users, 1,682 items | Hundreds of millions of users, millions of items |
| In-memory SVD (scipy) | Distributed training on Spark / Hadoop clusters |
| Full item scoring at inference | Approximate Nearest Neighbor search (FAISS, ScaNN) |
| Single-stage model | Two-stage pipeline: candidate generation → re-ranking |
| Offline evaluation only | Offline metrics → shadow testing → A/B experiment → online metrics |

---

## How to Run

**Google Colab (recommended — no setup needed):**
1. Upload `recommender_system.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → GPU *(optional — CPU works fine, ~3–5 min training)*
3. Runtime → Run All

**Local:**
```bash
pip install torch pandas numpy scikit-learn scipy matplotlib seaborn
jupyter notebook recommender_system.ipynb
```

Dataset downloads automatically inside the notebook. No manual steps.

---

## Generated Artifacts

After running the notebook, four charts are saved to the working directory:

| File | Description |
|------|-------------|
| `eda_analysis.png` | Rating distribution + long-tail user/item frequency plots |
| `training_curves.png` | NCF train vs validation loss across epochs |
| `model_comparison.png` | SVD vs NCF across Precision@K, NDCG@K, MAP@K |
| `embedding_space.png` | PCA 2D projection of learned item embeddings |

---

## Key Technical Concepts Demonstrated

- **Collaborative filtering** — learning from collective user behavior without requiring item content
- **Matrix factorization** — dimensionality reduction for sparse interaction data via SVD
- **Embedding layers** — dense vector representations for discrete entities (users and items)
- **Non-linear interaction modeling** — MLP over concatenated embeddings (NCF)
- **Ranking evaluation** — why NDCG, MAP, and Precision@K matter more than RMSE for recommenders
- **Cold start problem** — fundamental limitation of collaborative filtering and hybrid mitigation approaches
- **Regularization in recommendation** — dropout, weight decay, batch normalization on sparse data
- **Scalability considerations** — two-stage retrieval, ANN search, distributed training at platform scale

---

## References

- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW 2017.
- Koren, Y. et al. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- Harper & Konstan (2015). *The MovieLens Datasets: History and Context*. ACM TIIS.