
# Unsupervised Learning, Dimensionality Reduction & SVM-Based Classification

This repository contains a clean, modular implementation of three essential machine learning workflows:

- A **custom K-Means clustering algorithm** with benchmark comparisons
- An in-house **Principal Component Analysis (PCA)** module for high-dimensional data reduction
- A complete **SVM classification pipeline** applied to real-world Pokémon data

Each component is built to simulate how such tasks would be handled in practical machine learning and data science projects — from scratch implementations to comparisons with production-grade libraries.

---

## Use Cases

- Rapid prototyping of ML algorithms
- Educational tool for understanding unsupervised learning and PCA
- Demonstrating classification and model selection on real-world data
- Benchmarking custom implementations against `scikit-learn`

---

## Project Features

### 1. Custom K-Means Clustering

- Implements K-Means from scratch with convergence criteria and centroid updates
- Benchmarked against `sklearn.cluster.KMeans`
- Includes a visualization module to illustrate cluster separability

**Synthetic Dataset Generated From:**
- Resting: μ = [60, 10], Σ = [[20, 100], [100, 20]]
- Stressed: μ = [100, 80], Σ = [[50, 20], [20, 50]]

_Visual comparisons are included to verify clustering quality._

---

### 2. Principal Component Analysis (PCA)

- Custom PCA supporting:
  - Centering and scaling
  - Eigen decomposition
  - Dimensionality selection based on cumulative explained variance
- Validated against `sklearn.decomposition.PCA`
- Applied on the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)

> Automatically determines the optimal number of principal components to retain 95% variance.

---

### 3. Pokémon Legendary Classification (SVM)

- End-to-end classification pipeline:
  - Data loading & cleaning
  - Feature engineering
  - Model training, evaluation, and hyperparameter tuning
- Excludes all identifier fields (`name`, `name2`) to prevent data leakage
- Uses `sklearn.svm.SVC` with **grid search** over kernel and regularization parameters

> The classifier predicts whether a Pokémon is **Legendary** or **Non-Legendary**, leveraging combat stats and type information.

---

## Folder Structure

```
.
├── src/
│   ├── kmeans_custom.py           # Custom KMeans implementation
│   ├── pca_custom.py              # Custom PCA with threshold-based reduction
│   ├── svm_pipeline.py            # SVM training pipeline for Pokémon dataset
│   └── utils.py                   # Utility functions (standardization, plotting, etc.)
├── notebooks/
│   ├── kmeans_vs_sklearn.ipynb    # Visual comparison between custom & sklearn KMeans
│   ├── pca_projection.ipynb       # PCA projection with Iris dataset
│   └── pokemon_classifier.ipynb   # Pokémon SVM training, evaluation, and tuning
├── data/
│   ├── pokemon_dataset.csv        # Pokémon dataset with labeled legendary status
├── requirements.txt
└── README.md                      # You're here
```

---

## Getting Started

Clone the repo and install dependencies:

```bash
git clone https://github.com/-.git
cd -
pip install -r requirements.txt
```

Run the notebooks in `notebooks/` for a visual walkthrough, or execute scripts in `src/` to directly test models.

---

## Key Design Decisions

- **Modular Codebase:** Each algorithm is decoupled for testing, extension, and reuse.
- **Realistic Validation:** Every custom implementation is benchmarked against Scikit-learn’s production-grade version.
- **Explorability:** Each notebook includes interactive plotting, step-by-step logic, and data inspection.
- **Avoiding Leakage:** All classification models exclude identifying metadata or target-correlated features.

---

## Tech Stack

- Python 3.8+
- Numpy, Pandas, Matplotlib
- Scikit-learn
- Jupyter Notebook

---
