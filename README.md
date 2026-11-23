# 🏆 Kaggle S6E10: Airline Passenger Satisfaction Champion Solution

[![CI & Code Quality](https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction/actions/workflows/ci.yml/badge.svg)](https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction/actions/workflows/ci.yml)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Showcase-38bdf8.svg?style=flat&logo=github)](https://vijaikmathew.github.io/kaggle-s6e10-airline-satisfaction/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-261230.svg?style=flat&logo=python)](https://github.com/astral-sh/uv)
[![Kaggle Rank](https://img.shields.io/badge/Kaggle-Top%201%25%20Gold%20Medal-gold.svg)](https://www.kaggle.com/competitions/playground-series-s6e10)
[![Evaluation Metric](https://img.shields.io/badge/ROC--AUC-0.96342-success.svg)](#)

> Complete, leak-free, 1-command reproducible machine learning pipeline for the Kaggle Playground Series S6E10 competition, achieving **Top 1% (Gold Medal Tier)** out of 2,400+ international teams.
>
> 🌐 **[View Live GitHub Pages Showcase](https://vijaikmathew.github.io/kaggle-s6e10-airline-satisfaction/)** | 📖 **[Read the Full Gold-Tier Writeup](WRITEUP.md)**

---

## 📌 Executive Summary

Predicting airline passenger satisfaction requires extracting subtle interaction effects between service touchpoints while maintaining strict generalization against synthetic data quirks.

### Core Breakthroughs:
1. **Domain Discovery:** Identified that satisfaction rating `0` corresponds to structural non-experience rather than extreme dissatisfaction; separating structural zeros unlocked a $+45\text{u}$ gain.
2. **Multi-Family Diversity:** Combined tree-based models (LightGBM, CatBoost, XGBoost) with Deep Tabular Neural Networks (Residual MLP) to ensure orthogonal prediction residuals.
3. **Unbounded Logit Meta-Stacking:** Replaced naive probability averaging with an $L_2$-regularized Ridge meta-learner operating in unconstrained logit space ($z = \ln(p / (1 - p))$).
4. **Anti-Shakeout Validation:** Implemented a fixed 5-fold stratified cross-validation strategy with zero validation leakage, completely protecting the solution against private leaderboard shakeout.

---

## 📚 Deep-Dive Technical Documentation

| Document | Focus & Key Insights |
| :--- | :--- |
| 📖 **[WRITEUP.md](WRITEUP.md)** | **Full Gold-Tier Solution Writeup:** GBDT saturation barrier, structural zero semantic discovery, logit meta-stacking mathematics, and private leaderboard dominance. |
| 📓 **[notebooks/champion-pipeline/PS_S6E10_Champion_Pipeline.ipynb](notebooks/champion-pipeline/PS_S6E10_Champion_Pipeline.ipynb)** | **Publication Walkthrough Notebook:** End-to-end interactive demo reproducing feature generation, 5-fold cross-validation, and submission ensembling. |
| 🏛️ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | **Systems Architecture Specification:** Ingestion schemas, Polars memory downcasting, Level-1 diverse model topologies, Level-2 logit mechanics, and pre-flight verification gates. |
| 🔬 **[docs/ABLATION_STUDY.md](docs/ABLATION_STUDY.md)** | **Ablation & Failure Modes:** Feature ablation waterfall, empirical proof of the *Feature Dilution Trap*, regularization sensitivity sweeps ($C$), and tombstoned negative results. |
| 🌐 **[Live HTML Solution Preview](https://htmlpreview.github.io/?https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction/blob/main/reports/solution_preview.html)** | **Interactive Showcase:** Modern light-theme dashboard rendering metrics, architecture highlights, and ablation tables. |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    %% Styling
    classDef input fill:#f8fafc,stroke:#94a3b8,stroke-width:2px,color:#0f172a;
    classDef domain fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0f172a;
    classDef models fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#0f172a;
    classDef meta fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#0f172a;
    classDef out fill:#f1f5f9,stroke:#475569,stroke-width:2px,color:#0f172a;

    Raw["Raw Kaggle Ingestion via CLI"] --> Split["Strict 5-Fold Stratified Split (Seed 42)"]
    Split --> FeatEng["Domain Feature Engineering<br>• Structural Zero Semantic Masks<br>• External Super-Teacher Priors<br>• 3-Way Customer Interaction Profiles"]
    
    FeatEng --> M1["LightGBM GBDT (Hist Splitting)"]
    FeatEng --> M2["CatBoost GBDT (Ordered Categoricals)"]
    FeatEng --> M3["XGBoost GBDT (Exact Depth Reg)"]
    FeatEng --> M4["Tabular ResMLP (Quantile Normals)"]
    
    M1 --> OOF["5-Fold Out-of-Fold Matrix"]
    M2 --> OOF
    M3 --> OOF
    M4 --> OOF
    
    OOF --> Logit["Logit-Odds Projection: z = ln(p / 1-p)"]
    Logit --> Stacker["Level 2 Ridge Meta-Learner (C=0.001)"]
    Stacker --> FinalSub["Verified Final Submission (submission.csv)"]
    Stacker --> HTMLRep["Live Interactive Showcase (GitHub Pages)"]

    class Raw,Split input;
    class FeatEng domain;
    class M1,M2,M3,M4 models;
    class OOF,Logit,Stacker meta;
    class FinalSub,HTMLRep out;
```

---

## 📊 Validation & Leaderboard Progression

| Phase / Model Component | 5-Fold OOF ROC-AUC | Public LB | Private LB | Net Gain |
| :--- | :---: | :---: | :---: | :---: |
| **0. Raw Baseline (Tuned LightGBM)** | `0.95820` | `0.95790` | `0.95810` | Baseline |
| **1. + Domain & Structural Zero Features** | `0.96085` | `0.96040` | `0.96075` | $+265\text{u}$ |
| **2. + Multi-Family Orthogonal Models** | `0.96210` | `0.96180` | `0.96205` | $+125\text{u}$ |
| **3. + Level 2 Logit Meta-Stacker** | **`0.96342`** | **`0.96315`** | **`0.96338`** | **$+132\text{u}$** |

---

## 🚀 Quickstart & Reproducibility (via `make` & `uv`)

### 1. Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) and the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
pip install kaggle
```

### 2. Complete 1-Command Replication
```bash
git clone https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction.git
cd kaggle-s6e10-airline-satisfaction

# Run full pipeline: setup environment, fetch data, train, evaluate, generate submission and HTML preview
make all
```

*Individual targets are also available:*
- `make setup` — Configure virtualenv and sync locked dependencies
- `make data` — Download and extract dataset via Kaggle CLI
- `make train` — Fit Level-1 constituent models
- `make evaluate` — Audit OOF metrics and verification gates
- `make predict` — Fit meta-stacker and output `submission.csv`
- `make report` — Render standalone `reports/solution_preview.html`

---

## 💡 What Worked vs. What Didn't

### What Worked:
- **Structural zero indicator flags** for survey fields.
- **Logit-space stacking** instead of simple weighted probability averaging.
- **Aggressive feature pruning:** Ablating 32 noisy interaction features improved private score by $+18\text{u}$.

### What Didn't:
- **Target Encoding without shrinkage:** Caused validation fold leakage and degraded test score.
- **Stacking 10 similar LightGBM variants:** High collinearity ($\rho > 0.99$) yielded zero variance reduction.
- **KNN features:** Excessive memory footprint with negligible predictive gain.

---

## 👤 Author & Acknowledgments

- **Author:** [Vijai KM](https://www.linkedin.com/) — Senior Machine Learning Engineer
- **Competition:** [Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- **License:** Released under the [MIT License](LICENSE).
