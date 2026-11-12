# 🏆 Solution Writeup: Orthogonal Multi-Family Logit Stacking & The Gold Medal Summit
### Kaggle Playground Series Season 6 Episode 10 (`playground-series-s6e10`)

> **Performance Summary:**  
> - **Evaluation Metric:** Area Under the ROC Curve (**ROC-AUC**)  
> - **Verified 5-Fold Cross-Validation (CV):** **`0.962240`**  
> - **Public Leaderboard (LB):** **`0.96213`** (Peak Rank 11 / Top 15 Gold Tier)  
> - **Private Leaderboard (Final Evaluation):** **`0.96238`** (Top 5.8% Global Tier)  
> - **Baseline to Summit Gain:** **$+389\text{u}$** ($+0.00389$ ROC-AUC over pure LightGBM baseline)  
> - **Environment & Compute:** Apple Silicon M-series 10-core CPU, Astral `uv`, Polars columnar caching  

---

## Conventions & Notation

- **$1\text{u} = 10^{-5}$ ROC AUC** ($0.000010$ AUC). A $+300\text{u}$ gain denotes $+0.00300$ AUC.
- **Level 1 (L1):** Standalone models generating out-of-fold (OOF) predictions across our deterministic 5-fold split.
- **Level 2 (L2):** Logit-space stacking meta-learner mapping constituent predictions to log-odds:
  $$z = \ln\left(\frac{p}{1 - p}\right)$$
- **Level 3 (L3):** Multi-pillar consensus blending optimizing Spearman rank geometry on the test distribution.
- **Strictly Leak-Free:** Feature transformations, target encodings, and meta-learners fitted strictly inside training folds without validation fold bleed ($\text{shrink\_eb} = 0.0$).

---

## Visual Summary: Milestones, Weights & Sensitivity

### Figure 1: Constituent Models & Meta-Learner Stacking Weights
![Figure 1: Multi-Family Orthogonal Ensembling & Logit Meta-Weights](docs/images/figure1_models_and_weights.png)
*Left: OOF metric comparison across fundamentally distinct algorithmic families. Right: Meta-learner logit stacking weights showing positive drivers vs contrastive de-biasers.*

### Figure 2: The Feature Dilution Effect in GBDT Decision Trees
![Figure 2: The Feature Dilution Effect in GBDTs](docs/images/figure2_feature_ablation.png)
*Waterfall of monotonic metric gains for admitted candidate feature modules versus rejected/pruned noisy features.*

### Figure 3: Full Milestone Progression from Baseline to Gold Summit (Rank 11)
![Figure 3: Full Milestone Progression from Baseline to Gold Summit](docs/images/figure3_leaderboard_progression.png)
*End-to-end CV vs LB trajectory across major engineering phases with Gold, Silver, and Bronze cutoff lines.*

### Figure 4: Regularization Sensitivity ($C$) of the Logit Meta-Learner
![Figure 4: Regularization Sensitivity of the Logit Meta-Learner](docs/images/figure4_shrinkage_sensitivity.png)
*Diagnostic sweep identifying the optimal generalization operating point and illustrating the divergence between nominal CV and true test generalization.*

### Figure 5: Public vs Private Shakeout & Monte Carlo Variance Analysis
![Figure 5: Public vs Private Shakeout & Monte Carlo Variance Analysis](docs/images/figure5_private_shakeout_monte_carlo.png)
*Hanley-McNeil / Bootstrap simulation comparing the noisy public split against the stable private test split.*

---

## Executive Summary & Core Breakthroughs

1. **The Score Breakthrough:** On a CTGAN-generated synthetic dataset of 699,635 training rows where pure tree ensembles hit a mathematical glass ceiling at **$0.9582$ – $0.9588$**, our multi-family orthogonal stacking and consensus peak architecture propelled our out-of-fold cross-validation to **`0.962240`** and public leaderboard to **`0.96213`** (+$389\text{u}$ gain over the initial baseline), securing **Rank 11** inside the Top-15 Gold Medal cutoff.
2. **The GBDT Saturation Barrier:** Standard GBDTs (LightGBM, CatBoost, XGBoost) share a mutual Pearson rank correlation of $\rho > 0.998$. Equal-weight probability or percentile rank averaging produced a negligible variance reduction of just $+2.5\text{u}$ ($+0.000025$).
3. **The "Feature Dilution Trap":** Rigorous 5-fold ablation across 136 candidate interaction features proved that blindly engineering hundreds of interaction ratios, route hashes, and digit encodings degraded CV by $-47.4\text{u}$ due to greedy node split competition. Pruning down to an unpolluted **84-dimension core** was critical.
4. **Survey Rating 0 is Structural N/A:** In 13 passenger survey ratings (1–5 scale), a score of `0` is **NOT** extreme dissatisfaction—it represents a structural *"Not Applicable / Unexperienced"* service flag. Treating zeros as `NaN` during horizontal aggregation (`mean`, `std`, `min`, `max`) prevented decision boundary distortion.
5. **Multi-Family Orthogonal Fusion:** Rather than stacking more tree variants, the breakthrough came from combining genuinely non-tree representations: **TabPFN 3.5** (Bayesian prior tabular transformer), **RealMLP 10-fold** (deep neural network with quantile-normal marginals), and **Auxiliary Expected Value (AuxEV)** models.
6. **Unbounded Logit-Space Stacking:** Stacking in unbounded log-odds space using an $L_2$-penalized Logistic Regression ($C=0.0010$) completely outperformed probability and rank blending, giving the transformer ($+0.335$) and neural network ($+0.247$) the highest positive weights.

---

## Data in Brief & Problem Formulation

| Property | Value / Specification | Engineering Implications |
| :--- | :--- | :--- |
| **Training Sample Size** | 699,635 rows | Demands sub-second Polars columnar operations and vectorization |
| **Test Sample Size** | 299,844 rows | High-powered evaluation with tight standard error ($\sigma < 0.0002$) |
| **Feature Dimensions** | 22 raw columns (4 numeric, 13 survey, 4 text categorical, 1 id) | High interaction potential but severe feature dilution vulnerability |
| **Target Distribution** | 43.3% satisfied / 56.7% dissatisfied | Balanced binary classification; smooth ROC curve calibration |
| **Data Generation** | CTGAN synthetic generation from original dataset (129,880 rows) | Synthetic artifact smoothing; original dataset acts as Super Teacher |

---

## 1. Strict Validation Architecture & Engineering Invariants

To eliminate data leakage, ensure instantaneous iteration speeds, and guarantee full reproducibility across all CPU cores, we established strict repository-wide engineering constraints:

```mermaid
flowchart TD
    Raw[Raw CSV Files] --> Ingest[Polars Ingestion & Schema Cast]
    Ingest --> Split[Deterministic Stratified 5-Fold Split]
    Split --> Cache[Downcasted Columnar Parquet Cache]
    
    Cache --> TrainFolds[Training Folds k-1]
    Cache --> ValFold[Held-out Validation Fold k]
    
    TrainFolds --> FeatFit[Fit Feature Transforms & Imputers]
    FeatFit --> ValApply[Apply Strictly Frozen Transforms]
    
    ValApply --> Invariant[Automated Gatekeeper: Zero Bleed Check]
```

1. **Pre-Computed Deterministic Folds:**  
   Stratified 5-fold partition generated once using `seed=42` and stored as an immutable `fold` metadata column inside the Parquet cache.
2. **Zero-Bleed Target Boundaries:**  
   All frequency statistics, target encodings, and meta-learners were computed exclusively on the $k-1$ training folds.
3. **Automated Pre-Flight Invariants:**  
   Automated verification gates checking null absence, schema compliance, prediction entropy, and monotonic metric improvements prior to merging any candidate model.

---

## 2. Domain Discovery: The "Rating 0 is Structural N/A" Insight

Passenger satisfaction datasets contain 13 survey questions evaluated on an apparent 0–5 integer scale:
- `Inflight wifi service`, `Departure/Arrival time convenient`, `Ease of Online booking`
- `Gate location`, `Food and drink`, `Online boarding`, `Seat comfort`
- `Inflight entertainment`, `On-board service`, `Leg room service`
- `Baggage handling`, `Checkin service`, `Cleanliness`

### The Critical Realization
In airline survey forms, `0` is selected when a passenger did **not** use or experience a service (e.g. no inflight wifi on a regional hopper, or no food served on a 45-minute commuter flight).
- When standard models treat `0` as a continuous or ordinal step below `1` (dissatisfied), they introduce massive bias: a passenger who didn't eat is falsely classified as severely unhappy with the catering.
- **The Solution:** We constructed:
  1. A structural zero boolean mask per survey feature.
  2. A global `experienced_services_count` feature ($\sum \mathbb{I}(\text{rating} > 0)$).
  3. Horizontal aggregations (`rating_mean_experienced`, `rating_std_experienced`, `rating_min_experienced`) computed strictly over experienced $(\text{rating} \ge 1)$ features.
- **Impact:** Adding this single semantic insight yielded a **$+265.0\text{u}$** jump across all cross-validation folds.

---

## 3. The Feature Dilution Trap & Scientific Ablation

Decision trees split greedily on feature gain. When hundreds of noisy features are added, informative features suffer from **split competition**: tree nodes waste depth capacity on spurious splits.

| Feature Module | Dimensionality | 5-Fold OOF ROC-AUC | Delta ($\Delta\text{u}$) | Outcome & Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Core (LightGBM)** | 22 raw | `0.958200` | $0.0\text{u}$ | Initial baseline |
| **+ Structural Zero Mask & Experienced Stats** | +15 | `0.960850` | **$+265.0\text{u}$** | **KEPT:** High-signal domain discovery |
| **+ Super Teacher Original Dataset Priors** | +4 | `0.962064` | **$+121.4\text{u}$** | **KEPT:** Regularizes CTGAN distribution |
| **+ 3-Way Customer Profile Target Interaction** | +6 | `0.962130` | **$+6.6\text{u}$** | **KEPT:** Strong non-linear interaction |
| **+ Route Corridor Frequency Encodings** | +8 | `0.962155` | **$+2.5\text{u}$** | **KEPT:** High-support corridor clustering |
| **+ Naive Pairwise Interaction Explosion** | +48 | `0.961680` | **$-47.5\text{u}$** | **PRUNED:** Node split dilution |
| **+ Unregularized Target Encoding** | +12 | `0.960380` | **$-177.5\text{u}$** | **PRUNED:** Severe validation fold leakage |

---

## 4. Multi-Family Orthogonal Ensembling

Standard GBDTs share the same tree-inductive bias. Stacking LightGBM, CatBoost, and XGBoost alone yielded minimal variance reduction. The breakthrough came from forcing algorithmic diversity:

```mermaid
flowchart TD
    Data[Gold 84-Feature Representation] --> L1_LGB[1. LightGBM GBDT<br>Fast Histogram Binning]
    Data --> L1_CAT[2. CatBoost GBDT<br>Ordered Splits & Categoricals]
    Data --> L1_XGB[3. XGBoost GBDT<br>Exact Depth Regularization]
    Data --> L1_NN[4. RealMLP Deep Neural Net<br>Quantile-Normal Marginals]
    Data --> L1_TFM[5. TabPFN Tabular Transformer<br>In-Context Bayesian Prior]
    
    L1_LGB --> OOF[Out-of-Fold Predictions Matrix]
    L1_CAT --> OOF
    L1_XGB --> OOF
    L1_NN --> OOF
    L1_TFM --> OOF
    
    OOF --> LogitProj[Logit-Odds Transformation: z = ln p / 1-p]
    LogitProj --> Stacker[Level 2 Ridge Meta-Learner C=0.001]
    Stacker --> Consensus[Level 3 Summit Consensus Blending]
    Consensus --> FinalSub[Verified Final Submissions]
```

### Level-2 Meta-Learner: Unbounded Logit Stacking
Linear probability averaging compresses near boundary regions ($p \to 0$ or $p \to 1$). By projecting all predictions into logit-odds space:
$$z_m = \ln\left(\frac{p_m}{1 - p_m}\right)$$
We fit an $L_2$-regularized Logistic Regression meta-learner:
$$\hat{p} = \sigma\left(w_0 + \sum_{m=1}^{M} w_m z_m\right)$$

The optimal regularizer was found at $C=0.0010$, assigning positive weights to complementary families:
- **TabPFN 3.5 Transformer:** $+0.335$
- **RealMLP Deep Neural Net:** $+0.247$
- **CatBoost Classifier:** $+0.218$
- **LightGBM Classifier:** $+0.198$
- **XGBoost Classifier:** $+0.185$

---

## 5. Negative Results Ledger (What Did NOT Work)

Documenting failures is essential to rigorous machine learning:

1. **Direct Dataset Concatenation ($-13.3\text{u}$):** Appending Arseniy Shutko's original 129k rows directly to training data degraded performance due to covariate distribution shift introduced by the CTGAN generator. Using original data strictly as an external *prior feature* succeeded.
2. **High-Cardinality Interaction Crosses ($-47.4\text{u}$):** Multiplying survey ratings created sparse, noisy features that triggered tree split dilution.
3. **Unregularized Target Encoding ($-177.5\text{u}$):** In-fold target encoding without heavy empirical Bayes shrinkage caused overfitting on training folds and failed to generalize to test data.
4. **Tree Monoculture Stacking ($+2.5\text{u}$):** Blending 10 slightly tuned LightGBM variants achieved almost zero variance reduction due to extreme collinearity ($\rho > 0.998$).

---

## 6. Reproducibility & Verification

The complete pipeline is 1-command reproducible using Astral's `uv` and `make`:

```bash
# Clone repository
git clone https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction.git
cd kaggle-s6e10-airline-satisfaction

# Run complete end-to-end pipeline
make all
```

*Total execution runtime: ~8 minutes on an Apple Silicon M-series 10-core CPU.*

---

## 7. Author & Acknowledgments

- **Author:** [Vijai KM](https://www.linkedin.com/) — Senior Machine Learning Engineer
- **Competition:** [Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- **License:** Released under the [MIT License](LICENSE).
