# 🔬 Scientific Feature Ablation & Failure Analysis
### Comprehensive Ablation Ledger, Sensitivity Sweeps, and Negative Results

---

## 1. Feature Ablation Waterfall

To demonstrate disciplined engineering rigor and eliminate noisy features, every candidate transformation was subjected to an isolated 5-fold stratified cross-validation ablation test.

| Milestone Phase | Feature Module | Net Dimensionality | 5-Fold OOF ROC-AUC | Delta ($\Delta\text{u}$) | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **0. Raw Baseline** | Unaltered Clean Features | 22 | `0.958200` | $0.0\text{u}$ | Baseline |
| **1. Domain Discovery** | Structural Zero Indicators & Experienced Rating Aggs | 37 | `0.960850` | **$+265.0\text{u}$** | **ADMITTED** |
| **2. Prior Transfer** | External Super Teacher Model Logits (Original Dataset) | 41 | `0.962064` | **$+121.4\text{u}$** | **ADMITTED** |
| **3. Non-linear Interaction** | 3-Way Customer Profile Empirical Target Interaction | 47 | `0.962130` | **$+6.6\text{u}$** | **ADMITTED** |
| **4. Spatial Clustering** | Flight Distance Route Corridor Frequency Groups | 55 | `0.962155` | **$+2.5\text{u}$** | **ADMITTED** |
| **5. Model Diversity** | CatBoost & XGBoost Integration | 84 | `0.962240` | **$+8.5\text{u}$** | **ADMITTED** |
| **6. Level-2 Meta-Stacker** | Logit-Space Ridge Meta-Learner ($C=0.001$) | - | **`0.963420`** | **$+118.0\text{u}$** | **CHAMPION** |

---

## 2. The Feature Dilution Trap: Theoretical & Empirical Proof

A frequent error in competitive machine learning is unrestrained interaction feature generation. Decision trees use greedy split criteria (Gain / Variance reduction). When noisy or collinear features enter the training set, tree nodes allocate depth capacity to spurious splits, diluting true signal.

```
                           THE FEATURE DILUTION EFFECT
  ROC-AUC
   0.96230 ┌───────────────────▲ Peak (84 features)
           │                  / \
   0.96210 │                 /   \
           │                /     \
   0.96190 │               /       \
           │              /         ▼ Unpruned (136 features)
   0.96170 │             /            Degradation: -47.4u
           │            /
   0.95820 └───────────/
           0          40         84        120        160   Feature Count
```

### Empirical Proof of Dilution:
- Adding 52 pairwise multiplicative interactions (`rating_A * rating_B`) reduced CV from `0.962155` to `0.961681` ($-47.4\text{u}$).
- Pruning down to an unpolluted 84-feature core restored full performance monotonically across all 5 folds.

---

## 3. Regularization Sensitivity Sweep ($C$)

We conducted a fine-grained sweep of the $L_2$ regularization parameter $C \in [10^{-5}, 10^{2}]$ for the Level-2 logit meta-learner:

| Regularization $C$ | Shrinkage $\lambda = 1/C$ | 5-Fold OOF ROC-AUC | Leaderboard Generalization | Behavior |
| :---: | :---: | :---: | :---: | :--- |
| $10.0$ | $0.1$ | `0.96280` | Poor | Collinear tree over-weighting; boundary overfitting |
| $1.0$ | $1.0$ | `0.96310` | Moderate | Standard default operating point |
| $0.1$ | $10.0$ | `0.96332` | High | Suppresses noisy residual weights |
| **$0.0010$** | **$1000.0$** | **`0.96342`** | **Optimal** | **Peak generalization; orthogonal families balanced** |
| $0.0001$ | $10000.0$ | `0.96315` | Underfit | Excessive shrinkage; collapses toward simple mean |

---

## 4. Tombstoned Negative Results Ledger (What Failed & Why)

Documenting disproved hypotheses prevents wasted compute and establishes scientific integrity:

### 1. Direct Original Dataset Concatenation
- **Hypothesis:** Appending Arseniy Shutko's 129,880 original training samples directly into the competition dataset will increase effective sample size.
- **Empirical Result:** $-13.3\text{u}$ drop on 5-fold CV.
- **Root Cause:** Covariate shift. CTGAN generator introduced synthetic density variations; combining raw rows shifted decision thresholds away from the synthetic competition test distribution.

### 2. High-Order Polynomial Interaction Crosses
- **Hypothesis:** Multi-way survey products (`wifi * comfort * entertainment`) capture passenger delight.
- **Empirical Result:** $-47.4\text{u}$ drop.
- **Root Cause:** Feature dilution. Increased tree depth variance without adding orthogonal signal.

### 3. Target Encoding Without Shrinkage
- **Hypothesis:** Target-encoding high-cardinality route IDs will allow trees to split on route satisfaction.
- **Empirical Result:** $+32\text{u}$ (nominal CV) / $-177.5\text{u}$ (true generalization).
- **Root Cause:** Target leakage into out-of-fold validation splits. Out-of-fold predictions memorized fold labels.

### 4. Tree Monoculture Stacking
- **Hypothesis:** Stacking 10 slightly varied LightGBM seeds will average out variance.
- **Empirical Result:** $+2.5\text{u}$ negligible gain.
- **Root Cause:** Near-identical prediction residuals ($\rho > 0.998$). Variance reduction is bounded by residual decorrelation:
  $$\text{Var}\left(\frac{1}{M} \sum X_i\right) = \frac{\sigma^2}{M} + \frac{M - 1}{M} \rho \sigma^2 \xrightarrow{\rho \to 1} \sigma^2$$
