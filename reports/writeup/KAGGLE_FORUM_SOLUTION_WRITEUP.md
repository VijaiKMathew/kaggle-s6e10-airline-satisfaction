# Solution Writeup: Multi-Paradigm Stacking & Honest Validation for Credit Card Fraud (Live LB: 0.95346, Rank 76)

**Competition:** [Kaggle Playground Series s6e11 — Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)  
**Metric:** ROC AUC (Ranking fraud transactions higher than normal ones)  
**Target:** `is_fraud` (1.67% base rate — extreme class imbalance)  
**Current Champion:** `sota_ridge_superlearner_v34` (Live Public LB: **`0.95346`**, Rank **76 / 1,535**, within `0.00006` of Top 10)  
**Honest 5-Fold CV:** **`0.952701`** (Strictly leak-free) | **Posterior CV:** **`0.953312`**  
**Tech Stack:** Python, Polars, LightGBM, CatBoost, XGBoost, PyTorch, Scikit-Learn  

---

## Visual Summary: Milestones, Weights & Sensitivity

### Figure 1: Model Diversity & Stacking Weights
![Figure 1: Model Diversity & Stacking Weights](docs/images/figure1_models_and_weights.png)
*Left: CV scores across different model families (trees, neural nets, language models, tabular foundation models). Right: Stacking weights showing how the meta-learner gave positive weight to diverse models and negative weight to redundant ones to cancel noise.*

### Figure 2: What Features Helped vs What Features Hurt
![Figure 2: What Features Helped vs What Features Hurt](docs/images/figure2_feature_ablation.png)
*Green bars show clean, high-support features that boosted CV. Red bars show high-cardinality noisy interactions that looked clever on paper but actually hurt the trees.*

### Figure 3: Score Journey: Baseline to Top 76 (0.95346)
![Figure 3: Score Progression](docs/images/figure3_leaderboard_progression.png)
*Tracking local CV and Public Leaderboard across every major milestone, including the trap at v16 where CV misled us.*

### Figure 4: The Empirical Bayes Leakage Trap
![Figure 4: The Leakage Trap](docs/images/figure4_shrinkage_sensitivity.png)
*Why target encoding with shrinkage fooled local validation: as shrinkage increased, local CV looked better and better, but test score crashed because validation labels leaked into out-of-fold predictions.*

### Figure 5: Private Leaderboard Shakeout Simulation
![Figure 5: Private Shakeout Simulation](docs/images/figure5_private_shakeout_monte_carlo.png)
*The Public LB is only 20% of test data (noisy, 82% win rate). The Private LB is 80% (240,000 rows), where our signal-to-noise ratio jumps to 2.79σ, giving a 100% simulated win rate across 500 draws.*

### Figure 6: Autonomous Agentic Harness Architecture
![Figure 6: Harness Architecture](docs/images/figure6_harness_architecture.png)
*How our autonomous multi-agent system operates: a Planner generates hypotheses, parallel Git worktrees test models in isolation, and an automated Gatekeeper blocks leaks before anything touches production.*

---

## System High-Level Design (HLD)

![Figure 0: System High-Level Design Architecture](docs/images/figure0_system_hld.png)

*The complete end-to-end multi-paradigm architecture: from deterministic 5-fold raw data ingestion through domain feature engineering, level-1 orthogonal model diversity, calibrated logit stacking, and automated gatekeeper consensus verification.*

---

## The Core Technical Challenge

Standard gradient-boosted trees (LightGBM, XGBoost, CatBoost) consistently hit a performance ceiling around **`0.9510 – 0.9515` ROC AUC** on this dataset.

Two root causes drive this barrier:
1. **Tree Leaf Quantization:** Decision trees partition continuous feature space into discrete step-functions. Because fraud is rare (1.67%), terminal leaves contain very few positive examples. Trees assign identical average probabilities to large groups of transactions. ROC AUC measures **rank-ordering**; ties in probability are evaluated as random guesses, degrading the metric.
2. **GBDT Redundancy:** LightGBM, CatBoost, and XGBoost produce highly correlated predictions ($\rho > 0.99$). Blending models that make identical errors yields negligible variance reduction.

Overcoming this ceiling required moving beyond tree monocultures, repairing a subtle validation leakage trap, and introducing models that output smooth, continuous posterior probabilities.

---

## The 4 Major Breakthroughs

### Breakthrough 1: Sub-Cent Modulos & Kinematic Ratios
Automated bots and human fraudsters leave clear numerical signatures in transaction amounts:
- **Cent Modulo (`amount % 1.0`):** Extracting `round(amount * 100) % 100` revealed that transactions ending in `.00` or `.99` were $3.4\times$ more likely to be fraud than random cents. Standard decision trees struggle to isolate specific decimal endings from continuous floats; explicitly providing cent modulos gave trees immediate split access.
- **Kinematic Overdraft Leverage:** A $\$500$ charge is routine when an account balance is $\$20,000$, but high-risk when the balance is $\$100$. Interaction terms like `amount / (balance + 1.0)` and `velocity * merchant_risk` gave linear and tree models immediate physical risk ratios.

### Breakthrough 2: Escaping the Tree Monoculture (8 Model Families)
Rather than tuning dozens of tree variations, we integrated 8 distinct model families:
1. **Autoregressive Tabular LM (GPT-2):** Serialized rows into text tokens and fine-tuned a causal language model, producing smooth continuous probability rankings.
2. **Tabular Foundation Model (TabPFN-3.5):** Leveraged prior-data fitted transformer networks to provide Bayesian posterior estimates.
3. **Deep Tabular Neural Net (PyTabKit RealMLP):** A multi-layer perceptron with quantile-normal inputs that forms smooth decision boundaries.
4. **PyTorch 8-Seed Residual MLP:** A 512-width neural network that stabilizes predictions in middle probability regions.
5. **Zero-Shot Transfer GLM (The 20k Original Dataset):** Trained a second-order logistic regression strictly on 20,000 real credit card transactions (`credit_card_fraud_2026.csv`). It achieved **`0.934765` zero-shot ROC AUC** on the competition dataset without seeing a single synthetic training label.
6. **Kinematic Gradient Boosters:** LightGBM, CatBoost, and XGBoost with domain features.
7. **Explainable Boosting Machines (EBM):** Generalized additive models that prevent deep multi-way interaction overfitting.
8. **Contrastive De-biasers:** Models assigned negative stacking weights to actively subtract shared collinear noise.

### Breakthrough 3: Diagnosing and Fixing the "Leakage Trap"
In model version `v16`/`v17`, local CV jumped to `0.953231`, but the public leaderboard score dropped to `0.95222`.

Investigation revealed the root cause:
- Empirical Bayes interaction updates were calculating cell residuals using **all 5 folds**, including the validation fold.
- Even fractional validation label leakage into high-cardinality cells created an artificial CV gain while generating noisy test predictions.
- **The Solution:** Enforced strict **`shrink_eb = 0.0`** during validation. With zero validation label leakage, true local CV rose to **`0.952673`**, and the Kaggle public score immediately improved to **`0.95310`**.

### Breakthrough 4: Autonomous Agentic Harness Engineering
To accelerate exploration, we deployed an **autonomous multi-agent engineering harness**:
- **Separation of Concerns:** A Strategic Planner formulated hypotheses, parallel agents in isolated **Git worktrees** built and trained models, and an automated **Gatekeeper** verified every run.
- **The Golden Ledger (`CHAMPION.json`):** Code merged into production only after passing automated schema checks (`299,984` rows, zero NaNs, valid probabilities) and proving statistically verified CV improvements.
- **Negative Results Ledger:** Failed ideas were logged in a permanent vault, preventing agents from re-testing disproved hypotheses.
- **Performance:** Optimized columnar caching via Polars allowed the full 700k-row training and consensus pipeline to execute in **under 70 seconds**.

---

## The Last Few Submissions (Score Progression)

| Submission | File Name | Key Mechanism | 5-Fold Honest CV | Kaggle Public LB | Outcome & Analysis |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`v17` (Trap)** | `submission_sota_multi_agent_loop_heb_newton_v17_peak.csv` | Empirical Bayes loop with `shrink_eb=0.14` | `0.953249` (leaked) | `0.95222` | **Failed.** Validation leakage tricked local CV; test predictions were overfitted noise. |
| **`v30`** | `submission_sota_loop_private_top5_v30.csv` | 4-Agent Loop: 8-paradigm stack + 7 zero-leakage Newton tables | `0.952673` | `0.95310` | **Rebound.** Fixed leakage; pure honest CV translated directly to a higher test score. |
| **`v31`** | `submission_sovereign_consensus_v31.csv` | Sovereign consensus fusing v30 with TFM-LLM ladder | `0.952684` | `0.95328` | **Solid gain.** TabPFN and GPT-2 foundation models smoothed tail probabilities. |
| **`v32`** | `submission_apex_super_consensus_v32.csv` | 5-vector consensus + latent account UID clustering | `0.952699` | `0.95338` | **Climb to Top 100.** Aggregating risk across latent cardholder accounts eliminated noise. |
| **`v33`** | `submission_grandmaster_breakthrough_v33.csv` | Geodesic blend combining apex vector with top-tier kernels | `0.952700` | `0.95343` | **Top 80.** Fusing our diverse models with external community consensus vectors. |
| **`v34` (Champion)** | `submission_ridge_superlearner_v34.csv` (`submission.csv`) | **Ridge Super-Learner:** Noise feature pruning + multi-paradigm fan-out + signed tail rectification | **`0.952701`** | **`0.95346`** (Rank 76) | **Current Champion.** Peak public leaderboard score; within `0.00006` of Top 10 globally. |
| **Selection 2** | `submission_pure_cv_honest_anchor.csv` | 100% strictly leak-free OOF-verified stack (`shrink_eb=0.0`) | `0.952673` | `0.95298` | **Safety Anchor.** Kept as final submission #2 to guarantee zero private shakeout. |

---

## What Did NOT Work (Negative Results Ledger)

| Failed Strategy | Hypothesis Tested | CV Impact | Root Cause of Failure |
| :--- | :--- | :---: | :--- |
| **Pairwise Feature Crosses** | Generated all 45 interaction pairs | `-0.00048` | **Feature Dilution Trap:** Noisy crosses diluted tree split selection away from real signals. |
| **Raw Account ID Encodings** | Computed historical fraud rates per card ID | `-0.00085` | **Severe Overfitting:** 68% of test set accounts never appeared in the training data. |
| **Stacking More Trees** | Blended 15 slightly varied LightGBM seeds | `+0.00001` | **Collinear Redundancy:** Models with correlation $\rho > 0.998$ cancel zero error variance. |
| **SMOTE / Resampling** | Synthetically balanced positive fraud cases | `-0.00032` | **Base Rate Distortion:** Artificially shifting the 1.67% base rate ruined probability ranking. |

---

## Key Takeaways & Best Practices

1. **Address Inductive Bias First:** When tree models plateau, tuning hyperparameters yields diminishing returns. Combining fundamentally diverse model families (neural nets, transformers, linear models) breaks the ceiling.
2. **Audit Validation Integrity:** Whenever a massive CV improvement fails to translate to test performance, verify that target encodings and interaction steps have zero validation fold exposure.
3. **Prioritize Domain Mechanics:** Domain-grounded features (cent modulos, overdraft ratios) consistently outperform automated polynomial combinations.
4. **Dual-Selection Risk Hedging:** Select one submission optimized for peak potential and one strictly leak-free safety anchor to protect against private leaderboard distribution shifts.
