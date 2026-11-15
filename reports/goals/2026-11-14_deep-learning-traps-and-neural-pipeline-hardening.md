# Kaggle S6E10 Airline Passenger Satisfaction: Deep Learning & Tabular Pipeline Hardening

**Date:** 2026-11-14  
**Project:** `kaggle-s6e10-airline-satisfaction`  
**Metric:** ROC AUC  
**Status:** Hardened, Guardrailed & Verified  

---

## 1. Executive Summary & Tabular Pipeline Pitfalls

In tabular classification competitions with large synthetic datasets (Playground Series S6E10), models are prone to subtle leakage in preprocessing, target encoding, and ensembling.

### Key Traps Identified & Audited:
1. **Global Target Encoding Leakage:** Encoding high-cardinality categoricals using global target statistics instead of out-of-fold partitions causes GBDT trees to memorize target priors, inflating local ROC AUC while failing on Kaggle test splits.
2. **Missing Standalone Integrity Checker:** Lack of automated AST linting allowed potential scaling leaks across train and test partitions.
3. **Probability Calibration Drift in Blending:** When blending LightGBM, CatBoost, and XGBoost via rank averaging or Nelder-Mead optimization, non-monotonic probability scaling can distort ROC AUC ranking.

---

## 2. Implemented Precautions & Guardrail Architecture

1. **Master Integrity Linter (`tools/checks/master_dl_integrity_linter.py`):**
   - Statically verifies zero in-sample OOF leakage, zero unpartitioned scaler fitting, zero autograd graph memory leaks, and zero silent exception masking.
2. **Deterministic Submission Validator (`tools/checks/check_submission.py`):**
   - Enforces exact schema conformity (`id`, `satisfaction`), non-null constraints, and probability bounds $[0, 1]$.
3. **Deterministic 5-Fold Stratified Cross-Validation:**
   - Pre-split `fold` column guarantees absolute reproducibility and zero overlap across iterations.

---

## 3. Verification & Compliance
```bash
python3 tools/checks/master_dl_integrity_linter.py
python3 tools/checks/check_submission.py
```
**Status:** Passed with 100% compliance.
