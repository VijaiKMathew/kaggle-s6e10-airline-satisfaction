"""
Gold Medal Training and Inference Pipeline for Kaggle Playground S6E10.
Predicting Airline Passenger Satisfaction (ROC AUC).
Features:
- Deterministic 5-Fold Stratified Cross-Validation using pre-split 'fold' column
- External Super Teacher Priors & 3-Way Profile Mapping
- Tri-GBDT Ensemble: LightGBM + CatBoost + XGBoost with native categorical handling
- Percentile Rank-Averaging and SLSQP Optimal Rank Blend
- Outputs verified submission to submissions/submission.csv and submission.csv
- Saves individual model OOF and test prediction arrays for downstream meta-stacking
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

from src.features import create_gold_features

def parse_args():
    parser = argparse.ArgumentParser(description="Train S6E10 Gold Medal Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="submissions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["lgb", "cat", "xgb"], help="Models to train")
    return parser.parse_args()

def optimize_rank_blend(oof_preds: list, y_true: np.ndarray):
    """Find weights for prediction blend that maximize ROC AUC using SLSQP on percentile ranks."""
    n_models = len(oof_preds)
    n_samples = len(y_true)
    
    ranked_oofs = [rankdata(p) / n_samples for p in oof_preds]
    
    def loss(weights):
        w_norm = weights / np.sum(weights)
        blended = sum(w * r for w, r in zip(w_norm, ranked_oofs))
        return -roc_auc_score(y_true, blended)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
    
    res = minimize(loss, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)
    best_score = -res.fun
    return best_weights, best_score

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("artifacts/oofs")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sub_oofs_dir = Path("submissions/oofs")
    sub_oofs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Kaggle Playground Series S6E10: Predicting Airline Satisfaction")
    print("  GOLD MEDAL Super Teacher Tri-GBDT Pipeline (Apple Silicon / Multi-Core)")
    print(f"  Models: {args.models}, Seed: {args.seed}")
    print("=" * 80)

    start_time = time.time()
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    print(f"Loading pre-split data from {data_dir}...")
    train_df = pd.read_parquet(train_path, engine="fastparquet")
    test_df = pd.read_parquet(test_path, engine="fastparquet")

    test_ids = test_df["id"].values
    y = train_df["satisfaction"].astype(np.int32).values
    folds = train_df["fold"].values
    n_splits = len(np.unique(folds))

    print(f"Loaded {len(train_df):,} train rows across {n_splits} folds and {len(test_df):,} test rows.")
    print(f"Target positive rate: {np.mean(y):.4%}")

    print("\nEngineering features with Super Teacher Priors, Route Corridor, and Survey Semantics...")
    fe_start = time.time()
    train_feat, test_feat, cat_cols = create_gold_features(train_df, test_df)
    print(f"Feature engineering completed in {time.time() - fe_start:.2f}s.")

    drop_cols = ["id", "satisfaction", "fold"]
    feature_cols = [c for c in train_feat.columns if c not in drop_cols]
    print(f"Total engineered features: {len(feature_cols)} (Categorical: {len(cat_cols)})")

    X = train_feat[feature_cols].copy()
    X_test = test_feat[feature_cols].copy()

    oof_preds = {m: np.zeros(len(train_df), dtype=np.float32) for m in args.models}
    test_preds = {m: np.zeros(len(test_df), dtype=np.float32) for m in args.models}

    print("\n" + "-" * 80)
    print("Starting Deterministic Stratified Cross-Validation")
    print("-" * 80)

    for fold in range(n_splits):
        fold_start = time.time()
        print(f"\n>>> Fold {fold + 1} / {n_splits}")

        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # 1. LightGBM Model
        if "lgb" in args.models:
            lgb_t0 = time.time()
            lgb_model = lgb.LGBMClassifier(
                n_estimators=2200,
                learning_rate=0.035,
                num_leaves=127,
                max_depth=-1,
                min_child_samples=50,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.75,
                max_bin=512,
                random_state=args.seed + fold,
                n_jobs=-1,
                verbosity=-1,
            )
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
            )
            val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
            oof_preds["lgb"][val_idx] = val_preds_lgb
            test_preds["lgb"] += lgb_model.predict_proba(X_test)[:, 1] / n_splits
            auc_lgb = roc_auc_score(y_val, val_preds_lgb)
            print(f"  [Fold {fold + 1}] LightGBM ROC AUC: {auc_lgb:.5f} ({time.time() - lgb_t0:.1f}s)")

        # 2. CatBoost Model
        if "cat" in args.models:
            cat_t0 = time.time()
            cat_model = CatBoostClassifier(
                iterations=1800,
                learning_rate=0.05,
                depth=7,
                l2_leaf_reg=4,
                eval_metric="AUC",
                thread_count=-1,
                random_seed=args.seed + fold,
                verbose=False,
            )
            cat_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=60,
                verbose=False,
            )
            val_preds_cat = cat_model.predict_proba(X_val)[:, 1]
            oof_preds["cat"][val_idx] = val_preds_cat
            test_preds["cat"] += cat_model.predict_proba(X_test)[:, 1] / n_splits
            auc_cat = roc_auc_score(y_val, val_preds_cat)
            print(f"  [Fold {fold + 1}] CatBoost ROC AUC: {auc_cat:.5f} ({time.time() - cat_t0:.1f}s)")

        # 3. XGBoost Model
        if "xgb" in args.models:
            xgb_t0 = time.time()
            xgb_model = xgb.XGBClassifier(
                n_estimators=1800,
                learning_rate=0.035,
                max_depth=7,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=20,
                tree_method="hist",
                max_bin=512,
                eval_metric="auc",
                early_stopping_rounds=60,
                n_jobs=-1,
                random_state=args.seed + fold,
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
            oof_preds["xgb"][val_idx] = val_preds_xgb
            test_preds["xgb"] += xgb_model.predict_proba(X_test)[:, 1] / n_splits
            auc_xgb = roc_auc_score(y_val, val_preds_xgb)
            print(f"  [Fold {fold + 1}] XGBoost  ROC AUC: {auc_xgb:.5f} ({time.time() - xgb_t0:.1f}s)")

        fold_models = [oof_preds[m][val_idx] for m in args.models]
        fold_blend = sum(fold_models) / len(fold_models)
        print(f"  --> Fold {fold + 1} Blend AUC: {roc_auc_score(y_val, fold_blend):.5f} (Fold Total: {time.time() - fold_start:.1f}s)")

    print("\n" + "=" * 80)
    print("FINAL OUT-OF-FOLD (OOF) EVALUATION RESULTS")
    print("=" * 80)

    model_list = args.models
    oof_list = [oof_preds[m] for m in model_list]
    test_list = [test_preds[m] for m in model_list]

    for m in model_list:
        score = roc_auc_score(y, oof_preds[m])
        print(f"  * {m.upper():8s} OOF ROC AUC: {score:.5f}")

    # Equal Blend
    equal_blend_oof = sum(oof_list) / len(oof_list)
    equal_auc = roc_auc_score(y, equal_blend_oof)
    print(f"\n  * Equal Probability Blend OOF AUC: {equal_auc:.5f}")

    # Percentile Rank Equal Blend
    n_samples = len(y)
    ranked_oofs = [rankdata(p) / n_samples for p in oof_list]
    rank_equal_blend_oof = sum(ranked_oofs) / len(ranked_oofs)
    rank_equal_auc = roc_auc_score(y, rank_equal_blend_oof)
    print(f"  * Equal Percentile Rank Blend OOF AUC: {rank_equal_auc:.5f}")

    # SLSQP Optimized Rank Blend
    opt_weights, opt_auc = optimize_rank_blend(oof_list, y)
    print(f"\n  * Optimal SLSQP Rank Blend OOF AUC: {opt_auc:.5f}")
    weight_str = ", ".join(f"{m.upper()}: {w:.4f}" for m, w in zip(model_list, opt_weights))
    print(f"    Optimal Weights -> {weight_str}")

    n_test = len(test_df)
    ranked_test = [rankdata(p) / n_test for p in test_list]
    final_test_preds = sum(w * r for w, r in zip(opt_weights, ranked_test))
    final_oof = sum(w * r for w, r in zip(opt_weights, ranked_oofs))
    final_score = opt_auc

    print(f"\nFinal GBDT Cross-Validation Score: {final_score:.5f}")

    # Save OOF and test predictions for individual models and ensemble
    for m in model_list:
        np.save(artifacts_dir / f"oof_{m}_gold.npy", oof_preds[m])
        np.save(artifacts_dir / f"test_{m}_gold.npy", test_preds[m])
        np.save(sub_oofs_dir / f"oof_{m}_gold.npy", oof_preds[m])
        np.save(sub_oofs_dir / f"test_{m}_gold.npy", test_preds[m])

    np.save(artifacts_dir / "oof_gbdt_gold.npy", final_oof)
    np.save(artifacts_dir / "test_gbdt_gold.npy", final_test_preds)
    np.save(sub_oofs_dir / "oof_gbdt_gold.npy", final_oof)
    np.save(sub_oofs_dir / "test_gbdt_gold.npy", final_test_preds)

    # Save submission
    sub_df = pd.DataFrame({"id": test_ids, "satisfaction": final_test_preds})
    sub_path = output_dir / "submission_gbdt_gold.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved GBDT Gold submission: {sub_path}")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.1f}s")
    print("=" * 80)

    return final_score

if __name__ == "__main__":
    main()
