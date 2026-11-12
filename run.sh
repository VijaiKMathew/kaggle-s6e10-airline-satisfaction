#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo " Kaggle S6E10 Champion Pipeline Execution"
echo "================================================================================"

echo "==> Step 1: Syncing virtual environment via uv..."
uv sync

echo "==> Step 2: Training 5-fold ensemble (LightGBM + CatBoost + XGBoost)..."
mkdir -p submissions
uv run python src/train_predict.py --data-dir data/raw --output-dir submissions --models lgb cat xgb

echo "==> Step 3: Generating standalone interactive HTML report..."
mkdir -p reports
uv run python tools/generate_html_report.py --output reports/solution_preview.html

echo "================================================================================"
echo " SUCCESS: Solution pipeline completed successfully!"
echo " Submission file: submissions/submission.csv"
echo " Interactive HTML report: reports/solution_preview.html"
echo "================================================================================"
