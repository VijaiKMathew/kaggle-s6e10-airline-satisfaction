# ==============================================================================
# Kaggle Champion Solution Pipeline Makefile
# Powered by Astral 'uv' and the official Kaggle CLI
# ==============================================================================

SHELL := /bin/bash
PYTHON := uv run python
COMPETITION := playground-series-s6e10

.PHONY: help setup data train report clean all lint test

help:
	@echo "================================================================================"
	@echo " Kaggle Solution Reproduction & Pipeline Automation"
	@echo "================================================================================"
	@echo "  make setup     - Create virtual environment and install dependencies via uv"
	@echo "  make data      - Ingest competition dataset using the official Kaggle CLI"
	@echo "  make train     - Run feature engineering, 5-fold CV and ensemble inference"
	@echo "  make report    - Build standalone interactive HTML solution summary preview"
	@echo "  make test      - Verify Python syntax across src and tools"
	@echo "  make lint      - Check and format code with ruff"
	@echo "  make clean     - Remove temporary caches, build artifacts, and logs"
	@echo "  make all       - Run complete end-to-end pipeline (setup -> data -> train -> report)"
	@echo "================================================================================"

setup:
	@echo "==> Setting up environment with uv..."
	uv sync

test:
	@echo "==> Verifying Python syntax..."
	python3 -m py_compile src/*.py tools/*.py
	@echo "✓ Syntax validation passed."

data:
	@echo "==> Downloading competition data for $(COMPETITION) via Kaggle CLI..."
	mkdir -p data/raw
	kaggle competitions download -c $(COMPETITION) -p data/raw/
	unzip -o -q data/raw/$(COMPETITION).zip -d data/raw/
	@echo "==> Dataset successfully extracted to data/raw/"

train:
	@echo "==> Running leak-free feature engineering and 5-fold tri-GBDT ensemble..."
	mkdir -p submissions
	$(PYTHON) src/train_predict.py --data-dir data/raw --output-dir submissions --models lgb cat xgb

report:
	@echo "==> Generating standalone interactive HTML solution preview..."
	mkdir -p reports
	$(PYTHON) tools/generate_html_report.py --output reports/solution_preview.html
	@echo "==> Interactive report ready at reports/solution_preview.html"

lint:
	@echo "==> Running ruff code quality checks..."
	uv run ruff check src/ tools/ configs/
	uv run ruff format --check src/ tools/ configs/

clean:
	@echo "==> Cleaning transient files and pycache..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache
	@echo "==> Clean complete."

all: setup data train report
	@echo "==> Full end-to-end pipeline execution complete!"
