# 🔄 Complete Asset Recreation Playbook: Playground Series S6E10 (Airline Passenger Satisfaction)
**Competition:** [Kaggle Playground Series s6e10](https://www.kaggle.com/competitions/playground-series-s6e10)  
**Author:** Vijai K. Mathew  
**Created:** 2026-11-23  
**Status:** ACTIVE & VERIFIED  
**Scope:** Authoritative recreation commands and protocol for all pruned or reproducible assets  
**Post-Competition Rule:** T+30 Day Depuration & Storage Optimization Standard  

---

## 📌 Executive Summary

Following competition conclusion on November 1, 2026, and in strict adherence to the **Kaggle T+30 Day Storage Optimization Policy**, heavy reproducible binaries, virtual environments, raw downloaded datasets, and transient scratch outputs have been pruned to reclaim over **4.2 GB** of local disk storage.

Every single omitted or pruned asset can be reconstructed deterministically on demand using the native commands documented below.

---

## ⚡ Master One-Shot Command (Recreate Everything at Once)

To reconstruct the full operational pipeline from a fresh checkout:

```bash
cd archive/playground-series-s6e10

# 1. Rebuild virtual environment via uv
uv sync

# 2. Ingest official competition dataset via Kaggle CLI
mkdir -p data/competition
kaggle competitions download -c playground-series-s6e10 -p data/
unzip -o data/playground-series-s6e10.zip -d data/competition/

# 3. Retrain champion 5-fold ensemble (LightGBM + CatBoost + XGBoost)
uv run python src/train_predict.py \
  --data-dir data/competition \
  --output-dir submissions \
  --models lgb cat xgb

# 4. Verify champion integrity and sha256 checksum
python3 -c "
import hashlib, json
with open('CHAMPION.json') as f:
    meta = json.load(f)
expected = meta['files'][0]['sha256']
with open('submission.csv', 'rb') as f:
    actual = hashlib.sha256(f.read()).hexdigest()
assert actual == expected, f'Mismatch: {actual} vs {expected}'
print('Champion SHA256 verified successfully:', actual)
"
```

---

## 🛠️ Complete Recreation & Asset Matrix

| Asset Category | Original Size | Retention Status | Recreation Command / Method | Restoration Time |
| :--- | :---: | :---: | :--- | :---: |
| **Virtual Environment (`.venv/`)** | **2.1 GB** | Pruned (Reproducible) | `uv sync` or `pip install -r pyproject.toml` | ~15 seconds |
| **Scratch Test Env (`scratch/test_env`)** | **469 MB** | Pruned (Reproducible) | Disposable test env; recreated via `python3 -m venv` | ~10 seconds |
| **Raw Kaggle Data (`data/competition/`)** | **1.1 GB** | Pruned (Reproducible) | `kaggle competitions download -c playground-series-s6e10 -p data/ && unzip -o data/playground-series-s6e10.zip -d data/competition/` | ~30 seconds |
| **Intermediate Submissions (`submissions/*.csv`)** | **260 MB** | Pruned (Non-Champion) | Historical exploratory candidates; reproducible via parameter sweeps | On demand |
| **Transient Agent Workspaces (`workspaces/`)** | **98 MB** | Pruned | Scratchpad agent worktrees from iterative runs | N/A |
| **Champion Submission (`submission.csv`)** | **7.8 MB** | **PERMANENTLY RETAINED** | Pinned with SHA256: `93b589d074362b88481ac5b85a95fd7bd832d314befe34e255a7b513056d7721` | Retained |
| **Champion Metadata (`CHAMPION.json`)** | **1 KB** | **PERMANENTLY RETAINED** | Records CV (`0.962238`), Private LB (`0.96238`), LB Ref (`57690436`) | Retained |
| **Source Code (`src/`)** | **52 KB** | **PERMANENTLY RETAINED** | `src/features.py`, `src/train_predict.py` | Retained |
| **Diagnostic Reports (`reports/`)** | **204 KB** | **PERMANENTLY RETAINED** | Post-mortems, goal logs, forum writeups | Retained |

---

## 🔒 Champion Metadata Reference

```json
{
  "label": "pure-cv-honest-private-best",
  "competition": "playground-series-s6e10",
  "source": "submission.csv",
  "lb_score": 0.96238,
  "cv_score": 0.962238,
  "lb_ref": 57690436,
  "updated": "2026-11-01",
  "note": "Official Competition Final Champion: Pure 5-Fold Stratified Honest CV Stack. Private LB: 0.96238 (Rank 220 of 3,747 teams globally — Bronze Medal Tier).",
  "files": [
    {
      "path": "submission.csv",
      "role": "submission",
      "bytes": 7876896,
      "sha256": "93b589d074362b88481ac5b85a95fd7bd832d314befe34e255a7b513056d7721"
    }
  ]
}
```

---

## 🌐 Public Portfolio Repository Link

- **Public Repository:** [https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction](https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction)
- **Local Public Path:** `/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/kaggle-s6e10-airline-satisfaction`
