# 📜 Universal Kaggle Project Lifecycle & Asset Retention Policy
**Author:** Vijai K. Mathew  
**Scope:** All Kaggle Projects under `3-kaggle/` (Active, Public, and Archived)  
**Standard Version:** 2.0 (Post-Competition Dual-Track Architecture)  
**Date:** 2026-11-23  
**Status:** ENFORCED & UNIVERSALLY APPLIED  

---

## 1. Objective & Architectural Standard

To maintain high research agility during active competitions while ensuring clean public career visibility and avoiding multi-gigabyte disk sprawl, every Kaggle initiative in `3-kaggle/` must adhere to the **Dual-Track Repository Architecture** and the **T+30 Day Post-Competition Depuration Protocol**.

```
┌────────────────────────────────────────────────────────┐
│             PRIVATE REPOSITORY (Internal Lab)          │
│  Location: `archive/<competition>` or `<comp>-dev`     │
│  • Full experimental history, sweeps, exploratory code │
│  • Agent traces, diagnostic logs, research playbooks   │
│  • CHAMPION.json and SHA256-verified submission.csv    │
│  • RECREATION_PLAYBOOK.md (One-command restore guide)  │
└───────────────────────────┬────────────────────────────┘
                            │ One-way export & sanitize
                            ▼
┌────────────────────────────────────────────────────────┐
│             PUBLIC REPOSITORY (Career Anchor)          │
│  Location: `kaggle-<competition>-<slug>` (GitHub)      │
│  • Clean, 1-command reproducible champion pipeline     │
│  • Publication walkthrough notebook (notebooks/)       │
│  • Executive README with badges & system architecture  │
│  • Gold/Silver Solution WRITEUP.md with visual figures │
│  • Zero raw data, zero .venv, zero secrets             │
└────────────────────────────────────────────────────────┘
```

---

## 2. Universal Directory Hierarchy Rules

1. **Active / In-Flight Competitions:**
   - Development occurs in `<competition>-dev` or `<competition>`.
   - All environments are managed via Astral `uv` (`pyproject.toml` and `uv.lock`).
   - Champion configurations must be logged continuously in `CHAMPION.json`.
2. **Public Portfolio Repositories:**
   - Maintained under `kaggle-<competition>-<slug>` or `<competition>-public`.
   - Connected directly to a public GitHub remote (e.g. `https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction`).
   - Must contain:
     - `README.md` (Executive summary, badges, system diagram, progression table).
     - `WRITEUP.md` (Self-contained writeup with relative image paths).
     - `notebooks/champion-pipeline/` (Executable publication notebook + metadata).
     - `src/` & `configs/` (Clean, leak-free code).
     - `reports/solution_preview.html` (Standalone interactive browser preview).
     - `docs/images/` (High-res system diagrams and ablation charts).
     - `Makefile` & `run.sh` (Automated 1-command verification suite).
     - `.gitignore` (Strictly excludes `data/`, `.venv/`, `*.csv`, `*.parquet`, `.env`).
3. **No Stray Staging Folders:**
   - Intermediate export/staging folders (e.g. untracked duplicate directories at `3-kaggle/<competition>`) must never be left unmanaged.
   - Once public sync is complete, staging directories must be cleanly removed.

---

## 3. The T+30 Day Post-Competition Depuration Protocol

Competitions generate gigabytes of disposable weights, raw datasets, virtual environments, and scratch runs. **Thirty days (1 month) after competition close date**, projects undergo automated depuration:

### A. What MUST Be Permanently Retained
1. **Source Code & Pipelines:** All files in `src/`, `tools/checks/`, `configs/`.
2. **Authoritative Checkpoints:** `CHAMPION.json` and the official final `submission.csv` (pinned with SHA256 checksum).
3. **Documentation & Research:** `README.md`, `WRITEUP.md`, `reports/`, and post-mortem analyses.
4. **Interactive Reports:** Standalone HTML dashboards (`reports/solution_preview.html`).
5. **Recreation Guide:** `RECREATION_PLAYBOOK.md` detailing exact one-command restoration.

### B. What MUST Be Pruned (Storage Reclamation)
| Category | Disk Footprint | Pruning Action | Recreation Guarantee |
| :--- | :---: | :--- | :--- |
| **Virtual Environments (`.venv/`)** | 0.5 GB – 2.5 GB | **DELETE** | Instant 1-line rebuild via `uv sync` |
| **Scratch Environments (`scratch/test_env`)** | 0.5 GB – 1.0 GB | **DELETE** | Disposable test runners |
| **Raw Kaggle Datasets (`data/competition/*.csv`, `*.zip`)** | 1.0 GB – 10.0+ GB | **DELETE** | Re-downloadable anytime via `kaggle competitions download -c <id>` |
| **External Competitor OOF Dumps (`data/external/`)** | 0.5 GB – 2.0 GB | **DELETE** | Prune bulky non-champion ensembles |
| **Intermediate Submissions (`submissions/cand_*.csv`)** | 0.2 GB – 1.0 GB | **DELETE** | Prune intermediate sweeps, retain only `submission.csv` |
| **Old Agent Workspaces (`workspaces/`)** | 0.1 GB – 0.5 GB | **DELETE** | Scratchpad agent worktrees |
| **Stray Temporary Folders (`1.private/`, `2.public/`)** | Variable | **DELETE** | Unused directory placeholders |

**Expected Savings:** **4.0 GB to 15.0+ GB per completed competition.**

---

## 4. One-Command Deterministic Reproduction Protocol

Every archived project must guarantee zero-panic restorability. Anyone checking out the project can bring it back to full training capacity in under 2 minutes:

```bash
# 1. Restore locked virtual environment
uv sync

# 2. Ingest competition data via official Kaggle CLI
mkdir -p data/competition
kaggle competitions download -c <competition_id> -p data/
unzip -o data/<competition_id>.zip -d data/competition/

# 3. Execute champion pipeline
uv run python src/train_predict.py --data-dir data/competition --output-dir submissions

# 4. Validate output against CHAMPION.json
uv run python tools/checks/check_submission.py
```

---

## 5. Master Portfolio Repository Index

| Competition | Private Dev Archive | Public GitHub Portfolio | T+30 Depuration | Verified Score |
| :--- | :--- | :--- | :---: | :---: |
| **S6E10: Airline Satisfaction** | `archive/playground-series-s6e10` | [kaggle-s6e10-airline-satisfaction](https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction) | ✅ Complete (4.33 GB freed) | 0.96238 ROC AUC (Bronze) |
| **Kaggriculture** | `1.except-enveda/kaggriculture-dev` | `1.except-enveda/kaggriculture-public` | ✅ Complete | 3,094.0 Elo (Top 1%) |
| **Orbit Wars** | `1.except-enveda/orbit-wars-dev` | `1.except-enveda/orbit-wars-public` | ✅ Complete | 844.5 Score |
| **Enveda - CASMI 2026** | `1.except-enveda/enveda-casmi-dev` | `1.except-enveda/enveda-casmi-public` | In Progress | 0.456 MRR@25 |
| **S6E11: Fraud Detection** | `playground-series-s6e11` | In Progress | Active (Pending T+30) | 0.95487 (Rank 9) |
