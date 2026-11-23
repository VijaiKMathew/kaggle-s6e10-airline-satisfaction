#!/usr/bin/env python3
"""
tools/publish_public.py
Automated export pipeline from Private Kaggle Workspace to Public Portfolio Repo.

Usage:
    python tools/publish_public.py --dest /path/to/public-repo --commit-msg "feat: release champion pipeline" [--push]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ==============================================================================
# WHITELIST: Only these files and directories are permitted in the public repo.
# Everything else (scratchpads, harness, logs, agent dumps) is excluded.
# ==============================================================================
WHITELIST_FILES = [
    "README.md",
    "WRITEUP.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "run.sh",
    ".gitignore",
    "tools/generate_html_report.py",
]

WHITELIST_DIRS = [
    "src",               # Core production source code
    "configs",           # Model hyperparameters & training configs
    "notebooks",         # Curated single demo walkthrough notebook
    "docs",              # Technical architecture, ablation studies & visual diagrams
    "reports",           # HTML interactive preview reports
]

# Patterns that MUST NEVER be copied to the public repo
FORBIDDEN_EXTENSIONS = {
    ".csv", ".parquet", ".feather", ".h5", ".npy", ".pkl",  # Data / Heavy Weights
    ".env", ".key", ".pem", ".secret",                      # Credentials
    ".log", ".out", ".sqlite", ".db"                        # Cache / Run artifacts
}

FORBIDDEN_DIR_NAMES = {
    "__pycache__", ".git", ".idea", ".vscode",
    "harness", "scratch", "tmp", "wandb", "catboost_info"
}


def sanitize_directory(src_root: Path, dest_root: Path):
    """Copies whitelisted files/folders while enforcing security boundaries."""
    dest_root.mkdir(parents=True, exist_ok=True)

    # 1. Copy whitelisted top-level files
    for filename in WHITELIST_FILES:
        src_file = src_root / filename
        if src_file.is_file():
            dest_file = dest_root / filename
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            print(f" [COPIED] {filename}")
        else:
            print(f" [NOTICE] Optional file '{filename}' not present. Skipping.")

    # 2. Copy whitelisted directories recursively
    for dir_name in WHITELIST_DIRS:
        src_dir = src_root / dir_name
        if not src_dir.is_dir():
            print(f" [NOTICE] Whitelisted dir '{dir_name}' not present. Skipping.")
            continue

        for root, dirs, files in os.walk(src_dir):
            # Prune forbidden directories in-place
            dirs[:] = [d for d in dirs if d not in FORBIDDEN_DIR_NAMES]
            
            rel_path = Path(root).relative_to(src_root)
            target_dir = dest_root / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                ext = Path(file).suffix.lower()
                if ext in FORBIDDEN_EXTENSIONS:
                    print(f" [SAFETY SHIELD] Skipped forbidden file: {rel_path / file}")
                    continue
                
                src_path = Path(root) / file
                dest_path = target_dir / file
                shutil.copy2(src_path, dest_path)
                print(f" [COPIED] {rel_path / file}")


def verify_security(dest_root: Path):
    """Deep security audit to ensure no secrets or sensitive data leaked."""
    leaks = []
    for root, _, files in os.walk(dest_root):
        if ".git" in Path(root).parts:
            continue
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
                leaks.append(str(path))
            # Scan text files for potential API tokens
            if path.suffix.lower() in [".py", ".sh", ".yaml", ".json", ".md"]:
                content = path.read_text(errors="ignore")
                if "KAGGLE_KEY" in content or "AWS_SECRET" in content or "ghp_" in content:
                    leaks.append(f"Secret detected inside {path}")
    
    if leaks:
        print("\n❌ SECURITY AUDIT FAILED! Prohibited files detected in public export:")
        for leak in leaks:
            print(f"  - {leak}")
        sys.exit(1)
    print("\n✅ Security and Data Privacy Audit Passed.")


def git_commit_and_push(dest_root: Path, commit_msg: str, push: bool):
    """Executes git commit on the public repo."""
    try:
        subprocess.run(["git", "add", "."], cwd=dest_root, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=dest_root,
            capture_output=True,
            text=True,
            check=True
        )
        if not status.stdout.strip():
            print("ℹ️ No changes detected. Public repository is already up to date.")
            return

        subprocess.run(["git", "commit", "-m", commit_msg], cwd=dest_root, check=True)
        print(f"✅ Committed changes to public repo: '{commit_msg}'")

        if push:
            subprocess.run(["git", "push", "origin", "main"], cwd=dest_root, check=True)
            print("🚀 Successfully pushed to public remote!")
        else:
            print("ℹ️ Push skipped. Review changes locally, then run 'git push origin main'.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Export sanitized production files to public repo.")
    parser.add_argument("--dest", required=True, help="Absolute path to public git repository")
    parser.add_argument("--commit-msg", default="feat: release reproducible champion pipeline", help="Git commit message")
    parser.add_argument("--push", action="store_true", help="Push immediately to remote public repo")
    args = parser.parse_args()

    src_root = Path.cwd().resolve()
    dest_root = Path(args.dest).resolve()

    if not (dest_root / ".git").is_dir():
        print(f"❌ Error: Destination '{dest_root}' is not a valid git repository.")
        sys.exit(1)

    print(f"Starting export:\n  From: {src_root}\n  To:   {dest_root}\n")
    sanitize_directory(src_root, dest_root)
    verify_security(dest_root)
    git_commit_and_push(dest_root, args.commit_msg, args.push)


if __name__ == "__main__":
    main()
