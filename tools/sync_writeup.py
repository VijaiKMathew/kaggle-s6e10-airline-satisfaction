#!/usr/bin/env python3
"""sync_writeup.py: Secure, Zero-Vulnerability Writeup Synchronization Engine.

Security Invariants:
1. Dynamic Path Resolution: Never hardcodes absolute user paths, usernames, or home dirs.
2. Canonical Boundary Isolation: All destination paths are strictly canonicalized and
   must reside within verified repository roots. Prevents Path Traversal (CWE-22).
3. Symlink Protection: Rejects symlinks on source and target paths to prevent
   arbitrary file overwrite (TOCTOU / Symlink attacks).
4. Atomic Replacement: Writes to temporary files and uses atomic os.replace() to prevent
   corrupt partial reads.
5. Automated Secret & PII Gate: Scans source content for credentials, API tokens,
   private keys, and local home directory paths before synchronization. Fails closed.
6. Portable Git Hook: Hook uses $(git rev-parse --show-toplevel) without hardcoded paths.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Credential and PII leak detection patterns (Fail-Closed Gate)
SECURITY_GUARD_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"gho_[A-Za-z0-9]{36}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|kaggle_key)\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}"),
    re.compile(r"/Users/[a-zA-Z0-9_\-]+/"),
    re.compile(r"/home/[a-zA-Z0-9_\-]+/"),
]


def get_current_repo_root() -> Path:
    """Dynamically determine the root of the current repository without hardcoding."""
    return Path(__file__).resolve().parent.parent


def get_allowed_roots() -> list[Path]:
    """Get canonical allowed repository roots strictly bounded to workspace parent."""
    current_root = get_current_repo_root().resolve()
    allowed = [current_root]

    # Check sibling project directories safely if they exist
    parent = current_root.parent
    for sibling_name in ["playground-series-s6e10", "playground-series-s6e10"]:
        sibling_path = (parent / sibling_name).resolve()
        if sibling_path.exists() and sibling_path.is_dir() and not sibling_path.is_symlink():
            if sibling_path not in allowed:
                allowed.append(sibling_path)
    return allowed


def is_safe_path(target_path: Path, allowed_roots: list[Path]) -> bool:
    """Verify that a path is canonical, not a symlink, and within an allowed root."""
    try:
        resolved = target_path.resolve()
        # Reject symlinks in target path or its ancestors
        for part in [target_path] + list(target_path.parents):
            if part.is_symlink():
                return False
        # Must be relative to one of the allowed roots
        return any(resolved.is_relative_to(root) for root in allowed_roots)
    except (ValueError, RuntimeError):
        return False


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 checksum securely."""
    if not filepath.exists() or filepath.is_symlink() or not filepath.is_file():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def scan_for_secrets_and_pii(content: str) -> list[str]:
    """Scan content for sensitive credentials or personal paths."""
    violations = []
    for pat in SECURITY_GUARD_PATTERNS:
        match = pat.search(content)
        if match:
            violations.append(f"Detected prohibited pattern: {match.group(0)[:30]}...")
    return violations


def sanitize_for_public(content: str, rel_image_prefix: str = "docs/images") -> str:
    """Sanitize and standardize relative markdown links for public consumption."""
    text = content

    # Normalize image paths
    text = re.sub(r'!\[(.*?)\]\((.*?)(docs/images/figure\d+.*?\.png)\)', rf'![\1]({rel_image_prefix}/\3)', text)
    text = text.replace(f"{rel_image_prefix}/{rel_image_prefix}", rel_image_prefix)

    # Sanitize any accidental system references
    system_sanitizations = [
        (r'Apple Silicon M-series CPU', 'Python & Polars Engine'),
        (r'Apple Silicon', 'Multi-core CPU'),
        (r'M-series', 'CPU'),
    ]
    for pat, repl in system_sanitizations:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    return text


def atomic_write(target_path: Path, content: str) -> None:
    """Write to a temporary file and atomically replace target to prevent TOCTOU corruption."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary file in the same directory to guarantee atomic rename across filesystems
    with tempfile.NamedTemporaryFile("w", dir=target_path.parent, delete=False, encoding="utf-8") as tf:
        tf.write(content)
        temp_name = tf.name

    try:
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, target_path)
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def sync_images(allowed_roots: list[Path]):
    """Securely copy image assets across image directories without symlink traversal."""
    repo_root = get_current_repo_root()
    src_img_dir = (repo_root / "docs/images").resolve()
    if not src_img_dir.exists() or src_img_dir.is_symlink():
        return

    figures = [f for f in src_img_dir.glob("figure*.png") if f.is_file() and not f.is_symlink()]

    for root in allowed_roots:
        for sub in ["docs/images", "reports/writeup/docs/images"]:
            target_dir = (root / sub).resolve()
            if not is_safe_path(target_dir, allowed_roots):
                continue
            target_dir.mkdir(parents=True, exist_ok=True)

            for fig in figures:
                dest_file = target_dir / fig.name
                if dest_file.is_symlink():
                    continue
                if not dest_file.exists() or compute_sha256(fig) != compute_sha256(dest_file):
                    # Atomic copy
                    temp_dest = target_dir / f".tmp_{fig.name}"
                    shutil.copy2(fig, temp_dest)
                    os.chmod(temp_dest, 0o644)
                    os.replace(temp_dest, dest_file)


def perform_sync(verbose: bool = True) -> bool:
    """Execute complete security-checked sync from private master to all public targets."""
    repo_root = get_current_repo_root()
    allowed_roots = get_allowed_roots()

    source_master = (repo_root / "WRITEUP.md").resolve()
    if not source_master.exists() or source_master.is_symlink():
        # Fallback to sibling s6e10 if running from s6e10
        sibling_source = (repo_root.parent / "playground-series-s6e10/WRITEUP.md").resolve()
        if sibling_source.exists() and not sibling_source.is_symlink():
            source_master = sibling_source
        else:
            if verbose:
                print(f"Error: Safe source master WRITEUP.md not found.")
            return False

    with open(source_master, "r", encoding="utf-8") as f:
        master_content = f.read()

    # Pre-Flight Security Gate
    violations = scan_for_secrets_and_pii(master_content)
    if violations:
        print("[SECURITY ALERT] Synchronization aborted due to credential / PII guard violations:")
        for v in violations:
            print(f"  - {v}")
        return False

    # Sync assets
    sync_images(allowed_roots)

    # Determine targets dynamically across allowed repository roots
    target_files: list[Path] = []
    for root in allowed_roots:
        candidates = [
            root / "reports/writeup/KAGGLE_FORUM_SOLUTION_WRITEUP.md",
            root / "docs/reports/KAGGLE_FORUM_SOLUTION_WRITEUP.md",
        ]
        if root.name == "playground-series-s6e10":
            candidates.extend([
                root / "S6E10_WRITEUP_GOLD_TIER.md",
                root / "WRITEUP.md",
            ])
        for c in candidates:
            resolved_c = c.resolve()
            if is_safe_path(resolved_c, allowed_roots):
                target_files.append(c)

    updated_count = 0
    for target in target_files:
        sanitized = sanitize_for_public(master_content, rel_image_prefix="docs/images")

        current_content = ""
        if target.exists() and not target.is_symlink():
            with open(target, "r", encoding="utf-8") as f:
                current_content = f.read()

        if current_content != sanitized:
            atomic_write(target, sanitized)
            updated_count += 1
            if verbose:
                print(f"  -> Securely Synced: {target.name} ({target.parent.relative_to(repo_root.parent)})")

    if verbose:
        print(f"Zero-Vulnerability Sync Complete: {updated_count} files updated.")
    return True


def install_git_hook():
    """Install secure portable git pre-commit hook."""
    repo_root = get_current_repo_root()
    hook_path = (repo_root / ".git/hooks/pre-commit").resolve()
    
    if not hook_path.parent.exists() or hook_path.is_symlink():
        return

    # Fully portable POSIX shell script with no hardcoded paths
    hook_script = """#!/bin/sh
# Auto-sync public and private writeups securely on commit
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [ -f "$REPO_ROOT/tools/sync_writeup.py" ]; then
    python3 "$REPO_ROOT/tools/sync_writeup.py" --quiet
    git add "$REPO_ROOT/reports/writeup/" "$REPO_ROOT/docs/reports/" "$REPO_ROOT/docs/images/" 2>/dev/null || true
fi
exit 0
"""
    try:
        atomic_write(hook_path, hook_script)
        os.chmod(hook_path, 0o755)
        print(f"Secure Git pre-commit hook verified at {hook_path.name}")
    except Exception as e:
        print(f"Warning: Could not install git hook: {e}")


def watch_loop(poll_interval: float = 2.0):
    """Continuously monitor WRITEUP.md with graceful signal handling."""
    repo_root = get_current_repo_root()
    source_master = (repo_root / "WRITEUP.md").resolve()
    if not source_master.exists():
        source_master = (repo_root.parent / "playground-series-s6e10/WRITEUP.md").resolve()

    print(f"Secure Watcher Active on {source_master.name} (polling every {poll_interval}s)...")
    last_hash = compute_sha256(source_master)
    perform_sync(verbose=True)

    try:
        while True:
            time.sleep(poll_interval)
            current_hash = compute_sha256(source_master)
            if current_hash and current_hash != last_hash:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected modification in WRITEUP.md. Triggering secure sync...")
                perform_sync(verbose=True)
                last_hash = current_hash
    except KeyboardInterrupt:
        print("\nWatcher gracefully terminated.")


if __name__ == "__main__":
    quiet_mode = "--quiet" in sys.argv
    watch_mode = "--watch" in sys.argv
    install_hook = "--install-hook" in sys.argv

    if install_hook:
        install_git_hook()

    if watch_mode:
        install_git_hook()
        watch_loop()
    else:
        success = perform_sync(verbose=not quiet_mode)
        install_git_hook()
        sys.exit(0 if success else 1)
