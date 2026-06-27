"""Central config: reproducibility seeds, pinned model ids, and data/checkpoint paths."""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---
SEED = 42
DATA_SEED = 42

# --- Data volume (small for the first end-to-end run; scale via CLI --n-docs) ---
DEFAULT_N_DOCS = 500
TRAIN_SPLIT = 0.9

# --- Model ids (research-verified) ---
STUDENT_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-Extract"
# "main" for an iterative learning run; for a strictly reproducible run, capture the resolved
# commit SHA (model.config._commit_hash at load time) into the split manifest and pin it here.
STUDENT_REVISION = "main"
TEACHER_MODEL_ID = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # NOTE: no /v1 (research-verified)

# --- Teacher labeling ---
TEACHER_MAX_TOKENS = 2048
TEACHER_TEMPERATURE = 0  # ignored when thinking is enabled, but we disable thinking

# --- Evaluation ---
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# --- Paths (resolved relative to the repo root = two parents above this file) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
ADAPTER_DIR = CHECKPOINT_DIR / "adapter"
MERGED_DIR = CHECKPOINT_DIR / "merged"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Stage data files
CLEAN_JSONL = DATA_DIR / "clean.jsonl"
DIRTY_JSONL = DATA_DIR / "dirty.jsonl"
LABELED_JSONL = DATA_DIR / "labeled.jsonl"
QUARANTINE_JSONL = DATA_DIR / "quarantine.jsonl"
SFT_DIR = DATA_DIR / "sft"


def ensure_dirs() -> None:
    """Create all output directories. Safe to call repeatedly."""
    for p in (DATA_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR, ADAPTER_DIR, MERGED_DIR, SFT_DIR):
        p.mkdir(parents=True, exist_ok=True)
