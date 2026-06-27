"""Central config: reproducibility seeds, pinned model ids, and data/checkpoint paths."""

from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = value.split(" #", 1)[0].strip()
    return key, value


def load_project_env(path: Path) -> int:
    """Load KEY=VALUE lines from a project .env without overriding the shell environment."""
    if not path.exists():
        return 0

    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


# --- Reproducibility ---
SEED = 42
DATA_SEED = 42

# --- Data volume (small for the first end-to-end run; scale via CLI --n-docs) ---
DEFAULT_N_DOCS = 500
TRAIN_SPLIT = 0.9

# --- Model ids (research-verified) ---
GENERAL_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"
EXTRACT_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B-Extract"

# Backwards-compatible default for the original single-student pipeline.
STUDENT_MODEL_ID = EXTRACT_MODEL_ID
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
load_project_env(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
ADAPTER_DIR = CHECKPOINT_DIR / "adapter"
MERGED_DIR = CHECKPOINT_DIR / "merged"
GENERAL_CHECKPOINT_DIR = CHECKPOINT_DIR / "general"
GENERAL_ADAPTER_DIR = GENERAL_CHECKPOINT_DIR / "adapter"
GENERAL_MERGED_DIR = GENERAL_CHECKPOINT_DIR / "merged"
EXTRACT_CHECKPOINT_DIR = CHECKPOINT_DIR / "extract"
EXTRACT_ADAPTER_DIR = EXTRACT_CHECKPOINT_DIR / "adapter"
EXTRACT_MERGED_DIR = EXTRACT_CHECKPOINT_DIR / "merged"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_METRICS_PATH = ARTIFACTS_DIR / "comparison_metrics.json"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# Stage data files
CLEAN_JSONL = DATA_DIR / "clean.jsonl"
DIRTY_JSONL = DATA_DIR / "dirty.jsonl"
LABELED_JSONL = DATA_DIR / "labeled.jsonl"
QUARANTINE_JSONL = DATA_DIR / "quarantine.jsonl"
SFT_DIR = DATA_DIR / "sft"
ACTIVE_DIR = DATA_DIR / "active"
TRAIN_POOL_JSONL = ACTIVE_DIR / "train_pool.jsonl"
TEST_GOLD_JSONL = ACTIVE_DIR / "test_gold.jsonl"
HARD_CASES_JSONL = ACTIVE_DIR / "hard_cases.jsonl"
HARD_LABELED_JSONL = ACTIVE_DIR / "hard_labeled.jsonl"
HARD_QUARANTINE_JSONL = ACTIVE_DIR / "hard_quarantine.jsonl"
ACTIVE_SFT_DIR = ACTIVE_DIR / "sft"

GOLD_TRAIN_SPLIT = 0.8
MAX_TEACHER_LABELS = 100
HARD_PER_EASY = 2


def ensure_dirs() -> None:
    """Create all output directories. Safe to call repeatedly."""
    for p in (
        DATA_DIR,
        ARTIFACTS_DIR,
        CHECKPOINT_DIR,
        ADAPTER_DIR,
        MERGED_DIR,
        SFT_DIR,
        GENERAL_CHECKPOINT_DIR,
        GENERAL_ADAPTER_DIR,
        GENERAL_MERGED_DIR,
        EXTRACT_CHECKPOINT_DIR,
        EXTRACT_ADAPTER_DIR,
        EXTRACT_MERGED_DIR,
        PREDICTIONS_DIR,
        ACTIVE_DIR,
        ACTIVE_SFT_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)
