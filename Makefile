.PHONY: venv check test lint doctor data sample-data prepare train evaluate baseline all \
	active-data active-label train-general train-extract compare active-demo clean

# Load repo-root .env into the environment so recipe shells and guards see it.
# Python also loads .env at import (doc_extract.config), but the shell-level
# guards below run before Python starts, so Make must export the values too.
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
DOC_EXTRACT ?= $(PY) -m doc_extract.cli
N_DOCS ?= 500
SAMPLE_DOCS ?= 20
SEED ?= 42
MAX_TEACHER_LABELS ?= 100
MAX_NEW_TOKENS ?= 1024
GENERAL_MODEL ?= LiquidAI/LFM2.5-VL-1.6B
EXTRACT_MODEL ?= LiquidAI/LFM2.5-VL-1.6B-Extract

venv:
	$(PY) -m venv .venv
	@echo "Run: source .venv/bin/activate && pip install -e .[dev]"

check: lint test

test:
	$(PY) -m pytest -q -m "not slow and not network and not gpu and not model"

lint:
	$(PY) -m ruff check .

doctor:
	$(DOC_EXTRACT) doctor

data: data/clean.jsonl data/dirty.jsonl data/labeled.done

data/clean.jsonl:
	$(DOC_EXTRACT) generate --n-docs $(N_DOCS) --seed $(SEED) --out data/clean.jsonl

data/dirty.jsonl: data/clean.jsonl
	$(DOC_EXTRACT) corrupt --in data/clean.jsonl --out data/dirty.jsonl --seed $(SEED)

data/labeled.done: data/dirty.jsonl
	@test -n "$$DEEPSEEK_API_KEY" || (echo "Set DEEPSEEK_API_KEY"; exit 1)
	$(DOC_EXTRACT) label --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl --seed $(SEED)
	@touch $@

sample-data: data/sample/dirty.jsonl

data/sample/clean.jsonl:
	$(DOC_EXTRACT) generate --n-docs $(SAMPLE_DOCS) --seed $(SEED) --out data/sample/clean.jsonl

data/sample/dirty.jsonl: data/sample/clean.jsonl
	$(DOC_EXTRACT) corrupt --in data/sample/clean.jsonl --out data/sample/dirty.jsonl --seed $(SEED)

prepare: data/labeled.done
	$(DOC_EXTRACT) prepare --in data/labeled.jsonl --out-dir data/sft --seed $(SEED) --split 0.9

train: prepare
	$(DOC_EXTRACT) train

evaluate: train
	$(DOC_EXTRACT) evaluate

baseline: prepare
	$(DOC_EXTRACT) baseline

all: data prepare train evaluate

active-data: data/active/split.done

data/active/split.done: data/dirty.jsonl
	$(DOC_EXTRACT) split-gold --in data/dirty.jsonl --train-out data/active/train_pool.jsonl --test-out data/active/test_gold.jsonl --seed $(SEED) --split 0.8
	@touch $@

artifacts/predictions/base_general_train_pool.jsonl: data/active/split.done
	$(DOC_EXTRACT) predict --in data/active/train_pool.jsonl --out artifacts/predictions/base_general_train_pool.jsonl --run base_general --max-new-tokens $(MAX_NEW_TOKENS)

artifacts/predictions/base_extract_train_pool.jsonl: data/active/split.done
	$(DOC_EXTRACT) predict --in data/active/train_pool.jsonl --out artifacts/predictions/base_extract_train_pool.jsonl --run base_extract --max-new-tokens $(MAX_NEW_TOKENS)

data/active/hard_cases.jsonl: data/active/split.done artifacts/predictions/base_general_train_pool.jsonl artifacts/predictions/base_extract_train_pool.jsonl
	$(DOC_EXTRACT) mine-failures --train-pool data/active/train_pool.jsonl --general-predictions artifacts/predictions/base_general_train_pool.jsonl --extract-predictions artifacts/predictions/base_extract_train_pool.jsonl --out data/active/hard_cases.jsonl --max-labels $(MAX_TEACHER_LABELS)

data/active/hard_labeled.jsonl: data/active/hard_cases.jsonl
	@test -n "$$DEEPSEEK_API_KEY" || (echo "Set DEEPSEEK_API_KEY"; exit 1)
	$(DOC_EXTRACT) label-hard --in data/active/hard_cases.jsonl --out data/active/hard_labeled.jsonl --quarantine data/active/hard_quarantine.jsonl --max-labels $(MAX_TEACHER_LABELS) --seed $(SEED)

data/active/sft/train.jsonl: data/active/split.done data/active/hard_labeled.jsonl
	$(DOC_EXTRACT) prepare-active --train-pool data/active/train_pool.jsonl --hard-labels data/active/hard_labeled.jsonl --test-gold data/active/test_gold.jsonl --out-dir data/active/sft --seed $(SEED) --hard-per-easy 2

active-label: data/active/sft/train.jsonl

train-general: data/active/sft/train.jsonl
	$(DOC_EXTRACT) train --train-file data/active/sft/train.jsonl --base $(GENERAL_MODEL) --adapter-dir artifacts/checkpoints/general/adapter --merged-dir artifacts/checkpoints/general/merged

train-extract: data/active/sft/train.jsonl
	$(DOC_EXTRACT) train --train-file data/active/sft/train.jsonl --base $(EXTRACT_MODEL) --adapter-dir artifacts/checkpoints/extract/adapter --merged-dir artifacts/checkpoints/extract/merged

# Prerequisites matter under `make -j`: compare needs test_gold.jsonl + hard_cases.jsonl
# (built via the active-label -> split.done -> hard_cases chain) and the two fine-tuned
# merged checkpoints (ft_general/ft_extract load from the merged dirs). Without these
# edges parallel make races compare ahead of data generation -> FileNotFoundError.
compare: active-label train-general train-extract
	$(DOC_EXTRACT) compare --test-file data/active/test_gold.jsonl --hard-file data/active/hard_cases.jsonl --out artifacts/comparison_metrics.json --max-new-tokens $(MAX_NEW_TOKENS)

active-demo: active-label train-general train-extract compare

clean:
	rm -rf data/sft data/active data/labeled.done artifacts/checkpoints artifacts/predictions artifacts/metrics.json artifacts/comparison_metrics.json
