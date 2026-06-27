.PHONY: venv check test lint doctor data sample-data prepare train evaluate baseline all clean

PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
DOC_EXTRACT ?= $(PY) -m doc_extract.cli
N_DOCS ?= 500
SAMPLE_DOCS ?= 20
SEED ?= 42

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

clean:
	rm -rf data/sft data/labeled.done artifacts/checkpoints artifacts/metrics.json
