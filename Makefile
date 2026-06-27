.PHONY: venv data prepare train evaluate baseline all clean

PY ?= python
N_DOCS ?= 500
SEED ?= 42

venv:
	$(PY) -m venv .venv
	@echo "Run: source .venv/bin/activate && pip install -e .[dev]"

data: data/clean.jsonl data/dirty.jsonl data/labeled.jsonl

data/clean.jsonl:
	$(PY) -m doc_extract.generate --n-docs $(N_DOCS) --seed $(SEED) --out data/clean.jsonl

data/dirty.jsonl: data/clean.jsonl
	$(PY) -m doc_extract.corrupt --in data/clean.jsonl --out data/dirty.jsonl --seed $(SEED)

data/labeled.jsonl: data/dirty.jsonl
	@test -n "$$DEEPSEEK_API_KEY" || (echo "Set DEEPSEEK_API_KEY"; exit 1)
	$(PY) -m doc_extract.teacher_labeler --in data/dirty.jsonl --out data/labeled.jsonl --quarantine data/quarantine.jsonl

prepare: data/labeled.jsonl
	$(PY) -m doc_extract.prepare --in data/labeled.jsonl --out-dir data/sft --seed $(SEED) --split 0.9

train: prepare
	$(PY) -m doc_extract.train

evaluate: train
	$(PY) -m doc_extract.evaluate

baseline: prepare
	$(PY) -m doc_extract.evaluate --ft $(shell $(PY) -c "from doc_extract import config;print(config.STUDENT_MODEL_ID)")

all: data prepare train evaluate

clean:
	rm -rf data/sft artifacts/checkpoints artifacts/metrics.json
