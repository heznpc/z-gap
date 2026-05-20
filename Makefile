# Single entry point for reproducing the z-gap pilot.
# See experiments/README.md for the reproducibility envelope.

.PHONY: help setup smoke reproduce figures clean

PYTHON ?= python3
VENV   := experiments/.venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

help:
	@echo "Targets:"
	@echo "  setup      Create experiments/.venv and install requirements"
	@echo "  smoke      Validate imports and stimuli JSON without running the pipeline"
	@echo "  reproduce  Run the full pilot (scripts/run_all.py)"
	@echo "  figures    Re-run cross-experiment synthesis only"
	@echo "  clean      Remove venv and Python caches (keeps results/embeddings/)"

setup:
	cd experiments && $(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r experiments/requirements.txt
	@echo "Setup complete."
	@echo "Next: cp experiments/.env.example experiments/.env  # add OPENAI_API_KEY, MISTRAL_API_KEY"

smoke:
	$(PY) -c "import numpy, scipy, sklearn, matplotlib, seaborn, sentence_transformers, openai, requests, kiwipiepy; print('smoke OK')"
	$(PY) -c "import json, glob; [json.load(open(p)) for p in glob.glob('experiments/data/stimuli/*.json')]; print('stimuli JSON OK')"

reproduce:
	cd experiments && ../$(VENV)/bin/python scripts/run_all.py

figures:
	cd experiments && ../$(VENV)/bin/python scripts/run_cross_experiment_synthesis.py

clean:
	rm -rf experiments/.venv experiments/.pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "Caches cleared. experiments/results/ preserved."
