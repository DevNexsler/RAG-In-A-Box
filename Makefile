.PHONY: gate gate-fast gate-real test-unit test-integration test-e2e test-e2e-real test-live

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

gate:
	$(PYTHON) scripts/gate.py

gate-fast:
	$(PYTHON) scripts/gate.py --fast

# Full gate + a final real-API e2e pass (media + enrichment live). SPENDS MONEY;
# needs a real OPENROUTER_API_KEY. Runs only after every deterministic tier passes.
gate-real:
	$(PYTHON) scripts/gate.py --with-real-e2e

test-unit:
	$(PYTHON) -m pytest -m unit -q

test-integration:
	$(PYTHON) -m pytest -m integration -q

test-e2e:
	$(PYTHON) scripts/gate.py --only staging-e2e

# Just the real-API e2e stage (SPENDS MONEY; needs a real OPENROUTER_API_KEY).
test-e2e-real:
	$(PYTHON) scripts/gate.py --only e2e-real

test-live:
	$(PYTHON) scripts/gate.py --only live
