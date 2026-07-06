.PHONY: gate gate-fast gate-real test-unit test-integration test-e2e test-e2e-real test-live

gate:
	python scripts/gate.py

gate-fast:
	python scripts/gate.py --fast

# Full gate + a final real-API e2e pass (media + enrichment live). SPENDS MONEY;
# needs a real OPENROUTER_API_KEY. Runs only after every deterministic tier passes.
gate-real:
	python scripts/gate.py --with-real-e2e

test-unit:
	python -m pytest -m unit -q

test-integration:
	python -m pytest -m integration -q

test-e2e:
	python scripts/gate.py --only staging-e2e

# Just the real-API e2e stage (SPENDS MONEY; needs a real OPENROUTER_API_KEY).
test-e2e-real:
	python scripts/gate.py --only e2e-real

test-live:
	python scripts/gate.py --only live
