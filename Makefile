.PHONY: gate gate-fast test-unit test-integration test-e2e test-live

gate:
	python scripts/gate.py

gate-fast:
	python scripts/gate.py --fast

test-unit:
	python -m pytest -m unit -q

test-integration:
	python -m pytest -m integration -q

test-e2e:
	python scripts/gate.py --only staging-e2e

test-live:
	python scripts/gate.py --only live
