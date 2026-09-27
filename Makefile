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

# Extended, provider-free regression checks. Reports stay outside source control.
QUALITY_RUN_DIR ?= .evals/quality
.PHONY: test-quality test-properties test-mutation test-soak smoke-deployed

test-quality:
	$(PYTHON) scripts/rag_quality.py --output $(QUALITY_RUN_DIR)/quality.json

test-properties:
	$(PYTHON) -m pytest tests/test_queue_properties.int.test.py tests/test_outbox_properties.int.test.py -q

test-mutation:
	$(PYTHON) scripts/mutation_gate.py --output $(QUALITY_RUN_DIR)/mutation.json

test-soak:
	$(PYTHON) scripts/pipeline_soak.py --output $(QUALITY_RUN_DIR)/soak.json

# Explicit revision required: a healthy old process must not pass a release smoke.
smoke-deployed:
	@test -n "$(EXPECTED_REVISION)" || (echo 'Set EXPECTED_REVISION to the deployed Git commit'; exit 2)
	$(PYTHON) scripts/deployed_smoke.py --container $(or $(DEPLOYED_CONTAINER),doc-organizer) --expected-revision $(EXPECTED_REVISION) --output $(QUALITY_RUN_DIR)/deployed.json
