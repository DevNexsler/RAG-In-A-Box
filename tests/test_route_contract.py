"""Route contract observation (#1154).

The two response envelopes replayed here are the shapes captured either side of
the 2026-08-16 LiteLLM outage in
``.evals/llm-traces/<date>-litellm-ollama-deepseek-v4-pro.jsonl``: the same
alias, the same request, two opposite ``max_tokens`` contracts.
"""

import json
import logging
from unittest.mock import patch

import httpx
import pytest

from core.route_contract import (
    BUDGET_ENFORCED,
    BUDGET_UNBOUNDED,
    BUDGET_UNPROVEN,
    ROUTE_CONTRACTS,
    classify_budget_contract,
)
from providers.llm.litellm_llm import LiteLLMGenerator

_VALID_ENRICHMENT = json.dumps(
    {"summary": "A lease renewal notice.", "doc_type": ["email"], "topics": ["lease"]}
)


@pytest.fixture(autouse=True)
def _clean_registry():
    ROUTE_CONTRACTS.reset()
    yield
    ROUTE_CONTRACTS.reset()


def _cloud_overshoot_response(completion_tokens: int = 11902) -> httpx.Response:
    """The pre-outage shape: billed past the budget, still claims ``stop``."""
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    return httpx.Response(
        200,
        json={
            "id": "resp_0d3d365248292356016a7e63bacc088190a36f3c231af5ed28",
            "model": "ollama-deepseek-v4-pro",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": _VALID_ENRICHMENT,
                        "provider_specific_fields": {"refusal": None},
                    },
                }
            ],
            "usage": {"prompt_tokens": 15945, "completion_tokens": completion_tokens},
        },
        request=request,
    )


def _native_capped_response(completion_tokens: int = 5000) -> httpx.Response:
    """The post-outage shape: stopped at the budget, reasoning returned inline."""
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-390",
            "model": "ollama-deepseek-v4-pro",
            "system_fingerprint": "fp_ollama",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "We need answer as JSON only." * 400,
                    },
                }
            ],
            "usage": {"prompt_tokens": 10802, "completion_tokens": completion_tokens},
        },
        request=request,
    )


def _recovered_response() -> httpx.Response:
    request = httpx.Request("POST", "http://litellm.local/v1/chat/completions")
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-391",
            "model": "ollama-deepseek-v4-pro",
            "system_fingerprint": "fp_ollama",
            "choices": [
                {"finish_reason": "stop", "message": {"content": _VALID_ENRICHMENT}}
            ],
            "usage": {"prompt_tokens": 10802, "completion_tokens": 640},
        },
        request=request,
    )


@pytest.mark.parametrize(
    ("completion_tokens", "finish_reason", "expected"),
    [
        (11902, "stop", BUDGET_UNBOUNDED),
        (5001, "stop", BUDGET_UNBOUNDED),
        (5000, "length", BUDGET_ENFORCED),
        (5000, "stop", BUDGET_ENFORCED),
        (3021, "stop", BUDGET_UNPROVEN),
        (3021, "length", BUDGET_UNPROVEN),
        (None, "stop", BUDGET_UNPROVEN),
    ],
)
def test_classify_budget_contract(completion_tokens, finish_reason, expected):
    assert (
        classify_budget_contract(
            requested_tokens=5000,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
        == expected
    )


def test_classify_budget_contract_without_a_budget_proves_nothing():
    assert (
        classify_budget_contract(
            requested_tokens=None, completion_tokens=9000, finish_reason="stop"
        )
        == BUDGET_UNPROVEN
    )


def test_a_route_that_ignores_the_budget_is_reported_as_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="core.route_contract"):
        ROUTE_CONTRACTS.observe(
            "ollama-deepseek-v4-pro@http://litellm.local/v1",
            requested_tokens=5000,
            completion_tokens=11902,
            finish_reason="stop",
        )

    assert "ollama-deepseek-v4-pro" in caplog.text
    assert BUDGET_UNBOUNDED in caplog.text


def test_an_enforced_budget_is_recorded_without_a_warning(caplog):
    with caplog.at_level(logging.INFO, logger="core.route_contract"):
        ROUTE_CONTRACTS.observe(
            "ollama-deepseek-v4-pro@http://litellm.local/v1",
            requested_tokens=5000,
            completion_tokens=5000,
            finish_reason="length",
            backend="fp_ollama",
        )

    assert BUDGET_ENFORCED in caplog.text
    assert "fp_ollama" in caplog.text
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_repeated_observations_of_one_contract_stay_quiet(caplog):
    route = "ollama-deepseek-v4-pro@http://litellm.local/v1"
    with caplog.at_level(logging.INFO, logger="core.route_contract"):
        for _ in range(5):
            ROUTE_CONTRACTS.observe(
                route,
                requested_tokens=5000,
                completion_tokens=5000,
                finish_reason="length",
                backend="fp_ollama",
            )

    assert len(caplog.records) == 1


def test_unproven_budgets_are_not_recorded(caplog):
    route = "ollama-deepseek-v4-pro@http://litellm.local/v1"
    with caplog.at_level(logging.INFO, logger="core.route_contract"):
        ROUTE_CONTRACTS.observe(
            route,
            requested_tokens=5000,
            completion_tokens=3021,
            finish_reason="stop",
            backend="fp_ollama",
        )

    assert caplog.records == []


def test_a_contract_flip_on_one_route_is_reported_once(caplog):
    """The #1154 flip: one alias, two contracts, three days apart."""
    route = "ollama-deepseek-v4-pro@http://litellm.local/v1"
    with caplog.at_level(logging.WARNING, logger="core.route_contract"):
        ROUTE_CONTRACTS.observe(
            route,
            requested_tokens=5000,
            completion_tokens=11902,
            finish_reason="stop",
        )
        caplog.clear()
        for _ in range(3):
            ROUTE_CONTRACTS.observe(
                route,
                requested_tokens=5000,
                completion_tokens=5000,
                finish_reason="length",
                backend="fp_ollama",
            )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert BUDGET_UNBOUNDED in message and BUDGET_ENFORCED in message
    assert "fp_ollama" in message


def test_a_backend_swap_under_one_contract_is_reported(caplog):
    """The alias is unchanged and the budget still binds — a different server answered."""
    route = "ollama-deepseek-v4-pro@http://litellm.local/v1"
    with caplog.at_level(logging.WARNING, logger="core.route_contract"):
        ROUTE_CONTRACTS.observe(
            route,
            requested_tokens=5000,
            completion_tokens=5000,
            finish_reason="length",
            backend="fp_ollama",
        )
        ROUTE_CONTRACTS.observe(
            route,
            requested_tokens=5000,
            completion_tokens=5000,
            finish_reason="length",
            backend="fp_44709d6fcb83",
        )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "fp_44709d6fcb83" in warnings[0].getMessage()


def test_routes_are_tracked_independently(caplog):
    with caplog.at_level(logging.WARNING, logger="core.route_contract"):
        ROUTE_CONTRACTS.observe(
            "ollama-deepseek-v4-pro@http://litellm.local/v1",
            requested_tokens=5000,
            completion_tokens=5000,
            finish_reason="length",
            backend="fp_ollama",
        )
        ROUTE_CONTRACTS.observe(
            "openai/gpt-4.1-mini@https://openrouter.ai/api/v1",
            requested_tokens=5000,
            completion_tokens=5000,
            finish_reason="length",
            backend="fp_openrouter",
        )

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_litellm_generator_reports_the_route_contract_it_observed(caplog):
    """End of the wire: the flip is visible from the generator, per index run."""
    with patch(
        "providers.llm.litellm_llm.httpx.post",
        side_effect=[
            _cloud_overshoot_response(),
            _recovered_response(),
            _native_capped_response(),
            _recovered_response(),
        ],
    ):
        generator = LiteLLMGenerator(
            model="ollama-deepseek-v4-pro",
            base_url="http://litellm.local/v1",
            api_key="secret-key",
        )
        with caplog.at_level(logging.INFO, logger="core.route_contract"):
            generator.generate("a document", max_tokens=5000)
            generator.generate("another document", max_tokens=5000)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        BUDGET_UNBOUNDED in message and "ollama-deepseek-v4-pro" in message
        for message in messages
    )
    flip = [
        message
        for message in messages
        if BUDGET_UNBOUNDED in message and BUDGET_ENFORCED in message
    ]
    assert len(flip) == 1
    assert "fp_ollama" in flip[0]
