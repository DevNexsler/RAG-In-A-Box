"""What a provider route proves about the budget we send it.

An enrichment route reaches us as an alias — a model name plus a base URL — but
the contract behind that alias belongs to the provider. Around the 2026-08-16
LiteLLM outage the backend serving ``ollama-deepseek-v4-pro`` changed underneath
an unchanged route: before it billed reasoning past ``max_tokens`` and still
answered ``finish_reason=stop``, after it stopped generation exactly at
``max_tokens`` and answered ``length`` (#1154). No commit of ours moved, yet
every guard we calibrate against that route — truncation detection, budget
sizing, reasoning suppression — quietly changed which contract it was defending.

Each response carries the evidence: whether the budget we asked for bound
generation, and which backend answered. This module keeps that evidence per
route and reports it once, so a flip is a line in the run's log the day it
happens instead of an archaeology exercise over captured traces. It is
deliberately provider-agnostic — any OpenAI-compatible route can be observed
through it.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# The budget stopped generation: the provider honours what we send.
BUDGET_ENFORCED = "enforced"
# The provider billed more completion tokens than the budget allowed, so the
# budget never bound — output size and cost are the provider's decision.
BUDGET_UNBOUNDED = "unbounded"
# Generation finished below the budget: it says nothing either way.
BUDGET_UNPROVEN = "unproven"

_UNKNOWN_BACKEND = "unknown"


def classify_budget_contract(
    *,
    requested_tokens: int | None,
    completion_tokens: int | None,
    finish_reason: str = "",
) -> str:
    """Classify what one response proves about the ``max_tokens`` we sent.

    ``finish_reason`` is evidence, never the verdict: a route that cuts
    generation at the budget can still report ``stop``
    (``knowledge/runbooks/ollama-cloud-reasoning-truncation.md``), and a route
    that reports ``length`` below our budget was bound by something else.
    """
    if not requested_tokens or requested_tokens <= 0 or completion_tokens is None:
        return BUDGET_UNPROVEN
    if completion_tokens > requested_tokens:
        return BUDGET_UNBOUNDED
    if completion_tokens == requested_tokens:
        return BUDGET_ENFORCED
    return BUDGET_UNPROVEN


class RouteContracts:
    """Budget contracts observed per route in this process.

    Thread-safe: enrichment processes documents concurrently against one route.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # route -> observed (contract, backend) pairs, in the order first seen
        self._state: dict[str, list[tuple[str, str]]] = {}

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def observe(
        self,
        route: str,
        *,
        requested_tokens: int | None,
        completion_tokens: int | None,
        finish_reason: str = "",
        backend: str | None = None,
    ) -> str:
        """Record and report what this response proves about ``route``.

        Returns the contract observed. Silent for every repeat of a pair this
        route has already shown, so a route reports once per run and again the
        moment it starts behaving differently.
        """
        contract = classify_budget_contract(
            requested_tokens=requested_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
        if contract == BUDGET_UNPROVEN:
            return contract

        observed = (contract, backend or _UNKNOWN_BACKEND)
        with self._lock:
            seen = self._state.setdefault(route, [])
            if observed in seen:
                return contract
            previous = list(seen)
            seen.append(observed)

        if previous:
            logger.warning(
                "Enrichment route %s changed its max_tokens contract: now %s "
                "(backend=%s, completion_tokens=%s of %s), previously %s. Guards "
                "calibrated against the previous contract may no longer hold.",
                route,
                observed[0],
                observed[1],
                completion_tokens,
                requested_tokens,
                ", ".join(f"{name} (backend={fingerprint})" for name, fingerprint in previous),
            )
        elif contract == BUDGET_UNBOUNDED:
            logger.warning(
                "Enrichment route %s max_tokens contract: %s — it billed %s "
                "completion tokens against a %s budget (backend=%s), so the "
                "budget we send does not bind this route.",
                route,
                contract,
                completion_tokens,
                requested_tokens,
                observed[1],
            )
        else:
            logger.info(
                "Enrichment route %s max_tokens contract: %s at %s tokens "
                "(backend=%s).",
                route,
                contract,
                requested_tokens,
                observed[1],
            )
        return contract


ROUTE_CONTRACTS = RouteContracts()
