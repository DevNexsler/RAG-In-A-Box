"""The cross-encoder request is held to a token budget so a long payload cannot
outlast RERANK_TIMEOUT_SECONDS and degrade the search (2026-10-15: a 154k-token
request of accounting emails timed out on every attempt)."""

from __future__ import annotations

import httpx

from core.storage import SearchHit
from search_hybrid import (
    RERANK_DOC_CHARS,
    RERANK_PAIR_OVERHEAD_TOKENS,
    RERANK_TOKEN_BUDGET,
    DeepInfraReranker,
    _rerank_token_starts,
    fit_rerank_documents,
)


def estimated_request(query: str, docs: list[str]) -> int:
    per_pair = len(_rerank_token_starts(query)) + RERANK_PAIR_OVERHEAD_TOKENS
    return sum(len(_rerank_token_starts(doc)) + per_pair for doc in docs)


# An invoice digest: digit-dense, the shape that blew the budget in production.
INVOICE = ("Josh Hendricks 2 $958.90 Patrick Gallagher 1 $1,600 Krystal Knighton 1 $1,500 " * 60)


def test_digits_count_one_token_each():
    # rent | 1 | , | 6 0 0 | . | 2 5
    assert len(_rerank_token_starts("rent 1,600.25")) == 9
    assert len(_rerank_token_starts("the tenant called")) == 3


def test_a_request_within_budget_is_sent_whole():
    texts = ["water leak in the kitchen ceiling " * 20, "short note", "x" * (RERANK_DOC_CHARS + 50)]
    docs = fit_rerank_documents("leak", texts)
    assert docs == [text[:RERANK_DOC_CHARS] for text in texts]


def test_an_over_budget_request_is_cut_to_one_shared_length_that_fits():
    texts = [INVOICE] * 70 + ["Lease renewal for 12 Oak Street"]
    query = "6 tenant invoices are due today"
    assert estimated_request(query, [t[:RERANK_DOC_CHARS] for t in texts]) > RERANK_TOKEN_BUDGET

    docs = fit_rerank_documents(query, texts)

    assert len(docs) == len(texts)  # every hit is still scored
    cut = len(docs[0])
    assert 0 < cut < RERANK_DOC_CHARS
    assert all(doc == text[:cut] or doc == text for doc, text in zip(docs, texts))
    assert estimated_request(query, docs) <= RERANK_TOKEN_BUDGET
    # The shared length is the longest that fits: the next token would not.
    longer = [t[: cut + 1] for t in texts]
    assert estimated_request(query, longer) > RERANK_TOKEN_BUDGET


def test_the_reranker_sends_the_fitted_documents(monkeypatch):
    sent = {}

    def fake_post(url, **kwargs):
        sent.update(kwargs["json"])
        return httpx.Response(
            200,
            json={"scores": [0.5] * len(kwargs["json"]["documents"])},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("httpx.post", fake_post)
    hits = [
        SearchHit(doc_id=f"d{i}", loc="1", snippet="s", text=INVOICE, score=1.0)
        for i in range(70)
    ]
    ranked = DeepInfraReranker(api_key="k").rerank("invoices due today", hits)

    assert len(ranked) == 70
    assert sent["documents"] == fit_rerank_documents("invoices due today", [INVOICE] * 70)
    assert len(sent["documents"][0]) < RERANK_DOC_CHARS
