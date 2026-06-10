# Fast Multihop RAG Next Phase Improvement Plan

> Status: deferred capture for later design review.
> Created: 2026-05-13

## Goal

Add a fast multihop retrieval path that captures some benefits of agentic RAG without running a heavyweight LLM reasoning loop for every search step.

## Core Idea

Use a small, specialized pipeline for common linear multihop questions:

```text
User question
-> lightweight query decomposer
-> vector search per hop
-> context compressor
-> lightweight reader/extractive QA per hop
-> confidence gate
-> optional heavier agentic fallback
```

This keeps the common path fast and predictable while reserving expensive agentic reasoning for low-confidence or broken search paths.

## Problem To Solve

Regular RAG works for direct queries, but struggles when the answer depends on hidden intermediate facts.

Example:

```text
Question: What 1993 dinosaur movie was directed by the maker of the 1975 shark film?

Hop 0: 1975 shark film -> Jaws
Hop 1: Director of Jaws -> Steven Spielberg
Hop 2: 1993 dinosaur movie by Steven Spielberg -> Jurassic Park
```

A single vector search may miss this because no one document has all indirect references. The system needs search, read, substitute intermediate answer, then search again.

## Proposed Architecture

### 1. Query Decomposer

Use a small sequence-to-sequence model, such as T5, to turn a complex question into ordered search hops.

Example output:

```text
h0: Who played the "Alright, alright, alright" character in Dazed and Confused?
h1: What space wormhole movie starred Matthew McConaughey?
h2: Who directed Interstellar?
h3: What 2010 dream-heist movie was directed by Christopher Nolan?
```

Desired properties:

- Fast local or low-latency inference.
- Consistent structured output.
- Linear hop plans for common `A -> B -> C -> answer` questions.
- No heavyweight planning loop on the fast path.

### 2. Hop Search

Each hop becomes a query into the existing document index.

Search should reuse existing retrieval primitives where possible:

- Vector search for semantic recall.
- Keyword or FTS search for exact entity and date anchors.
- Existing reranking where latency budget allows.
- Metadata filters when the decomposer can identify entity type, date, source, or document class.

### 3. Context Compressor

Compress retrieved text before sending it to the reader.

Purpose:

- Remove irrelevant context.
- Reduce reader latency.
- Improve extraction accuracy.
- Avoid naive truncation.

Candidate approaches:

- LLMLingua2-style compression.
- Local extractive sentence selector.
- Score-based chunk trimming using query terms plus embedding similarity.

### 4. Lightweight Reader

Use a small reader model to extract intermediate answers from retrieved evidence.

Options:

- Extractive QA model that returns an answer span.
- Small local LLM constrained to short JSON output.
- Hybrid: extractive reader first, small LLM only when span extraction is weak.

Reader output should include:

```json
{
  "answer": "Christopher Nolan",
  "confidence": 0.91,
  "evidence_doc_id": "example-doc-id",
  "evidence_span": "Interstellar was directed by Christopher Nolan..."
}
```

### 5. Confidence Gate

Keep the fast path only when evidence is strong enough.

Signals:

- Reader confidence.
- Retrieval score margin.
- Evidence span contains expected entity/date/type.
- Hop answer successfully substitutes into the next hop.
- Final answer supported by a cited source.

If confidence is low, escalate to fallback.

### 6. Agentic Fallback

Use heavier agentic RAG only when needed.

Fallback triggers:

- Decomposer emits invalid or ambiguous hops.
- Retrieval returns weak evidence.
- Reader cannot extract a reliable answer.
- Later hop contradicts earlier hop.
- Final answer lacks source support.
- User question requires branching, comparison, conflict resolution, or synthesis.

Fallback behavior:

```text
Fast pipeline runs first.
If confidence is low, a larger agent reviews evidence, repairs the plan, or performs dynamic replanning.
```

## Expected Benefits

- Lower latency for common multihop questions.
- Lower API cost than full agentic search.
- More predictable performance.
- Easier observability per hop.
- Clear fallback boundary for hard cases.

Target benchmark from reference idea:

```text
T5 decomposition: about 200 ms
Each hop: about 200-240 ms
Network/stream overhead: about 150 ms
Four-hop total: about 1 second
```

These numbers are aspirational and need local benchmarking against this repo's actual index, embedding provider, reranker, and reader choices.

## Known Tradeoffs

This works best for mostly linear chains.

It may struggle with:

- Multiple branches.
- Ambiguous references.
- Conflicting evidence.
- Questions that are underspecified.
- Broad synthesis across many documents.
- Bad first-hop decomposition.

The design should treat the fast path as a high-confidence shortcut, not as a replacement for full agentic reasoning.

## Possible Implementation Phases

### Phase 1: Design Spike

- Identify current retrieval entry points and reusable search APIs.
- Choose decomposer model candidate.
- Choose reader model candidate.
- Define hop plan JSON schema.
- Define confidence scoring fields.
- Add offline benchmark set of multihop questions.

### Phase 2: Prototype Fast Path

- Add decomposer wrapper.
- Add hop executor.
- Add answer substitution between hops.
- Add reader interface.
- Add simple compressor or top-k chunk trimming.
- Return final answer with hop trace and citations.

### Phase 3: Confidence And Fallback

- Add confidence gate.
- Add failure reasons.
- Route low-confidence cases to existing heavier search or future agentic repair path.
- Log per-hop timings and evidence.

### Phase 4: Benchmark And Tune

- Measure latency per stage.
- Compare against current RAG behavior.
- Track answer correctness on direct vs multihop questions.
- Tune hop count limits, top-k, compression budget, and fallback threshold.

## Open Questions For Later Review

- Should decomposer be a local T5 model, a small hosted model, or a fine-tuned local model?
- Is extractive QA enough for the repo's document style, or is a small LLM reader required?
- Which current search path should hop search reuse?
- Should Graph RAG entity edges be used as optional evidence when available?
- What confidence threshold avoids false certainty without sending too many queries to fallback?
- How should this surface through MCP responses: normal answer only, or answer plus hop trace?

## Acceptance Criteria Draft

- A four-hop benchmark query can run without a heavyweight LLM planner on the fast path.
- Each hop records query, retrieved docs, compressed context size, reader answer, confidence, and timing.
- Final response includes source citations.
- Low-confidence runs escalate instead of returning unsupported answers.
- Unit tests cover decomposition schema parsing, answer substitution, confidence gating, and fallback trigger logic.
- Benchmark report compares latency and correctness against existing search behavior.
