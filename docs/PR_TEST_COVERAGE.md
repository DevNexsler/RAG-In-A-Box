# Merged PR regression checks

Baseline: `f6a8264c223652fa728dd6f3739b1bb905ff6dbd`, following the 37-PR cleanup and enrichment correction PR #201. Every original PR is accounted for below. A targeted mutation proves sensitivity to its named fault, not every possible regression in that PR.

Run `make test-mutation` for the seven safety mutations plus the PR-specific cases in `tests/fixtures/pr_regression_mutations.json`. The runner uses disposable copies of tracked working-tree files: stage new files first. Every selector must pass without skips before mutation; changed test inventories, collection errors, crashes, timeouts, syntax errors, and invalid anchors fail the campaign. JSON/JUnit/log evidence identifies the selector and observed failure.

| PR | Regression checked | Mutation or other verification | Tests |
|---|---|---|---|
| #198 | Expose exact platform ID lookup in Context Builder | `platform-id-authority` | `tests/contracts/test_factbook_identity.int.test.py` |
| #197 | fix(search): trip the circuit on a 402 refusal and report it in provider health (#2906) | `account-refusal-circuit` | `tests/test_endpoint_circuit.py::test_account_refusal_trips_on_first_failure_and_is_reported` |
| #196 | fix(#2830): stop duplicate document.indexed delivery during dedupe sweeps | `callback-dispatch-ownership` | `tests/hooks/test_delivery.py::test_send_enqueued_only_sends_rows_from_this_dispatch` |
| #195 | fix(store): widen stale-read recovery for fresh-index schema bursts (#2626) | `schema-burst-recovery` | `tests/test_store.py::test_read_survives_a_burst_of_schema_swaps_during_recovery` |
| #194 | fix(index): compact duplicate chunk rows left by mid-sweep queue races (#3174) | `duplicate-newest-row` | `tests/test_duplicate_indexing.int.test.py::test_compact_duplicate_chunk_rows_keeps_newest_physical_row` |
| #193 | fix(index): prevent duplicate chunk rows when mid-sweep queue serves a doc (#3143) | `sweep-insert-idempotence` | `tests/test_store.py::test_known_absent_insert_skips_after_same_session_upsert` |
| #192 | fix(enrichment): keep context-derived facts out of primary key_facts/keywords (#2611) | `context-fact-provenance` | `tests/test_enrichment.py::TestContextOnlyFactsLeavePrimaryFields` |
| #191 | fix(backup): promote first ISO-week backup when odd-day schedule skips Sunday (#2325) | `first-weekly-backup` | `tests/test_backup_retention.py::test_first_backup_in_iso_week_promoted_when_sunday_is_even_date` |
| #190 | chore: untrack generated superpowers scratch and ignore it | Tracked-file hygiene assertion in normal unit gate; source mutation not applicable | `tests/test_tracked_files_hygiene.py` |
| #189 | fix(enrichment): constrain enr_doc_type to controlled vocabulary (#3050) | `controlled-doc-type` | `tests/test_doc_type_vocabulary.py::test_constrain_doc_type_all_unknown_becomes_unclassified_sentinel` |
| #188 | fix(taxonomy): skip no-op folder sync that cost ~40s per indexer run | `bulk-taxonomy-sync` | `tests/test_taxonomy_store.py::TestFolderSync::test_unchanged_folder_tree_skips_per_entry_lookups` |
| #187 | fix(index): count mid-sweep served writes on run completion (#2692) | `queue-write-accounting` | `tests/test_index_run_served_accounting.py` |
| #186 | fix(health): expose live vector index metadata and drift (#2226) | `live-index-health` | `tests/test_vector_index_observability.py::test_health_reports_live_index_without_store_init_or_writer_lock` |
| #185 | fix(indexer): reconcile and report terminal degraded state (#2022) | `retired-ledger-reconciliation`, `retired-terminal-residue` | `tests/test_degraded_ledger.py::test_namespaced_ledger_clears_legacy_bare_retirement`; `tests/test_degraded_ledger.py::test_retired_terminal_residue_clears_but_unknown_entry_stays` |
| #184 | fix: use human message titles in embedded headers (#2691) | `human-message-title` | `tests/test_message_titles.py` |
| #182 | fix(enrichment): store facts only nearby context supports as context facts (Maint #2562) | `context-fact-provenance`; superseded implementation shares replacement regression | `tests/test_enrichment.py::TestContextOnlyFactsLeavePrimaryFields` |
| #181 | fix(enrichment): keep card-suffix claims to cards the source shows (Maint #2526) | `card-grounding` (existing safety mutation) | `tests/test_enrichment_invariants.py` |
| #179 | fix: preserve policy questions during enrichment (#2259) | `question-modality-contract`; prompt contract only. Real language behavior remains a live quality check. | `tests/test_enrichment_modality.py::test_modality_contract_reaches_generator` |
| #171 | fix(index): count terminal failures on the run's skip roll-up (#2184) | `terminal-skip-rollup` | `tests/test_index_run_accounting.int.test.py::test_terminal_failure_counts_the_same_on_both_skip_roll_ups` |
| #170 | fix(index): model a capped degraded doc as terminal, not retry-pending (#2101) | `terminal-retry-budget` | `tests/test_degraded_ledger.py::test_terminal_entry_covers_both_budgets` |
| #169 | fix(index): log every terminal skip per document, from the ledger seam (#2100) | `per-document-skip-log` | `tests/test_index_run_accounting.int.test.py::test_every_skipped_document_is_named_in_the_log` |
| #168 | fix(dedupe): never elect a terminal-skip document as a cohort canonical (#2097) | `terminal-dedupe-canonical` | `tests/test_content_dedupe_flow.py::test_terminal_skip_document_is_not_elected_dedupe_canonical` |
| #167 | feat(#2074): clear registry-retired ids from the terminal degraded ledger | `retired-ledger-reconciliation`, `retired-terminal-residue`; superseded implementation shares replacement regression | `tests/test_degraded_ledger.py::test_namespaced_ledger_clears_legacy_bare_retirement`; `tests/test_degraded_ledger.py::test_retired_terminal_residue_clears_but_unknown_entry_stays` |
| #166 | fix(index): isolate a failing source's scan from the rest of the run (#2020) | `failed-source-partial-records` | `tests/test_source_scan_isolation.py::test_failed_source_contributes_no_partial_records` |
| #165 | test: use speech fixture for live transcription (#1943) | Live speech fixture; excluded from hermetic mutation campaign | `tests/test_media_live.py` |
| #164 | fix: classify empty OpenRouter enrichment responses (#1944) | `empty-enrichment-retry` | `tests/test_benchmark_runner.py::test_openrouter_retries_an_empty_enrichment_summary` |
| #163 | fix(media): route audio models by capability | `audio-route-capability` | `tests/test_media.py::test_openrouter_whisper_uses_transcription_endpoint` |
| #162 | maint/1918 doc organizer accepts incomplete enrichment json a | `structured-response-validation` | `tests/test_enrichment_structured_retry.py::test_structured_response_contract_rejects_wrong_types_ranges_and_schema_tokens` |
| #161 | fix(#1876): keep queue progress visible to health probe | `queue-progress-heartbeat` | `tests/test_index_queue_fairness.py::test_sweep_queue_service_refreshes_heartbeat_between_requests` |
| #160 | fix: bound tracking-heavy HTML email chunks (#1882) | `document-chunk-bound` | `tests/test_scan.py::test_document_chunk_budget_caps_and_reports_degradation` |
| #159 | fix(index): date the foreign heartbeat so a no-evidence reconcile stays non-blocking (#1827) | `predecessor-heartbeat-age` | `tests/test_index_run_supervisor.py::test_startup_reconciles_counterless_run_behind_predecessor_heartbeat` |
| #158 | fix: ground explicit enrichment corrections | `explicit-correction-grounding` | `tests/test_enrichment_postprocess.py::test_explicit_email_correction_and_semicolon_inversion_are_grounded` |
| #157 | fix(#1690): distinguish transient index deferrals in logs | `transient-deferral-severity` | `tests/test_skip_ledger.py::test_transient_processing_failures_are_logged_as_deferred` |
| #156 | maint/1688 doc organizer index maintenance is coupled to run c | `independent-idle-maintenance` | `tests/test_index_scheduler.py::test_maintenance_runs_after_seeded_clean_boot_without_a_sweep_completion` |
| #155 | fix(#1687): substitute, don't send, an embedding input with no text | `empty-embedding-substitution` | `tests/test_embed_input_limit.py::test_text_free_input_is_substituted_not_dropped` |
| #141 | fix(index): finalize full FTS rebuild maintenance (#1630) | `fts-rebuild-maintenance` | `tests/test_store.py::test_full_fts_rebuild_finishes_index_maintenance` |
| #79 | #0584 record extraction provenance so content-free docs are not indexed as success | `missing-primary-content` | `tests/test_incomplete_indexing.int.test.py::test_failed_image_description_is_consumer_visible_and_stable` |

PR #201 shares `explicit-correction-grounding`; corrected-order and retraction regressions also run in `tests/test_enrichment_postprocess.py`. The new migration fix has `schema-index-preservation`: fresh and concurrent readers must retain ANN and FTS without an end-of-run repair.

## Collection completeness

Pytest importlib can synthesize a parent module from `test_name.int.test.py` and silently hide `test_name.py` in the same package. Integration filenames now have distinct stems. `scripts/pytest_collection_guard.py`, loaded by `tests/conftest.py`, validates every collected module against its actual file, including modules with no collected items. Incorrect identity aborts collection even when a tier marker deselects its cases. The normal gate again includes all 23 supervisor and five source-isolation unit cases.

## Search during schema changes

`tests/test_schema_search_availability.int.test.py` exercises public writes against concurrent and fresh readers, rollback after a failed index subprocess, retry, ANN settings, native FTS phrase behavior, and scalar indexes. Replacement indexes are built before promotion in a worker guarded by the existing cgroup memory ceiling. Unsupported index types and rebuild failures abort the migration while preserving the serving table. Insufficient memory therefore defers the metadata write; operators must provide headroom before retrying. Initial indexing of an unindexed table retains its existing behavior.

Consumer HTTP contracts still need provider-side Factbook/CDS tests. Live model semantics, real deployment smoke checks, and long soak memory limits remain distinct release evidence. This campaign is not a whole-repository mutation score.
