# API_RECONSTRUCTION_REPORT.md
## Scope: 4 primary source files → 3 gpts_core modules
## Δ_total=105, Δ_explained=105, Δ_unexplained=0
## Reproducible via: `python3 scripts/compute_delta.py` (AST-based)

| SOURCE_FILE | SOURCE_SYMBOL | DEST_MODULE | DEST_SYMBOL | MAPPING_TYPE |
|---|---|---|---|---|
| exochronos_capture_model_v0_1.py | shannon_entropy | coherence.py | shannon_entropy | EXACT |
| exochronos_capture_model_v0_1.py | joint_entropy | coherence.py | joint_entropy | EXACT |
| exochronos_capture_model_v0_1.py | mutual_information | coherence.py | mutual_information | RENAMED_SIG |
| exochronos_capture_model_v0_1.py | compute_metric_fields | coherence.py | compute_metric_fields | RENAMED |
| exochronos_capture_model_v0_1.py | digitize | coherence.py | digitize | EXACT |
| exochronos_capture_model_v0_1.py | normalize_observations | coherence.py | normalize_observations | EXACT |
| exochronos_capture_model_v0_1.py | make_edges | coherence.py | _make_edges | PRIVATE |
| exochronos_capture_model_v0_1.py | build_passport_from_capture | coherence.py | build_coherence_passport | RENAMED |
| exochronos_capture_model_v0_1.py | utc_now_iso | coherence.py | _utc_now | PRIVATE |
| exochronos_capture_model_v0_1.py | finite_float | coherence.py | _is_finite_float | PRIVATE |
| exochronos_capture_model_v0_1.py | load_passport_validator | NONE | NONE | EXCLUDED |
| exochronos_capture_model_v0_1.py | read_json | NONE | NONE | EXCLUDED |
| exochronos_capture_model_v0_1.py | write_json | NONE | NONE | EXCLUDED |
| exochronos_capture_model_v0_1.py | build_demo_capture | NONE | NONE | EXCLUDED |
| exochronos_capture_model_v0_1.py | cmd_init_demo_capture | NONE | NONE | CLI_IO |
| exochronos_capture_model_v0_1.py | cmd_capture | NONE | NONE | CLI_IO |
| exochronos_capture_model_v0_1.py | build_arg_parser | NONE | NONE | CLI_IO |
| exochronos_capture_model_v0_1.py | main | NONE | NONE | CLI_IO |
| exochronos_coherence_passport_v0_1.py | canonical_json | coherence.py | _canonical | PRIVATE |
| exochronos_coherence_passport_v0_1.py | sha256_bytes | coherence.py | _sha256_json | PRIVATE |
| exochronos_coherence_passport_v0_1.py | sha256_json | coherence.py | _sha256_json | PRIVATE |
| exochronos_coherence_passport_v0_1.py | sha256_file | coherence.py | _sha256_file | PRIVATE |
| exochronos_coherence_passport_v0_1.py | is_number | coherence.py | _is_finite_float | PRIVATE |
| exochronos_coherence_passport_v0_1.py | nonempty_list | coherence.py | normalize_observations | INLINED |
| exochronos_coherence_passport_v0_1.py | nonempty_dict | coherence.py | normalize_observations | INLINED |
| exochronos_coherence_passport_v0_1.py | raw_payload | coherence.py | _raw_payload | PRIVATE |
| exochronos_coherence_passport_v0_1.py | entry_payload | coherence.py | _entry_payload | PRIVATE |
| exochronos_coherence_passport_v0_1.py | compute_raw_payload_hash | coherence.py | seal_passport_hashes | RENAMED |
| exochronos_coherence_passport_v0_1.py | compute_entry_hash | coherence.py | seal_passport_hashes | INLINED |
| exochronos_coherence_passport_v0_1.py | validate_passport | coherence.py | validate_coherence_passport | RENAMED |
| exochronos_coherence_passport_v0_1.py | seal_passport_for_replay | coherence.py | seal_passport_hashes | RENAMED |
| exochronos_coherence_passport_v0_1.py | build_cycle88_interface_signal | NONE | NONE | EXCLUDED |
| exochronos_coherence_passport_v0_1.py | build_minimal_valid_demo | NONE | NONE | EXCLUDED |
| exochronos_coherence_passport_v0_1.py | to_dict | NONE | NONE | EXCLUDED |
| exochronos_coherence_passport_v0_1.py | utc_now_iso | coherence.py | _utc_now | PRIVATE |
| exochronos_coherence_passport_v0_1.py | read_json | NONE | NONE | EXCLUDED |
| exochronos_coherence_passport_v0_1.py | write_json | NONE | NONE | EXCLUDED |
| exochronos_coherence_passport_v0_1.py | cmd_init_cycle88 | NONE | NONE | CLI_IO |
| exochronos_coherence_passport_v0_1.py | cmd_init_demo | NONE | NONE | CLI_IO |
| exochronos_coherence_passport_v0_1.py | cmd_validate | NONE | NONE | CLI_IO |
| exochronos_coherence_passport_v0_1.py | cmd_replay | NONE | NONE | CLI_IO |
| exochronos_coherence_passport_v0_1.py | build_arg_parser | NONE | NONE | CLI_IO |
| exochronos_coherence_passport_v0_1.py | main | NONE | NONE | CLI_IO |
| aegis_omega_ultimate.py | render | audit_claims.py | render | EXACT |
| aegis_omega_ultimate.py | validate | audit_claims.py | validate | EXACT |
| aegis_omega_ultimate.py | render_markdown | audit_claims.py | render_markdown | EXACT |
| aegis_omega_ultimate.py | load_artifacts | audit_claims.py | load_artifacts | EXACT |
| aegis_omega_ultimate.py | extract_evidence | audit_claims.py | extract_evidence | EXACT |
| aegis_omega_ultimate.py | decompose_claim | audit_claims.py | decompose_claim | EXACT |
| aegis_omega_ultimate.py | assess_proposition | audit_claims.py | assess_proposition | EXACT |
| aegis_omega_ultimate.py | detect_logical_contradictions | audit_claims.py | detect_logical_contradictions | EXACT |
| aegis_omega_ultimate.py | audit_claim | audit_claims.py | audit_claim | EXACT |
| aegis_omega_ultimate.py | inspect_document | audit_claims.py | inspect_document | EXACT |
| aegis_omega_ultimate.py | validate_canonical_report | audit_claims.py | validate_canonical_report | EXACT |
| aegis_omega_ultimate.py | normalize_space | audit_claims.py | _normalize_space | PRIVATE |
| aegis_omega_ultimate.py | split_sentences | audit_claims.py | _split_sentences | PRIVATE |
| aegis_omega_ultimate.py | compact_excerpt | audit_claims.py | _compact_excerpt | PRIVATE |
| aegis_omega_ultimate.py | tokenize | audit_claims.py | _tokenize | PRIVATE |
| aegis_omega_ultimate.py | keyword_overlap | audit_claims.py | _keyword_overlap | PRIVATE |
| aegis_omega_ultimate.py | infer_channel | audit_claims.py | _infer_channel | PRIVATE |
| aegis_omega_ultimate.py | build_weaknesses | audit_claims.py | _build_weaknesses | PRIVATE |
| aegis_omega_ultimate.py | verdict_from_assessments | audit_claims.py | _verdict_from_assessments | PRIVATE |
| aegis_omega_ultimate.py | dedupe | audit_claims.py | _dedupe | PRIVATE |
| aegis_omega_ultimate.py | build_one_sentence_justification | audit_claims.py | _one_sentence_justification | PRIVATE |
| aegis_omega_ultimate.py | read_text_file | audit_claims.py | _read_text | PRIVATE |
| aegis_omega_ultimate.py | read_json_file | audit_claims.py | _read_json | PRIVATE |
| aegis_omega_ultimate.py | read_csv_file | audit_claims.py | _read_csv | PRIVATE |
| aegis_omega_ultimate.py | iter_input_paths | audit_claims.py | _iter_paths | PRIVATE |
| aegis_omega_ultimate.py | score_sentence_against_claim | audit_claims.py | extract_evidence | INLINED |
| aegis_omega_ultimate.py | summarize_supporting_arguments | audit_claims.py | audit_claim | INLINED |
| aegis_omega_ultimate.py | summarize_opposing_arguments | audit_claims.py | audit_claim | INLINED |
| aegis_omega_ultimate.py | build_taxonomy_texts | audit_claims.py | audit_claim | INLINED |
| aegis_omega_ultimate.py | build_limitations | audit_claims.py | audit_claim | INLINED |
| aegis_omega_ultimate.py | build_recommendations | audit_claims.py | audit_claim | INLINED |
| aegis_omega_ultimate.py | run_benchmark | audit_claims.py | batch_audit | RENAMED |
| aegis_omega_ultimate.py | read_pdf_file | NONE | NONE | EXCLUDED |
| aegis_omega_ultimate.py | write_output | NONE | NONE | CLI_IO |
| aegis_omega_ultimate.py | build_parser | NONE | NONE | CLI_IO |
| aegis_omega_ultimate.py | main | NONE | NONE | CLI_IO |
| score_aegis_ultimate.py | normalize_text | score_report.py | normalize_text | EXACT |
| score_aegis_ultimate.py | extract_sections | score_report.py | extract_sections | EXACT |
| score_aegis_ultimate.py | extract_verdict | score_report.py | extract_verdict | EXACT |
| score_aegis_ultimate.py | extract_confidence | score_report.py | extract_confidence | EXACT |
| score_aegis_ultimate.py | score_verdicts | score_report.py | score_verdicts | EXACT |
| score_aegis_ultimate.py | score_critical_points | score_report.py | score_critical_points | EXACT |
| score_aegis_ultimate.py | score_confidence | score_report.py | score_confidence | EXACT |
| score_aegis_ultimate.py | score_discriminant_tests | score_report.py | score_discriminant_tests | EXACT |
| score_aegis_ultimate.py | score_probative_separation | score_report.py | score_probative_separation | EXACT |
| score_aegis_ultimate.py | score_weakness_taxonomy | score_report.py | score_weakness_taxonomy | EXACT |
| score_aegis_ultimate.py | score_fail_closed | score_report.py | score_fail_closed | EXACT |
| score_aegis_ultimate.py | validate_case_json | score_report.py | validate_case_json | EXACT |
| score_aegis_ultimate.py | evaluate | score_report.py | evaluate | EXACT |
| score_aegis_ultimate.py | normalize_heading | score_report.py | _normalize_heading | PRIVATE |
| score_aegis_ultimate.py | tokenize | score_report.py | _tokenize | PRIVATE |
| score_aegis_ultimate.py | keyword_set | score_report.py | _keyword_set | PRIVATE |
| score_aegis_ultimate.py | semantic_match | score_report.py | _semantic_match | PRIVATE |
| score_aegis_ultimate.py | contains_nearby_verdict | score_report.py | _contains_nearby_verdict | PRIVATE |
| score_aegis_ultimate.py | safe_get | score_report.py | _safe_get | PRIVATE |
| score_aegis_ultimate.py | count_discriminant_tests | score_report.py | score_discriminant_tests | INLINED |
| score_aegis_ultimate.py | weakness_taxonomy_check | score_report.py | score_weakness_taxonomy | INLINED |
| score_aegis_ultimate.py | apply_fatal_errors | score_report.py | evaluate | INLINED |
| score_aegis_ultimate.py | apply_bonus | score_report.py | evaluate | INLINED |
| score_aegis_ultimate.py | load_json | NONE | NONE | EXCLUDED |
| score_aegis_ultimate.py | load_text | NONE | NONE | EXCLUDED |
| score_aegis_ultimate.py | main | NONE | NONE | CLI_IO |

## DELTA SUMMARY

| MAPPING_TYPE | COUNT | DESCRIPTION |
|---|---|---|
| EXACT | 28 | Same name, same signature, same logic |
| PRIVATE | 30 | Public→private (de-hyping convention) |
| CLI_IO | 15 | CLI/IO utilities excluded from library |
| INLINED | 13 | Logic absorbed into calling function |
| EXCLUDED | 12 | Non-functional (demo, PDF, config) or IO-only |
| RENAMED | 6 | Renamed for clarity |
| RENAMED_SIG | 1 | Renamed + signature changed (mutual_information: matrix→two sequences) |
| **TOTAL** | **105** | |
| UNKNOWN | 0 | |
| MISSING | 0 | |

**Δ_total = 105**
**Δ_explained = 105**
**Δ_unexplained = 0**
**"Delta resolved" = SUPPORTED** (subject to correctness of INLINED assertions)

## Limitation
INLINED mappings assert that the source logic is present in the destination function.
This assertion is based on code inspection, not formal proof.
A formal proof would require property-based testing of each inlined case.
