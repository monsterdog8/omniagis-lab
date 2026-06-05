"""
gpts_core — Unified validation and analysis toolkit.

Modules:
  signals      — time-series structure analysis (entropy, spectral, AR2, structure score)
  classifier   — metric text classification (observed/computed/reported/simulated/symbolic/unsupported)
  evidence     — raw-record validation, evidence gating, CSV/ZIP audit
  gate         — claim classification (BLOCKED/BOUNDED/UNKNOWN) + evidence scoring
  ledger       — append-only SHA-256 chained audit log + citation lock
  adjudication — confusion matrix scoring and precision/recall
  manifest     — file hashing, type classification, manifest building
  benchmark    — stdlib-only linear model, MSE scoring, prediction lock comparator
  promotion    — promotion gate engine, SEAL artifact continuity, JSONL ledger replay
  cli          — unified command-line interface
"""
from gpts_core.signals import analyze as analyze_signal, structure_score
from gpts_core.classifier import classify_metric, LABELS
from gpts_core.evidence import (
    build_raw_record,
    validate_raw_record,
    validate_raw_records,
    summarize_csv,
    audit_zip,
    sha256_text,
    sha256_file,
)
from gpts_core.gate import (
    classify_claim,
    compute_evidence_score,
    from_gate_counts,
    maturity_map,
    proof_firewall,
    EvidenceInput,
    EvidenceScore,
)
from gpts_core.ledger import AuditLedger, ContextCitationLock
from gpts_core.adjudication import score_adjudication, build_confusion_matrix
from gpts_core.manifest import build_manifest, write_manifest, sha256_file as hash_file, kind_for
from gpts_core.benchmark import (
    train_linear,
    predict_linear,
    mse_score,
    compare_prediction_lock,
)
from gpts_core.promotion import (
    evaluate_promotion,
    replay_jsonl_ledger,
    validate_seal_ledger,
)

__all__ = [
    # signals
    "analyze_signal", "structure_score",
    # classifier
    "classify_metric", "LABELS",
    # evidence
    "build_raw_record", "validate_raw_record", "validate_raw_records",
    "summarize_csv", "audit_zip", "sha256_text", "sha256_file",
    # gate
    "classify_claim", "compute_evidence_score", "from_gate_counts",
    "maturity_map", "proof_firewall", "EvidenceInput", "EvidenceScore",
    # ledger
    "AuditLedger", "ContextCitationLock",
    # adjudication
    "score_adjudication", "build_confusion_matrix",
    # manifest
    "build_manifest", "write_manifest", "hash_file", "kind_for",
    # benchmark
    "train_linear", "predict_linear", "mse_score", "compare_prediction_lock",
    # promotion
    "evaluate_promotion", "replay_jsonl_ledger", "validate_seal_ledger",
]
