"""gpts_ultimate — Unified scientific validation framework.

Single-import access to the full toolkit: signals, classification, evidence,
gate, ledger, manifest, benchmark, promotion, adjudication, world-agnostic
metrics, BERZERKER ablation engine, Lorenz generator, mathematical metrics,
tribunal verdict engine, and end-to-end pipeline.

Claim ceiling: LOCAL_SIMULATION_ONLY for all Phase Omega components.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
from gpts_ultimate.signals import (
    analyze,
    structure_score,
    entropy_norm,
    compression_ratio,
    autocorr,
    spectral_analysis,
    motif_share,
    window_stats,
    ar2_fit,
    step_metrics,
    generate_null,
    bootstrap_ci,
    # world-agnostic shortcuts
    memory,
    coherence,
    direction,
    spectral_entropy_agnostic,
    # null surrogates
    phase_scramble,
    iid_shuffle,
    block_shuffle,
)

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
from gpts_ultimate.classifier import classify_metric, LABELS

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
from gpts_ultimate.evidence import (
    sha256_text,
    build_raw_record,
    validate_raw_record,
    validate_raw_records,
    audit_zip,
    summarize_csv,
    RecordVerdict,
    RawGateReport,
)

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
from gpts_ultimate.gate import (
    classify_claim,
    ClaimResult,
    ClaimStatus,
    compute_evidence_score,
    EvidenceInput,
    EvidenceScore,
    from_gate_counts,
    maturity_map,
    proof_firewall,
    validate_metric,
    classify_claim_scientific,
)

# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
from gpts_ultimate.ledger import (
    AuditLedger,
    ContextCitationLock,
    # re-exported from math_metrics
    RealityLedger,
    Prediction,
    HazardRegistry,
    HazardStatus,
)

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
from gpts_ultimate.manifest import (
    sha256_file,
    build_manifest,
    aggregate_hash,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
from gpts_ultimate.benchmark import (
    minmax_fit,
    minmax_transform,
    train_linear,
    predict_linear,
    mse_score,
    compare_prediction_lock,
)

# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
from gpts_ultimate.promotion import (
    evaluate_promotion,
    validate_seal_record,
    validate_seal_ledger,
    replay_jsonl_ledger,
)

# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------
from gpts_ultimate.adjudication import (
    build_confusion_matrix,
    score_label,
    score_adjudication,
)

# ---------------------------------------------------------------------------
# World-agnostic (Phase Omega)
# ---------------------------------------------------------------------------
from gpts_ultimate.agnostic import (
    WorldAgnosticMetrics,
    WorldAgnosticNulls,
    compute_metrics,
    tribunal_verdict as agnostic_tribunal_verdict,
)

# ---------------------------------------------------------------------------
# BERZERKER (Phase Omega)
# ---------------------------------------------------------------------------
from gpts_ultimate.berzerker import (
    base_coupling,
    shuffle_coupling,
    logistic,
    simulate,
    metrics_from,
    ablation_effects,
    null_threshold,
    tribunal_verdict as berzerker_tribunal_verdict,
    ORGANS,
    METRIC_KEYS,
)

# ---------------------------------------------------------------------------
# Lorenz (Phase Omega)
# ---------------------------------------------------------------------------
from gpts_ultimate.lorenz import FastLorenzGenerator

# ---------------------------------------------------------------------------
# Mathematical Metrics (Phase Omega)
# ---------------------------------------------------------------------------
from gpts_ultimate.math_metrics import (
    FormulaRegistry,
    MetricComputer,
    ValidationPipeline,
)

# ---------------------------------------------------------------------------
# Tribunal — unified verdict engine (NEW)
# ---------------------------------------------------------------------------
from gpts_ultimate.tribunal import (
    run_berzerker_tribunal,
    run_agnostic_tribunal,
    run_full_tribunal,
)

# ---------------------------------------------------------------------------
# Pipeline — end-to-end (NEW)
# ---------------------------------------------------------------------------
from gpts_ultimate.pipeline import GptsUltimatePipeline

__all__ = [
    # signals
    "analyze", "structure_score", "entropy_norm", "compression_ratio",
    "autocorr", "spectral_analysis", "motif_share", "window_stats",
    "ar2_fit", "step_metrics", "generate_null", "bootstrap_ci",
    "memory", "coherence", "direction", "spectral_entropy_agnostic",
    "phase_scramble", "iid_shuffle", "block_shuffle",
    # classifier
    "classify_metric", "LABELS",
    # evidence
    "sha256_text", "build_raw_record", "validate_raw_record",
    "validate_raw_records", "audit_zip", "summarize_csv",
    "RecordVerdict", "RawGateReport",
    # gate
    "classify_claim", "ClaimResult", "ClaimStatus",
    "compute_evidence_score", "EvidenceInput", "EvidenceScore",
    "from_gate_counts", "maturity_map", "proof_firewall",
    "validate_metric", "classify_claim_scientific",
    # ledger
    "AuditLedger", "ContextCitationLock",
    "RealityLedger", "Prediction", "HazardRegistry", "HazardStatus",
    # manifest
    "sha256_file", "build_manifest", "aggregate_hash", "write_manifest",
    # benchmark
    "minmax_fit", "minmax_transform", "train_linear", "predict_linear",
    "mse_score", "compare_prediction_lock",
    # promotion
    "evaluate_promotion", "validate_seal_record",
    "validate_seal_ledger", "replay_jsonl_ledger",
    # adjudication
    "build_confusion_matrix", "score_label", "score_adjudication",
    # agnostic
    "WorldAgnosticMetrics", "WorldAgnosticNulls",
    "compute_metrics", "agnostic_tribunal_verdict",
    # berzerker
    "base_coupling", "shuffle_coupling", "logistic", "simulate",
    "metrics_from", "ablation_effects", "null_threshold",
    "berzerker_tribunal_verdict", "ORGANS", "METRIC_KEYS",
    # lorenz
    "FastLorenzGenerator",
    # math_metrics
    "FormulaRegistry", "MetricComputer", "ValidationPipeline",
    # tribunal
    "run_berzerker_tribunal", "run_agnostic_tribunal", "run_full_tribunal",
    # pipeline
    "GptsUltimatePipeline",
]
