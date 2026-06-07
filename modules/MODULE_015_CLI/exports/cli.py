"""Unified CLI for gpts_core.

Subcommands:
  analyze-signal      -- compute structure metrics for a CSV signal column
  classify-metric     -- classify a metric text row
  validate-records    -- validate a JSONL file of raw output records
  classify-claim      -- classify a claim string as BLOCKED/BOUNDED/UNKNOWN
  evidence-score      -- compute evidence score from gate counts
  replay-ledger       -- replay a JSONL event ledger (hash chain)
  seal-audit          -- validate a SEAL continuity JSONL ledger
  promote             -- evaluate promotion candidates from CSV matrices
  build-manifest      -- build a file manifest for a directory
  adjudicate          -- score a manual adjudication CSV
  coherence-passport  -- build and validate a coherence passport from module observations JSON
  audit-claim         -- audit a text claim against a corpus (7-section canonical report)
  score-report        -- score a canonical audit report markdown against an oracle case JSON
  spectral-gap        -- compute Ulam spectral gap for Pomeau-Manneville map over alpha grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _out(obj: object, pretty: bool = False) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None))


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_analyze_signal(args: argparse.Namespace) -> int:
    import csv
    from gpts_core.signals import analyze

    path = Path(args.file)
    col = args.column
    sr = args.sr
    values = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row[col]))
            except (ValueError, KeyError):
                pass
    if not values:
        print(f"ERROR: no numeric values found in column {col!r}", file=sys.stderr)
        return 2
    result = analyze(values, sr=sr)
    _out(result, args.pretty)
    return 0


def cmd_classify_metric(args: argparse.Namespace) -> int:
    from gpts_core.classifier import classify_metric

    label, reasons, conf = classify_metric(
        metric_name=args.name,
        value_raw=args.value,
        context=args.context or "",
        row_type=args.row_type,
    )
    _out({"label": label, "reasons": reasons, "confidence": conf}, args.pretty)
    return 0


def cmd_validate_records(args: argparse.Namespace) -> int:
    from gpts_core.evidence import load_jsonl, validate_raw_records

    path = Path(args.file)
    records = load_jsonl(path)
    report = validate_raw_records(
        run_id=args.run_id or path.stem,
        expected=args.expected,
        records=records,
        strict=not args.no_strict,
    )
    _out(report.to_dict(), args.pretty)
    return 0 if report.pass_gate else 2


def cmd_classify_claim(args: argparse.Namespace) -> int:
    from gpts_core.gate import classify_claim

    result = classify_claim(args.text)
    _out(result.to_dict(), args.pretty)
    return 0 if result.status != "BLOCKED" else 2


def cmd_evidence_score(args: argparse.Namespace) -> int:
    from gpts_core.gate import from_gate_counts, compute_evidence_score

    inp = from_gate_counts(
        expected=args.expected,
        present=args.present,
        valid=args.valid,
        strict_sidecars=args.sidecars,
        safety=args.safety,
        scoring=args.scoring,
        replay=args.replay,
        independence=args.independence,
    )
    score = compute_evidence_score(inp)
    _out(score.to_dict(), args.pretty)
    return 0 if score.public_claim_allowed else 2


def cmd_replay_ledger(args: argparse.Namespace) -> int:
    from gpts_core.promotion import replay_jsonl_ledger

    result = replay_jsonl_ledger(Path(args.file))
    _out(result, args.pretty)
    return 0 if result.get("status") == "PASS" else 1


def cmd_seal_audit(args: argparse.Namespace) -> int:
    from gpts_core.promotion import validate_seal_ledger

    result = validate_seal_ledger(
        ledger_path=Path(args.ledger),
        root=Path(args.root),
    )
    _out(result, args.pretty)
    return 0 if result["all_valid"] else 1


def cmd_promote(args: argparse.Namespace) -> int:
    from gpts_core.promotion import evaluate_promotion

    result = evaluate_promotion([Path(p) for p in args.matrices])
    _out(result, args.pretty)
    return 0 if result["candidates"] else 1


def cmd_build_manifest(args: argparse.Namespace) -> int:
    from gpts_core.manifest import write_manifest

    result = write_manifest(
        root=Path(args.root),
        out_dir=Path(args.out),
        label=args.label,
    )
    _out({"file_count": result["file_count"], "aggregate_hash": result["aggregate_hash"]},
         args.pretty)
    return 0


def cmd_adjudicate(args: argparse.Namespace) -> int:
    from gpts_core.adjudication import read_adjudication_csv, score_adjudication

    rows = read_adjudication_csv(Path(args.file))
    result = score_adjudication(rows)
    _out(result, args.pretty)
    return 0


def cmd_coherence_passport(args: argparse.Namespace) -> int:
    from gpts_core.coherence import (
        build_coherence_passport, seal_passport_hashes, validate_coherence_passport
    )

    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    passport = build_coherence_passport(
        cycle_id=payload.get("cycle_id", "UNKNOWN"),
        module_observations=payload.get("module_observations", {}),
        system_version=payload.get("system_version", "LOCAL_CAPTURE"),
        bins=args.bins,
    )
    passport = seal_passport_hashes(passport)
    report = validate_coherence_passport(passport)
    _out({"passport": passport, "validation": report}, args.pretty)
    return 0 if report["replay_status"] == "PASS" else 2


def cmd_audit_claim(args: argparse.Namespace) -> int:
    from gpts_core.audit_claims import audit_claim

    report = audit_claim(
        claim=args.claim,
        title=args.title or f"Audit: {args.claim[:60]}",
        inputs=args.inputs or [],
    )
    if args.markdown:
        print(report.render_markdown())
    else:
        _out(report.to_dict(), args.pretty)
    return 0


def cmd_score_report(args: argparse.Namespace) -> int:
    from gpts_core.score_report import evaluate, score_audit_report

    report_text = Path(args.report).read_text(encoding="utf-8")
    if args.case:
        case = json.loads(Path(args.case).read_text(encoding="utf-8"))
        result = evaluate(case, report_text, strict=args.strict)
    else:
        result = score_audit_report(report_text)
    _out(result, args.pretty)
    return 0 if result["total"] >= args.min_score else 2


def cmd_spectral_gap(args: argparse.Namespace) -> int:
    from gpts_core.spectral_gap import run_pipeline
    import numpy as np

    alpha_grid = np.geomspace(args.alpha_min, args.alpha_max, args.n_alpha)
    result = run_pipeline(
        alpha_grid,
        n_bins=args.n_bins,
        n_traj=args.n_traj,
        seed=args.seed,
    )
    _out(result, args.pretty)
    return 0 if result.get("regression_fail") is None else 2


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpts-core",
        description="Unified validation and analysis toolkit for GPTs systems.",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    sub = p.add_subparsers(dest="cmd", required=True)

    # analyze-signal
    s = sub.add_parser("analyze-signal", help="Compute structure metrics for a signal column")
    s.add_argument("file", help="CSV file path")
    s.add_argument("--column", default="value", help="Column name (default: value)")
    s.add_argument("--sr", type=float, default=256.0, help="Sampling rate Hz")

    # classify-metric
    s = sub.add_parser("classify-metric", help="Classify a metric text row")
    s.add_argument("--name", default="", help="Metric name")
    s.add_argument("--value", default="", help="Value string")
    s.add_argument("--context", default="", help="Context string")
    s.add_argument("--row-type", default="metric", help="Row type (metric/equation/recalculation)")

    # validate-records
    s = sub.add_parser("validate-records", help="Validate JSONL raw output records")
    s.add_argument("file", help="JSONL file")
    s.add_argument("--expected", type=int, required=True, help="Expected record count")
    s.add_argument("--run-id", default="", help="Run ID label")
    s.add_argument("--no-strict", action="store_true", help="Disable strict sidecar requirement")

    # classify-claim
    s = sub.add_parser("classify-claim", help="Classify a claim string")
    s.add_argument("text", help="Claim text to classify")

    # evidence-score
    s = sub.add_parser("evidence-score", help="Compute evidence score from gate counts")
    s.add_argument("--expected", type=int, required=True)
    s.add_argument("--present", type=int, required=True)
    s.add_argument("--valid", type=int, required=True)
    s.add_argument("--sidecars", type=int, default=0)
    s.add_argument("--safety", type=float, default=0.7)
    s.add_argument("--scoring", action="store_true")
    s.add_argument("--replay", action="store_true")
    s.add_argument("--independence", action="store_true")

    # replay-ledger
    s = sub.add_parser("replay-ledger", help="Replay JSONL event ledger (hash chain check)")
    s.add_argument("file", help="JSONL ledger file")

    # seal-audit
    s = sub.add_parser("seal-audit", help="Validate SEAL continuity JSONL ledger")
    s.add_argument("ledger", help="SEAL ledger JSONL file")
    s.add_argument("--root", default=".", help="Root directory for resolving artifact paths")

    # promote
    s = sub.add_parser("promote", help="Evaluate promotion candidates from CSV matrices")
    s.add_argument("matrices", nargs="+", help="CSV matrix files")

    # build-manifest
    s = sub.add_parser("build-manifest", help="Build file manifest for a directory")
    s.add_argument("root", help="Directory to scan")
    s.add_argument("--out", required=True, help="Output directory")
    s.add_argument("--label", default="manifest", help="Output file prefix")

    # adjudicate
    s = sub.add_parser("adjudicate", help="Score a manual adjudication CSV")
    s.add_argument("file", help="Adjudication CSV with label and human_label columns")

    # coherence-passport
    s = sub.add_parser("coherence-passport", help="Build and validate a coherence passport")
    s.add_argument("file", help="JSON with cycle_id, module_observations, system_version")
    s.add_argument("--bins", type=int, default=8, help="Discretization bins (default: 8)")

    # audit-claim
    s = sub.add_parser("audit-claim", help="Audit a claim against a corpus (7-section report)")
    s.add_argument("--claim", required=True, help="Claim text to audit")
    s.add_argument("--title", default="", help="Report title")
    s.add_argument("--inputs", nargs="*", default=[], help="File or directory paths (corpus)")
    s.add_argument("--markdown", action="store_true", help="Output rendered markdown instead of JSON")

    # score-report
    s = sub.add_parser("score-report", help="Score a canonical audit report markdown")
    s.add_argument("report", help="Markdown report file")
    s.add_argument("--case", default="", help="Oracle case JSON file (optional)")
    s.add_argument("--strict", action="store_true", help="Zero score if canonical validation fails")
    s.add_argument("--min-score", type=int, default=0, help="Exit code 2 if total < min-score")

    # spectral-gap
    s = sub.add_parser("spectral-gap", help="Compute Ulam spectral gap over alpha grid")
    s.add_argument("--alpha-min", type=float, default=0.02)
    s.add_argument("--alpha-max", type=float, default=0.20)
    s.add_argument("--n-alpha", type=int, default=10)
    s.add_argument("--n-bins", type=int, default=80)
    s.add_argument("--n-traj", type=int, default=20000)
    s.add_argument("--seed", type=int, default=42)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "analyze-signal": cmd_analyze_signal,
        "classify-metric": cmd_classify_metric,
        "validate-records": cmd_validate_records,
        "classify-claim": cmd_classify_claim,
        "evidence-score": cmd_evidence_score,
        "replay-ledger": cmd_replay_ledger,
        "seal-audit": cmd_seal_audit,
        "promote": cmd_promote,
        "build-manifest": cmd_build_manifest,
        "adjudicate": cmd_adjudicate,
        "coherence-passport": cmd_coherence_passport,
        "audit-claim": cmd_audit_claim,
        "score-report": cmd_score_report,
        "spectral-gap": cmd_spectral_gap,
    }
    handler = dispatch.get(args.cmd)
    if handler is None:
        print(f"Unknown command: {args.cmd}", file=sys.stderr)
        return 2
    try:
        return handler(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
