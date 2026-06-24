"""Master CLI for gpts_ultimate.

Subcommands (10 from gpts_core + 4 new):
  --- gpts_core commands ---
  analyze-signal   -- compute structure metrics for a CSV signal column
  classify-metric  -- classify a metric text row
  validate-records -- validate a JSONL file of raw output records
  classify-claim   -- classify a claim string as BLOCKED/BOUNDED/UNKNOWN
  evidence-score   -- compute evidence score from gate counts
  replay-ledger    -- replay a JSONL event ledger (hash chain)
  seal-audit       -- validate a SEAL continuity JSONL ledger
  promote          -- evaluate promotion candidates from CSV matrices
  build-manifest   -- build a file manifest for a directory
  adjudicate       -- score a manual adjudication CSV
  --- new Phase Omega commands ---
  lorenz           -- generate Lorenz ensemble trajectories
  berzerker        -- run BERZERKER tribunal on logistic lattice
  math-metrics     -- compute a metric formula by ID
  run-pipeline     -- end-to-end pipeline on a trajectory CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _out(obj: object, pretty: bool = False) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None, default=str))


# ---------------------------------------------------------------------------
# gpts_core subcommand handlers (delegate to gpts_ultimate modules)
# ---------------------------------------------------------------------------

def cmd_analyze_signal(args: argparse.Namespace) -> int:
    import csv
    from gpts_ultimate.signals import analyze

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
    from gpts_ultimate.classifier import classify_metric

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
    from gpts_ultimate.gate import classify_claim

    result = classify_claim(args.text)
    _out(result.to_dict(), args.pretty)
    return 0 if result.status != "BLOCKED" else 2


def cmd_evidence_score(args: argparse.Namespace) -> int:
    from gpts_ultimate.gate import from_gate_counts, compute_evidence_score

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
    from gpts_ultimate.manifest import write_manifest

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


# ---------------------------------------------------------------------------
# New Phase Omega subcommand handlers
# ---------------------------------------------------------------------------

def cmd_lorenz(args: argparse.Namespace) -> int:
    """Generate Lorenz ensemble and optionally export as CSV."""
    from gpts_ultimate.lorenz import FastLorenzGenerator

    gen = FastLorenzGenerator(
        t_max=args.t_max,
        discard_transient=args.discard,
        random_seed=args.seed,
    )
    ensemble = gen.generate_ensemble(n_nodes=args.nodes)
    if args.out:
        import csv as _csv
        out_path = Path(args.out)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow([f"node_{i}" for i in range(ensemble.shape[1])])
            for row in ensemble:
                writer.writerow([f"{v:.8f}" for v in row])
        print(f"Ensemble written to {out_path} ({ensemble.shape[0]} steps, {ensemble.shape[1]} nodes)",
              file=sys.stderr)
    else:
        _out({"shape": list(ensemble.shape), "nodes": args.nodes,
              "t_max": args.t_max}, args.pretty)
    return 0


def cmd_berzerker(args: argparse.Namespace) -> int:
    """Run BERZERKER tribunal on the logistic lattice."""
    from gpts_ultimate.berzerker import base_coupling, tribunal_verdict

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    C = base_coupling(seed=args.coupling_seed)
    out_path = args.out or None
    result = tribunal_verdict(C, seeds, output_path=out_path)
    _out(result, args.pretty)
    return 0 if result["verdict"] in ("EMPIRICAL_SIGNAL", "PASS_LOCAL") else 1


def cmd_math_metrics(args: argparse.Namespace) -> int:
    """Look up a formula by ID and optionally compute a metric."""
    from gpts_ultimate.math_metrics import FormulaRegistry, ValidationPipeline

    formula_id = args.formula_id.upper()
    try:
        formula = FormulaRegistry.get_formula(formula_id)
    except KeyError:
        available = FormulaRegistry.list_all()
        print(f"ERROR: formula {formula_id!r} not found. Available: {available}", file=sys.stderr)
        return 2

    if args.result is not None:
        verdict = ValidationPipeline.validate_metric(
            metric_name=args.name or formula_id,
            formula_id=formula_id,
            result=float(args.result),
            r_squared=float(args.r2) if args.r2 is not None else None,
        )
        _out({"formula": formula, "verdict": verdict}, args.pretty)
    else:
        _out(formula, args.pretty)
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    """Run end-to-end gpts_ultimate pipeline on a trajectory CSV."""
    from gpts_ultimate.pipeline import GptsUltimatePipeline

    ledger_path = Path(args.ledger) if args.ledger else None
    pipe = GptsUltimatePipeline(ledger_path=ledger_path, sr=args.sr)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    report = pipe.run(
        trajectory=Path(args.file),
        run_id=args.run_id or Path(args.file).stem,
        csv_column=args.column,
        seeds=seeds,
        n_null=args.n_null,
    )
    _out(report, args.pretty)
    return 0 if report["tribunal"]["final_verdict"] in ("EMPIRICAL_SIGNAL", "PASS_LOCAL") else 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpts-ultimate",
        description="Unified validation, analysis, and tribunal toolkit — gpts_ultimate.",
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

    # lorenz
    s = sub.add_parser("lorenz", help="Generate Lorenz ensemble trajectories")
    s.add_argument("--nodes", type=int, default=20, help="Number of nodes (default: 20)")
    s.add_argument("--t-max", type=float, default=500.0, help="Integration time (default: 500)")
    s.add_argument("--discard", type=float, default=50.0, help="Transient discard time")
    s.add_argument("--seed", type=int, default=42, help="Random seed")
    s.add_argument("--out", default="", help="Output CSV path (omit to print metadata)")

    # berzerker
    s = sub.add_parser("berzerker", help="Run BERZERKER tribunal on logistic lattice")
    s.add_argument("--seed-start", type=int, default=5, help="First bootstrap seed")
    s.add_argument("--n-seeds", type=int, default=20, help="Number of seeds")
    s.add_argument("--coupling-seed", type=int, default=0, help="Coupling matrix seed")
    s.add_argument("--out", default="", help="Output JSON path for report")

    # math-metrics
    s = sub.add_parser("math-metrics", help="Look up or validate a formula by ID")
    s.add_argument("formula_id", help="Formula ID (e.g. BRIER_SCORE)")
    s.add_argument("--name", default="", help="Metric name for validation")
    s.add_argument("--result", type=float, default=None, help="Metric result value")
    s.add_argument("--r2", type=float, default=None, help="R² for validation")

    # run-pipeline
    s = sub.add_parser("run-pipeline", help="End-to-end pipeline on a trajectory CSV")
    s.add_argument("file", help="CSV file with trajectory")
    s.add_argument("--column", default="value", help="Column name (default: value)")
    s.add_argument("--run-id", default="", help="Run identifier")
    s.add_argument("--sr", type=float, default=256.0, help="Sampling rate Hz")
    s.add_argument("--seed-start", type=int, default=5)
    s.add_argument("--n-seeds", type=int, default=10)
    s.add_argument("--n-null", type=int, default=20, help="Null surrogates")
    s.add_argument("--ledger", default="", help="Ledger JSONL output path")

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
        "lorenz": cmd_lorenz,
        "berzerker": cmd_berzerker,
        "math-metrics": cmd_math_metrics,
        "run-pipeline": cmd_run_pipeline,
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
