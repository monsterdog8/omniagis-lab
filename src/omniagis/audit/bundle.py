"""Bundle manifest auditor — append-only chain and hash validation.

GO OMNIÆGIS Bundle Audit (v5.1)
================================

Verifies that every artifact declared in a bundle manifest:

1. **Exists** on the filesystem (PRESENCE CHECK).
2. **Matches** its declared SHA-256 hash (HASH CHECK).
3. Can trace its **append-only chain** to its predecessor artifact
   (CHAIN CHECK): if ``prev_artifact_path`` is declared, that file
   must be present *and* its SHA-256 must equal ``prev_artifact_sha256``.

Any failure triggers **FAIL_CLOSED** — the bundle is not considered
strictly replayable regardless of partial successes.

Manifest format (JSON)
----------------------
::

    {
        "version": "5.1",
        "name": "my-bundle",
        "description": "...",
        "artifacts": [
            {
                "name": "main_output",
                "path": "/abs/or/relative/path/to/file.json",
                "sha256": "<hex>",
                "prev_artifact_path": "/path/to/previous.json",   # optional
                "prev_artifact_sha256": "<hex>",                   # optional
                "synthetic_demo": false                            # optional
            }
        ]
    }

Usage
-----
As a module::

    from omniagis.audit.bundle import BundleAuditor
    auditor = BundleAuditor()
    report = auditor.audit("/path/to/manifest.json")
    print(report.global_verdict)   # "PASS" | "PARTIAL PASS" | "FAIL_CLOSED"
    print(auditor.render(report))

As a CLI::

    python -m omniagis bundle /path/to/manifest.json
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

PASS = "PASS"
PARTIAL_PASS = "PARTIAL PASS"
FAIL_CLOSED = "FAIL_CLOSED"

# Chain / hash reason codes
REASON_HASH_VALID = "HASH_VALID"
REASON_HASH_INVALID = "HASH_INVALID"
REASON_HASH_MISSING = "HASH_MISSING"
REASON_HASH_SKIPPED = "HASH_SKIPPED"          # artifact absent — hash not checked
REASON_CHAIN_OK = "CHAIN_OK"
REASON_CHAIN_NO_PREV = "CHAIN_NO_PREV"        # no predecessor declared (base)
REASON_CHAIN_PREV_MISSING = "CHAIN_PREVIOUS_ARTIFACT_MISSING"
REASON_CHAIN_HASH_MISMATCH = "CHAIN_HASH_MISMATCH"
REASON_CHAIN_SKIPPED = "CHAIN_SKIPPED"        # artifact absent — chain not checked


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArtifactSpec:
    """One artifact entry as declared in the manifest."""

    name: str
    path: str
    sha256: Optional[str] = None
    prev_artifact_path: Optional[str] = None
    prev_artifact_sha256: Optional[str] = None
    synthetic_demo: bool = False


@dataclass
class ArtifactResult:
    """Audit result for a single artifact."""

    spec: ArtifactSpec
    present: bool
    hash_reason: str
    chain_reason: str
    actual_sha256: Optional[str] = None   # computed when file is present

    @property
    def is_ok(self) -> bool:
        """True only when presence, hash, and chain are all verified and valid."""
        if not self.present:
            return False
        hash_ok = self.hash_reason == REASON_HASH_VALID
        chain_ok = self.chain_reason in (REASON_CHAIN_OK, REASON_CHAIN_NO_PREV)
        return hash_ok and chain_ok

    @property
    def verdict(self) -> str:
        if not self.present:
            return FAIL_CLOSED
        if self.hash_reason == REASON_HASH_INVALID:
            return FAIL_CLOSED
        if self.chain_reason in (REASON_CHAIN_PREV_MISSING, REASON_CHAIN_HASH_MISMATCH):
            return FAIL_CLOSED
        # Present, no critical failure; PASS only when hash is explicitly verified
        if self.hash_reason == REASON_HASH_VALID and self.chain_reason in (
            REASON_CHAIN_OK, REASON_CHAIN_NO_PREV
        ):
            return PASS
        return PARTIAL_PASS


@dataclass
class BundleManifest:
    """Parsed bundle manifest."""

    version: str
    name: str
    description: str
    artifacts: List[ArtifactSpec] = field(default_factory=list)
    source_path: str = ""


@dataclass
class BundleAuditReport:
    """Full bundle audit report."""

    manifest: BundleManifest
    results: List[ArtifactResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def n_declared(self) -> int:
        return len(self.results)

    @property
    def n_present(self) -> int:
        return sum(1 for r in self.results if r.present)

    @property
    def n_missing(self) -> int:
        return self.n_declared - self.n_present

    @property
    def n_hash_valid(self) -> int:
        return sum(1 for r in self.results if r.hash_reason == REASON_HASH_VALID)

    @property
    def n_chain_ok(self) -> int:
        return sum(
            1 for r in self.results
            if r.chain_reason in (REASON_CHAIN_OK, REASON_CHAIN_NO_PREV)
        )

    @property
    def global_verdict(self) -> str:
        """Fail-closed global verdict.

        PASS       — all artifacts present, all hashes valid, all chains intact.
        FAIL_CLOSED — any artifact missing, hash mismatch, or broken chain.
        PARTIAL PASS — all present; some hashes/chains unverified (no declared hash/prev).
        """
        if any(r.verdict == FAIL_CLOSED for r in self.results):
            return FAIL_CLOSED
        if all(r.verdict == PASS for r in self.results):
            return PASS
        return PARTIAL_PASS

    @property
    def fail_reasons(self) -> List[str]:
        """Unique top-level failure reasons across all artifacts."""
        reasons: list[str] = []
        for r in self.results:
            if not r.present:
                reasons.append(REASON_CHAIN_PREV_MISSING)
            elif r.hash_reason == REASON_HASH_INVALID:
                reasons.append(REASON_HASH_INVALID)
            elif r.chain_reason in (REASON_CHAIN_PREV_MISSING, REASON_CHAIN_HASH_MISMATCH):
                reasons.append(r.chain_reason)
        seen: set[str] = set()
        deduped: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Compute the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str, manifest_dir: str) -> str:
    """Resolve *path* against *manifest_dir* when it is relative."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(manifest_dir, path))


# ---------------------------------------------------------------------------
# BundleAuditor
# ---------------------------------------------------------------------------

class BundleAuditor:
    """Manifest-based bundle auditor with hash and chain validation."""

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def load_manifest(self, manifest_path: str) -> BundleManifest:
        """Parse a bundle manifest JSON file.

        Parameters
        ----------
        manifest_path:
            Absolute or relative path to the manifest ``.json`` file.

        Returns
        -------
        BundleManifest
        """
        manifest_path = os.path.abspath(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        artifacts: List[ArtifactSpec] = []
        for entry in data.get("artifacts", []):
            artifacts.append(
                ArtifactSpec(
                    name=str(entry.get("name", "")),
                    path=str(entry.get("path", "")),
                    sha256=entry.get("sha256") or None,
                    prev_artifact_path=entry.get("prev_artifact_path") or None,
                    prev_artifact_sha256=entry.get("prev_artifact_sha256") or None,
                    synthetic_demo=bool(entry.get("synthetic_demo", False)),
                )
            )

        return BundleManifest(
            version=str(data.get("version", "unknown")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            artifacts=artifacts,
            source_path=manifest_path,
        )

    # ------------------------------------------------------------------
    # Core audit logic
    # ------------------------------------------------------------------

    def _check_artifact(
        self,
        spec: ArtifactSpec,
        manifest_dir: str,
    ) -> ArtifactResult:
        resolved_path = _resolve(spec.path, manifest_dir)
        present = os.path.isfile(resolved_path)

        if not present:
            return ArtifactResult(
                spec=spec,
                present=False,
                hash_reason=REASON_HASH_SKIPPED,
                chain_reason=REASON_CHAIN_SKIPPED,
            )

        # Compute SHA-256
        actual_sha256 = _sha256_file(resolved_path)

        # Hash check
        if spec.sha256 is None:
            hash_reason = REASON_HASH_MISSING
        elif actual_sha256 == spec.sha256:
            hash_reason = REASON_HASH_VALID
        else:
            hash_reason = REASON_HASH_INVALID

        # Chain check
        if spec.prev_artifact_path is None:
            chain_reason = REASON_CHAIN_NO_PREV
        else:
            prev_resolved = _resolve(spec.prev_artifact_path, manifest_dir)
            if not os.path.isfile(prev_resolved):
                chain_reason = REASON_CHAIN_PREV_MISSING
            elif spec.prev_artifact_sha256 is None:
                chain_reason = REASON_CHAIN_OK
            else:
                prev_actual = _sha256_file(prev_resolved)
                if prev_actual == spec.prev_artifact_sha256:
                    chain_reason = REASON_CHAIN_OK
                else:
                    chain_reason = REASON_CHAIN_HASH_MISMATCH

        return ArtifactResult(
            spec=spec,
            present=True,
            hash_reason=hash_reason,
            chain_reason=chain_reason,
            actual_sha256=actual_sha256,
        )

    def audit(self, manifest_path: str) -> BundleAuditReport:
        """Run the full bundle audit for *manifest_path*.

        Parameters
        ----------
        manifest_path:
            Path to the bundle manifest ``.json`` file.

        Returns
        -------
        BundleAuditReport
        """
        manifest = self.load_manifest(manifest_path)
        manifest_dir = os.path.dirname(manifest.source_path)

        results = [
            self._check_artifact(spec, manifest_dir)
            for spec in manifest.artifacts
        ]

        return BundleAuditReport(manifest=manifest, results=results)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, report: BundleAuditReport) -> str:
        """Render *report* as a human-readable text string."""
        divider = "=" * 72
        thin = "-" * 72
        m = report.manifest

        # Header
        lines: List[str] = [
            divider,
            f"OMNIÆGIS BUNDLE AUDIT — v{m.version}",
            thin,
            f"Bundle   : {m.name}",
            f"Manifest : {m.source_path}",
            f"Timestamp: {report.timestamp}",
            f"",
            f"Declared : {report.n_declared}  artifacts",
            f"Present  : {report.n_present}",
            f"Missing  : {report.n_missing}",
            f"Hash OK  : {report.n_hash_valid}",
            f"Chain OK : {report.n_chain_ok}",
            f"",
            f"GLOBAL VERDICT: {report.global_verdict}",
        ]

        if report.fail_reasons:
            lines.append(f"FAIL REASONS  : {', '.join(report.fail_reasons)}")

        # Artifact detail table
        lines += [divider, "ARTIFACT DETAIL", thin]
        headers = ["Name", "Present", "Hash", "Chain", "Verdict"]
        col_widths = [max(len(h), max((len(str(_cell(r, i))) for r in report.results), default=0))
                      for i, h in enumerate(headers)]

        def fmt_row(cells: List[str]) -> str:
            return "  ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))

        lines.append(fmt_row(headers))
        lines.append(thin)
        for r in report.results:
            lines.append(fmt_row([
                _cell(r, 0), _cell(r, 1), _cell(r, 2), _cell(r, 3), _cell(r, 4),
            ]))

        # Weakness summary
        missing = [r for r in report.results if not r.present]
        hash_fails = [r for r in report.results if r.hash_reason == REASON_HASH_INVALID]
        chain_fails = [r for r in report.results if r.chain_reason in (
            REASON_CHAIN_PREV_MISSING, REASON_CHAIN_HASH_MISMATCH
        )]

        lines += [divider, "WEAKNESSES", thin]
        if not (missing or hash_fails or chain_fails):
            lines.append("  None detected.")
        else:
            for r in missing:
                lines.append(f"  [MISSING] {r.spec.name} — path: {r.spec.path}")
            for r in hash_fails:
                lines.append(
                    f"  [HASH_INVALID] {r.spec.name} "
                    f"declared={r.spec.sha256} actual={r.actual_sha256}"
                )
            for r in chain_fails:
                lines.append(
                    f"  [{r.chain_reason}] {r.spec.name} "
                    f"prev={r.spec.prev_artifact_path}"
                )

        lines += [divider, f"GLOBAL VERDICT: {report.global_verdict}", divider]
        return "\n".join(lines)


def _cell(r: ArtifactResult, col: int) -> str:
    """Extract display cell for artifact result row."""
    if col == 0:
        return r.spec.name
    if col == 1:
        return "YES" if r.present else "NO"
    if col == 2:
        return r.hash_reason
    if col == 3:
        return r.chain_reason
    return r.verdict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="OMNIÆGIS bundle manifest auditor (hash + chain validation)."
    )
    p.add_argument("manifest", help="Path to bundle manifest JSON file")
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = p.parse_args(argv)

    auditor = BundleAuditor()
    try:
        report = auditor.audit(args.manifest)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.output == "json":
        import dataclasses

        def _artifact_to_dict(r: ArtifactResult) -> dict:
            return {
                "name": r.spec.name,
                "path": r.spec.path,
                "present": r.present,
                "sha256_declared": r.spec.sha256,
                "sha256_actual": r.actual_sha256,
                "hash_reason": r.hash_reason,
                "chain_reason": r.chain_reason,
                "verdict": r.verdict,
                "synthetic_demo": r.spec.synthetic_demo,
            }

        data = {
            "bundle_name": report.manifest.name,
            "bundle_version": report.manifest.version,
            "manifest": report.manifest.source_path,
            "timestamp": report.timestamp,
            "n_declared": report.n_declared,
            "n_present": report.n_present,
            "n_missing": report.n_missing,
            "global_verdict": report.global_verdict,
            "fail_reasons": report.fail_reasons,
            "artifacts": [_artifact_to_dict(r) for r in report.results],
        }
        print(json.dumps(data, indent=2))
    else:
        print(auditor.render(report))

    verdict_to_code = {PASS: 0, PARTIAL_PASS: 1, FAIL_CLOSED: 2}
    sys.exit(verdict_to_code.get(report.global_verdict, 2))


if __name__ == "__main__":
    main()
