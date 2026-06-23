"""
BERZERKER Omega — Decoupled Ablation Engine v2

FAIL-CLOSED | NULL-MODEL-FIRST | NON-TAUTOLOGICAL

Addresses CH-33/CH-34/CH-35/CH-36:
  CH-33: organs != metrics. Organs = coupled substrate.
         Metrics = emergent observers on trajectory.
  CH-34: threshold derived from NULL model, not hardcoded. Bootstrap CI.
  CH-35: NULL_ENTITY (permuted coupling) present before promotion.
  CH-36: effect size + CI + bootstrap_std, no binary flags.

CLAIM CEILING: LOCAL_SIMULATION_ONLY
Does NOT demonstrate: consciousness, agent, intention, external validation.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ORGANS: List[str] = ["cortex", "ganglion", "filaments", "nucleus"]
N: int = len(ORGANS)
METRIC_KEYS: List[str] = ["memory", "coherence", "direction", "attractor"]


def base_coupling(seed: int = 0) -> np.ndarray:
    """Directed, weighted coupling matrix for the logistic lattice."""
    rng = np.random.default_rng(seed)
    C = np.array([
        [0.0,  0.6,  0.1,  0.0],
        [0.0,  0.0,  0.7,  0.1],
        [0.0,  0.1,  0.0,  0.8],
        [0.05, 0.0,  0.0,  0.0],
    ], dtype=float)
    C += rng.normal(0, 0.02, size=C.shape)
    C = np.clip(C, 0, None)
    return C


def shuffle_coupling(C: np.ndarray, seed: int) -> np.ndarray:
    """NULL: permute off-diagonal weights — destroys causal structure, preserves coupling mass."""
    rng = np.random.default_rng(seed)
    mask = ~np.eye(N, dtype=bool)
    vals = C[mask].copy()
    rng.shuffle(vals)
    Cn = C.copy()
    Cn[mask] = vals
    return Cn


def logistic(x: np.ndarray, r: float = 3.82) -> np.ndarray:
    """Logistic map: f(x) = r·x·(1−x)."""
    return r * x * (1.0 - x)


def simulate(
    C: np.ndarray,
    ablate: Optional[int] = None,
    steps: int = 3000,
    transient: int = 1000,
    seed: int = 0,
    eps: float = 0.25,
) -> np.ndarray:
    """
    Simulate coupled logistic lattice.
    Returns time series array of shape (steps-transient, N).
    ablate: organ index frozen to 0 (and outgoing contribution silenced).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 0.8, size=N)
    if ablate is not None:
        x[ablate] = 0.0
    hist = np.zeros((steps - transient, N))
    for t in range(steps):
        contrib = C @ x
        x_new = (1 - eps) * logistic(x) + eps * contrib
        x_new = np.clip(x_new, 0.0, 1.0)
        if ablate is not None:
            x_new[ablate] = 0.0
        x = x_new
        if t >= transient:
            hist[t - transient] = x
    return hist


def metrics_from(hist: np.ndarray, ablate: Optional[int] = None) -> Dict[str, float]:
    """
    Emergent observers on the global trajectory.
    No metric belongs to a single organ.
    Returns dict with keys: memory, coherence, direction, attractor.
    """
    cols = [i for i in range(N) if i != ablate] if ablate is not None else list(range(N))
    A = hist[:, cols]
    s = A.mean(axis=1)

    s0, s1 = s[:-1], s[1:]
    memory = (
        0.0 if s0.std() < 1e-9 or s1.std() < 1e-9
        else float(abs(np.corrcoef(s0, s1)[0, 1]))
    )

    if A.shape[1] < 2:
        coherence = 0.0
    else:
        Cm = np.corrcoef(A.T)
        iu = np.triu_indices(A.shape[1], k=1)
        coherence = float(np.nanmean(np.abs(Cm[iu])))

    d = np.diff(s)
    denom = float(np.sum(np.abs(d)))
    direction = float(abs(float(np.sum(d)))) / denom if denom > 1e-9 else 0.0

    late = s[-max(1, len(s) // 3):]
    attractor = 1.0 / (1.0 + float(np.var(late)) * 100)

    return dict(memory=memory, coherence=coherence, direction=direction, attractor=attractor)


def ablation_effects(
    C: np.ndarray, ablate_idx: int, seeds: Sequence[int]
) -> Dict[str, np.ndarray]:
    """Bootstrap ablation effects over multiple seeds."""
    out: Dict[str, list] = {k: [] for k in METRIC_KEYS}
    for sd in seeds:
        base = metrics_from(simulate(C, ablate=None, seed=sd))
        abl = metrics_from(simulate(C, ablate=ablate_idx, seed=sd), ablate=ablate_idx)
        for k in METRIC_KEYS:
            out[k].append(base[k] - abl[k])
    return {k: np.array(v) for k, v in out.items()}


def ci(arr: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns (mean, lo_2.5pct, hi_97.5pct, std)."""
    m = float(np.mean(arr))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return m, float(lo), float(hi), float(np.std(arr))


def null_threshold(
    C: np.ndarray, seeds: Sequence[int], n_null: int = 40, base_seed: int = 9000
) -> Dict[str, float]:
    """
    95th percentile of |effect| under permuted coupling across all ablations.
    This is the empirical threshold — not hardcoded.
    """
    pool: Dict[str, list] = {k: [] for k in METRIC_KEYS}
    for j in range(n_null):
        Cn = shuffle_coupling(C, seed=base_seed + j)
        for idx in range(N):
            eff = ablation_effects(Cn, idx, seeds[:3])
            for k in METRIC_KEYS:
                pool[k].extend(np.abs(eff[k]).tolist())
    return {k: float(np.percentile(pool[k], 95)) for k in METRIC_KEYS}


def tribunal_verdict(
    C: np.ndarray,
    seeds: Sequence[int],
    thr: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Full BERZERKER verdict.
    Verdicts: REFUTED_VS_NULL | OPEN_PROBLEM | EMPIRICAL_SIGNAL | PASS_LOCAL
    claim_ceiling: LOCAL_SIMULATION_ONLY
    """
    if thr is None:
        thr = null_threshold(C, seeds)

    atlas = []
    total_dd = 0
    sig_matrix: Dict[str, set] = {}

    for idx, organ in enumerate(ORGANS):
        eff = ablation_effects(C, idx, seeds)
        signature = []
        dd = 0
        for k in METRIC_KEYS:
            m, lo, hi, _ = ci(eff[k])
            if abs(m) > thr[k] and (lo > 0 or hi < 0):
                dd += 1
                signature.append(k)
        total_dd += dd
        sig_matrix[organ] = set(signature)
        atlas.append(dict(organ=organ, delta_d=dd, signature=signature,
                          effects={k: ci(eff[k]) for k in METRIC_KEYS}))

    distinct = len({frozenset(sig_matrix[o]) for o in ORGANS})

    if total_dd == 0:
        verdict = "REFUTED_VS_NULL"
    elif distinct <= 1:
        verdict = "OPEN_PROBLEM"
    elif total_dd < 4:
        verdict = "EMPIRICAL_SIGNAL"
    else:
        verdict = "PASS_LOCAL"

    report = dict(
        verdict=verdict,
        total_delta_d=total_dd,
        distinct_signatures=distinct,
        null_threshold=thr,
        atlas=atlas,
        claim_ceiling="LOCAL_SIMULATION_ONLY",
        blocked_claims=["CONSCIOUSNESS", "AUTONOMOUS_AGENT", "INTENT", "EXTERNAL_VALIDATION"],
    )

    raw = json.dumps(report, sort_keys=True, default=str)
    report["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    report["timestamp"] = time.time()

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report
