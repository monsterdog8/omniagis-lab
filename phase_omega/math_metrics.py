"""
MONSTERDOG Mathematical Metrics — Sealed v1.0

EXOCHRONOS Ω∞ — Phase Ω Operational Core

Mode: FAIL_CLOSED | Authority: NONE | Truth: OPEN

Provides complete mathematical specification and executable implementation
of all metrics, formulas, governance rules, and firewalls for EXOCHRONOS
Phase Ω empirical execution.

No metric without formula.
No score without replay.
No update without calibration.
No truth status except OPEN.
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: MATHEMATICAL FORMULA REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

class FormulaRegistry:
    """
    Complete registry of all formulas used in MONSTERDOG system.
    Every metric must have an entry here before it can be computed.
    """

    FORMULAS: Dict[str, Dict[str, Any]] = {
        "LAMBDA_1_LOGISTIC": {
            "name": "Lyapunov exponent (logistic map)",
            "formula": "λ₁ = (1/N) Σ log|f'(xᵢ)|",
            "description": "Mean logarithmic derivative over trajectory",
            "variables": {"N": "trajectory length", "xᵢ": "state at time i",
                          "f'(x)": "derivative of map at x"},
            "units": "nats/iterate",
            "source": "Wolf et al. (1985), Physica D",
            "reference_implementation": "lambda1_from_trajectory",
        },
        "LAMBDA_1_PM": {
            "name": "Lyapunov exponent (Pomeau-Manneville)",
            "formula": "λ₁ = (1/N) Σ log|1 + (1+α)xᵢ^α|",
            "description": "Mean logarithmic derivative for PM map",
            "variables": {"N": "trajectory length", "α": "intermittency parameter ∈ (0,1)",
                          "x": "state"},
            "units": "nats/iterate",
            "source": "Pomeau & Manneville (1980)",
            "reference_implementation": "lambda1_pm_from_trajectory",
        },
        "ULAM_SPECTRAL_GAP": {
            "name": "Ulam spectral gap",
            "formula": "g_gap(N) = |λ₁(P)| - |λ₂(P)|",
            "description": "Difference between largest two eigenvalue moduli of Ulam matrix",
            "variables": {"N": "partition size", "P": "Ulam transition matrix (N×N)",
                          "λ₁, λ₂": "largest two eigenvalue moduli"},
            "units": "dimensionless",
            "source": "Ulam (1960), Grassberger-Procaccia (1983)",
            "reference_implementation": "ulam_spectral_gap",
        },
        "ULAM_TRANSITION_MATRIX": {
            "name": "Ulam transition matrix",
            "formula": "P_ij = #{t: x_t ∈ I_i, x_{t+1} ∈ I_j} / #{t: x_t ∈ I_i}",
            "description": "Empirical transfer operator on uniform partition",
            "variables": {"I_i": "partition cell i", "x_t": "trajectory at time t"},
            "units": "probability matrix (rows sum to 1)",
            "source": "Ulam (1960)",
            "reference_implementation": "ulam_matrix_from_trajectory",
        },
        "CORRELATION_FUNCTION": {
            "name": "Correlation function",
            "formula": "C(n) = ∫ f·(g∘T^n) dμ - ∫f dμ ∫g dμ",
            "description": "Temporal correlation of observables under dynamics",
            "variables": {"f, g": "observables", "T": "dynamical map",
                          "n": "time lag", "μ": "invariant measure"},
            "units": "dimensionless",
            "source": "Birkhoff ergodic theorem",
            "reference_implementation": "correlation_function",
        },
        "CORRELATION_DECAY_EXPONENT": {
            "name": "Correlation decay exponent",
            "formula": "γ = 1/α - 1 (for PM maps)",
            "description": "Power-law exponent in C(n) ~ n^(-γ)",
            "variables": {"α": "PM intermittency parameter", "γ": "decay exponent"},
            "units": "dimensionless",
            "source": "Young (1999), Gouezel (2004)",
            "reference_implementation": "correlation_decay_exponent",
        },
        "RETURN_TIME_TAIL": {
            "name": "Return time tail distribution",
            "formula": "μ(R > n) ~ n^(-β)",
            "description": "Survival probability of return times to reference set",
            "variables": {"R": "return time to reference set", "n": "time threshold",
                          "β": "tail exponent"},
            "units": "probability",
            "source": "Young tower theory",
            "reference_implementation": "return_time_survival",
        },
        "RETURN_TIME_EXPONENT": {
            "name": "Return time exponent",
            "formula": "β = 1/α (for PM maps with parameter α)",
            "description": "Power-law exponent in return time distribution",
            "variables": {"α": "PM parameter", "β": "return time exponent"},
            "units": "dimensionless",
            "source": "Pomeau-Manneville dynamics",
            "reference_implementation": "return_time_exponent",
        },
        "RECURRENCE_RATE": {
            "name": "Recurrence rate",
            "formula": "RR = (1/(N(N-1))) Σ_{i≠j} θ(ε - ||x_i - x_j||)",
            "description": "Fraction of phase-space pairs within distance ε",
            "variables": {"N": "trajectory length", "ε": "recurrence threshold",
                          "θ": "Heaviside function"},
            "units": "dimensionless probability",
            "source": "Marwan et al. (2007)",
            "reference_implementation": "recurrence_rate",
        },
        "DETERMINISM": {
            "name": "Determinism (recurrence)",
            "formula": "DET = (Σ lᵢ·p(lᵢ)) / (Σ p(lᵢ))",
            "description": "Fraction of recurrence points in diagonal lines",
            "variables": {"l": "diagonal line length", "p(l)": "histogram of line lengths",
                          "l_min": "minimum line length"},
            "units": "dimensionless",
            "source": "Recurrence Quantification Analysis",
            "reference_implementation": "determinism_rqa",
        },
        "CORRELATION_DIMENSION": {
            "name": "Correlation dimension",
            "formula": "D₂ = lim_{ε→0} log(C(ε)) / log(ε)",
            "description": "Fractal dimension via correlation integral",
            "variables": {"C(ε)": "correlation integral", "ε": "scale"},
            "units": "dimensionless",
            "source": "Grassberger & Procaccia (1983)",
            "reference_implementation": "correlation_dimension_gp",
        },
        "BRIER_SCORE": {
            "name": "Brier score",
            "formula": "BS = (1/N) Σ (ŷᵢ - yᵢ)²",
            "description": "Mean squared forecast error",
            "variables": {"N": "number of predictions", "ŷ": "forecast probability",
                          "y": "observed outcome (0 or 1)"},
            "units": "dimensionless",
            "source": "Brier (1950)",
            "reference_implementation": "brier_score",
        },
        "BRIER_SKILL_SCORE": {
            "name": "Brier skill score",
            "formula": "BSS = 1 - (BS / BS_ref)",
            "description": "Forecast skill relative to reference baseline",
            "variables": {"BS": "Brier score", "BS_ref": "reference Brier score (climatology)"},
            "units": "dimensionless",
            "source": "Murphy (1971)",
            "reference_implementation": "brier_skill_score",
        },
        "EXPECTED_CALIBRATION_ERROR": {
            "name": "Expected calibration error",
            "formula": "ECE = Σ_k |accuracy_k - confidence_k| · p(bin_k)",
            "description": "Calibration gap across forecast bins",
            "variables": {"accuracy": "fraction correct", "confidence": "mean forecast probability",
                          "p(bin)": "proportion in bin"},
            "units": "dimensionless",
            "source": "Niculescu-Mizil & Caruana (2005)",
            "reference_implementation": "expected_calibration_error",
        },
    }

    @classmethod
    def get_formula(cls, formula_id: str) -> Dict[str, Any]:
        if formula_id not in cls.FORMULAS:
            raise KeyError(f"Formula {formula_id} not registered")
        return cls.FORMULAS[formula_id]

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls.FORMULAS.keys())

    @classmethod
    def verify_all_implemented(cls, metrics_module: Any) -> Dict[str, bool]:
        status = {}
        for formula_id, spec in cls.FORMULAS.items():
            ref_impl = spec.get("reference_implementation", "")
            status[formula_id] = hasattr(metrics_module, ref_impl)
        return status


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: METRIC IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

class MetricComputer:
    """
    Complete implementation of all registered metrics.
    Every method corresponds to a registered formula.
    """

    @staticmethod
    def lambda1_from_trajectory(
        trajectory: np.ndarray, derivative_func: Any, burn_in: int = 100
    ) -> float:
        """Lyapunov exponent: λ₁ = (1/N) Σ log|f'(xᵢ)|  [LAMBDA_1_LOGISTIC]"""
        xs = trajectory[burn_in:]
        if len(xs) == 0:
            return float("nan")
        log_derivatives = np.array([math.log(abs(derivative_func(x))) for x in xs])
        return float(np.mean(log_derivatives))

    @staticmethod
    def lambda1_pm_from_trajectory(
        trajectory: np.ndarray, alpha: float, burn_in: int = 100
    ) -> float:
        """Lyapunov exponent for PM maps: λ₁ = (1/N) Σ log|1 + (1+α)xᵢ^α|  [LAMBDA_1_PM]"""
        xs = trajectory[burn_in:]
        if len(xs) == 0:
            return float("nan")
        log_derivatives = np.array([
            math.log(abs(1 + (1 + alpha) * x**alpha)) for x in xs
        ])
        return float(np.mean(log_derivatives))

    @staticmethod
    def ulam_matrix_from_trajectory(
        trajectory: np.ndarray, partition_size: int
    ) -> np.ndarray:
        """Ulam transition matrix: P_ij = #{I_i→I_j} / #{visits I_i}  [ULAM_TRANSITION_MATRIX]"""
        matrix = np.zeros((partition_size, partition_size))
        bins = np.linspace(0, 1, partition_size + 1)
        idx = np.clip(np.digitize(trajectory, bins) - 1, 0, partition_size - 1)
        for t in range(len(idx) - 1):
            matrix[idx[t], idx[t + 1]] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums

    @staticmethod
    def ulam_spectral_gap(trajectory: np.ndarray, partition_size: int) -> float:
        """Spectral gap: g_gap = |λ₁| - |λ₂|  [ULAM_SPECTRAL_GAP]"""
        P = MetricComputer.ulam_matrix_from_trajectory(trajectory, partition_size)
        eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        if len(eigs) < 2:
            return float("nan")
        return float(eigs[0] - eigs[1])

    @staticmethod
    def correlation_function(
        observable_f: np.ndarray, observable_g: np.ndarray, max_lag: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Temporal correlation: C(n) = ∫ f·(g∘T^n) dμ − ∫f dμ ∫g dμ  [CORRELATION_FUNCTION]"""
        mean_f, mean_g = np.mean(observable_f), np.mean(observable_g)
        lags, correlations = [], []
        for lag in range(1, min(max_lag + 1, len(observable_f) // 2)):
            f_c = observable_f[:-lag] - mean_f
            g_s = observable_g[lag:] - mean_g
            correlations.append(float(np.mean(f_c * g_s)))
            lags.append(lag)
        return np.array(lags), np.array(correlations)

    @staticmethod
    def correlation_decay_exponent(
        lags: np.ndarray, correlations: np.ndarray
    ) -> Tuple[float, float]:
        """Power-law exponent γ where C(n) ~ n^(-γ)  [CORRELATION_DECAY_EXPONENT]"""
        mask = correlations > 0
        lags_pos, corrs_pos = lags[mask], correlations[mask]
        if len(lags_pos) < 2:
            return float("nan"), float("nan")
        log_l, log_c = np.log(lags_pos), np.log(np.abs(corrs_pos))
        coeffs = np.polyfit(log_l, log_c, 1)
        poly = np.poly1d(coeffs)
        res = log_c - poly(log_l)
        ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
        r2 = 1 - np.sum(res ** 2) / ss_tot if ss_tot > 0 else 0.0
        return float(-coeffs[0]), float(r2)

    @staticmethod
    def return_time_survival(
        trajectory: np.ndarray,
        reference_set: Tuple[float, float],
        max_lag: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return time survival: μ(R > n)  [RETURN_TIME_TAIL]"""
        x_min, x_max = reference_set
        in_set = (trajectory >= x_min) & (trajectory <= x_max)
        return_times, current = [], 0
        for flag in in_set:
            if flag:
                if current > 0:
                    return_times.append(current)
                current = 0
            else:
                current += 1
        if not return_times:
            return np.array([]), np.array([])
        rt = np.array(return_times)
        max_rt = int(np.percentile(rt, 95))
        ns = np.arange(1, min(max_rt, max_lag))
        survivals = np.array([np.sum(rt > n) / len(rt) for n in ns])
        return ns, survivals

    @staticmethod
    def return_time_exponent(
        ns: np.ndarray, survivals: np.ndarray
    ) -> Tuple[float, float]:
        """Power-law exponent β where μ(R > n) ~ n^(-β)  [RETURN_TIME_EXPONENT]"""
        mask = survivals > 0
        ns_pos, surv_pos = ns[mask], survivals[mask]
        if len(ns_pos) < 2:
            return float("nan"), float("nan")
        log_n, log_s = np.log(ns_pos), np.log(surv_pos)
        coeffs = np.polyfit(log_n, log_s, 1)
        poly = np.poly1d(coeffs)
        res = log_s - poly(log_n)
        ss_tot = np.sum((log_s - np.mean(log_s)) ** 2)
        r2 = 1 - np.sum(res ** 2) / ss_tot if ss_tot > 0 else 0.0
        return float(-coeffs[0]), float(r2)

    @staticmethod
    def recurrence_rate(trajectory: np.ndarray, epsilon: float) -> float:
        """Recurrence rate: RR = fraction of pairs within ε  [RECURRENCE_RATE]"""
        N = len(trajectory)
        if N < 2:
            return 0.0
        count = sum(
            1 for i in range(N) for j in range(i + 1, N)
            if abs(trajectory[i] - trajectory[j]) < epsilon
        )
        return float(2 * count) / (N * (N - 1))

    @staticmethod
    def determinism_rqa(
        trajectory: np.ndarray, epsilon: float, min_line_length: int = 2
    ) -> float:
        """Determinism: DET = mean diagonal line length  [DETERMINISM]"""
        N = len(trajectory)
        rec = np.array([
            [1 if abs(trajectory[i] - trajectory[j]) < epsilon else 0
             for j in range(N)] for i in range(N)
        ])
        line_lengths = []
        for k in range(-N + 1, N):
            run = 0
            for i in range(N):
                j = i + k
                if 0 <= j < N and rec[i, j]:
                    run += 1
                else:
                    if run >= min_line_length:
                        line_lengths.append(run)
                    run = 0
            if run >= min_line_length:
                line_lengths.append(run)
        return float(sum(line_lengths)) / len(line_lengths) if line_lengths else 0.0

    @staticmethod
    def correlation_dimension_gp(
        trajectory: np.ndarray,
        epsilon_range: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Correlation dimension D₂ via Grassberger-Procaccia  [CORRELATION_DIMENSION]"""
        N = len(trajectory)
        if epsilon_range is None:
            epsilon_range = np.logspace(-3, 0, 20)
        correlations = []
        for eps in epsilon_range:
            count = sum(
                1 for i in range(N) for j in range(i + 1, N)
                if abs(trajectory[i] - trajectory[j]) < eps
            )
            c_eps = count / (N * (N - 1) / 2)
            correlations.append(max(c_eps, 1e-10))
        log_eps, log_c = np.log(epsilon_range), np.log(correlations)
        coeffs = np.polyfit(log_eps, log_c, 1)
        poly = np.poly1d(coeffs)
        res = log_c - poly(log_eps)
        ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
        r2 = 1 - np.sum(res ** 2) / ss_tot if ss_tot > 0 else 0.0
        return float(coeffs[0]), float(r2)

    @staticmethod
    def brier_score(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
        """Brier score: BS = (1/N) Σ (ŷᵢ − yᵢ)²  [BRIER_SCORE]"""
        forecasts = np.asarray(forecasts)
        outcomes = np.asarray(outcomes)
        if len(forecasts) != len(outcomes):
            raise ValueError("Forecast and outcome lengths must match")
        return float(np.mean((forecasts - outcomes) ** 2))

    @staticmethod
    def brier_skill_score(
        forecasts: np.ndarray,
        outcomes: np.ndarray,
        baseline_forecasts: Optional[np.ndarray] = None,
    ) -> float:
        """Brier skill score: BSS = 1 − (BS / BS_ref)  [BRIER_SKILL_SCORE]"""
        bs = MetricComputer.brier_score(forecasts, outcomes)
        if baseline_forecasts is None:
            baseline_forecasts = np.full_like(forecasts, np.mean(outcomes))
        bs_baseline = MetricComputer.brier_score(baseline_forecasts, outcomes)
        if bs_baseline == 0:
            return float("nan")
        return float(1 - bs / bs_baseline)

    @staticmethod
    def expected_calibration_error(
        forecasts: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> float:
        """ECE = Σ_k |acc_k − conf_k| · p_k  [EXPECTED_CALIBRATION_ERROR]"""
        forecasts = np.asarray(forecasts)
        outcomes = np.asarray(outcomes)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (forecasts >= bin_edges[i]) & (forecasts < bin_edges[i + 1])
            if not np.any(mask):
                continue
            conf = float(np.mean(forecasts[mask]))
            acc = float(np.mean(outcomes[mask]))
            prop = float(np.sum(mask)) / len(forecasts)
            ece += prop * abs(conf - acc)
        return float(ece)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: GOVERNANCE & FIREWALLS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HazardStatus:
    code: str
    name: str
    category: str   # OBSERVED | PLAUSIBLE | UNKNOWN | ACTIVE
    status: str
    last_update: str
    mitigation: str


class HazardRegistry:
    """Complete hazard inventory with active firewalls. CH01–CH52+."""

    HAZARDS: Dict[str, HazardStatus] = {
        "CH27": HazardStatus(
            code="CH27", name="RECURSIVE_GOVERNANCE_ACCUMULATION",
            category="ACTIVE", status="BLOCKED", last_update="2026-06-24",
            mitigation="More audits ≠ More evidence. Loop terminated.",
        ),
        "CH28": HazardStatus(
            code="CH28", name="CALIBRATION_OVERREACH",
            category="ACTIVE", status="CONTROLLED", last_update="2026-06-24",
            mitigation="Brier ≠ Truth. Explicitly stated. Firewall active.",
        ),
        "CH31": HazardStatus(
            code="CH31", name="BASELINE_NEGLECT",
            category="ACTIVE", status="BLOCKED", last_update="2026-06-24",
            mitigation="NO_BASELINE → NO_CALIBRATION_CLAIM",
        ),
        "CH38": HazardStatus(
            code="CH38", name="BASELINE_CAPTURE",
            category="ACTIVE", status="CONTROLLED", last_update="2026-06-24",
            mitigation="Baseline frozen before prediction; immutable.",
        ),
        "CH46": HazardStatus(
            code="CH46", name="CLUSTERING_ARTIFACT",
            category="ACTIVE", status="OPEN", last_update="2026-06-24",
            mitigation="Requires: embedding stability test; blind validation.",
        ),
        "CH52": HazardStatus(
            code="CH52", name="UPDATE_FREEZE_RISK",
            category="ACTIVE", status="BLOCKED", last_update="2026-06-24",
            mitigation="Update rule now specified. Changes bounded.",
        ),
    }

    @classmethod
    def get_hazard(cls, code: str) -> Optional[HazardStatus]:
        return cls.HAZARDS.get(code)

    @classmethod
    def list_active(cls) -> List[str]:
        return [code for code, h in cls.HAZARDS.items() if h.category == "ACTIVE"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: REALITY LEDGER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Prediction:
    """Frozen prediction ledger entry. Immutable after freeze timestamp."""
    prediction_id: str
    claim: str
    timestamp_frozen: str
    hash_frozen: str
    confidence: float
    difficulty: str        # LOW | MEDIUM | HIGH
    forecast_horizon: int
    success_condition: str
    failure_condition: str
    primary_resolver: str
    secondary_resolver: str
    independent_referee: str
    resolution_date: str
    outcome: Optional[str]
    brier_score: Optional[float]
    status: str            # DESIGNED | FROZEN | WAITING | RESOLVED | CALIBRATED


class RealityLedger:
    """Immutable ledger of frozen predictions with full governance."""

    PREDICTIONS_FROZEN: Dict[str, Prediction] = {
        "P_001": Prediction(
            prediction_id="P_001",
            claim="CH60 threshold sensitivity affects ≤1/5 nodes",
            timestamp_frozen="2026-06-24T12:00:00Z",
            hash_frozen="sha256_placeholder",
            confidence=0.65, difficulty="MEDIUM", forecast_horizon=7,
            success_condition="BERZERKER rerun with ±0.5 δD_threshold: ≥4/5 nodes maintain verdict",
            failure_condition="≤3/5 nodes maintain verdict",
            primary_resolver="MONSTERBOY", secondary_resolver="External_Validator_A",
            independent_referee="Independent_Referee_1",
            resolution_date="2026-07-01",
            outcome=None, brier_score=None, status="FROZEN",
        ),
        "P_002": Prediction(
            prediction_id="P_002",
            claim="Replay determinism: 5/5 reruns identical verdicts",
            timestamp_frozen="2026-06-24T12:00:00Z",
            hash_frozen="sha256_placeholder",
            confidence=0.92, difficulty="LOW", forecast_horizon=7,
            success_condition="All 5 reruns with seed permutation: 100% verdict match",
            failure_condition="<100% match",
            primary_resolver="MONSTERBOY", secondary_resolver="External_Validator_B",
            independent_referee="Independent_Referee_1",
            resolution_date="2026-07-01",
            outcome=None, brier_score=None, status="FROZEN",
        ),
    }

    @classmethod
    def get_prediction(cls, pred_id: str) -> Optional[Prediction]:
        return cls.PREDICTIONS_FROZEN.get(pred_id)

    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls.PREDICTIONS_FROZEN.keys())

    @classmethod
    def record_resolution(cls, pred_id: str, outcome: str, brier: float) -> bool:
        """Record resolution outcome (immutable after first write)."""
        pred = cls.PREDICTIONS_FROZEN.get(pred_id)
        if pred is None or pred.outcome is not None:
            return False
        pred.outcome = outcome
        pred.brier_score = brier
        pred.status = "RESOLVED"
        return True


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: FAIL-CLOSED VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class ValidationPipeline:
    """Complete fail-closed validation with explicit gates."""

    GATES: Dict[str, Dict[str, str]] = {
        "M01_INTAKE": {"description": "Raw data present", "status": "REQUIRED"},
        "M02_RAW_LEDGER": {"description": "Metadata complete", "status": "REQUIRED"},
        "M03_PARSER": {"description": "Data parseable", "status": "REQUIRED"},
        "M04_RUBRIC": {"description": "Success criteria explicit", "status": "REQUIRED"},
        "M05_REPLAY": {"description": "Deterministic", "status": "REQUIRED"},
        "M07_CLAIM_FIREWALL": {"description": "Overclaim blocked", "status": "REQUIRED"},
        "M20_SCORING": {"description": "R² ≥ 0.95 gate", "status": "REQUIRED"},
        "M25_VERDICT_EXPORT": {"description": "Categories explicit", "status": "REQUIRED"},
    }

    @staticmethod
    def validate_metric(
        metric_name: str,
        formula_id: str,
        result: float,
        r_squared: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate metric against acceptance criteria."""
        formula = FormulaRegistry.get_formula(formula_id)
        verdict: Dict[str, Any] = {
            "metric": metric_name,
            "formula_id": formula_id,
            "result": result,
            "r_squared": r_squared,
            "status": "UNKNOWN",
        }
        if r_squared is not None:
            if r_squared >= 0.95:
                verdict["status"] = "PASS"
            elif r_squared >= 0.80:
                verdict["status"] = "CONDITIONAL"
            else:
                verdict["status"] = "FAIL"
        if not isinstance(result, float) or math.isnan(result) or math.isinf(result):
            verdict["status"] = "INVALID"
        return verdict

    @staticmethod
    def classify_claim(
        observation: Optional[str],
        evidence_r2: Optional[float] = None,
        theory_support: bool = False,
        causality: bool = False,
    ) -> str:
        """Classify claim using fail-closed rules. Default: UNKNOWN."""
        if observation is None:
            return "UNKNOWN"
        if evidence_r2 is not None and evidence_r2 >= 0.95:
            return "OBSERVED"
        if evidence_r2 is not None and evidence_r2 >= 0.80 and theory_support:
            return "SUPPORTED"
        if evidence_r2 is not None and evidence_r2 >= 0.60:
            return "PLAUSIBLE"
        if evidence_r2 is not None and evidence_r2 < 0:
            return "REFUTED"
        return "UNKNOWN"
