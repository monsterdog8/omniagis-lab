"""OMNIÆGIS — Fail-closed scientific validation framework.

GO OMNIÆGIS (Generalized Omni-Aegis) provides:
- ε-robustness validation for dynamical system trajectories
- Return-time (Poincaré recurrence) statistics for scalar time series
- Fail-closed verdict classification
- Mode MAVERICK code audit with M1–M12 scorecard

Public API:
    Core validation components:
        - EpsilonRobustnessValidator: Trajectory validation via L2 distance
        - ReturnTimeStatistics: Poincaré recurrence analysis
        - FailClosedClassifier: Fail-closed verdict aggregation

    Audit components:
        - FileInventory: File type classification and SHA-256 inventory
        - ParsabilityChecker: Python syntax and import validation
        - build_scorecard: M1–M12 metric scorecard generator
        - ColdPass: Mode MAVERICK orchestrator (outputs A–G)

Usage:
    >>> from omniagis import EpsilonRobustnessValidator
    >>> import numpy as np
    >>> validator = EpsilonRobustnessValidator(epsilon=1e-3)
    >>> result = validator.validate(trajectory, reference)
    >>> print(result.status)  # "PASS" | "PARTIAL PASS" | "NO PASS"

    >>> from omniagis import ColdPass
    >>> auditor = ColdPass()
    >>> report = auditor.run("/path/to/project")
    >>> print(report.global_verdict)
"""

__version__ = "0.1.0"

# Core validation components
from .core import (
    EpsilonRobustnessValidator,
    ReturnTimeStatistics,
    FailClosedClassifier,
)

# Audit components
from .audit import (
    FileInventory,
    ParsabilityChecker,
    build_scorecard,
    ColdPass,
    BundleAuditor,
)

# Scientific validation engine (validatorgate_full)
from .validatorgate_full import (
    ValidationConfig,
    PowerLawFit,
    PlateauResult,
    MultiScaleCI,
    compute_return_times,
    survival_function,
    fit_power_law_tail,
    detect_plateau,
    bootstrap_ci,
    multi_scale_ci,
    validate,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "EpsilonRobustnessValidator",
    "ReturnTimeStatistics",
    "FailClosedClassifier",
    # Audit
    "FileInventory",
    "ParsabilityChecker",
    "build_scorecard",
    "ColdPass",
    "BundleAuditor",
    # Scientific validation engine
    "ValidationConfig",
    "PowerLawFit",
    "PlateauResult",
    "MultiScaleCI",
    "compute_return_times",
    "survival_function",
    "fit_power_law_tail",
    "detect_plateau",
    "bootstrap_ci",
    "multi_scale_ci",
    "validate",
]
