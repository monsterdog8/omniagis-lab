"""OMNIÆGIS — fail-closed scientific validation framework for intermittent dynamical systems."""

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
