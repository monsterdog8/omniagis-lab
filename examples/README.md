# GO OMNIÆGIS Examples

This directory contains comprehensive examples demonstrating the full capabilities
of the GO OMNIÆGIS (Generalized Omni-Aegis) framework.

## Examples Overview

### Example 1: Epsilon-Robustness Validation
**File:** `example_1_epsilon_robustness.py`

Demonstrates trajectory validation using pointwise L2 distance metrics.
Shows PASS, PARTIAL PASS, and NO PASS verdicts for various trajectory deviations.

**Run:**
```bash
python examples/example_1_epsilon_robustness.py
```

### Example 2: Return-Time Statistics
**File:** `example_2_return_time.py`

Demonstrates Poincaré recurrence analysis on scalar time series.
Analyzes periodic, damped, and chaotic-like signals.

**Run:**
```bash
python examples/example_2_return_time.py
```

### Example 3: Fail-Closed Classification
**File:** `example_3_fail_closed.py`

Demonstrates fail-closed verdict aggregation logic.
Shows how multiple component verdicts combine into system-wide verdicts.

**Run:**
```bash
python examples/example_3_fail_closed.py
```

### Example 4: Mode MAVERICK Code Audit
**File:** `example_4_mode_maverick.py`

Demonstrates comprehensive code quality auditing with M1–M12 scorecard
and A–G section output. Creates a sample project and audits it.

**Run:**
```bash
python examples/example_4_mode_maverick.py
```

### Example 5: Complete Workflow Integration
**File:** `example_5_full_workflow.py`

Demonstrates the COMPLETE GO OMNIÆGIS workflow integrating all components:
- Dynamical system validation
- Return-time analysis
- Fail-closed classification
- Mode MAVERICK audit

This example showcases the full power of the framework.

**Run:**
```bash
python examples/example_5_full_workflow.py
```

## Requirements

All examples require:
```bash
pip install numpy
```

The framework itself can be installed with:
```bash
pip install -e .
```

## GO OMNIÆGIS Philosophy

These examples demonstrate the **fail-closed** validation philosophy:
- Any component failure → system fails
- Any partial success → system partially succeeds
- Only all successes → system succeeds

This ensures **conservative, rigorous validation** suitable for scientific
and safety-critical applications.
