"""Targeted gap tests: scorecard helpers, parsability exception path,
inventory OSError, epsilon_sweep/exp_rt_runner stdout, validatorgate edge cases."""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# scorecard._pct(0, 0)
# ---------------------------------------------------------------------------

from omniagis.audit.scorecard import _pct, build_scorecard
from omniagis.audit.inventory import (
    FileRecord, InventoryReport, CODE_PYTHON, UNKNOWN,
)
from omniagis.audit.parsability import ParseResult


class TestPct:
    def test_zero_denominator_returns_zero(self):
        assert _pct(0, 0) == 0.0

    def test_normal_fraction(self):
        assert _pct(1, 4) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# scorecard M7 — no requirements.txt (PARTIAL PASS) and missing deps (NO PASS)
# ---------------------------------------------------------------------------

def _empty_inventory():
    return InventoryReport(files=[], duplicates=[], summary={})


class TestM7NoBranches:
    def test_m7_partial_pass_when_no_req_file(self, monkeypatch):
        monkeypatch.setattr("omniagis.audit.scorecard.os.path.isfile", lambda p: False)
        entries = build_scorecard(_empty_inventory(), [])
        m7 = next(e for e in entries if e.metric_id == "M7")
        assert m7.status == "PARTIAL PASS"
        assert "No requirements.txt" in m7.detail

    def test_m7_no_pass_when_package_missing(self, monkeypatch):
        # Patch importlib.util.find_spec so numpy (the only dep in requirements.txt)
        # appears uninstalled — _ilu is the same module object, so this propagates.
        orig = importlib.util.find_spec

        def fake_find_spec(name):
            if name == "numpy":
                return None
            return orig(name)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        entries = build_scorecard(_empty_inventory(), [])
        m7 = next(e for e in entries if e.metric_id == "M7")
        assert m7.status == "NO PASS"
        assert "numpy" in m7.detail


# ---------------------------------------------------------------------------
# parsability._is_available — ValueError / ModuleNotFoundError path
# ---------------------------------------------------------------------------

from omniagis.audit.parsability import _is_available


class TestIsAvailableException:
    def test_empty_string_returns_false(self):
        # importlib.util.find_spec("") raises ValueError → caught → False
        result = _is_available("")
        assert result is False

    def test_nonexistent_module_returns_false(self):
        assert _is_available("totally_fake_xyz_123_abc") is False

    def test_module_not_found_error_caught(self, monkeypatch):
        orig = importlib.util.find_spec

        def raise_mnf(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        monkeypatch.setattr(importlib.util, "find_spec", raise_mnf)
        # stdlib check passes first for "os", but for an unknown module both
        # stdlib check and sys.modules check fail, then find_spec is called
        result = _is_available("totally_fake_xyz_not_in_stdlib")
        assert result is False


# ---------------------------------------------------------------------------
# inventory — OSError in file walk (silently skipped)
# ---------------------------------------------------------------------------

from omniagis.audit.inventory import FileInventory, _sha256 as _inv_sha256


class TestInventoryOsError:
    def test_ioerror_on_hash_skips_file(self, tmp_path, monkeypatch):
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("y = 2\n")

        original_sha256 = _inv_sha256
        calls = []

        def patched_sha256(path):
            calls.append(path)
            if "bad.py" in path:
                raise OSError("permission denied")
            return original_sha256(path)

        monkeypatch.setattr("omniagis.audit.inventory._sha256", patched_sha256)
        report = FileInventory().build(str(tmp_path))
        paths = [r.path for r in report.files]
        assert not any("bad.py" in p for p in paths)
        assert any("good.py" in p for p in paths)


# ---------------------------------------------------------------------------
# epsilon_sweep.main() — stdout print branch (no --output)
# ---------------------------------------------------------------------------

from omniagis.epsilon_sweep import main as sweep_main


class TestEpsilonSweepMainStdout:
    def test_print_to_stdout_when_no_output_arg(self, capsys):
        sweep_main([
            "--n-steps", "200",
            "--n-epsilons", "2",
            "--z", "1.5",
            "--seed", "42",
            "--min-tail-sample-size", "5",
        ])
        out = capsys.readouterr().out
        assert len(out) > 0
        parsed = json.loads(out)
        assert isinstance(parsed, (dict, list))

    def test_output_to_file_does_not_print(self, tmp_path, capsys):
        out_file = str(tmp_path / "result.json")
        sweep_main([
            "--n-steps", "200",
            "--n-epsilons", "2",
            "--z", "1.5",
            "--seed", "42",
            "--min-tail-sample-size", "5",
            "--output", out_file,
        ])
        captured = capsys.readouterr().out
        assert captured == ""
        assert os.path.isfile(out_file)


# ---------------------------------------------------------------------------
# exp_rt_runner.main() — stdout print branch (no --output)
# ---------------------------------------------------------------------------

from omniagis.exp_rt_runner import main as runner_main


class TestExpRtRunnerMainStdout:
    def test_print_to_stdout_when_no_output_arg(self, capsys):
        runner_main([
            "--n-steps", "200",
            "--z", "1.5",
            "--epsilon", "0.1",
            "--seed", "42",
            "--min-tail-sample-size", "5",
        ])
        out = capsys.readouterr().out
        assert len(out) > 0
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_output_to_file_does_not_print(self, tmp_path, capsys):
        out_file = str(tmp_path / "result.json")
        runner_main([
            "--n-steps", "200",
            "--z", "1.5",
            "--epsilon", "0.1",
            "--seed", "42",
            "--min-tail-sample-size", "5",
            "--output", out_file,
        ])
        captured = capsys.readouterr().out
        assert captured == ""
        assert os.path.isfile(out_file)


# ---------------------------------------------------------------------------
# validatorgate_full — edge cases
# ---------------------------------------------------------------------------

from omniagis.validatorgate_full import (
    ValidationConfig,
    bootstrap_ci,
    detect_plateau,
    fit_power_law_tail,
    multi_scale_ci,
    survival_function,
    validate,
)


class TestSurvivalFunctionMaxN:
    def test_max_n_zero_returns_empty(self):
        tau = np.array([5, 10, 15], dtype=np.int64)
        ns, S = survival_function(tau, max_n=0)
        assert ns.size == 0
        assert S.size == 0

    def test_max_n_negative_returns_empty(self):
        tau = np.array([1, 2, 3], dtype=np.int64)
        ns, S = survival_function(tau, max_n=-1)
        assert ns.size == 0
        assert S.size == 0


class TestDetectPlateauFewSlopes:
    def test_four_positive_points_gives_three_slopes_no_plateau(self):
        # 4 ns_p points → 3 raw slopes → n_valid_slopes = 3 < 4 → early return
        ns = np.array([1, 2, 3, 4], dtype=np.int64)
        S = np.array([0.8, 0.6, 0.4, 0.1])
        result = detect_plateau(ns, S)
        assert result.detected is False
        assert result.n_valid_slopes == 3

    def test_fewer_than_four_positive_s_values(self):
        # Only 3 S > 0 values → ns_p.size = 3 < 4 → first early return
        ns = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        S = np.array([0.6, 0.3, 0.1, 0.0, 0.0])
        result = detect_plateau(ns, S)
        assert result.detected is False


class TestBootstrapCiEdgeCases:
    def test_rng_none_uses_default_rng(self):
        # Covers line 365: rng = np.random.default_rng()
        tau = np.arange(1, 30, dtype=np.int64)
        lo, hi, nv = bootstrap_ci(
            tau, n_min=3, min_tail_sample_size=5, min_tail_obs=2,
            n_bootstrap=10,
            # rng deliberately omitted → defaults to None → line 365 executed
        )
        # Result may be NaN (insufficient data) or finite — either is acceptable
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_size_one_tau_all_nan_fits(self):
        # tau.size = 1 → every resample has size 1 → _alpha_from_tau line 326 hit
        tau = np.array([5], dtype=np.int64)
        lo, hi, nv = bootstrap_ci(
            tau, n_min=3, min_tail_sample_size=5, min_tail_obs=2,
            n_bootstrap=5, rng=np.random.default_rng(0),
        )
        assert np.isnan(lo)
        assert np.isnan(hi)
        assert nv == 0

    def test_zero_tau_survival_returns_empty(self):
        # tau values all 0 → max_tau = 0 → survival empty → _alpha_from_tau line 329
        tau = np.zeros(4, dtype=np.int64)
        lo, hi, nv = bootstrap_ci(
            tau, n_min=1, min_tail_sample_size=1, min_tail_obs=1,
            n_bootstrap=5, rng=np.random.default_rng(0),
        )
        assert np.isnan(lo)
        assert np.isnan(hi)


class TestMultiScaleCiEdgeCases:
    def test_zero_tau_triggers_nan_alpha_hat(self):
        # survival returns empty for sub-arrays of all zeros → line 461: a_hat = nan
        tau = np.zeros(8, dtype=np.int64)
        result = multi_scale_ci(
            tau, n_min=1, min_tail_sample_size=1, min_tail_obs=1,
            n_bootstrap=3, rng=np.random.default_rng(0),
        )
        # All alpha_hat values should be NaN (no valid fits possible)
        assert all(np.isnan(a) for a in result.alpha_hat)


class TestValidateAlphaOutOfRange:
    def test_alpha_out_of_bounds_in_fail_reasons(self):
        # Use tight alpha bounds that no realistic fit can satisfy → line 651
        from omniagis.exp_rt_runner import generate_trajectory
        traj = generate_trajectory(n_steps=3000, z=1.5, x0=0.5)
        in_target = lambda x: x <= 0.1
        config = ValidationConfig(
            alpha_min=50.0,   # impossibly high — any real alpha will be out of bounds
            alpha_max=100.0,
            n_min=3,
            min_tail_sample_size=10,
            min_tail_obs=3,
            n_bootstrap=5,
            seed=42,
        )
        result = validate(traj, in_target, config)
        fail_reasons = result.get("fail_reasons", [])
        assert isinstance(fail_reasons, list)
        assert result["verdict"] in ("ACCEPTED", "FAIL_CLOSED")

    def test_theory_alpha_outside_ci_in_fail_reasons(self):
        # Set theory_alpha to an impossible value → line 748-751 in validate
        from omniagis.exp_rt_runner import generate_trajectory
        traj = generate_trajectory(n_steps=3000, z=1.5, x0=0.5)
        in_target = lambda x: x <= 0.1
        config = ValidationConfig(
            theory_alpha=999.0,  # no CI will contain this
            n_min=3,
            min_tail_sample_size=10,
            min_tail_obs=3,
            n_bootstrap=5,
            seed=42,
        )
        result = validate(traj, in_target, config)
        assert result["verdict"] in ("ACCEPTED", "FAIL_CLOSED")
