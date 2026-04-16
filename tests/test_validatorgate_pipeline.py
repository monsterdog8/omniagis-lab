from __future__ import annotations

import json

import numpy as np
import pytest

from omniagis import epsilon_sweep, exp_rt_runner, validatorgate_full


def test_run_experiment_rejects_invalid_z() -> None:
    with pytest.raises(ValueError, match="z must be > 1.0"):
        exp_rt_runner.run_experiment(exp_rt_runner.ExperimentParams(z=1.0))


@pytest.mark.parametrize(
    "params,match",
    [
        (epsilon_sweep.SweepParams(z=1.0), "z must be > 1.0"),
        (
            epsilon_sweep.SweepParams(epsilon_min=0.0, epsilon_max=0.2),
            "epsilon_min and epsilon_max must be > 0",
        ),
        (
            epsilon_sweep.SweepParams(epsilon_min=0.3, epsilon_max=0.2),
            "epsilon_min must be <= epsilon_max",
        ),
    ],
)
def test_run_epsilon_sweep_rejects_invalid_inputs(
    params: epsilon_sweep.SweepParams, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        epsilon_sweep.run_epsilon_sweep(params)


def test_exp_rt_runner_main_output_file_is_json_only(tmp_path, capsys, monkeypatch) -> None:
    out_path = tmp_path / "result.json"
    monkeypatch.setattr(
        exp_rt_runner,
        "run_experiment",
        lambda _params: {"ok": True},
    )

    exp_rt_runner.main(["--output", str(out_path)])

    assert json.loads(out_path.read_text()) == {"ok": True}
    assert capsys.readouterr().err == ""


def test_epsilon_sweep_main_output_file_is_json_only(tmp_path, capsys, monkeypatch) -> None:
    out_path = tmp_path / "sweep.json"
    monkeypatch.setattr(
        epsilon_sweep,
        "run_epsilon_sweep",
        lambda _params: {"ok": True},
    )

    epsilon_sweep.main(["--output", str(out_path)])

    assert json.loads(out_path.read_text()) == {"ok": True}
    assert capsys.readouterr().err == ""


def test_multi_scale_ci_allows_equal_widths_when_scales_repeat(monkeypatch) -> None:
    tau = np.array([1, 2, 3, 4], dtype=np.int64)  # scales become [4, 4, 4]

    monkeypatch.setattr(
        validatorgate_full,
        "fit_power_law_tail",
        lambda *args, **kwargs: validatorgate_full.PowerLawFit(
            alpha=1.0,
            log_c=0.0,
            r_squared=1.0,
            n_tail_points=3,
            tail_sample_size=4,
            n_min=1,
            min_tail_obs=0,
        ),
    )
    monkeypatch.setattr(
        validatorgate_full,
        "bootstrap_ci",
        lambda *args, **kwargs: (0.5, 1.5, 5),  # constant CI width = 1.0
    )

    out = validatorgate_full.multi_scale_ci(
        tau,
        n_min=1,
        min_tail_sample_size=0,
        min_tail_obs=0,
        n_bootstrap=3,
    )
    assert out.scales == [4, 4, 4]
    assert out.ci_shrinks is True


def test_validate_accepts_explicit_rng() -> None:
    trajectory = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9], dtype=np.float64)
    config = validatorgate_full.ValidationConfig(
        n_min=1,
        min_tail_sample_size=1,
        min_tail_obs=1,
        n_bootstrap=5,
    )
    report = validatorgate_full.validate(
        trajectory,
        lambda x: x <= 0.2,
        config=config,
        rng=np.random.default_rng(123),
    )
    assert report["verdict"] in {"ACCEPTED", "FAIL_CLOSED"}
