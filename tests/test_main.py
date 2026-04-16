from __future__ import annotations

from omniagis.__main__ import main


def test_main_delegates_to_audit_cli(monkeypatch):
    captured_args: dict[str, list[str]] = {}

    def fake_audit(argv):
        captured_args["argv"] = argv

    monkeypatch.setattr("omniagis.cli.main", fake_audit)

    main(["."])

    assert captured_args["argv"] == ["."]


def test_main_delegates_to_benchmark_runner(monkeypatch):
    captured_args: dict[str, list[str]] = {}

    def fake_benchmark(argv):
        captured_args["argv"] = argv

    monkeypatch.setattr("omniagis.exp_rt_runner.main", fake_benchmark)

    main(["benchmark", "--n-steps", "128"])

    assert captured_args["argv"] == ["--n-steps", "128"]


def test_main_delegates_to_epsilon_sweep(monkeypatch):
    captured_args: dict[str, list[str]] = {}

    def fake_sweep(argv):
        captured_args["argv"] = argv

    monkeypatch.setattr("omniagis.epsilon_sweep.main", fake_sweep)

    main(["benchmark-sweep", "--n-epsilons", "5"])

    assert captured_args["argv"] == ["--n-epsilons", "5"]
