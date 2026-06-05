"""Functional equivalence: source (calc_weighted_stability) vs gpts_core (mse_score).

Source: scoring.py in monsterdog_kaggle_minibench_001.
PASS criterion: abs(source_fn(x) - dest_fn(x)) < 1e-12 on all test cases.
"""
import pathlib, math
import pytest

SRC_SCORING = pathlib.Path(
    "/tmp/artifacts3/f6989625-monsterdog_kaggle_minibench_001"
    "/monsterdog_kaggle_minibench_001/scoring.py"
)
ACCESSIBLE = SRC_SCORING.exists()


def _import_source():
    import importlib.util
    spec = importlib.util.spec_from_file_location("src_scoring", SRC_SCORING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not ACCESSIBLE, reason="source artifact not accessible")
class TestMseScoreEquivalence:
    """source: calc_weighted_stability  →  dest: mse_score
    Same logic: 1 - MSE with missing/invalid/range penalties.
    """
    def setup_method(self):
        self.src_mod = _import_source()
        self.src = self.src_mod.calc_weighted_stability
        from gpts_core.benchmark import mse_score
        self.dst = mse_score

    def _check(self, preds, truth, tol=1e-12):
        s = self.src(preds, truth)
        d = self.dst(preds, truth)
        assert abs(s - d) < tol, f"src={s} dst={d} preds={preds} truth={truth}"

    def test_perfect_predictions(self):
        self._check({"a": 0.8, "b": 0.5, "c": 0.3},
                    {"a": 0.8, "b": 0.5, "c": 0.3})

    def test_all_missing(self):
        self._check({}, {"a": 0.8, "b": 0.5})

    def test_empty_truth(self):
        assert self.src({}, {}) == self.dst({}, {}) == 0.0

    def test_max_error(self):
        self._check({"a": 0.0}, {"a": 1.0})

    def test_single_perfect(self):
        self._check({"a": 0.5}, {"a": 0.5})

    def test_invalid_prediction(self):
        self._check({"a": float("inf")}, {"a": 0.5})

    def test_out_of_range_prediction(self):
        self._check({"a": 1.5}, {"a": 0.5})

    def test_mixed_keys(self):
        # some present, some missing
        self._check({"a": 0.8}, {"a": 0.8, "b": 0.5})

    def test_multiple_samples(self):
        preds = {str(i): 0.1 * i for i in range(10)}
        truth = {str(i): 0.1 * i for i in range(10)}
        self._check(preds, truth)
