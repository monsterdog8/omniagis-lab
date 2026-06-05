"""Functional equivalence tests: source vs gpts_core for coherence module.

Runs source_fn(x) and dest_fn(x) on identical inputs and asserts equality.
Claim ceiling: LOCAL_EQUIVALENCE_ONLY — not proof of exhaustive correctness.
"""
import sys, pathlib

SRC_CAPTURE = pathlib.Path(
    "/tmp/artifacts/DEZIPPED_NEXT_BATCH_SAFE/DEZIPPED_NEXT_BATCH_SAFE"
    "/EXOCHRONOS_CAPTURE_MODEL_v0_1_BUNDLE/exochronos_capture_model_v0_1.py"
)
ACCESSIBLE = SRC_CAPTURE.exists()


def _import_source():
    import importlib.util
    spec = importlib.util.spec_from_file_location("src_capture", SRC_CAPTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import pytest

@pytest.mark.skipif(not ACCESSIBLE, reason="source artifact not accessible")
class TestShannonEntropyEquivalence:
    def setup_method(self):
        self.src = _import_source()
        from gpts_core.coherence import shannon_entropy
        self.dest = shannon_entropy

    def test_uniform_binary(self):
        x = [0, 1] * 50
        assert abs(self.src.shannon_entropy(x) - self.dest(x)) < 1e-12

    def test_single_symbol(self):
        x = [0] * 20
        assert abs(self.src.shannon_entropy(x) - self.dest(x)) < 1e-12

    def test_four_symbols(self):
        x = [0, 1, 2, 3] * 25
        assert abs(self.src.shannon_entropy(x) - self.dest(x)) < 1e-12

    def test_skewed_distribution(self):
        x = [0] * 80 + [1] * 15 + [2] * 5
        assert abs(self.src.shannon_entropy(x) - self.dest(x)) < 1e-12

    def test_empty_input(self):
        assert self.src.shannon_entropy([]) == self.dest([]) == 0.0


@pytest.mark.skipif(not ACCESSIBLE, reason="source artifact not accessible")
class TestJointEntropyEquivalence:
    def setup_method(self):
        self.src = _import_source()
        from gpts_core.coherence import joint_entropy
        self.dest = joint_entropy

    def test_independent(self):
        import random
        rng = random.Random(99)
        x = [rng.randint(0, 3) for _ in range(200)]
        y = [rng.randint(0, 3) for _ in range(200)]
        assert abs(self.src.joint_entropy(x, y) - self.dest(x, y)) < 1e-12

    def test_identical_sequences(self):
        x = [0, 1, 2, 3] * 25
        assert abs(self.src.joint_entropy(x, x) - self.dest(x, x)) < 1e-12

    def test_complementary(self):
        x = [0] * 50 + [1] * 50
        y = [1] * 50 + [0] * 50
        assert abs(self.src.joint_entropy(x, y) - self.dest(x, y)) < 1e-12


@pytest.mark.skipif(not ACCESSIBLE, reason="source artifact not accessible")
class TestDigitizeEquivalence:
    def setup_method(self):
        self.src = _import_source()
        from gpts_core.coherence import digitize
        self.dest = digitize

    def test_linear_range(self):
        x = [float(i) / 10 for i in range(10)]
        src_result = self.src.digitize(x, bins=8)
        dst_result = self.dest(x, bins=8)
        assert src_result == dst_result, f"src={src_result} dst={dst_result}"

    def test_constant_input(self):
        x = [0.5] * 10
        assert self.src.digitize(x, bins=8) == self.dest(x, bins=8)

    def test_two_values(self):
        x = [0.0, 1.0] * 5
        assert self.src.digitize(x, bins=4) == self.dest(x, bins=4)


@pytest.mark.skipif(not ACCESSIBLE, reason="source artifact not accessible")
class TestGlobalCoherenceEquivalence:
    """global_coherence is defined in destination only (computed from i_mutual/h_total inline in source)."""
    def test_formula_holds(self):
        from gpts_core.coherence import global_coherence
        assert global_coherence(0.6, 1.2) == pytest.approx(0.5, abs=1e-9)
        assert global_coherence(0.0, 0.0) == 0.0
        assert global_coherence(1.0, 2.0) == pytest.approx(0.5, abs=1e-9)
