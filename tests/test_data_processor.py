"""Tests for omniagis.data_processor — process_data and main."""
from __future__ import annotations

import hashlib
import json
from io import StringIO
from unittest.mock import patch

import pytest

from omniagis.data_processor import main, process_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ERROR_KEYS = {"error", "clean_count", "transformed", "sum", "sum_squared", "mean",
              "consistency_check", "hash"}
SUCCESS_KEYS = {"clean_count", "transformed", "sum", "sum_squared", "mean",
                "consistency_check", "hash"}


def _expected_hash(values_squared: list) -> str:
    sorted_sq = sorted(values_squared)
    return hashlib.sha256("".join(str(v) for v in sorted_sq).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Normal (happy-path) processing
# ---------------------------------------------------------------------------

class TestProcessDataNormal:
    def test_returns_success_keys(self):
        result = process_data([{"id": 1, "value": 4}])
        assert SUCCESS_KEYS == result.keys()

    def test_clean_count_excludes_none(self):
        data = [{"id": 1, "value": 10}, {"id": 2, "value": None}, {"id": 3, "value": 5}]
        result = process_data(data)
        assert result["clean_count"] == 2

    def test_transformation_value_squared(self):
        data = [{"id": 1, "value": 3}, {"id": 2, "value": 4}]
        result = process_data(data)
        for entry in result["transformed"]:
            assert entry["value_squared"] == entry["value"] ** 2

    def test_transformation_preserves_id_and_value(self):
        data = [{"id": 99, "value": 7}]
        result = process_data(data)
        assert result["transformed"][0]["id"] == 99
        assert result["transformed"][0]["value"] == 7

    def test_sum_is_correct(self):
        data = [{"id": i, "value": i * 10} for i in range(1, 6)]
        result = process_data(data)
        assert result["sum"] == sum(i * 10 for i in range(1, 6))

    def test_sum_squared_is_correct(self):
        data = [{"id": i, "value": i} for i in range(1, 4)]
        result = process_data(data)
        assert result["sum_squared"] == sum(i ** 2 for i in range(1, 4))

    def test_mean_is_correct(self):
        data = [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
        result = process_data(data)
        assert result["mean"] == pytest.approx(15.0)

    def test_consistency_check_is_consistent_for_integers(self):
        data = [{"id": i, "value": i} for i in range(1, 5)]
        result = process_data(data)
        assert result["consistency_check"] == "CONSISTENT"

    def test_hash_is_64_hex_chars(self):
        result = process_data([{"id": 1, "value": 5}])
        assert len(result["hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["hash"])

    def test_hash_deterministic(self):
        data = [{"id": 1, "value": 3}, {"id": 2, "value": 4}]
        assert process_data(data)["hash"] == process_data(data)["hash"]

    def test_hash_matches_expected_algorithm(self):
        data = [{"id": 1, "value": 3}, {"id": 2, "value": 4}]
        result = process_data(data)
        assert result["hash"] == _expected_hash([9, 16])

    def test_hash_is_order_independent(self):
        """Hash is based on *sorted* value_squared, so entry order doesn't matter."""
        data_a = [{"id": 1, "value": 3}, {"id": 2, "value": 4}]
        data_b = [{"id": 2, "value": 4}, {"id": 1, "value": 3}]
        assert process_data(data_a)["hash"] == process_data(data_b)["hash"]

    def test_none_values_are_excluded_from_transform(self):
        data = [{"id": 1, "value": None}, {"id": 2, "value": 5}]
        result = process_data(data)
        assert len(result["transformed"]) == 1
        assert result["transformed"][0]["id"] == 2

    def test_single_entry(self):
        result = process_data([{"id": 1, "value": 7}])
        assert result["clean_count"] == 1
        assert result["mean"] == pytest.approx(7.0)
        assert result["sum"] == 7
        assert result["sum_squared"] == 49

    def test_canonical_fixture_from_docs(self):
        """Matches the hardcoded data in main()."""
        data = [
            {"id": 1, "value": 10},
            {"id": 2, "value": 25},
            {"id": 3, "value": 40},
            {"id": 4, "value": None},
            {"id": 5, "value": 15},
            {"id": 6, "value": 60},
        ]
        result = process_data(data)
        assert result["clean_count"] == 5
        assert result["sum"] == 150
        assert result["mean"] == pytest.approx(30.0)
        assert result["consistency_check"] == "CONSISTENT"

    def test_float_values_accepted(self):
        data = [{"id": 1, "value": 1.5}, {"id": 2, "value": 2.5}]
        result = process_data(data)
        assert result["sum"] == pytest.approx(4.0)
        assert result["sum_squared"] == pytest.approx(1.5 ** 2 + 2.5 ** 2)


# ---------------------------------------------------------------------------
# Empty-after-cleaning branch
# ---------------------------------------------------------------------------

class TestProcessDataAllNone:
    def test_all_none_returns_error_key(self):
        data = [{"id": 1, "value": None}, {"id": 2, "value": None}]
        result = process_data(data)
        assert "error" in result

    def test_all_none_error_message(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["error"] == "Empty list after cleaning"

    def test_all_none_clean_count_zero(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["clean_count"] == 0

    def test_all_none_empty_transformed(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["transformed"] == []

    def test_all_none_sums_zero(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["sum"] == 0
        assert result["sum_squared"] == 0

    def test_all_none_mean_zero(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["mean"] == 0.0

    def test_all_none_consistency_inconsistent(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["consistency_check"] == "INCONSISTENT"

    def test_all_none_hash_empty_string(self):
        result = process_data([{"id": 1, "value": None}])
        assert result["hash"] == ""

    def test_empty_list_input_also_triggers_error(self):
        result = process_data([])
        assert result["error"] == "Empty list after cleaning"

    def test_returns_all_error_keys(self):
        result = process_data([])
        assert ERROR_KEYS == result.keys()


# ---------------------------------------------------------------------------
# Exception / malformed-input fallback branch
# ---------------------------------------------------------------------------

class TestProcessDataExceptionFallback:
    def test_missing_id_key_raises_caught_exception(self):
        # Entry has value but no 'id' — transform step raises KeyError, caught by except
        data = [{"value": 5}]
        result = process_data(data)
        assert "error" in result
        assert result["clean_count"] == 0
        assert result["transformed"] == []
        assert result["consistency_check"] == "INCONSISTENT"
        assert result["hash"] == ""

    def test_non_numeric_value_raises_caught_exception(self):
        # value * value on a string raises TypeError
        data = [{"id": 1, "value": "bad"}]
        result = process_data(data)
        assert "error" in result

    def test_exception_result_has_all_error_keys(self):
        data = [{"value": 5}]  # missing 'id'
        result = process_data(data)
        assert ERROR_KEYS == result.keys()


# ---------------------------------------------------------------------------
# main() — output and JSON validity
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_prints_valid_json(self, capsys):
        main()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)

    def test_main_output_has_expected_keys(self, capsys):
        main()
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "clean_count" in result
        assert "hash" in result
        assert "consistency_check" in result

    def test_main_clean_count_is_five(self, capsys):
        main()
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["clean_count"] == 5

    def test_main_consistency_consistent(self, capsys):
        main()
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["consistency_check"] == "CONSISTENT"

    def test_main_hash_is_valid_sha256(self, capsys):
        main()
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert len(result["hash"]) == 64
