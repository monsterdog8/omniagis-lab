#!/usr/bin/env python3
"""OMNIÆGIS data processing and validation script.

This script processes data through cleaning, transformation, aggregation,
and validation steps, returning a JSON report with SHA-256 hash.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional


def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process data through cleaning, transformation, aggregation, and validation.

    Args:
        data: List of dictionaries with 'id' and 'value' keys

    Returns:
        Dictionary with processing results in JSON-serializable format

    Raises:
        ValueError: If input data is empty after cleaning
    """
    try:
        # Step 1: Cleaning - remove entries where value is None
        cleaned_data = [entry for entry in data if entry.get("value") is not None]
        clean_count = len(cleaned_data)

        # Step 5: Error handling - check if list is empty
        if clean_count == 0:
            return {
                "error": "Empty list after cleaning",
                "clean_count": 0,
                "transformed": [],
                "sum": 0,
                "sum_squared": 0,
                "mean": 0.0,
                "consistency_check": "INCONSISTENT",
                "hash": ""
            }

        # Step 2: Transformation - create new list with id, value, value_squared
        transformed = []
        for entry in cleaned_data:
            value = entry["value"]
            transformed.append({
                "id": entry["id"],
                "value": value,
                "value_squared": value * value
            })

        # Step 3: Aggregation
        sum_value = sum(entry["value"] for entry in transformed)
        sum_squared = sum(entry["value_squared"] for entry in transformed)
        mean_value = sum_value / clean_count

        # Step 4: Validation - check that mean * count ≈ sum (tolerance 1e-6)
        consistency_check = "CONSISTENT" if abs(mean_value * clean_count - sum_value) < 1e-6 else "INCONSISTENT"

        # Step 6: Hash final - SHA-256 based on concat of value_squared sorted
        sorted_values_squared = sorted(entry["value_squared"] for entry in transformed)
        hash_input = "".join(str(v) for v in sorted_values_squared)
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()

        return {
            "clean_count": clean_count,
            "transformed": transformed,
            "sum": sum_value,
            "sum_squared": sum_squared,
            "mean": mean_value,
            "consistency_check": consistency_check,
            "hash": hash_digest
        }

    except Exception as e:
        # Return structured error
        return {
            "error": str(e),
            "clean_count": 0,
            "transformed": [],
            "sum": 0,
            "sum_squared": 0,
            "mean": 0.0,
            "consistency_check": "INCONSISTENT",
            "hash": ""
        }


def main():
    """Main entry point - process predefined data and output JSON."""
    # Input data (as specified in instructions)
    data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 25},
        {"id": 3, "value": 40},
        {"id": 4, "value": None},
        {"id": 5, "value": 15},
        {"id": 6, "value": 60}
    ]

    result = process_data(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
