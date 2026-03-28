"""Fail-closed verdict classifier."""

from __future__ import annotations

from typing import List

VALID_VERDICTS = {"PASS", "PARTIAL PASS", "NO PASS"}


class FailClosedClassifier:
    """Combine multiple validation verdicts using fail-closed logic.

    Fail-closed means: in the absence of sufficient evidence of correctness,
    the system is considered invalid.

    Rules (in priority order):
    1. Empty list or any unknown value → "NO PASS"
    2. Any "NO PASS" in the list → "NO PASS"
    3. Any "PARTIAL PASS" (with no NO PASS) → "PARTIAL PASS"
    4. All "PASS" → "PASS"
    """

    def combine(self, results: List[str]) -> str:
        """Combine a list of verdict strings.

        Parameters
        ----------
        results:
            List of verdict strings.  Each must be one of
            ``"PASS"``, ``"PARTIAL PASS"``, or ``"NO PASS"``.

        Returns
        -------
        str
            Combined verdict.
        """
        if not results:
            return "NO PASS"

        # Any unknown value triggers fail-closed NO PASS
        if any(r not in VALID_VERDICTS for r in results):
            return "NO PASS"

        if any(r == "NO PASS" for r in results):
            return "NO PASS"

        if any(r == "PARTIAL PASS" for r in results):
            return "PARTIAL PASS"

        return "PASS"
