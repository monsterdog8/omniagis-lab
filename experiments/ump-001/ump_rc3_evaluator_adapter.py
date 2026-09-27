#!/usr/bin/env python3
from __future__ import annotations
from ump_rc3_core import proper_score_skill, uniform_null

CLASSES=("U0","U1","U2")
NULL=uniform_null(CLASSES)

def score_trial(model_probs, outcome):
    if outcome not in CLASSES:
        raise ValueError("outcome must be U0/U1/U2")
    result=proper_score_skill(model_probs,NULL,outcome)
    result["primary_endpoint"]="BRIER_SKILL_VS_EXACT_UNIFORM_THREE_STATE_NULL"
    result["secondary_log_skill"]=result["log_skill"]
    return result
