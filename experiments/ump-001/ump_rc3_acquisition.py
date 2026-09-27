#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib

ROOT=pathlib.Path(__file__).resolve().parent
REQUIRED=[
"OBSERVATION_DEVICE_MANUFACTURER","OBSERVATION_DEVICE_MODEL","OBSERVATION_PRECISION","CALIBRATION_METHOD",
"SAMPLING_RATE_HZ","MEASURED_R_OHM","MEASURED_C_FARAD","GROUND_TRUTH_AUTHORITY","ACQUISITION_METHOD","ACQUISITION_PATH",
"VOLTAGE_U0","VOLTAGE_U1","VOLTAGE_U2","SAFE_VOLTAGE_RANGE","SHAM_INTERVENTION","RESET_PROCEDURE",
"INTERVENTION_LATENCY","REVERSIBILITY"
]
BLOCKERS={"TBD","NOT_MEASURED","NOT_COMPUTABLE",None,""}

def load_hardware_bind():
    return json.loads((ROOT/"UMP_001_RC3_HARDWARE_BIND_V1.json").read_text(encoding="utf-8"))

def missing_physical_fields(binding=None):
    b=binding or load_hardware_bind()
    p=b["physical_bind"]
    return [k for k in REQUIRED if p.get(k) in BLOCKERS]

def preflight(binding=None):
    missing=missing_physical_fields(binding)
    return {"status":"READY_FOR_ENGINEERING_CALIBRATION" if not missing else "BLOCKED_REQUIRED_MEASUREMENT",
            "missing_physical_fields":missing,"GTAB_REAL":"BOUND" if not missing else "NOT_BOUND"}

if __name__=="__main__":
    print(json.dumps(preflight(),sort_keys=True,indent=2))
