#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from ump_rc3_core import canonical_json_bytes, commit_prediction, verify_commitment

def payload_sha256(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def chain_entry(entry, prev_entry_sha256):
    row=dict(entry)
    row["prev_entry_sha256"]=prev_entry_sha256
    row_bytes=canonical_json_bytes(row)
    row["entry_sha256"]=hashlib.sha256(row_bytes).hexdigest()
    return row

__all__=["commit_prediction","verify_commitment","payload_sha256","chain_entry"]
