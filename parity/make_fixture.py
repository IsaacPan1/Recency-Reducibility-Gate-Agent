"""Generate the cross-language parity fixture for recency_cv.

Defines a small set of CANONICAL inputs covering the ENTIRE public surface
(standardized_mean_shift, drift_diagnostic, decide_scheme — every branch),
writes them verbatim to parity/fixture_inputs.json, runs the reference Python
implementation on them, and writes full-precision results to parity/expected.json.

fixture_inputs.json + expected.json are the SOURCE OF TRUTH: a JS port must read
the inputs, run its own implementation, and reproduce expected.json exactly. Run
this script only to regenerate the fixture when the reference behavior changes.

Non-finite floats (NaN) are serialized as the string "NaN" so the JSON stays
strictly valid and parseable by a JS port (which must map it back to NaN).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recency_cv import standardized_mean_shift, drift_diagnostic, decide_scheme
from recency_cv.thresholds import RECENT_PERIODS

HERE = Path(__file__).resolve().parent


def _sanitize(obj):
    """Recursively replace non-finite floats with sentinel strings so the
    output is strictly-valid JSON (json with allow_nan=False would reject NaN)."""
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 1. standardized_mean_shift — cover every branch
# ─────────────────────────────────────────────────────────────────────────────
SMM_CASES = [
    {"name": "identical", "a": [1.0, 2.0, 3.0, 4.0, 5.0],
     "b": [1.0, 2.0, 3.0, 4.0, 5.0]},                        # -> 0.0
    {"name": "too_few_left", "a": [1.0, 2.0, 3.0, 4.0],
     "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},                   # <5 valid -> NaN
    {"name": "hand_case", "a": [1.0, 2.0, 3.0, 4.0, 5.0],
     "b": [3.0, 4.0, 5.0, 6.0, 7.0]},                        # -> sqrt(2)
    {"name": "normal", "a": [10.0, 12.0, 11.0, 13.0, 9.0, 10.5],
     "b": [14.0, 15.0, 13.5, 16.0, 14.5]},                   # -> normal value
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. drift_diagnostic — ONE deterministic panel, non-trivial verdict
#    6 numeric features, 25 rows (one per period 0..24). Features f0..f2 shift to
#    a new regime over the recent window and val sits in that regime (recency-
#    REDUCIBLE); f3..f5 are stationary (no improvement). The mix exercises both
#    per-feature branches (improvement above and below the threshold).
# ─────────────────────────────────────────────────────────────────────────────
def _build_panel():
    rng = np.random.default_rng(42)
    n_periods = 25
    n_features = 6
    cutoff = n_periods - RECENT_PERIODS  # = 11; recent window is t >= 11
    bases = rng.uniform(-2.0, 2.0, n_features)
    shifts = np.array([4.0, 4.0, 4.0, 0.0, 0.0, 0.0])  # first 3 drift, last 3 stay

    train_records = []
    for t in range(n_periods):
        row = {"t": int(t)}
        for i in range(n_features):
            level = bases[i] + (shifts[i] if t >= cutoff else 0.0)
            row[f"f{i}"] = float(level + rng.normal(0.0, 1.0))
        train_records.append(row)

    val_records = []
    for _ in range(8):  # val rows all in the new regime
        row = {}
        for i in range(n_features):
            row[f"f{i}"] = float(bases[i] + shifts[i] + rng.normal(0.0, 1.0))
        val_records.append(row)

    return {"time_col": "t", "train": train_records, "val": val_records}


# ─────────────────────────────────────────────────────────────────────────────
# 3. decide_scheme — real calibration dicts + synthetic sliding + fallback
# ─────────────────────────────────────────────────────────────────────────────
DECIDE_CASES = [
    {"name": "award_A",
     "diagnostic": {"ok": True, "rel": 0.623, "frac_improved": 0.556, "n_features_scanned": 9}},
    {"name": "retail_sales",
     "diagnostic": {"ok": True, "rel": 0.022, "frac_improved": 0.200, "n_features_scanned": 5}},
    {"name": "energy_load",
     "diagnostic": {"ok": True, "rel": -0.514, "frac_improved": 0.500, "n_features_scanned": 4}},
    {"name": "synthetic_sliding",
     "diagnostic": {"ok": True, "rel": 0.40, "frac_improved": 0.75, "n_features_scanned": 20}},
    {"name": "fallback_not_ok",
     "diagnostic": {"ok": False, "reason": "no numeric rank could be derived"}},
]


def main():
    panel = _build_panel()

    inputs = {
        "standardized_mean_shift": SMM_CASES,
        "drift_diagnostic": panel,
        "decide_scheme": DECIDE_CASES,
    }

    # ---- run the reference implementation ----
    smm_out = [
        {"name": c["name"], "result": standardized_mean_shift(c["a"], c["b"])}
        for c in SMM_CASES
    ]

    train_df = pd.DataFrame(panel["train"])
    val_df = pd.DataFrame(panel["val"])
    diag_out = drift_diagnostic(train_df, val_df, panel["time_col"])

    decide_out = [
        {"name": c["name"], "result": decide_scheme(c["diagnostic"])}
        for c in DECIDE_CASES
    ]

    expected = {
        "standardized_mean_shift": smm_out,
        "drift_diagnostic": diag_out,
        "decide_scheme": decide_out,
    }

    inputs_path = HERE / "fixture_inputs.json"
    expected_path = HERE / "expected.json"
    with open(inputs_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(inputs), f, indent=2, allow_nan=False)
        f.write("\n")
    with open(expected_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(expected), f, indent=2, allow_nan=False)
        f.write("\n")

    print(f"wrote {inputs_path.relative_to(HERE.parent)}")
    print(f"wrote {expected_path.relative_to(HERE.parent)}")
    print("  smm:", [(o["name"], o["result"]) for o in smm_out])
    print(f"  drift: ok={diag_out.get('ok')} n={diag_out.get('n_features_scanned')} "
          f"frac={diag_out.get('frac_improved')} rel={diag_out.get('rel')}")
    print("  decide:", [(o["name"], o["result"]["scheme"]) for o in decide_out])


if __name__ == "__main__":
    main()
