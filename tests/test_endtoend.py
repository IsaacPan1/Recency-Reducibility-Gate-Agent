"""End-to-end tests: build small panels and assert the full verdict."""

import numpy as np
import pandas as pd

from recency_cv import drift_diagnostic, decide_scheme

N_PERIODS = 30
ROWS_PER_PERIOD = 25
N_FEATURES = 15
VAL_PERIODS = 5


def _stationary(seed):
    """No drift: train and val drawn from the same per-feature distribution."""
    rng = np.random.default_rng(seed)
    bases = rng.uniform(-5.0, 5.0, N_FEATURES)

    tr = []
    for t in range(N_PERIODS):
        for _ in range(ROWS_PER_PERIOD):
            row = {"t": t}
            for i in range(N_FEATURES):
                row[f"f{i}"] = bases[i] + rng.normal(0.0, 1.0)
            tr.append(row)
    train = pd.DataFrame(tr)

    vl = []
    for _ in range(VAL_PERIODS * ROWS_PER_PERIOD):
        row = {}
        for i in range(N_FEATURES):
            row[f"f{i}"] = bases[i] + rng.normal(0.0, 1.0)
        vl.append(row)
    val = pd.DataFrame(vl)
    return train, val


def _drifting(seed):
    """Drift concentrated in the recent block: the distribution shifts to a new
    level over the last `recent_periods` and val continues at that new level.
    The recent train block therefore lands on val while the older block does
    not — the standardized-shift metric flags this as recency-reducible (a
    smooth linear trend would not: its in-window variance normalizes the shift
    away, leaving recent and full equidistant from val)."""
    rng = np.random.default_rng(seed)
    bases = rng.uniform(-5.0, 5.0, N_FEATURES)
    shifts = rng.uniform(3.0, 6.0, N_FEATURES)  # new-regime offset, per feature
    cutoff = N_PERIODS - 14  # matches RECENT_PERIODS: recent window is t >= 16

    tr = []
    for t in range(N_PERIODS):
        for _ in range(ROWS_PER_PERIOD):
            row = {"t": t}
            for i in range(N_FEATURES):
                level = bases[i] + (shifts[i] if t >= cutoff else 0.0)
                row[f"f{i}"] = level + rng.normal(0.0, 1.0)
            tr.append(row)
    train = pd.DataFrame(tr)

    vl = []
    for _ in range(VAL_PERIODS * ROWS_PER_PERIOD):  # val sits in the new regime
        row = {}
        for i in range(N_FEATURES):
            row[f"f{i}"] = bases[i] + shifts[i] + rng.normal(0.0, 1.0)
        vl.append(row)
    val = pd.DataFrame(vl)
    return train, val


def test_stationary_expands():
    train, val = _stationary(seed=1234)
    diag = drift_diagnostic(train, val, "t")
    assert diag["ok"]
    assert diag["n_features_scanned"] == N_FEATURES
    assert decide_scheme(diag)["scheme"] == "expanding"


def test_drift_into_val_slides():
    train, val = _drifting(seed=1234)
    diag = drift_diagnostic(train, val, "t")
    assert diag["ok"]
    assert diag["n_features_scanned"] == N_FEATURES
    # recent train must be closer to val than full train
    assert diag["mean_dist_recent"] < diag["mean_dist_full"]
    assert decide_scheme(diag)["scheme"] == "sliding"
