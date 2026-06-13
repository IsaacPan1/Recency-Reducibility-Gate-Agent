"""Recency-reducibility drift gate: expanding-vs-sliding CV scheme decision.

The default time-series CV scheme is expanding (use ALL available history). This
gate switches to sliding ONLY when the train→val drift is genuinely
recency-REDUCIBLE — that is, when restricting train to its most recent block
measurably brings the covariate distribution closer to val. That is a different
and much stronger condition than "drift is large." The classic drift detectors —
an adversarial train-vs-val AUC, or a per-feature KS statistic — answer "are
train and val distinguishable?", and on a forecasting holdout that question is
already settled before any drift exists: val is the future window by
construction, so a time index, seasonal phase, or any slow trend makes train and
val trivially separable. Those detectors saturate near their maximum on every
forecasting set, whether or not recency would help, and a gate built on them
picks sliding almost always — discarding history for no measurable gain.

This gate instead asks the question sliding can actually act on: does the recent
slice of train look more like val than the full slice does? For each shared
numeric covariate it computes a standardized mean shift, |mu_train - mu_val| /
pooled_std, twice — full train vs val and the last RECENT_PERIODS ranks of train
vs val — and measures the per-feature improvement. It aggregates that into
frac_improved (the breadth of features that move closer) and rel (the relative
size of the average move). Because the metric is an effect size on the very
quantity sliding manipulates, it does NOT saturate on "val is the future": a set
whose drift is pure trend shows little recent-vs-full improvement and stays on
expanding, while only a set where recency genuinely closes the gap trips the
sliding branch. Engineered seasonality / time-index columns are excluded from
the scan precisely because they encode "val is the future" rather than reducible
drift, and the val TARGET is never read — only competition-provided val
covariates participate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .thresholds import (
    RECENT_PERIODS,
    DRIFT_REL_THRESHOLD,
    DRIFT_FRAC_IMPROVED_THRESHOLD,
    MIN_FEATURES_FOR_SLIDING,
    DRIFT_IMPROVE_PER_FEATURE_THRESHOLD,
)

__all__ = ["standardized_mean_shift", "drift_diagnostic", "decide_scheme"]

# Engineered seasonality / time-index columns are excluded from the drift scan:
# they are deterministic functions of period rank, so they separate train from
# val by construction and encode "val is the future" rather than reducible drift.
_WINDOW_FEATURES = {
    "period_id_ord", "period_id_trend", "period_id_of_cycle", "horizon",
    "period_id_sin", "period_id_cos", "period_id_sin2", "period_id_cos2",
    "period_id_quarter", "period_id_month", "month_of_year",
    "quarter_of_year", "is_quarter_start",
}
_NON_FEATURES = {"adversarial_weights"}


def standardized_mean_shift(a, b) -> float:
    """|mean(a) - mean(b)| / pooled_std. Effect size; 0 = identical; does NOT
    saturate the way an adversarial AUC does.

    Drops NaNs from each side. If either side has fewer than 5 valid values,
    returns nan. pooled_std = sqrt((var(a) + var(b)) / 2) using population
    variance (np.var, ddof=0); if pooled_std is 0 returns 0.0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    sp = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
    if sp == 0.0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(b)) / sp)


def drift_diagnostic(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    time_col: str,
    *,
    recent_periods: int = RECENT_PERIODS,
    exclude=None,
) -> dict:
    """Recent-vs-full drift diagnostic.

    Returns a verdict dict carrying the summary stats and an ``ok`` flag. When
    ``ok`` is False, ``reason`` explains why the diagnostic could not run — the
    caller falls back to expanding in that case (NEVER to sliding on missing
    evidence).

    The recent rank is derived directly from ``train_df[time_col]`` via
    ``pd.to_numeric(..., errors="coerce")``. If no numeric rank can be derived,
    if the rank span is shorter than ``recent_periods``, or if there are no
    shared numeric non-excluded features, the diagnostic returns ``ok=False``.

    ``_WINDOW_FEATURES`` and ``{"adversarial_weights"}`` are excluded from the
    scan by default; ``exclude`` adds further column names to that set. The val
    TARGET is never read — only shared numeric covariates participate.
    """
    excluded = set(_WINDOW_FEATURES) | set(_NON_FEATURES)
    if exclude:
        excluded |= set(exclude)

    if val_df is None or train_df is None:
        return {"ok": False, "reason": "no val frame available"}
    if not time_col or time_col not in train_df.columns:
        return {"ok": False, "reason": f"time_col {time_col!r} not present in train frame"}

    common = [
        c for c in train_df.columns
        if c in val_df.columns
        and pd.api.types.is_numeric_dtype(train_df[c])
        and c not in excluded
    ]
    if not common:
        return {"ok": False, "reason": "no shared numeric non-window features between train and val"}

    tr_rank = pd.to_numeric(train_df[time_col], errors="coerce")
    tr_rank_valid = tr_rank.dropna()
    if tr_rank_valid.empty:
        return {"ok": False, "reason": "could not derive numeric rank for train rows"}

    rank_max = int(tr_rank_valid.max())
    rank_span = rank_max + 1
    if rank_span < recent_periods:
        return {"ok": False,
                "reason": f"train spans {rank_span} periods < RECENT_PERIODS={recent_periods}"}
    cutoff = rank_max - recent_periods + 1
    recent_mask = (tr_rank >= cutoff).fillna(False).to_numpy()

    rows = []
    for c in common:
        a_full = train_df[c].to_numpy(dtype=float, copy=False)
        a_recent = train_df.loc[recent_mask, c].to_numpy(dtype=float, copy=False)
        b_val = val_df[c].to_numpy(dtype=float, copy=False)
        d_full = standardized_mean_shift(a_full, b_val)
        d_recent = standardized_mean_shift(a_recent, b_val)
        if np.isnan(d_full) or np.isnan(d_recent):
            continue
        rows.append((c, d_full, d_recent, d_full - d_recent))

    if not rows:
        return {"ok": False, "reason": "no feature yielded a valid distance (too many NaNs?)"}

    mean_full = float(np.mean([r[1] for r in rows]))
    mean_recent = float(np.mean([r[2] for r in rows]))
    mean_impr = float(np.mean([r[3] for r in rows]))
    frac_improved = float(np.mean([r[3] > DRIFT_IMPROVE_PER_FEATURE_THRESHOLD for r in rows]))
    rel = float(mean_impr / mean_full) if mean_full > 0 else 0.0

    return {
        "ok":                  True,
        "n_features_scanned":  len(rows),
        "rank_max":            rank_max,
        "recent_periods":      recent_periods,
        "recent_cutoff_rank":  cutoff,
        "mean_dist_full":      mean_full,
        "mean_dist_recent":    mean_recent,
        "mean_improvement":    mean_impr,
        "rel":                 rel,
        "frac_improved":       frac_improved,
        "per_feature": [
            {"feature": c, "dist_full": float(d_full),
             "dist_recent": float(d_recent), "improvement": float(d_full - d_recent)}
            for (c, d_full, d_recent, _) in rows
        ],
    }


def decide_scheme(diagnostic: dict) -> dict:
    """Map a drift diagnostic onto the CV scheme.

    Expanding is the structural default. Sliding is chosen ONLY when ALL three
    affirmative conditions hold:
        frac_improved >= DRIFT_FRAC_IMPROVED_THRESHOLD  (primary gate)
        rel           >= DRIFT_REL_THRESHOLD            (secondary gate)
        n_features    >= MIN_FEATURES_FOR_SLIDING       (evidence-breadth floor)
    Any failure — including a non-runnable diagnostic — falls back to expanding.

    Returns ``{"scheme", "reason", "gates": {frac_ok, rel_ok, nfeat_ok}}``.
    """
    if not diagnostic.get("ok"):
        return {
            "scheme": "expanding",
            "reason": (
                f"expanding (fallback - drift diagnostic could not run: "
                f"{diagnostic.get('reason', 'unknown')})"
            ),
            "gates": {"frac_ok": False, "rel_ok": False, "nfeat_ok": False},
        }

    rel = diagnostic["rel"]
    frac = diagnostic["frac_improved"]
    n_feat = int(diagnostic.get("n_features_scanned") or 0)
    frac_ok = frac >= DRIFT_FRAC_IMPROVED_THRESHOLD
    rel_ok = rel >= DRIFT_REL_THRESHOLD
    nfeat_ok = n_feat >= MIN_FEATURES_FOR_SLIDING
    gates = {"frac_ok": frac_ok, "rel_ok": rel_ok, "nfeat_ok": nfeat_ok}

    if frac_ok and rel_ok and nfeat_ok:
        reason = (
            f"sliding (drift is recency-reducible AND evidence is broad enough: "
            f"frac_improved={frac:.3f} >= {DRIFT_FRAC_IMPROVED_THRESHOLD} "
            f"AND rel={rel:.3f} >= {DRIFT_REL_THRESHOLD} "
            f"AND n_features={n_feat} >= {MIN_FEATURES_FOR_SLIDING})"
        )
        return {"scheme": "sliding", "reason": reason, "gates": gates}

    fails = []
    if not nfeat_ok:
        fails.append(
            f"insufficient features to assess drift breadth "
            f"(n_features={n_feat} < {MIN_FEATURES_FOR_SLIDING})"
        )
    if not frac_ok:
        fails.append(
            f"frac_improved={frac:.3f} < {DRIFT_FRAC_IMPROVED_THRESHOLD} (primary gate)"
        )
    if not rel_ok:
        fails.append(
            f"rel={rel:.3f} < {DRIFT_REL_THRESHOLD} (secondary gate)"
        )
    reason = (
        "expanding (conservative default; sliding not affirmatively justified - "
        + "; ".join(fails)
        + ")"
    )
    return {"scheme": "expanding", "reason": reason, "gates": gates}
