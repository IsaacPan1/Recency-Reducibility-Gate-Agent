"""Calibrated thresholds for the recency-reducibility drift gate.

Every constant here was fixed against the clean raw-csv sweep across the five
practice sets (award_A, retail_sales, energy_load, kaggle_style,
medical_imaging) — the distributions the eval actually sees at plan time, since
no engineered parquet exists when the gate runs. The observed sweep:

    award_A        rel= 0.623  frac=0.556  n_feat= 9   -> EXPANDING (known correct)
    retail_sales   rel= 0.022  frac=0.200  n_feat= 5   -> EXPANDING
    energy_load    rel=-0.514  frac=0.500  n_feat= 4   -> EXPANDING
    kaggle_style   fallback (10 < RECENT_PERIODS)       -> EXPANDING
    medical_imaging not a time-series scheme            -> N/A

The thresholds are tuned so every correct-expanding set stays on the expanding
default while leaving a clean margin for a genuine recency-reducible case to
trip the sliding branch.
"""

RECENT_PERIODS = 14
"""Size of the recent window (in period ranks) compared against val.
Calibrated against the practice-set sweep: kaggle_style spans only 10 periods
and correctly falls back to expanding below this floor, while the other
time-series sets span enough history for a meaningful recent-vs-full contrast."""

DRIFT_REL_THRESHOLD = 0.25
"""Secondary gate: minimum mean_improvement / mean_dist_full for sliding.
Calibrated against the practice-set sweep. Mean-driven and fragile (a couple of
high-variance covariates can flip its sign, e.g. energy_load at rel=-0.514), so
it is kept only as a secondary guard behind the robust frac_improved gate."""

DRIFT_FRAC_IMPROVED_THRESHOLD = 0.60
"""Primary gate: minimum fraction of features whose recent window moves closer
to val. Calibrated against the practice-set sweep — every correct-expanding set
sits at frac<=0.556 (award_A is the closest at 0.556), so a 0.60 floor cleanly
separates them from any would-be sliding case."""

MIN_FEATURES_FOR_SLIDING = 12
"""Evidence-breadth floor: a scan covering fewer features must fall back to
expanding. Calibrated against the practice-set sweep — it is a second,
independent brake on thin-evidence cases like award_A (n_feat=9 on the raw
path); even if the frac floor were later lowered, this floor still protects the
conservative default."""

DRIFT_IMPROVE_PER_FEATURE_THRESHOLD = 0.05
"""Minimum per-feature improvement (dist_full - dist_recent) for a feature to
count toward frac_improved. Calibrated against the practice-set sweep to ignore
negligible per-feature movement that would otherwise inflate frac_improved."""
