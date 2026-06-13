"""recency_cv: the recency-reducibility CV gate.

Detects whether train->val drift is recency-REDUCIBLE and decides between an
expanding and a sliding time-series CV scheme.
"""

from .gate import standardized_mean_shift, drift_diagnostic, decide_scheme

__version__ = "0.0.1"

__all__ = ["standardized_mean_shift", "drift_diagnostic", "decide_scheme"]
