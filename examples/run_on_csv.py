#!/usr/bin/env python3
"""Run the recency-reducibility gate on two CSVs — no code to write.

Point it at a train CSV and a validation CSV, name the period/time column, and
it prints the CV-scheme verdict plus the three gate values the decision rests on.

    python examples/run_on_csv.py train.csv val.csv --time-col week

    python examples/run_on_csv.py \
        tests/fixtures/covariates_train.csv \
        tests/fixtures/covariates_val.csv \
        --time-col week

The gate reads only shared numeric covariates and never the target. ``--target``
just names a column to keep out of the scan explicitly; the gate is safe to run
before fitting either way.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from recency_cv import drift_diagnostic, decide_scheme


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("train_csv", help="path to the training CSV (past periods)")
    p.add_argument("val_csv", help="path to the validation CSV (future holdout, covariates only)")
    p.add_argument("--time-col", required=True, help="name of the numeric period/time column")
    p.add_argument("--target", default=None,
                   help="optional target column name to exclude from the scan")
    args = p.parse_args(argv)

    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)

    exclude = [args.target] if args.target else None
    diag = drift_diagnostic(train, val, args.time_col, exclude=exclude)
    verdict = decide_scheme(diag)

    print(f"scheme : {verdict['scheme'].upper()}")
    print(f"reason : {verdict['reason']}")
    if diag.get("ok"):
        print(f"frac   : {diag['frac_improved']:.2f}  (breadth gate)")
        print(f"rel    : {diag['rel']:.2f}  (depth gate)")
        print(f"n      : {diag['n_features_scanned']}  (evidence floor)")
    else:
        print("frac   : n/a  (diagnostic did not run)")
        print("rel    : n/a")
        print("n      : n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
