"""
outlier_sampling_demo.py — tighter, better-calibrated confidence intervals by
choosing rows more carefully (outlier-separated / "measure-biased" sampling).

Motivation
----------
SUM/AVG over a heavy-tailed column (here `Quantity`) have unstable confidence
intervals under plain uniform sampling: a few extreme rows carry most of the
sum AND most of the variance, so the interval width lurches depending on
whether those rows happen to land in the sample.

Fix
---
Split the table once (offline) into:
  * OUTLIERS — the heaviest rows by |value| (top 0.5% here). Keep them EXACT.
  * BODY     — everything else. Sample the body uniformly at rate r.

Estimator (SUM):
    SUM_hat = sum(outliers)            # exact, zero variance
            + (1/r) * sum(sampled body)

    Var(SUM_hat) ~ (1 - r)/r^2 * sum_over_sampled_body( y^2 )

Because the body excludes the extreme values, its sum-of-squares is far
smaller, so the standard error collapses and the estimator's sampling
distribution is much closer to normal -> tighter interval, coverage near 95%.

Cost (be honest): identifying the outliers needs one offline pass over the
column to fix the cutoff and precompute the exact outlier sum (an "outlier
index"). In a real AQE this is amortized across many queries; at query time you
only scan the sampled body. On a small in-memory table like this it is not a
speed win — it is an ACCURACY / CALIBRATION win.

Run:
    python outlier_sampling_demo.py
"""

from __future__ import annotations

import math
import os

import duckdb

PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB", "data.parquet")

Z_95 = 1.959963984540054
COLUMN = "Quantity"
OUTLIER_QUANTILE = 0.995   # keep the top 0.5% by |value| exact
TARGET_SAMPLE_ROWS = 50_000
N_TRIALS = 400


def uniform_sum_ci(con, rate: float) -> tuple[float, float, float]:
    """Plain uniform-sample SUM estimate with a Horvitz-Thompson CI."""
    row = con.execute(
        f"SELECT SUM({COLUMN}) s, SUM({COLUMN}*{COLUMN}) sq "
        f"FROM data USING SAMPLE {rate*100:.4f}% (bernoulli)"
    ).fetchone()
    s, sq = float(row[0] or 0), float(row[1] or 0)
    est = s / rate
    se = math.sqrt(max((1 - rate) / (rate * rate) * sq, 0.0))
    return est, est - Z_95 * se, est + Z_95 * se


def outlier_sum_ci(con, rate: float, cutoff: float, outlier_sum: float) -> tuple[float, float, float]:
    """Outlier-separated SUM: outliers exact, body sampled. Variance from body only."""
    row = con.execute(
        f"SELECT SUM({COLUMN}) s, SUM({COLUMN}*{COLUMN}) sq "
        f"FROM (SELECT * FROM data WHERE abs({COLUMN}) < {cutoff}) "
        f"USING SAMPLE {rate*100:.4f}% (bernoulli)"
    ).fetchone()
    s, sq = float(row[0] or 0), float(row[1] or 0)
    est = outlier_sum + s / rate
    se = math.sqrt(max((1 - rate) / (rate * rate) * sq, 0.0))
    return est, est - Z_95 * se, est + Z_95 * se


def evaluate(name: str, sampler, truth: float) -> None:
    covered = rel_err = half_width = 0.0
    for _ in range(N_TRIALS):
        est, lo, hi = sampler()
        if lo <= truth <= hi:
            covered += 1
        rel_err += abs(est - truth) / abs(truth)
        half_width += (hi - lo) / 2 / max(abs(est), 1e-12)
    print(
        f"  {name:22s} coverage={covered/N_TRIALS:5.1%}  "
        f"mean_rel_err={rel_err/N_TRIALS:6.2%}  "
        f"mean_CI_halfwidth={half_width/N_TRIALS:6.2%}"
    )


def main() -> None:
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW data AS SELECT * FROM '{PARQUET_PATH}'")

    n = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    truth = float(con.execute(f"SELECT SUM({COLUMN}) FROM data").fetchone()[0])
    rate = min(1.0, TARGET_SAMPLE_ROWS / n)

    # Offline outlier index.
    cutoff = float(con.execute(
        f"SELECT quantile_cont(abs({COLUMN}), {OUTLIER_QUANTILE}) FROM data"
    ).fetchone()[0])
    outlier_sum, n_out = con.execute(
        f"SELECT SUM({COLUMN}), COUNT(*) FROM data WHERE abs({COLUMN}) >= {cutoff}"
    ).fetchone()
    outlier_sum = float(outlier_sum)

    print(f"SUM({COLUMN}) over {n:,} rows | true value = {truth:,.0f} | sample rate = {rate:.2%}")
    print(f"outlier index: |{COLUMN}| >= {cutoff:.0f} kept exact ({n_out:,} rows, {n_out/n:.2%})\n")
    print(f"Averaged over {N_TRIALS} independent samples:")

    evaluate("uniform (current)", lambda: uniform_sum_ci(con, rate), truth)
    evaluate("outlier-separated", lambda: outlier_sum_ci(con, rate, cutoff, outlier_sum), truth)


if __name__ == "__main__":
    main()
