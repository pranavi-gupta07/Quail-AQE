"""
demo.py — run the Quail AQE engine end-to-end without the API server.

Loads the bundled UCI Online Retail parquet, then for a set of representative
queries runs BOTH exact and approximate execution and prints, side by side:

    strategy chosen, exact value, approximate value, relative error,
    the 95% confidence interval, and wall-clock time for each path.

Usage:
    pip install -r requirements.txt
    python demo.py
"""

from __future__ import annotations

import os
import time

import duckdb

from parser import parse
from features import extract
from classify import make_sampling_plan
from query_engine import QueryEngine
from catalog import CatalogClient

PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB", "data.parquet")
HASH_KEY = "InvoiceNo"

DEMO_QUERIES = [
    "SELECT COUNT(*) FROM data",
    "SELECT SUM(Quantity) FROM data",
    "SELECT AVG(UnitPrice) FROM data",
    "SELECT COUNT(*), SUM(Quantity), AVG(UnitPrice) FROM data",
    "SELECT COUNT(*) FROM data WHERE Quantity > 5",
    "SELECT Country, SUM(Quantity) FROM data GROUP BY Country",
]


def _time(fn, repeats: int = 3) -> tuple[object, float]:
    """Return (result_of_last_call, best_wall_time_seconds)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def main() -> None:
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW data AS SELECT * FROM '{PARQUET_PATH}'")
    row_count = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    print(f"Loaded 'data' with {row_count:,} rows from {os.path.basename(PARQUET_PATH)}\n")

    catalog = CatalogClient(con)
    engine = QueryEngine(con)

    for sql in DEMO_QUERIES:
        print("=" * 78)
        print(f"QUERY: {sql}")

        ir = parse(sql)
        plan = make_sampling_plan(*extract(ir, catalog), ir, hash_key=HASH_KEY)

        exact, t_exact = _time(lambda: engine.execute_exact(sql))
        approx, t_approx = _time(
            lambda: engine.execute_approx(sql, plan=plan, confidence_level=0.95)
        )

        print(f"  strategy      : {plan.strategy}  ({plan.reason})")
        print(f"  sample rate   : {plan.sample_rate:.2%}")
        print(f"  exact  ({t_exact*1000:6.1f} ms): {exact.rows[:4]}")
        print(f"  approx ({t_approx*1000:6.1f} ms): {approx.rows[:4]}")

        # Relative error on the first numeric metric, when comparable.
        if (
            exact.rows
            and approx.rows
            and not ir.groupby_cols
            and isinstance(exact.rows[0][0], (int, float))
        ):
            for i in range(len(exact.rows[0])):
                ev = float(exact.rows[0][i])
                av = float(approx.rows[0][i])
                rel = abs(av - ev) / max(abs(ev), 1e-12)
                print(f"    metric[{i}] exact={ev:,.3f}  approx={av:,.3f}  rel_err={rel:.2%}")

        if approx.confidence:
            # Metrics are in SELECT order, so metric i lines up with column i.
            for i, m in enumerate(approx.confidence.metrics):
                true_val = float(exact.rows[0][i])
                covers = m.ci_low <= true_val <= m.ci_high
                print(
                    f"    CI {m.name}: [{m.ci_low:,.2f}, {m.ci_high:,.2f}] "
                    f"(±{m.rel_half_width:.2%})  covers_true={covers}"
                )
        if approx.warnings:
            print(f"    warnings: {approx.warnings}")
    print("=" * 78)


if __name__ == "__main__":
    main()
