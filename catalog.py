"""
catalog.py — lightweight statistics catalog for the Quail AQE engine.

The feature-extraction layer (``features.py``) needs cardinality and
distribution statistics to estimate selectivity, variance and group counts.
In a production DBMS these would come from a persisted statistics catalog
(histograms, sketches, ndv estimates). Here we compute them on demand from
DuckDB, which is fast enough for an interactive engine and keeps the project
dependency-free.

``CatalogClient`` is the object that ``features.extract`` and
``classify.make_sampling_plan`` expect. It is duck-typed: any object exposing
the same method surface as ``TableStats`` will work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import duckdb


@dataclass(frozen=True)
class OutlierIndex:
    """Offline-computed split point for measure-biased sampling.

    ``cutoff`` is the |value| above which rows are treated as heavy hitters and
    kept exact; ``total_count`` is the table cardinality (used for AVG scaling).
    The cutoff is a property of the column distribution, independent of any
    query predicate, so it is safe to cache.
    """

    column: str
    cutoff: float
    total_count: int


class TableStats:
    """Per-table statistics, computed lazily against a live DuckDB connection."""

    def __init__(self, con: duckdb.DuckDBPyConnection, table_ref: str, row_count: int):
        # ``table_ref`` is already quoted/escaped by CatalogClient when needed
        # (e.g. a parquet path becomes 'path/to/file.parquet').
        self.con = con
        self.table_name = table_ref
        self.row_count = row_count

    def histogram_frequency(self, col: str, val: Any) -> float:
        """Fraction of rows where ``col = val`` (equality selectivity)."""
        try:
            res = self.con.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE {col} = ?",
                (val,),
            ).fetchone()
            return (res[0] if res else 0) / max(self.row_count, 1)
        except Exception:
            # Unknown column / type mismatch: fall back to a neutral estimate.
            return 0.1

    def histogram_range_fraction(self, col: str, val: Any) -> float:
        """Approximate selectivity of a range predicate.

        This is a deliberate heuristic (0.3). A production engine would read an
        equi-depth histogram to compute the fraction of the domain covered by
        the range. The AQE pipeline only uses this to *pick a strategy*, not to
        scale results, so a coarse estimate is acceptable here.
        """
        return 0.3

    def column_variance(self, cols: List[str]) -> float:
        """Sample variance of the first column (used as a spread signal)."""
        if not cols:
            return 0.0
        try:
            res = self.con.execute(
                f"SELECT VAR_SAMP({cols[0]}) FROM {self.table_name}"
            ).fetchone()
            return float(res[0]) if res and res[0] is not None else 0.0
        except Exception:
            return 1.0

    def ndistinct(self, cols: List[str]) -> int:
        """Number of distinct value-combinations across ``cols`` (group count)."""
        if not cols:
            return 1
        try:
            res = self.con.execute(
                f"SELECT COUNT(DISTINCT {', '.join(cols)}) FROM {self.table_name}"
            ).fetchone()
            return res[0] if res else 1
        except Exception:
            return 100


class CatalogClient:
    """Resolves a table name to a :class:`TableStats` object."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self._outlier_cache: Dict[Tuple[str, str, float], OutlierIndex] = {}

    def outlier_index(
        self, table_ref: str, column: str, quantile: float = 0.995
    ) -> OutlierIndex:
        """Compute (and cache) the heavy-hitter cutoff for ``column``.

        One scan fixes the |value| cutoff at the given quantile and records the
        table cardinality. In a production AQE this is built at load time; here
        it is computed lazily on first use and cached for later queries.
        """
        key = (table_ref, column, quantile)
        if key not in self._outlier_cache:
            cutoff = self.con.execute(
                f"SELECT quantile_cont(abs({column}), {quantile}) FROM {table_ref}"
            ).fetchone()[0]
            total = self.con.execute(
                f"SELECT COUNT(*) FROM {table_ref}"
            ).fetchone()[0]
            self._outlier_cache[key] = OutlierIndex(column, float(cutoff), int(total))
        return self._outlier_cache[key]

    def get_stats(self, table_name: str) -> TableStats:
        # A raw parquet/csv path must be quoted so DuckDB reads it as a file.
        needs_quoting = ".parquet" in table_name.lower() or "/" in table_name
        table_ref = f"'{table_name}'" if needs_quoting else table_name
        try:
            res = self.con.execute(f"SELECT COUNT(*) FROM {table_ref}").fetchone()
            return TableStats(self.con, table_ref, res[0] if res else 1_000_000)
        except Exception:
            # Unknown table: assume a large table so we still sample rather than
            # accidentally running exact on something huge.
            return TableStats(self.con, table_ref, 1_000_000)
