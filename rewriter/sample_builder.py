# rewriter/sample_builder.py
from __future__ import annotations
from dataclasses import dataclass


from parser import QueryIR
from classify import SamplingPlan, Strategy


# ─────────────────────────────────────────────
# REAL CODE STARTS HERE
# ─────────────────────────────────────────────

MIN_SAMPLE_N = 30  # below this, results are statistically unreliable


class SampleSizeError(Exception):
    """Raised when estimated sample n is below the statistical minimum."""
    pass


@dataclass
class SampleFragment:
    """
    The output of SampleBuilder.
    Carries everything the rewriter needs to inject the sample into the query.
    
    cte_sql     : the WITH clause body. None for TABLESAMPLE (uniform).
    table_ref   : what goes in the FROM clause after injection.
    sample_rate : actual rate used (may differ from plan if adjusted).
    strategy    : which method was used.
    """
    cte_sql: str | None
    table_ref: str
    sample_rate: float
    strategy: str


class SampleSizeGate:
    """
    Checks whether the planned sample is large enough to be statistically valid.
    
    Why 30? Central Limit Theorem - below 30 samples, estimates become 
    unreliable regardless of which sampling method you use.
    """

    def __init__(self, min_n: int = MIN_SAMPLE_N):
        self.min_n = min_n

    def passes(self, plan: SamplingPlan, row_count: int) -> bool:
        estimated_n = int(plan.sample_rate * row_count)
        return estimated_n >= self.min_n

    def adjusted_rate(self, row_count: int) -> float:
        """Returns the minimum rate that would give us exactly min_n rows."""
        return self.min_n / max(row_count, 1)


class SampleBuilder:
    """
    Builds the SQL fragment that represents the sample.

    3 strategies:
      uniform    — DuckDB USING SAMPLE, db-native, fast, unbiased
      stratified — CTE with ROW_NUMBER() OVER (PARTITION BY group_col), ordered by hash(key)
      hash       — deterministic hash(key) predicate, reproducible
    """

    def __init__(self, gate: SampleSizeGate | None = None):
        self.gate = gate or SampleSizeGate()

    def build(self, ir: QueryIR, plan: SamplingPlan) -> SampleFragment:
        row_count = plan.table_row_count or 1_000_000  # fallback estimate

        # Gate check — is n large enough?
        if not self.gate.passes(plan, row_count):
            raise SampleSizeError(
                f"Estimated n={int(plan.sample_rate * row_count)} "
                f"is below minimum {self.gate.min_n}. "
                f"Fall back to exact execution."
            )

        table = ir.tables[0]
        hash_key = plan.hash_key or "id"

        if plan.strategy == "SAMPLING":
            return self._uniform(table, plan.sample_rate)

        elif plan.strategy == "FILTERED_SAMPLING":
            strat_key = plan.stratify_on[0] if plan.stratify_on else (ir.groupby_cols[0] if ir.groupby_cols else None)
            if strat_key:
                return self._stratified(table, strat_key, plan.sample_rate, hash_key=hash_key)
            return self._uniform(table, plan.sample_rate)

        elif plan.strategy == "HYBRID":
            strat_key = plan.stratify_on[0] if plan.stratify_on else (ir.groupby_cols[0] if ir.groupby_cols else None)
            if strat_key:
                return self._stratified(table, strat_key, plan.sample_rate, hash_key=hash_key)
            return self._hash(table, plan.sample_rate, hash_key=hash_key)

        elif plan.strategy == "EXACT":
            raise SampleSizeError("EXACT strategy requested; no sampling fragment should be built.")

        else:
            raise ValueError(f"Unknown strategy: {plan.strategy!r}")

    # ── 3 private methods, one per strategy ──────────────────────────

    def _uniform(self, table: str, rate: float) -> SampleFragment:
        """
        Uses DuckDB's native Bernoulli sampling, wrapped in a subquery.

        Each row is included independently with probability ``rate`` — exactly
        the model the confidence-interval estimator assumes. Wrapping the sample
        in a subquery (rather than ``table USING SAMPLE ...``) means the sample
        is applied first and any trailing WHERE / GROUP BY clause attaches to
        the subquery — DuckDB rejects a WHERE placed directly after a
        table-level ``USING SAMPLE``.

        Example output table_ref:
            (SELECT * FROM orders USING SAMPLE 10% (bernoulli)) AS _aqe_uniform
        """
        pct = round(rate * 100, 4)
        table_ref = (
            f"(SELECT * FROM {table} USING SAMPLE {pct}% (bernoulli)) AS _aqe_uniform"
        )
        return SampleFragment(
            cte_sql=None,
            table_ref=table_ref,
            sample_rate=rate,
            strategy="uniform",
        )

    def _stratified(self, table: str, strat_col: str, rate: float, *, hash_key: str) -> SampleFragment:
        """
        CTE-based stratified sample.
        Guarantees every group gets proportional rows — none disappear.
        
        CEIL ensures at least 1 row per group even at very low rates.
        
        Example: 3 regions, 10% rate → each region loses exactly 90% of rows.
        """
        # Optimization: avoid ORDER BY random() (expensive). Ordering by a stable hash
        # gives a pseudo-random, reproducible permutation within each group.
        cte = (
            f"_aqe_strat AS (\n"
            f"    SELECT *,\n"
            f"        ROW_NUMBER() OVER (\n"
            f"            PARTITION BY {strat_col} ORDER BY abs(hash({hash_key}))\n"
            f"        ) AS _aqe_rn,\n"
            f"        COUNT(*) OVER (\n"
            f"            PARTITION BY {strat_col}\n"
            f"        ) AS _aqe_gsz\n"
            f"    FROM {table}\n"
            f")"
        )
        return SampleFragment(
            cte_sql=cte,
            table_ref=f"_aqe_strat WHERE _aqe_rn <= CEIL(_aqe_gsz * {rate})",
            sample_rate=rate,
            strategy="stratified",
        )

    def _hash(self, table: str, rate: float, *, hash_key: str) -> SampleFragment:
        """
        Deterministic hash-based sample.
        No random() = same query always returns the same rows.

        DuckDB-friendly approach: take abs(hash(key)) modulo a big number M,
        then keep rows below floor(rate * M).
        """
        modulus = 1_000_000
        bucket_threshold = int(rate * modulus)
        cte = (
            f"_aqe_hash AS (\n"
            f"    SELECT * FROM {table}\n"
            f"    WHERE (abs(hash({hash_key})) % {modulus}) < {bucket_threshold}\n"
            f")"
        )
        return SampleFragment(
            cte_sql=cte,
            table_ref="_aqe_hash",
            sample_rate=rate,
            strategy="hash",
        )