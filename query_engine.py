from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import sqlglot.expressions as exp

from bypass import should_approximate
from parser import parse, QueryIR
from classify import SamplingPlan
from catalog import CatalogClient
from rewriter.sample_builder import SampleBuilder, SampleFragment, SampleSizeError
from rewriter.rewriter import StringRewriter, RewriteResult
from confidence import ConfidenceEstimator, ConfidenceReport, MetricCI


@dataclass(frozen=True)
class QueryResponse:
    original_sql: str
    executed_sql: str
    is_approx: bool
    strategy: str
    sample_rate: float
    rows: List[Tuple[Any, ...]]
    columns: List[str]
    warnings: List[str]
    confidence: ConfidenceReport | None


class QueryEngine:
    """
    Minimal AQE query runner for DuckDB.

    - Parses SQL to IR
    - Applies bypass rules
    - If approximate: uses provided SamplingPlan + SampleBuilder + StringRewriter
    - Executes final SQL in DuckDB
    - Computes confidence intervals using a companion stats query
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection | None = None,
        *,
        catalog: CatalogClient | None = None,
        builder: SampleBuilder | None = None,
        rewriter: StringRewriter | None = None,
        estimator: ConfidenceEstimator | None = None,
    ):
        self.con = con or duckdb.connect(database=":memory:")
        self.catalog = catalog or CatalogClient(self.con)
        self.builder = builder or SampleBuilder()
        self.rewriter = rewriter or StringRewriter()
        self.estimator = estimator or ConfidenceEstimator()

    def execute_exact(self, sql: str) -> QueryResponse:
        cur = self.con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return QueryResponse(
            original_sql=sql,
            executed_sql=sql,
            is_approx=False,
            strategy="EXACT",
            sample_rate=1.0,
            rows=rows,
            columns=cols,
            warnings=[],
            confidence=None,
        )

    def execute_approx(
        self,
        sql: str,
        *,
        plan: SamplingPlan,
        confidence_level: float = 0.95,
    ) -> QueryResponse:
        ir: QueryIR = parse(sql)
        ok, reason = should_approximate(ir)
        if not ok or plan.strategy == "EXACT":
            resp = self.execute_exact(sql)
            return replace(resp, warnings=[reason])

        # Measure-biased path: keep heavy hitters exact, sample the body.
        # If the query is outside its scope (e.g. the aggregate is an
        # expression rather than a simple column), fall back to uniform.
        if plan.strategy == "MEASURE_BIASED":
            resp = self._execute_measure_biased(sql, ir, plan, confidence_level)
            if resp is not None:
                return resp
            plan = replace(plan, strategy="SAMPLING", reason="measure_biased_fallback")

        try:
            fragment: SampleFragment = self.builder.build(ir, plan)
        except SampleSizeError as e:
            resp = self.execute_exact(sql)
            return replace(resp, warnings=[str(e)])

        rewrite: RewriteResult = self.rewriter.rewrite(sql, fragment)

        # Execute rewritten SQL
        cur = self.con.execute(rewrite.sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []

        # Confidence: compute stats query using same fragment (same sample).
        # Per-group confidence intervals for GROUP BY queries are not supported
        # (the companion stats query is not grouped), so we skip CI computation
        # rather than report a single, misleading interval for one group.
        conf: ConfidenceReport | None = None
        warnings: List[str] = list(rewrite.warnings)
        if ir.groupby_cols:
            warnings.append(
                "Per-group confidence intervals are not computed for GROUP BY "
                "queries; only point estimates are approximate."
            )
        elif rows:
            stats_sql, ci_warnings = self.estimator.build_stats_sql(sql, fragment)
            warnings.extend(ci_warnings)
            stats_cur = self.con.execute(stats_sql)
            stats_row_tuple = stats_cur.fetchone()
            stats_cols = [d[0] for d in stats_cur.description] if stats_cur.description else []
            stats_row: Dict[str, Any] = dict(zip(stats_cols, stats_row_tuple)) if stats_row_tuple else {}

            conf = self.estimator.compute(
                original_sql=sql,
                fragment=fragment,
                approx_result_row=rows[0],
                stats_row=stats_row,
                confidence_level=confidence_level,
            )

        return QueryResponse(
            original_sql=sql,
            executed_sql=rewrite.sql,
            is_approx=True,
            strategy=plan.strategy,
            sample_rate=fragment.sample_rate,
            rows=rows,
            columns=cols,
            warnings=warnings,
            confidence=conf,
        )

    def _execute_measure_biased(
        self,
        sql: str,
        ir: QueryIR,
        plan: SamplingPlan,
        confidence_level: float,
    ) -> QueryResponse | None:
        """Outlier-separated estimation for a single SUM/AVG over one column.

        Returns None (signalling "fall back to uniform") when the query is
        outside scope: more than one aggregate, an aggregate over an expression
        rather than a simple column, or a full-table sample (rate >= 1).
        """
        if plan.sample_rate >= 1.0:
            return None

        node = ir.ast.find(exp.Sum) or ir.ast.find(exp.Avg)
        if node is None or not isinstance(node.this, exp.Column):
            return None
        func = "SUM" if isinstance(node, exp.Sum) else "AVG"
        col = node.this.name
        table = ir.tables[0]
        label = node.sql(dialect="duckdb")

        idx = self.catalog.outlier_index(table, col)
        cutoff, r = idx.cutoff, plan.sample_rate

        where = ir.ast.find(exp.Where)
        cond = where.this.sql(dialect="duckdb") if where else None
        pred = f"({cond}) AND " if cond else ""

        # Heavy hitters: kept exact (respecting any WHERE predicate).
        out_s, _out_c = self.con.execute(
            f"SELECT COALESCE(SUM({col}), 0), COUNT(*) FROM {table} "
            f"WHERE {pred}abs({col}) >= {cutoff}"
        ).fetchone()
        out_s = float(out_s or 0)

        # Body: sampled uniformly. sum + sum-of-squares drive the estimate + CI.
        body_from = (
            f"(SELECT * FROM {table} WHERE {pred}abs({col}) < {cutoff}) "
            f"USING SAMPLE {r * 100:.6f}% (bernoulli)"
        )
        body_sql = (
            f"SELECT COALESCE(SUM({col}), 0), "
            f"COALESCE(SUM(({col}) * ({col})), 0) FROM {body_from}"
        )
        body_s, body_sq = self.con.execute(body_sql).fetchone()
        body_s, body_sq = float(body_s or 0), float(body_sq or 0)

        # Total (population) estimate = exact outliers + rescaled body sample.
        total_hat = out_s + body_s / r
        var_total = (1.0 - r) / (r * r) * body_sq          # variance from body only
        z = self.estimator._z_value(confidence_level)

        if func == "SUM":
            estimate = total_hat
            se = math.sqrt(max(var_total, 0.0))
            value: Any = round(estimate)
        else:  # AVG = total / count; count matching WHERE is known exactly.
            m = self.con.execute(
                f"SELECT COUNT(*) FROM {table}" + (f" WHERE {cond}" if cond else "")
            ).fetchone()[0]
            m = max(int(m), 1)
            estimate = total_hat / m
            se = math.sqrt(max(var_total, 0.0)) / m
            value = estimate

        half = z * se
        metric = MetricCI(
            name=label,
            estimate=estimate,
            se=se,
            ci_low=estimate - half,
            ci_high=estimate + half,
            rel_half_width=half / max(abs(estimate), 1e-12),
        )
        report = ConfidenceReport(
            confidence_level=confidence_level, metrics=[metric], warnings=[]
        )
        return QueryResponse(
            original_sql=sql,
            executed_sql=body_sql,
            is_approx=True,
            strategy="MEASURE_BIASED",
            sample_rate=r,
            rows=[(value,)],
            columns=[label],
            warnings=[f"outlier-separated: |{col}| >= {cutoff:.0f} kept exact"],
            confidence=report,
        )

