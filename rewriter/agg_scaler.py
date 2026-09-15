# rewriter/agg_scaler.py
from __future__ import annotations
from dataclasses import dataclass


# Aggregates that MUST be scaled by 1/rate
# COUNT(*) on 10% sample → multiply by 10
# SUM(revenue) on 10% sample → multiply by 10
SCALE_UP = {"COUNT", "SUM"}

# Aggregates that must NOT be scaled — they are distribution statistics
# AVG of a 10% sample is still the correct average
# MIN/MAX of any sample is still min/max of that sample
NO_SCALE = {"AVG", "MIN", "MAX", "MEDIAN", "PERCENTILE_CONT", "PERCENTILE_DISC"}

# COUNT DISTINCT needs special treatment — accuracy degrades below this rate
COUNT_DISTINCT_MIN_RATE = 0.30


@dataclass
class ScaleResult:
    """
    Output of the scaler.
    
    scaled_sql      : the aggregate expression after scaling
    was_scaled      : whether any scaling was applied
    warnings        : e.g. COUNT DISTINCT rate warning
    """
    scaled_sql: str
    was_scaled: bool
    warnings: list[str]


class AggScaler:
    """
    Handles aggregate scaling after sampling.
    
    Works on raw SQL strings of individual aggregate expressions,
    not the full query AST — the rewriter handles AST injection.
    
    Rules:
      COUNT(*) at rate 0.1  → ROUND(COUNT(*) * 10)
      SUM(x)   at rate 0.1  → ROUND(SUM(x) * 10)
      AVG(x)               → AVG(x)         [unchanged]
      MIN(x)               → MIN(x)         [unchanged]
      MAX(x)               → MAX(x)         [unchanged]
    """

    def scale(self, agg_expr: str, rate: float) -> ScaleResult:
        """
        Main entry point.
        
        agg_expr : raw aggregate SQL string e.g. "COUNT(*)", "SUM(revenue)"
        rate     : sample rate used e.g. 0.1 for 10%
        
        Returns ScaleResult with the corrected expression.
        """
        warnings = []

        # Non-aggregate SELECT items (GROUP BY keys, bare columns, literals)
        # have no function call to scale — pass them through untouched. This is
        # the normal case for the grouping columns in a GROUP BY query, so it
        # must not emit a warning.
        if "(" not in agg_expr:
            return ScaleResult(scaled_sql=agg_expr, was_scaled=False, warnings=[])

        func_name = self._extract_func_name(agg_expr)

        # ── COUNT DISTINCT special case ───────────────────────────────────────
        if self._is_count_distinct(agg_expr):
            if rate < COUNT_DISTINCT_MIN_RATE:
                warnings.append(
                    f"COUNT DISTINCT detected with rate={rate:.2f}. "
                    f"Accuracy degrades below {COUNT_DISTINCT_MIN_RATE:.2f}. "
                    f"Consider bumping rate or using exact execution."
                )
            # COUNT DISTINCT is NOT scaled — distinct values in sample
            # are already a sample of distinct values in population
            return ScaleResult(
                scaled_sql=agg_expr,
                was_scaled=False,
                warnings=warnings,
            )

        # ── Scale up aggregates ───────────────────────────────────────────────
        if func_name in SCALE_UP:
            inverse = round(1.0 / rate, 6)
            scaled = f"ROUND({agg_expr} * {inverse})"
            return ScaleResult(
                scaled_sql=scaled,
                was_scaled=True,
                warnings=warnings,
            )

        # ── Invariant aggregates — return unchanged ───────────────────────────
        if func_name in NO_SCALE:
            return ScaleResult(
                scaled_sql=agg_expr,
                was_scaled=False,
                warnings=warnings,
            )

        # ── Unknown aggregate — warn and return unchanged ─────────────────────
        warnings.append(
            f"Unknown aggregate '{func_name}' — not scaled. "
            f"Verify manually whether scaling is needed."
        )
        return ScaleResult(
            scaled_sql=agg_expr,
            was_scaled=False,
            warnings=warnings,
        )

    def scale_many(self, agg_exprs: list[str], rate: float) -> list[ScaleResult]:
        """
        Convenience method — scale a list of aggregates at once.
        Used when a query has multiple aggregates e.g. SELECT COUNT(*), SUM(x), AVG(y)
        """
        return [self.scale(expr, rate) for expr in agg_exprs]

    # ── private helpers ───────────────────────────────────────────────────────

    def _extract_func_name(self, agg_expr: str) -> str:
        """
        Pulls the function name out of an expression string.
        
        "COUNT(*)"           → "COUNT"
        "SUM(revenue)"       → "SUM"
        "AVG(price)"         → "AVG"
        "count(*)"           → "COUNT"  (case-insensitive)
        """
        return agg_expr.strip().split("(")[0].upper()

    def _is_count_distinct(self, agg_expr: str) -> bool:
        """
        Detects COUNT(DISTINCT col) pattern.
        
        "COUNT(DISTINCT user_id)"  → True
        "COUNT(*)"                 → False
        "COUNT(user_id)"           → False
        """
        normalized = agg_expr.upper().replace(" ", "")
        return "COUNT(DISTINCT" in normalized