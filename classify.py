from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from features import Phase1Features, Phase2Features
from parser import QueryIR

Strategy = Literal["EXACT", "SAMPLING", "FILTERED_SAMPLING", "HYBRID", "MEASURE_BIASED"]


@dataclass(frozen=True)
class SamplingPlan:
    """
    Output contract of the classifier/planner for the rewriter layer.

    strategy        : which approximation strategy to use
    sample_rate     : fraction in (0, 1], e.g. 0.1 for 10%
    stratify_on     : optional list of columns to stratify on (e.g. GROUP BY keys)
    table_row_count : optional row count estimate for sample-size gating
    hash_key        : stable key column used for deterministic hash sampling/order
    reason          : human-readable reason label from the classifier
    """

    strategy: Strategy
    sample_rate: float
    stratify_on: list[str] | None = None
    table_row_count: int | None = None
    hash_key: str | None = "id"
    reason: str = ""


def make_sampling_plan(
    p1: Phase1Features,
    p2: Phase2Features,
    ir: QueryIR,
    *,
    hash_key: str | None = "id",
) -> SamplingPlan:
    """
    Convenience wrapper: choose strategy + derive a plan the rewriter can use.
    """
    strategy, reason = choose_strategy(p1, p2, ir)

    stratify_on = ir.groupby_cols if ir.groupby_cols else None
    sample_rate = float(p2.sample_size)

    # Clamp to sane bounds for safety.
    if sample_rate <= 0.0:
        sample_rate = 0.01
    if sample_rate > 1.0:
        sample_rate = 1.0

    return SamplingPlan(
        strategy=strategy,
        sample_rate=sample_rate,
        stratify_on=stratify_on,
        table_row_count=p2.table_row_count,
        hash_key=hash_key,
        reason=reason,
    )


def choose_strategy(
    p1: Phase1Features,
    p2: Phase2Features,
    ir: QueryIR,
) -> tuple[Strategy, str]:
    """Pick an approximation strategy from the extracted features.

    Design principle: the confidence-interval math in ``confidence.py`` assumes
    **independent, row-level (Bernoulli) inclusion**. Uniform sampling satisfies
    that assumption, so plain aggregate queries are routed to uniform sampling
    to keep the reported CIs statistically valid. Cluster/hash sampling on a
    key that groups related rows (e.g. all line-items of one invoice) violates
    that assumption and is therefore avoided for ungrouped aggregates.

    Returns ``(strategy, reason)``.
    """

    # 1. If the predicate is so selective the sample would rarely contain any
    #    qualifying rows, approximation is unreliable -> run exact.
    if p2.selectivity < 0.01:
        return "EXACT", "very_low_selectivity"

    # 2. Multi-join queries are outside the single-table rewriter's scope.
    #    Run exact so results stay correct rather than silently wrong.
    if p1.num_joins >= 2:
        return "EXACT", "multi_join_unsupported"

    # 3. GROUP BY -> stratified sampling so that no group vanishes from the
    #    result set. (Per-group CIs are not reported; see confidence.py.)
    if p1.has_groupby:
        return "HYBRID", "grouped_stratified"

    # 4. A single SUM/AVG over a heavy-tailed column benefits from
    #    outlier-separated ("measure-biased") sampling: keep the heavy hitters
    #    exact, sample only the body. The engine verifies the aggregate is a
    #    simple column and falls back to uniform sampling if not.
    if (
        p1.num_aggregations == 1
        and ir.aggregations
        and ir.aggregations[0] in ("SUM", "AVG")
    ):
        return "MEASURE_BIASED", "heavy_tail_outlier_separation"

    # 5. Range-filtered single-table aggregate. With no GROUP BY key to
    #    stratify on, FILTERED_SAMPLING falls back to uniform sampling, which
    #    keeps the CI math valid.
    has_filters = p1.num_filters >= 1
    has_range = any(p["is_range"] for p in ir.predicates)
    if has_filters and has_range:
        return "FILTERED_SAMPLING", "range_filter_sampling"

    # 6. Default: plain single-table aggregate -> uniform Bernoulli sampling.
    #    High variance is fine here; it just widens the (still valid) CI.
    reason = "high_variance_uniform" if p2.variance > 1e5 else "uniform_sampling"
    return "SAMPLING", reason