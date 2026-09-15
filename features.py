from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parser import QueryIR
    from catalog import CatalogClient


# Sampling-rate tuning. Aim for ~50k sampled rows: tight CIs, real speedup.
TARGET_SAMPLE_ROWS = 50_000
MIN_SAMPLE_RATE = 0.02


@dataclass
class Phase1Features:
    num_filters:      int
    num_joins:        int
    has_groupby:      bool
    num_aggregations: int
    has_subquery:     bool
    join_types:       list[str]


def extract_phase1(ir: "QueryIR") -> Phase1Features:
    return Phase1Features(
        num_filters      = len(ir.predicates),
        num_joins        = len(ir.joins),
        has_groupby      = bool(ir.groupby_cols),
        num_aggregations = len(ir.aggregations),
        has_subquery     = ir.has_subquery,
        join_types       = [j["type"] for j in ir.joins],
    )


def phase1_sufficient(p1: Phase1Features) -> bool:
    clearly_sampling = (
        p1.num_joins == 0
        and p1.num_filters == 0
        and not p1.has_groupby
    )

    cross_join_present = "CROSS" in p1.join_types

    return clearly_sampling or cross_join_present


@dataclass
class Phase2Features:
    selectivity:     float
    variance:        float
    group_count:     int
    table_row_count: int

    @property
    def sample_size(self) -> float:
        """Target sampling fraction.

        We aim for roughly ``TARGET_SAMPLE_ROWS`` rows: enough for tight
        confidence intervals while still giving a real speedup on large tables.
        Bounded to [MIN_SAMPLE_RATE, 1.0].
        """
        if self.table_row_count == 0:
            return 1.0
        base = min(1.0, TARGET_SAMPLE_ROWS / max(self.table_row_count, 1))
        return max(base, MIN_SAMPLE_RATE)


def _estimate_selectivity(predicates: list, stats) -> float:
    if not predicates:
        return 1.0

    sel = 1.0

    for pred in predicates:
        op = pred["op"]

        if op == "EQ":
            col_sel = stats.histogram_frequency(pred["col"], pred["val"])

        elif op in ("GT", "LT", "GTE", "LTE", "BETWEEN"):
            # fallback: approximate range selectivity
            col_sel = stats.histogram_range_fraction(pred["col"], pred["val"])

        else:
            col_sel = 0.5

        sel *= col_sel

    return max(0.0, min(1.0, sel))


def extract_phase2(ir: "QueryIR", catalog: "CatalogClient") -> Phase2Features:
    stats = catalog.get_stats(ir.tables[0])

    selectivity = _estimate_selectivity(ir.predicates, stats)

    # safer variance logic
    if ir.groupby_cols:
        variance = stats.column_variance(ir.groupby_cols)
    else:
        variance = 0.0

    group_count = (
        stats.ndistinct(ir.groupby_cols)
        if ir.groupby_cols
        else 1
    )

    return Phase2Features(
        selectivity     = selectivity,
        variance        = variance,
        group_count     = group_count,
        table_row_count = stats.row_count,
    )


def extract(
    ir: "QueryIR",
    catalog: "CatalogClient",
) -> tuple[Phase1Features, Phase2Features]:

    p1 = extract_phase1(ir)

    if phase1_sufficient(p1):
        try:
            row_count = catalog.get_stats(ir.tables[0]).row_count
        except IndexError:
            row_count = 1000000

        p2 = Phase2Features(
            selectivity     = 1.0,
            variance        = 0.0,
            group_count     = 1,
            table_row_count = row_count,
        )
    else:
        p2 = extract_phase2(ir, catalog)

    return p1, p2