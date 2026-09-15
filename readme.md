# Quail-AQE — An Approximate Query Engine

Quail-AQE is a small **Approximate Query Engine (AQE)** built on top of
[DuckDB](https://duckdb.org/). Instead of scanning every row to answer an
aggregate query, it runs the query on a **statistical sample** of the data,
**rescales** the result back to full-table scale, and reports a **confidence
interval** so you know how much to trust the answer.

The whole point of the project is the question in the middle:

> Given an arbitrary SQL aggregate query, *how do we approximate it* — which
> rows to sample, how to sample them, how to correct the aggregate, and how to
> quantify the error?

This README explains how that pipeline works and how to run it.

---

## 1. Why approximate query processing?

Analytical (OLAP) queries such as `SELECT SUM(revenue) FROM sales` touch huge
tables. Often you don't need the *exact* answer — a number that is within ~1%
with a stated confidence is enough to draw a business conclusion, and it can be
computed from a fraction of the data. This is the idea behind systems like
BlinkDB and the approximate-aggregation features in Presto/Spark.

Quail-AQE is a compact, readable implementation of that idea over a single
DuckDB table, meant to make each stage of the process explicit and inspectable.

---

## 2. Architecture

The engine is a linear pipeline. A query flows top to bottom; at any stage it
may "bypass" to exact execution if approximation is unsafe or unsupported.

```
        SQL string
            |
            v
   [ parse ]            parser.py     SQL -> QueryIR (via sqlglot)
            |
            v
   [ bypass rules ] --- bypass ---> run EXACT
            |                       (DDL/DML, DISTINCT, no aggregate, ...)
            v
   [ feature extract ]  features.py + catalog.py
            |                       structural + statistical features
            v
   [ classifier ]       classify.py   choose_strategy() -> SamplingPlan
            |                         (EXACT / SAMPLING / FILTERED / HYBRID)
            v
   [ sample builder ]   rewriter/sample_builder.py
            |                         uniform / stratified / hash fragment
            v
   [ SQL rewriter ]     rewriter/rewriter.py + agg_scaler.py
            |                         inject sample into FROM, scale COUNT/SUM
            v
   [ execute + CI ]     query_engine.py + confidence.py
            |                         run rewritten SQL, companion stats -> CIs
            v
   QueryResponse (rows, strategy, sample_rate, confidence, warnings)
```

### Module map

| File | Responsibility |
|------|----------------|
| `parser.py` | Parse SQL into a `QueryIR` (tables, predicates, joins, group-by, aggregations) using `sqlglot`. |
| `bypass.py` | Decide whether a query is even approximable. |
| `catalog.py` | Table statistics (row count, equality/range selectivity, variance, distinct counts). |
| `features.py` | Turn the IR + catalog stats into features and a target sample rate. |
| `classify.py` | Choose a sampling strategy and produce a `SamplingPlan`. |
| `rewriter/sample_builder.py` | Turn a plan into a concrete SQL sample fragment. |
| `rewriter/agg_scaler.py` | Scale aggregates after sampling (`COUNT`/`SUM` x 1/rate; `AVG`/`MIN`/`MAX` unchanged). |
| `rewriter/rewriter.py` | String-rewrite the original query: inject the sample, apply scaling. |
| `confidence.py` | Run a companion stats query and compute confidence intervals. |
| `query_engine.py` | Orchestrate the whole pipeline; expose `execute_exact` / `execute_approx`. |
| `api/app.py` | FastAPI wrapper with `/query/direct` and `/query/approximation`. |
| `demo.py` | Standalone runner that compares exact vs approximate on sample queries. |

---

## 3. How the approximation works (the core)

### 3.1 Should we approximate at all? (`bypass.py`)

The engine runs **exact** when approximation is meaningless or unsafe:

- DDL/DML (`INSERT`, `CREATE`, `UPDATE`, ...) — not read queries.
- `DISTINCT` queries — sampling distorts distinct counts.
- Queries with **no aggregate** — sampling a row listing just returns fewer rows.
- Deeply nested multi-table subqueries — out of scope for the rewriter.

### 3.2 Features and a target sample rate (`features.py`, `catalog.py`)

*Structural* features come straight from the IR (number of filters, joins,
group-by presence, aggregation count). *Statistical* features come from the
catalog: estimated **selectivity** of the predicates, column **variance**, and
**group count**.

The target sampling fraction aims for roughly `TARGET_SAMPLE_ROWS` (50,000)
sampled rows, bounded below by `MIN_SAMPLE_RATE` (2%):

```
rate = clamp( TARGET_SAMPLE_ROWS / table_row_count , 0.02 , 1.0 )
```

On the bundled 541,909-row dataset this yields a **~9.2%** sample.

### 3.3 Choosing a strategy (`classify.py`)

`choose_strategy()` picks one of four strategies. The guiding constraint is
that the confidence-interval math (section 3.6) assumes **independent,
row-level (Bernoulli) inclusion**, so plain aggregate queries are routed to
*uniform* sampling to keep the reported intervals valid.

| Condition | Strategy | Why |
|-----------|----------|-----|
| selectivity < 1% | `EXACT` | too few qualifying rows to sample reliably |
| >= 2 joins | `EXACT` | multi-join is outside the single-table rewriter |
| has `GROUP BY` | `HYBRID` -> stratified | guarantee no group disappears |
| single `SUM`/`AVG` | `MEASURE_BIASED` -> outlier-separated | tighter, better-calibrated CIs on skewed columns |
| range filter present | `FILTERED_SAMPLING` -> uniform | sample then filter |
| otherwise | `SAMPLING` -> uniform | plain aggregate; valid CIs |

### 3.4 Building the sample (`rewriter/sample_builder.py`)

Three concrete sampling methods:

- **Uniform** — DuckDB native Bernoulli sampling, wrapped in a subquery so
  trailing `WHERE`/`GROUP BY` clauses attach correctly:
  `(SELECT * FROM t USING SAMPLE 9.23% (bernoulli)) AS _aqe_uniform`.
  Every row is included independently with probability *r*.
- **Stratified** — a CTE using `ROW_NUMBER() OVER (PARTITION BY group_col
  ORDER BY hash(key))`, keeping `ceil(group_size x r)` rows per group, so each
  group keeps a proportional share and none vanishes.
- **Hash** — deterministic `abs(hash(key)) % M < r*M`, giving a reproducible
  sample (same query -> same rows). Used only where determinism is wanted.

A **sample-size gate** refuses to approximate when the estimated sample would
have fewer than 30 rows (below the central-limit rule of thumb), falling back
to exact execution.

#### Measure-biased (outlier-separated) sampling

A single `SUM`/`AVG` over a heavy-tailed column is the hard case: a few extreme
rows carry most of the sum *and* most of the variance, so uniform-sample
intervals are wide and erratic. The `MEASURE_BIASED` strategy splits the table
into **heavy hitters** (top 0.5% by `|value|`, kept **exact**) and the
**body** (sampled at rate *r*), then combines them:

```
SUM_hat = sum(outliers)            # exact, zero variance
        + (1/r) * sum(sampled body)

Var(SUM_hat) ~ (1 - r)/r^2 * sum_over_sampled_body( y^2 )
```

Because the body excludes the extremes, its sum-of-squares is far smaller, so
the standard error collapses and the estimator is much closer to normal. The
cutoff is computed once (an "outlier index" in `catalog.py`) and cached. The
engine applies this only to a single `SUM`/`AVG` over a plain column and falls
back to uniform sampling otherwise. `outlier_sampling_demo.py` reproduces the
comparison in isolation.

### 3.5 Scaling the aggregate (`rewriter/agg_scaler.py`)

After sampling at rate *r*, aggregates must be corrected:

| Aggregate | Correction | Reason |
|-----------|-----------|--------|
| `COUNT`, `SUM` | multiply by `1/r` | the sample holds ~*r* of the total mass |
| `AVG`, `MIN`, `MAX`, percentiles | unchanged | a sample mean estimates the population mean; sample min/max need no scaling |
| `COUNT(DISTINCT ...)` | unchanged, warned | distinct-count from a sample is biased; a warning fires below 30% rate |

Non-aggregate select items (the grouping columns of a `GROUP BY`) are passed
through untouched.

### 3.6 Confidence intervals (`confidence.py`)

For a sample at rate *r*, the engine runs a **companion stats query** on the
same sampled relation and forms a normal-approximation interval
`estimate +/- z * SE` (`z = 1.96` for 95%).

Using a Horvitz-Thompson / Bernoulli variance estimator with equal inclusion
probability *r*:

```
COUNT :  Var(N_hat) ~ (1 - r)/r^2  * n_sample
SUM   :  Var(T_hat) ~ (1 - r)/r^2  * sum_sample( y^2 )
AVG   :  SE(mean)   ~ sqrt( var_samp / n ) * sqrt(1 - r)
```

The report includes each metric's estimate, standard error, interval bounds,
and relative half-width, so a wide interval visibly flags a shaky estimate.

---

## 4. Getting started

```bash
pip install -r requirements.txt

# Option A - standalone demo (no server): exact vs approximate side by side
python demo.py

# Option B - run the API (backend), then run the frontend separately
uvicorn api.app:app --reload
# API at http://localhost:8000 : /query/direct and /query/approximation (POST).
```

### Frontend (AQE-Frontend)

The user interface is a separate Vite + React app (the `AQE-Frontend` repo).
It calls this backend at `http://localhost:8000`, so run the two together:

```bash
# terminal 1 - backend
uvicorn api.app:app --reload

# terminal 2 - frontend
cd ../AQE-Frontend
npm install
npm run dev            # Vite dev server, e.g. http://localhost:5173
```

Open the Vite URL, type a query, set a target confidence, and hit Execute. The
app fires both an exact (`/query/direct`) and an approximate
(`/query/approximation`) request, then shows the result rows, the actual
rewritten SQL, timing for each path, and confidence/error gauges. CORS is open
on the backend so the browser can call it from the Vite origin.

Example API calls:

```bash
curl -s -X POST localhost:8000/query/direct \
  -H 'Content-Type: application/json' \
  -d '{"query":"SELECT COUNT(*), SUM(Quantity) FROM data"}'

curl -s -X POST localhost:8000/query/approximation \
  -H 'Content-Type: application/json' \
  -d '{"query":"SELECT COUNT(*), SUM(Quantity), AVG(UnitPrice) FROM data","confidence_level":0.95}'
```

The table is exposed as the view `data` (loaded from `DB/data.parquet`).

---

## 5. Example results

Representative output from `python demo.py` on the bundled dataset
(541,909 rows, ~9.2% sample). Point estimates and intervals vary between runs
because the sample is drawn fresh each time.

```
QUERY: SELECT COUNT(*), SUM(Quantity), AVG(UnitPrice) FROM data
  strategy   : SAMPLING (uniform_sampling)   sample rate: 9.23%
  exact      : COUNT=541,909   SUM=5,176,450   AVG=4.611
  approx     : COUNT=539,741   SUM=5,080,820   AVG=4.696
  rel error  : 0.40%           1.85%           1.85%
  95% CI     : COUNT [535,217 , 544,265]   (+/-0.84%)
               SUM   [4,896,180 , 5,265,460] (+/-3.63%)
               AVG   [3.81 , 5.58]          (+/-18.83%)

QUERY: SELECT COUNT(*) FROM data WHERE Quantity > 5
  strategy   : FILTERED_SAMPLING            sample rate: 9.23%
  exact 213,867  approx 215,626  rel error 0.82%  CI [212,790 , 218,462]

QUERY: SELECT Country, SUM(Quantity) FROM data GROUP BY Country
  strategy   : HYBRID (grouped_stratified)  sample rate: 9.23%
  (point estimates per country; per-group CIs not reported - see Limitations)
```

---

## 6. Evaluation (honest coverage numbers)

A 95% confidence interval should contain the true value ~95% of the time. Over
**200 independent runs** of the combined query at a ~9.2% sample:

| Metric | Empirical CI coverage | Mean relative error |
|--------|----------------------|---------------------|
| `COUNT(*)` | ~92% | ~0.4% |
| `SUM(Quantity)` | ~77% | ~6% |
| `AVG(UnitPrice)` | ~88% | ~7% |

Under *uniform* sampling, `COUNT` intervals are close to nominal, but `SUM` and
`AVG` **undercover** and are wide, because `Quantity` and `UnitPrice` are
heavy-tailed (bulk orders, returns as large negatives, extreme prices) and the
normal-approximation interval converges slowly for skewed data.

The `MEASURE_BIASED` strategy targets exactly this. Over 300 runs of single
`SUM`/`AVG` queries at the same ~9.2% budget:

| Query | Uniform err / CI half-width | Measure-biased err / CI half-width / coverage |
|-------|------------------------------|-----------------------------------------------|
| `SUM(Quantity)` | ~6% / ~14% | **~0.6% / ~1.5% / ~95%** |
| `AVG(UnitPrice)` | ~7% / (wide) | **~0.4% / ~0.8% / ~94%** |

Keeping the top 0.5% of rows exact makes the estimate ~10x more accurate, the
interval ~9x tighter, and coverage land on target. This applies to single
`SUM`/`AVG` queries; mixed-aggregate and grouped queries still use uniform /
stratified sampling, so their `SUM`/`AVG` intervals keep the caveat above.

> Note on speed: on this ~0.5M-row **in-memory** table, exact queries already
> run in a few milliseconds, and the sampling + companion stats query make the
> approximate path *slower* here, not faster. Approximation pays off when the
> exact scan is the bottleneck — very large or on-disk tables — where reading
> ~9% of the data dominates the cost. This demo showcases estimate quality and
> calibrated error, not wall-clock speedup at this scale.

---

## 7. Limitations and future work

- **Single-table rewriter.** The string rewriter targets single-table
  `SELECT ... FROM ... [WHERE] [GROUP BY]`. Multi-join queries route to exact.
  An AST-based rewriter would lift this.
- **Per-group confidence intervals** are not computed for `GROUP BY` queries;
  only point estimates are approximate. A grouped companion stats query would
  add them.
- **Heavy-tailed `SUM`/`AVG`** are handled for *single*-aggregate queries by
  the `MEASURE_BIASED` strategy (section 6). Mixed-aggregate and grouped queries
  still fall back to uniform/stratified sampling, so their skewed `SUM`/`AVG`
  intervals remain wide; extending measure-biased to those is future work.
- **Range selectivity** in the catalog is a fixed heuristic; an equi-depth
  histogram would make strategy selection sharper.
- **`COUNT(DISTINCT)`** is not corrected (only warned); sketch-based estimators
  (e.g. HyperLogLog) are the natural extension.

---

## 8. Project layout

```
Quail-AQE/
├── parser.py                 # SQL -> QueryIR
├── bypass.py                 # approximate-or-not rules
├── catalog.py                # table statistics
├── features.py               # feature + sample-rate extraction
├── classify.py               # strategy selection -> SamplingPlan
├── confidence.py             # confidence-interval estimation
├── query_engine.py           # pipeline orchestrator (+ measure-biased path)
├── demo.py                   # standalone exact-vs-approx runner
├── outlier_sampling_demo.py  # uniform vs measure-biased comparison
├── rewriter/
│   ├── sample_builder.py     # uniform / stratified / hash sample fragments
│   ├── agg_scaler.py         # aggregate rescaling
│   └── rewriter.py           # inject sample + scaling into the query
├── api/
│   └── app.py                # FastAPI endpoints (consumed by AQE-Frontend)
├── DB/
│   └── data.parquet          # demo dataset
└── requirements.txt
```

---

## 9. Dataset

The bundled `DB/data.parquet` is the **Online Retail** transactional dataset
(UK online retailer, ~541k rows: `InvoiceNo`, `StockCode`, `Description`,
`Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`), originally
published in the UCI Machine Learning Repository. It is used here purely as a
realistic table to demonstrate approximate aggregation.
