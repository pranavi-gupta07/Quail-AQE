"""
api/app.py — FastAPI front-end for the Quail AQE engine.

Two endpoints:
  POST /query/direct          -> exact execution (baseline / ground truth)
  POST /query/approximation   -> approximate execution with confidence intervals

The heavy lifting lives in the engine modules; this file only wires the
pipeline to HTTP and loads the demo dataset.
"""

import os
import sys
import time

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Make the project root importable when running `uvicorn api.app:app`.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_engine import QueryEngine
from parser import parse
from classify import make_sampling_plan
from features import extract
from catalog import CatalogClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(PROJECT_ROOT, "DB", "data.parquet")

# The demo dataset (UCI Online Retail) has no surrogate key; InvoiceNo is the
# closest thing to a stable identifier and is only used for stratified ordering.
DEFAULT_HASH_KEY = "InvoiceNo"

# --- App setup ---

app = FastAPI(title="Quail AQE API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

con = duckdb.connect(database=":memory:")
catalog = CatalogClient(con)
engine = QueryEngine(con, catalog=catalog)


@app.on_event("startup")
async def startup_event():
    if os.path.exists(PARQUET_PATH):
        con.execute(f"CREATE VIEW data AS SELECT * FROM '{PARQUET_PATH}'")
        print(f"VIEW 'data' -> {PARQUET_PATH}")
    else:
        con.execute("CREATE TABLE data (id INTEGER, val DOUBLE)")
        print("Parquet not found, created empty 'data' table.")


# --- Models ---

class QueryRequest(BaseModel):
    query: str
    confidence_level: Optional[float] = 0.95


# --- Routes ---

@app.get("/")
def root():
    return {"message": "Quail AQE API", "status": "ready"}


@app.post("/query/direct")
def query_direct(req: QueryRequest):
    try:
        t = time.time()
        return {"response": engine.execute_exact(req.query), "duration_s": time.time() - t}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query/approximation")
def query_approximation(req: QueryRequest):
    try:
        t = time.time()
        ir = parse(req.query)
        plan = make_sampling_plan(*extract(ir, catalog), ir, hash_key=DEFAULT_HASH_KEY)
        print(
            f"[AQE] Strategy: {plan.strategy} | Reason: {plan.reason} "
            f"| SampleRate: {plan.sample_rate}"
        )
        resp = engine.execute_approx(
            req.query, plan=plan, confidence_level=req.confidence_level
        )
        print(f"[AQE] executed_sql: {resp.executed_sql}")
        print(f"[AQE] warnings: {resp.warnings}")
        return {"response": resp, "duration_s": time.time() - t}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
