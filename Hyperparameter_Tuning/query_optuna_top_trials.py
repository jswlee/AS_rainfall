#!/usr/bin/env python3
"""
Query Optuna's PostgreSQL RDB backend to fetch the parameters of the top-N trials
for a given study, ranked by an objective value.

- Works directly against the database tables (trials, trial_values, trial_params, studies)
- No dependency on Optuna's Python API (but compatible with its default schema)
- Supports multiple objectives (choose objective index)
- Outputs to stdout as a pretty table, or to JSON/CSV files

Usage examples:

python -m Hyperparameter_Tuning.query_optuna_top_trials \
  --db-url postgresql+psycopg2://user:pass@host:5432/optuna \
  --study-name land_daily_focused_lr \
  --top-n 10 \
  --direction minimize \
  --objective-index 0 \
  --out-format table

python -m Hyperparameter_Tuning.query_optuna_top_trials \
  --db-url postgresql+psycopg2://user:pass@host:5432/optuna \
  --study-id 7 \
  --top-n 20 \
  --out-format json \
  --out-path output/top_trials.json

Notes on schema (Optuna default):
- studies (study_id, study_name, ...)
- trials (trial_id, number, study_id, state, datetime_start, datetime_complete)
- trial_values (trial_value_id, trial_id, objective, value, value_type)
- trial_params (param_id, trial_id, param_name, param_value, distribution_json)

This script assumes numeric parameters in trial_params.param_value (float8).
If your Optuna version stores categorical params differently, extend the SQL accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_study_id(engine: Engine, study_name: Optional[str], study_id: Optional[int]) -> int:
    if study_id is not None:
        return study_id
    if not study_name:
        raise ValueError("Either --study-name or --study-id must be provided")
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT study_id FROM studies WHERE study_name = :name"),
            {"name": study_name},
        ).mappings().fetchone()
        if not row:
            raise ValueError(f"Study not found: {study_name}")
        return int(row["study_id"])


def fetch_top_trials(
    engine: Engine,
    study_id: int,
    top_n: int,
    objective_index: int,
    direction: str,
) -> List[Dict[str, Any]]:
    order = "ASC" if direction.lower() == "minimize" else "DESC"
    sql = text(
        """
        WITH completed AS (
            SELECT t.trial_id, t.number, t.study_id, v.value AS objective_value
            FROM trials t
            JOIN trial_values v ON v.trial_id = t.trial_id AND v.objective = :objective
            WHERE t.study_id = :study_id AND t.state = 'COMPLETE'
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY objective_value {order}) AS rn
            FROM completed
        ), top AS (
            SELECT trial_id, number, objective_value
            FROM ranked
            WHERE rn <= :top_n
        )
        SELECT
            top.trial_id,
            top.number AS trial_number,
            top.objective_value,
            p.param_name,
            p.param_value
        FROM top
        LEFT JOIN trial_params p ON p.trial_id = top.trial_id
        ORDER BY objective_value {order}, trial_number ASC, p.param_name ASC
        """.format(order=order)
    )

    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {"study_id": study_id, "top_n": top_n, "objective": objective_index},
        ).mappings().all()

    # Pivot params per trial
    trials: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        tid = int(r["trial_id"])
        if tid not in trials:
            trials[tid] = {
                "trial_id": tid,
                "trial_number": int(r["trial_number"]),
                "objective_value": float(r["objective_value"]),
                "params": {},
            }
        if r["param_name"] is not None:
            trials[tid]["params"][r["param_name"]] = float(r["param_value"]) if r["param_value"] is not None else None

    return sorted(trials.values(), key=lambda x: x["objective_value"], reverse=(order == "DESC"))


def print_table(trials: List[Dict[str, Any]]) -> None:
    if not trials:
        print("No trials found.")
        return
    # Collect all param names across top trials (stable order)
    param_names: List[str] = []
    for t in trials:
        for k in t["params"].keys():
            if k not in param_names:
                param_names.append(k)

    # Header
    header = ["rank", "trial_number", "trial_id", "objective_value"] + param_names
    widths = [max(len(h), 5) for h in header]

    def fmt_row(cols: List[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cols, widths))

    print(fmt_row(header))
    print("-+-".join("-" * w for w in widths))

    for i, t in enumerate(trials, start=1):
        row = [
            str(i),
            str(t["trial_number"]),
            str(t["trial_id"]),
            f"{t['objective_value']:.6f}",
        ]
        for p in param_names:
            v = t["params"].get(p)
            row.append("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
        print(fmt_row(row))


def save_json(trials: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2)
    print(f"Saved JSON: {path}")


def save_csv(trials: List[Dict[str, Any]], path: str) -> None:
    if not trials:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        print(f"Saved empty CSV: {path}")
        return
    # Union of all params
    param_names: List[str] = []
    for t in trials:
        for k in t["params"].keys():
            if k not in param_names:
                param_names.append(k)
    cols = ["rank", "trial_number", "trial_id", "objective_value"] + param_names
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for i, t in enumerate(trials, start=1):
            fields: List[str] = [
                str(i),
                str(t["trial_number"]),
                str(t["trial_id"]),
                f"{t['objective_value']:.10g}",
            ]
            for p in param_names:
                v = t["params"].get(p)
                fields.append("" if v is None else f"{v:.10g}")
            f.write(",".join(fields) + "\n")
    print(f"Saved CSV: {path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Query Optuna Postgres for top-N trials and their parameters")
    parser.add_argument("--db-url", default="postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/optuna_daily", help="SQLAlchemy DB URL")
    parser.add_argument("--study-name", default="daily_3x3_1km4km_one_hot_3_1", help="Optuna study name (alternative to --study-id)")
    parser.add_argument("--study-id", type=int, help="Optuna study id (alternative to --study-name)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top trials to return")
    parser.add_argument("--objective-index", type=int, default=0, help="Objective index (0 for single-objective)")
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize", help="Optimization direction")
    parser.add_argument("--out-format", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--out-path", help="Path to save JSON/CSV output (required if out-format != table)")

    args = parser.parse_args(argv)

    if args.out_format in ("json", "csv") and not args.out_path:
        parser.error("--out-path is required when --out-format is json or csv")

    engine = create_engine(args.db_url)
    sid = get_study_id(engine, args.study_name, args.study_id)

    trials = fetch_top_trials(
        engine=engine,
        study_id=sid,
        top_n=args.top_n,
        objective_index=args.objective_index,
        direction=args.direction,
    )

    if args.out_format == "table":
        print_table(trials)
    elif args.out_format == "json":
        save_json(trials, args.out_path)
    else:
        save_csv(trials, args.out_path)


if __name__ == "__main__":
    main()
