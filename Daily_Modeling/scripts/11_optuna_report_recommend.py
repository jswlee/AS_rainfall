import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import sqlite3


def _load_study(tuning_dir: Path, study_name: str | None) -> optuna.Study:
    db_path = tuning_dir / "optuna_study.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing optuna db: {db_path}")

    if study_name is None:
        study_name = tuning_dir.name

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        # Fallback: auto-detect a single study name from the DB
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT study_name FROM studies LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row is None:
            raise RuntimeError(f"No studies found in {db_path}")
        detected_name = row[0]
        print(f"Detected study name: {detected_name}")
        return optuna.load_study(study_name=detected_name, storage=storage)


def _completed_trials_df(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append({"trial": t.number, "value": t.value, **t.params})
    if not rows:
        return pd.DataFrame(columns=["trial", "value"])
    df = pd.DataFrame(rows)
    df = df.sort_values("value", ascending=True).reset_index(drop=True)
    return df


def _quantile_band(values: np.ndarray, lo_q: float, hi_q: float) -> tuple[float, float]:
    lo = float(np.quantile(values, lo_q))
    hi = float(np.quantile(values, hi_q))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _recommend_ranges(
    df: pd.DataFrame,
    importances: dict[str, float],
    *,
    top_frac: float = 0.15,
    dominant_threshold: float = 0.70,
) -> dict:
    if df.empty:
        return {"notes": ["No completed trials found."]}

    top_k = max(5, int(len(df) * float(top_frac)))
    top_df = df.head(top_k)

    rec: dict[str, object] = {
        "notes": [],
        "dominant_params": [],
        "tighten": {},
        "fix_or_deprioritize": {},
        "categorical": {},
        "two_stage_plan": [],
    }

    # Identify dominant params
    dominant = [k for k, v in importances.items() if float(v) >= float(dominant_threshold)]
    if dominant:
        rec["dominant_params"] = [{"param": k, "importance": float(importances[k])} for k in dominant]
        rec["notes"].append(
            "At least one hyperparameter dominates the objective variance; consider a two-stage tuning plan."
        )

    # Heuristics:
    # - For log-like scalars (lr/wd), tighten using quantile band of top trials.
    # - For other continuous, tighten similarly.
    # - For low-importance, recommend fixing to best-trial value.
    best = df.iloc[0].to_dict()

    for p in df.columns:
        if p in ("trial", "value"):
            continue
        if p not in top_df.columns:
            continue

        imp = float(importances.get(p, 0.0))
        series = top_df[p]

        # Detect categorical-ish (small unique count) columns
        uniq = series.dropna().unique()
        is_categorical = len(uniq) <= 6 and all(np.isfinite(np.array(uniq, dtype=float)))

        if is_categorical:
            # Recommend keeping only values that appear in the top set
            counts = series.value_counts(dropna=True).to_dict()
            best_val = best.get(p)
            rec["categorical"][p] = {
                "best": best_val,
                "top_values": [
                    {"value": (float(k) if isinstance(k, (int, float, np.integer, np.floating)) else k), "count": int(v)}
                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
                ],
                "recommendation": "Prefer values that recur among top trials; drop rarely-good categories.",
            }
            continue

        # Continuous numeric
        vals = series.dropna().to_numpy(dtype=float)
        if len(vals) < 5:
            continue

        # Log-like params: use log10 band for stability
        if p in ("base_lr", "learning_rate", "weight_decay"):
            safe = np.clip(vals, 1e-12, None)
            logv = np.log10(safe)
            lo, hi = _quantile_band(logv, 0.10, 0.90)
            # Add small padding
            pad = 0.15 * (hi - lo + 1e-12)
            lo2 = lo - pad
            hi2 = hi + pad
            rec["tighten"][p] = {
                "importance": imp,
                "suggest_log10_range": [float(lo2), float(hi2)],
                "suggest_range": [float(10 ** lo2), float(10 ** hi2)],
                "based_on": f"top_{top_k}_trials_10-90pct_with_padding",
            }
        else:
            lo, hi = _quantile_band(vals, 0.10, 0.90)
            pad = 0.15 * (hi - lo + 1e-12)
            rec["tighten"][p] = {
                "importance": imp,
                "suggest_range": [float(lo - pad), float(hi + pad)],
                "based_on": f"top_{top_k}_trials_10-90pct_with_padding",
            }

        # Low-importance: propose fixing
        if imp <= 0.02:
            rec["fix_or_deprioritize"][p] = {
                "importance": imp,
                "fix_to": best.get(p),
                "reason": "Very low estimated importance; fix to best value or remove from search until later.",
            }

    # Two-stage plan template
    if "base_lr" in importances:
        rec["two_stage_plan"].append(
            {
                "stage": 1,
                "goal": "Stabilize and localize learning-rate region",
                "tune": ["base_lr"],
                "fix": [p for p in df.columns if p not in ("trial", "value", "base_lr")],
                "trials": 50,
                "notes": "Use a moderately wide log range around the current best; keep early stopping on for speed.",
            }
        )
        rec["two_stage_plan"].append(
            {
                "stage": 2,
                "goal": "Fine-tune secondary hyperparameters once LR is localized",
                "tune": [p for p in df.columns if p not in ("trial", "value")],
                "trials": 100,
                "notes": "Narrow base_lr to the band from stage 1; only then widen architecture/regularization search if needed.",
            }
        )

    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuning-dir",
        required=True,
        help="Path to tuning output dir containing optuna_study.db (e.g. Daily_Modeling/output/tuning/land_daily_tweedie_mae_100t)",
    )
    parser.add_argument("--study-name", default=None, help="Optuna study name (default: folder name)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top trials to print/save")
    parser.add_argument(
        "--top-frac",
        type=float,
        default=0.15,
        help="Fraction of top trials used to infer tighter ranges (default: 0.15)",
    )
    parser.add_argument(
        "--dominant-threshold",
        type=float,
        default=0.70,
        help="Importance threshold for declaring a dominant param (default: 0.70)",
    )
    args = parser.parse_args()

    tuning_dir = Path(args.tuning_dir)
    study = _load_study(tuning_dir, args.study_name)

    df = _completed_trials_df(study)
    if df.empty:
        raise RuntimeError("No completed trials found in study.")

    top_n = int(args.top_n)
    top_df = df.head(top_n).copy()

    # Importance (fANOVA / default evaluator)
    importances = optuna.importance.get_param_importances(study)

    rec = _recommend_ranges(
        df,
        importances,
        top_frac=float(args.top_frac),
        dominant_threshold=float(args.dominant_threshold),
    )

    # Save artifacts
    out_top = tuning_dir / f"top_{top_n}_trials.csv"
    out_imp = tuning_dir / "hp_importance.json"
    out_rec = tuning_dir / "hp_space_recommendations.json"

    top_df.to_csv(out_top, index=False)
    out_imp.write_text(json.dumps({k: float(v) for k, v in importances.items()}, indent=2))
    out_rec.write_text(json.dumps(rec, indent=2))

    # Console summary
    print(f"Study: {study.study_name}")
    print(f"Completed trials: {len(df)}")
    print("\nTop trials:")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(top_df)

    print("\nHyperparameter importance:")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>20s} : {float(v):.3f}")

    print(f"\nWrote: {out_top}")
    print(f"Wrote: {out_imp}")
    print(f"Wrote: {out_rec}")


if __name__ == "__main__":
    main()
