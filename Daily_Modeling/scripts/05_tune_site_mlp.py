"""
Step 5: Hyperparameter tuning for the site-specific MLP using Optuna.

One independent Optuna study is run per loss type so that objective values
are always comparable within a study.  Results are saved to separate
subdirectories: <study_name>/mse/, <study_name>/log_mse/, etc.
The best_hyperparameters.json for each loss type can then be passed directly
to 08_train_site_mlp.py via --hp-dir.

Speed optimisations:
  - GPU-only: sequential per-station training within each trial.
    (CUDA context is not thread-safe; the GPU is already massively
    parallel within each forward/backward pass, so threading adds
    overhead without benefit.)
  - 18 predefined architecture combos spanning 1, 2, and 3 hidden layers
    (depth is a key HP: shallow nets generalise better for small/noisy stations)
  - Random station sampling per trial (--n-stations, default 5)
  - Raised batch sizes (128-1024)
  - Reduced tuning epochs (80) and patience (12)

Outputs per loss type (under <study_name>/<loss_type>/):
  - best_hyperparameters.json  (includes loss_type key)
  - all_trials.csv
  - top10_trials.csv  (printed + saved)
  - hp_distribution_top10.png
  - hp_importance.png, optimization_history.png, parallel_coordinate.png

Usage:
    python -m Daily_Modeling.scripts.05_tune_site_mlp [--n-trials 30] [--n-stations 8]
    python -m Daily_Modeling.scripts.05_tune_site_mlp --loss-types mse log_mse
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import (
    load_tensors_from_npz, normalize_tensors, RainfallDataset,
)
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, compute_station_year_ranges, compute_year_boundaries,
    spatiotemporal_split, station_proportional_split,
)
from Daily_Modeling.models.site_mlp import SiteMLP, SiteGLU, build_model, compute_input_size
from Daily_Modeling.utils.io_utils import save_json
from Daily_Modeling.utils.training import get_criterion, train_model

# Architecture candidates spanning 1, 2, and 3 hidden layers.
# Depth is a key HP: shallow nets generalise better for small/noisy stations;
# deeper nets have more capacity for data-rich stations.
_ARCH_CANDIDATES = [
    # --- 1 hidden layer ---
    [64],               # 1L tiny
    [128],              # 1L small
    [256],              # 1L medium
    [512],              # 1L large
    # --- 2 hidden layers ---
    [128, 64],          # 2L tiny tapered
    [128, 128],         # 2L small
    [256, 128],         # 2L tapered
    [256, 256],         # 2L medium
    [512, 256],         # 2L tapered large
    [512, 512],         # 2L large
    # --- 3 hidden layers ---
    [128, 128, 128],    # 3L small
    [256, 128, 128],    # 3L tapered small
    [256, 256, 256],    # 3L medium
    [512, 256, 128],    # 3L tapered medium
    [512, 256, 256],    # 3L medium-large
    [512, 512, 256],    # 3L large tapered
    [512, 512, 512],    # 3L large (paper default)
    [256, 512, 256],    # 3L bottleneck
]

# Tuning-specific constants
_TUNE_MAX_EPOCHS = 80
_TUNE_PATIENCE = 12


class _FlatDataset(torch.utils.data.Dataset):
    """Wraps RainfallDataset to return flattened feature vector."""
    def __init__(self, base: RainfallDataset):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        feats, target = self.base[idx]
        parts = [feats[k].view(-1) for k in ("climate", "local_dem", "regional_dem", "temporal")]
        return torch.cat(parts), target


def _get_metadata(tensors):
    c = tensors["climate"]
    return {
        "climate_shape": tuple(c.shape[1:]),
        "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
        "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
        "num_month_features": int(tensors["temporal"].shape[1]),
    }


def _sorted_indices(indices, years, months, days):
    return sorted(indices, key=lambda i: (int(years[i]), int(months[i]), int(days[i])))


def _expanding_time_folds(indices_sorted, n_folds: int):
    """Forward-chaining (expanding-window) folds.

    Fold k trains on the earliest chunk and validates on the next contiguous
    chunk, so validation is always later in time than the training data.
    """
    if n_folds <= 1:
        return []
    n = len(indices_sorted)
    val_size = max(n // (n_folds + 1), 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = k * val_size
        val_end = min((k + 1) * val_size, n)
        if train_end >= n or train_end >= val_end:
            break
        folds.append((indices_sorted[:train_end], indices_sorted[train_end:val_end]))
    return folds


def _train_one_station(st, tensors, stations, years, months, days, target_scale,
                       dem_crop, input_size, hp, criterion, device, cv_folds: int = 1):
    """Train a single station model and return mean val RMSE (mm) or None if skipped.

    The model is trained with *criterion* (which may be Tweedie, MSE, etc.),
    but the value returned to the Optuna objective is always **validation
    RMSE in physical units (mm)**.  This ensures that the objective is
    scale-invariant across different loss functions and Tweedie p values.
    """
    sp = station_proportional_split(stations, years, months, days, st)
    trainval_idx = list(sp["train"]) + list(sp["val"])
    if len(trainval_idx) < 70:
        return None

    # Build CV folds over train+val; if disabled, use the original split
    if cv_folds and cv_folds > 1:
        tv_sorted = _sorted_indices(trainval_idx, years, months, days)
        folds = _expanding_time_folds(tv_sorted, cv_folds)
    else:
        folds = [(sp["train"], sp["val"])]

    rmses = []
    for tr_idx, va_idx in folds:
        n_train, n_val = len(tr_idx), len(va_idx)
        if n_train < 50 or n_val < 10:
            continue

        train_ds = _FlatDataset(RainfallDataset(tensors, tr_idx, target_scale,
                                                dem_crop_config=dem_crop))
        val_ds = _FlatDataset(RainfallDataset(tensors, va_idx, target_scale,
                                              dem_crop_config=dem_crop))
        tl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True,
                        pin_memory=False)
        vl = DataLoader(val_ds, batch_size=hp["batch_size"], pin_memory=False)

        model = build_model(
            hp.get("arch_type", "mlp"), input_size, hp["hidden_sizes"], hp["dropout_rate"]
        ).to(device)
        train_model(model, tl, vl, device, epochs=_TUNE_MAX_EPOCHS, patience=_TUNE_PATIENCE,
                    learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
                    criterion=criterion, verbose=0)

        # Compute validation RMSE in mm (scale-invariant across loss types / p)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, t in vl:
                x = torch.nan_to_num(x.to(device))
                out = model(x)
                preds.append(out.cpu().numpy().ravel())
                targets.append(t.cpu().numpy().ravel())
        yp = np.concatenate(preds) * target_scale
        yt = np.concatenate(targets) * target_scale
        rmses.append(float(np.sqrt(np.mean((yp - yt) ** 2))))

    if len(rmses) == 0:
        return None
    return float(np.mean(rmses))


def objective(trial, tensors, train_stations, stations, years, months, days,
              metadata, device, target_scale, n_stations, loss_type, tweedie_p,
              tune_arch_type=False, cv_folds: int = 1):
    local_candidates = config.DEM_LOCAL_CANDIDATES
    regional_candidates = config.DEM_REGIONAL_CANDIDATES
    local_idx = trial.suggest_int("local_dem_cfg", 0, len(local_candidates) - 1)
    regional_idx = trial.suggest_int("regional_dem_cfg", 0, len(regional_candidates) - 1)
    lp, lk = local_candidates[local_idx]
    rp, rk = regional_candidates[regional_idx]
    dem_crop = {
        "local_patch_size": lp, "local_km": lk,
        "regional_patch_size": rp, "regional_km": rk,
    }

    arch_idx = trial.suggest_int("arch_idx", 0, len(_ARCH_CANDIDATES) - 1)
    hidden_sizes = _ARCH_CANDIDATES[arch_idx]

    # Optionally tune arch_type (mlp vs lstm) as a HP
    if tune_arch_type:
        arch_type = trial.suggest_categorical("arch_type", ["mlp", "glu"])
    else:
        arch_type = "mlp"

    # Tune tweedie_p within the objective when loss is tweedie
    if loss_type == "tweedie":
        tweedie_p = trial.suggest_float("tweedie_p", 1.05, 1.95, step=0.05)

    hp = {
        "hidden_sizes": hidden_sizes,
        "arch_type": arch_type,
        "dropout_rate": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("bs", [128, 256, 512, 1024]),
        "local_dem_patch": lp, "local_dem_km": lk,
        "regional_dem_patch": rp, "regional_dem_km": rk,
    }
    criterion = get_criterion(loss_type, p=tweedie_p) if loss_type == "tweedie" else get_criterion(loss_type)

    input_size = compute_input_size(
        climate_shape=metadata["climate_shape"],
        local_dem_shape=(lp, lp),
        regional_dem_shape=(rp, rp),
        num_month=metadata["num_month_features"],
    )

    # Randomly sample stations per trial
    rng = np.random.RandomState(config.RANDOM_SEED + trial.number)
    n_pick = min(n_stations, len(train_stations))
    picked = rng.choice(train_stations, n_pick, replace=False).tolist()

    # GPU-only: run sequentially — CUDA context is not thread-safe and the GPU
    # is already massively parallel within each forward/backward pass.
    val_losses = []
    for st in picked:
        result = _train_one_station(
            st, tensors, stations, years, months, days,
            target_scale, dem_crop, input_size, hp, criterion, device,
            cv_folds=cv_folds,
        )
        if result is not None:
            val_losses.append(result)

    return float(np.mean(val_losses)) if val_losses else float("inf")


_ALL_LOSS_TYPES = ["mse", "log_mse", "tweedie"]


def _run_one_study(loss_type, tensors, meta, train_stations, stations, years,
                   metadata, stats, target_scale, base_out_dir, args, tweedie_p):
    """Run a single Optuna study for one loss type and save all outputs."""
    out_dir = base_out_dir / loss_type
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss_type == "tweedie":
        # tweedie_p is tuned inside the Optuna objective (suggest_float), so don't
        # print a misleading fixed p here. The CLI tweedie_p is only a fallback
        # when tuning is disabled or not used.
        p_str = "  p=tuned[1.05..1.95 step 0.05]"
    else:
        p_str = ""
    print(f"\n{'='*60}")
    print(f"  Loss: {loss_type}{p_str}  |  Device: {device}  |  Trials: {args.n_trials}")
    print(f"{'='*60}")

    study = optuna.create_study(
        study_name=f"{args.study_name}_{loss_type}", direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    study.optimize(
        lambda t: objective(
            t, tensors, train_stations, stations, years,
            meta["months"], meta["days"], metadata, device, target_scale,
            n_stations=args.n_stations, loss_type=loss_type, tweedie_p=tweedie_p,
            tune_arch_type=args.tune_arch_type,
            cv_folds=args.cv_folds,
        ),
        n_trials=args.n_trials, show_progress_bar=True,
    )

    print(f"\n[{loss_type}] Best: trial {study.best_trial.number}  val={study.best_value:.6f}")

    best_hp = dict(study.best_params)
    best_hp["loss_type"] = loss_type
    # tweedie_p: if tuned inside objective it's already in best_params; else use CLI default
    if loss_type == "tweedie" and "tweedie_p" not in best_hp:
        best_hp["tweedie_p"] = tweedie_p
    # arch_type: if not tuned, default to mlp
    if "arch_type" not in best_hp:
        best_hp["arch_type"] = "mlp"
    dem_crop = config.resolve_dem_crop(best_hp)
    if dem_crop is not None:
        best_hp["local_dem_patch"] = dem_crop["local_patch_size"]
        best_hp["local_dem_km"] = dem_crop["local_km"]
        best_hp["regional_dem_patch"] = dem_crop["regional_patch_size"]
        best_hp["regional_dem_km"] = dem_crop["regional_km"]
    if "arch_idx" in best_hp:
        best_hp["hidden_sizes"] = _ARCH_CANDIDATES[best_hp["arch_idx"]]
    save_json(best_hp, out_dir / "best_hyperparameters.json")
    save_json(stats, out_dir / "normalization_stats.json")

    rows = [{"trial": t.number, "value": t.value, **t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    all_df = pd.DataFrame(rows)
    all_df.to_csv(out_dir / "all_trials.csv", index=False)

    _save_top10(all_df, out_dir, title_suffix=f" [{loss_type}]")
    _save_tuning_visuals(study, out_dir)

    print(f"[{loss_type}] Saved to {out_dir}")
    return study.best_value, best_hp


def _objective_single_station(trial, tensors, station, stations, years, months, days,
                               metadata, device, target_scale, loss_type, tweedie_p,
                               tune_arch_type=False, cv_folds: int = 1):
    """Optuna objective for a single station — tune HPs for that station only."""
    local_candidates = config.DEM_LOCAL_CANDIDATES
    regional_candidates = config.DEM_REGIONAL_CANDIDATES
    local_idx = trial.suggest_int("local_dem_cfg", 0, len(local_candidates) - 1)
    regional_idx = trial.suggest_int("regional_dem_cfg", 0, len(regional_candidates) - 1)
    lp, lk = local_candidates[local_idx]
    rp, rk = regional_candidates[regional_idx]
    dem_crop = {
        "local_patch_size": lp, "local_km": lk,
        "regional_patch_size": rp, "regional_km": rk,
    }
    arch_idx = trial.suggest_int("arch_idx", 0, len(_ARCH_CANDIDATES) - 1)

    # Optionally tune arch_type (mlp vs glu)
    if tune_arch_type:
        arch_type = trial.suggest_categorical("arch_type", ["mlp", "glu"])
    else:
        arch_type = "mlp"

    # Tune tweedie_p within the objective when loss is tweedie
    if loss_type == "tweedie":
        tweedie_p = trial.suggest_float("tweedie_p", 1.05, 1.95, step=0.05)

    hp = {
        "hidden_sizes": _ARCH_CANDIDATES[arch_idx],
        "arch_type": arch_type,
        "dropout_rate": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("bs", [128, 256, 512, 1024]),
        "local_dem_patch": lp, "local_dem_km": lk,
        "regional_dem_patch": rp, "regional_dem_km": rk,
    }
    criterion = get_criterion(loss_type, p=tweedie_p) if loss_type == "tweedie" else get_criterion(loss_type)
    input_size = compute_input_size(
        climate_shape=metadata["climate_shape"],
        local_dem_shape=(lp, lp),
        regional_dem_shape=(rp, rp),
        num_month=metadata["num_month_features"],
    )
    result = _train_one_station(
        station, tensors, stations, years, months, days,
        target_scale, dem_crop, input_size, hp, criterion, device,
        cv_folds=cv_folds,
    )
    return result if result is not None else float("inf")


def _run_per_station_tuning(tensors, meta, all_stations, stations, years,
                             metadata, stats, target_scale, base_out_dir, args):
    """Run a separate Optuna study per station × loss_type combination."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    months = meta["months"]
    days = meta["days"]
    tweedie_p = args.tweedie_p

    station_summary = {}

    def _tune_one_station(station: str, loss_type: str):
        out_dir = base_out_dir / "per_station" / loss_type / station
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check station has enough data
        sp = station_proportional_split(stations, years, months, days, station)
        if len(sp["train"]) < 50 or len(sp["val"]) < 10:
            return (station, loss_type, None, f"SKIP: insufficient data (n_train={len(sp['train'])}, n_val={len(sp['val'])})")

        study = optuna.create_study(
            study_name=f"{args.study_name}_{station}_{loss_type}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        )
        study.optimize(
            lambda t, _st=station, _lt=loss_type, _p=tweedie_p: _objective_single_station(
                t, tensors, _st, stations, years, months, days,
                metadata, device, target_scale, _lt, _p,
                tune_arch_type=args.tune_arch_type,
            ),
            n_trials=args.per_station_trials, show_progress_bar=False,
        )

        best_hp = dict(study.best_params)
        best_hp["loss_type"] = loss_type
        # tweedie_p: if tuned inside objective it's already in best_params; else use CLI default
        if loss_type == "tweedie" and "tweedie_p" not in best_hp:
            best_hp["tweedie_p"] = tweedie_p
        # arch_type: if not tuned, default to mlp
        if "arch_type" not in best_hp:
            best_hp["arch_type"] = "mlp"
        dem_crop = config.resolve_dem_crop(best_hp)
        if dem_crop is not None:
            best_hp["local_dem_patch"] = dem_crop["local_patch_size"]
            best_hp["local_dem_km"] = dem_crop["local_km"]
            best_hp["regional_dem_patch"] = dem_crop["regional_patch_size"]
            best_hp["regional_dem_km"] = dem_crop["regional_km"]
        if "arch_idx" in best_hp:
            best_hp["hidden_sizes"] = _ARCH_CANDIDATES[best_hp["arch_idx"]]

        save_json(best_hp, out_dir / "best_hyperparameters.json")
        save_json(stats, out_dir / "normalization_stats.json")

        rows = [{"trial": t.number, "value": t.value, **t.params}
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        all_df = pd.DataFrame(rows)
        all_df.to_csv(out_dir / "all_trials.csv", index=False)
        _save_top10(all_df, out_dir, title_suffix=f" [{station} | {loss_type}]")

        return (station, loss_type, float(study.best_value), str(out_dir))

    tasks = []
    skipped = 0
    for lt in args.loss_types:
        for st in all_stations:
            if getattr(args, "per_station_resume", False):
                hp_path = base_out_dir / "per_station" / lt / st / "best_hyperparameters.json"
                if hp_path.exists():
                    skipped += 1
                    continue
            tasks.append((st, lt))
    total = len(tasks)
    parallel = max(int(getattr(args, "per_station_parallel", 1)), 1)
    if device.type == "cuda" and parallel > 1:
        print("WARNING: --per-station-parallel > 1 with CUDA may OOM or run slower due to GPU contention.")

    if getattr(args, "per_station_resume", False):
        print(f"Resume enabled: skipping {skipped} already-completed station/loss tasks")

    if parallel == 1:
        done = 0
        for station, loss_type in tasks:
            done += 1
            print(f"\n[{done}/{total}] Station={station}  Loss={loss_type}")
            st, lt, best_val, info = _tune_one_station(station, loss_type)
            if best_val is None:
                print(f"  {info}")
                continue
            station_summary[(st, lt)] = best_val
            print(f"  BEST val={best_val:.6f}  saved -> {info}")
    else:
        print(f"\nRunning per-station tuning with {parallel} parallel workers (total tasks: {total})")
        done = 0
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_tune_one_station, st, lt): (st, lt) for st, lt in tasks}
            for fut in as_completed(futures):
                st, lt = futures[fut]
                done += 1
                try:
                    st, lt, best_val, info = fut.result()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"\n[{done}/{total}] Station={st}  Loss={lt}  FAILED: {e}")
                    continue
                if best_val is None:
                    print(f"\n[{done}/{total}] Station={st}  Loss={lt}  {info}")
                    continue
                station_summary[(st, lt)] = best_val
                print(f"\n[{done}/{total}] Station={st}  Loss={lt}  BEST val={best_val:.6f}  saved -> {info}")

    # Print final summary table
    print(f"\n{'='*70}")
    print("  PER-STATION TUNING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Station':<22s}  {'Loss':<10s}  {'Best val':>10s}  {'HP dir'}")
    for (st, lt), val in sorted(station_summary.items()):
        hp_dir = base_out_dir / "per_station" / lt / st
        print(f"  {st:<22s}  {lt:<10s}  {val:>10.6f}  {hp_dir}")
    print(f"\nPer-station HPs saved under {base_out_dir}/per_station/<loss_type>/<station>/")
    print("Pass --hp-dir to 08_train_site_mlp.py pointing at a specific station dir.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="site_mlp_daily")
    parser.add_argument("--n-stations", type=int, default=5,
                        help="Stations sampled per trial (more = slower but more robust)")
    parser.add_argument("--loss-types", nargs="+", default=_ALL_LOSS_TYPES,
                        choices=_ALL_LOSS_TYPES,
                        help="Which loss functions to tune (one study per loss)")
    parser.add_argument("--tweedie-p", type=float, default=1.5,
                        help="Tweedie power parameter p in (1, 2) (default: 1.5)")
    parser.add_argument("--per-station-tuning", action="store_true",
                        help="Run a separate Optuna study per station (ignores --n-stations). "
                             "Saves per-station best HPs under <study>/per_station/<loss>/<station>/")
    parser.add_argument("--per-station-trials", type=int, default=30,
                        help="Trials per station when --per-station-tuning is set (default: 20)")
    parser.add_argument("--per-station-parallel", type=int, default=1,
                        help="When --per-station-tuning is set, number of stations to tune concurrently. "
                             "Uses threads; >1 may increase GPU/CPU/RAM usage.")
    parser.add_argument("--per-station-resume", action="store_true",
                        help="When --per-station-tuning is set, skip station/loss tasks that already have "
                             "best_hyperparameters.json saved under <study>/per_station/<loss>/<station>/.")
    parser.add_argument("--tune-arch-type", action="store_true",
                        help="Include arch_type (mlp vs glu) as a tunable HP in Optuna. "
                             "Adds ~2x trial cost but lets Optuna discover if GLU gating helps.")
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Time-based expanding-window CV folds per station during tuning (1 disables CV; default: 1)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — tuning will be slow on CPU.")

    tensors, meta = load_tensors_from_npz(device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    stations = meta["stations"]
    years = meta["years"]

    train_yr, val_yr, test_yr = compute_year_boundaries(years)
    yr_ranges = compute_station_year_ranges(stations, years)
    groups = assign_station_groups(
        sorted(set(str(s) for s in stations)),
        station_year_ranges=yr_ranges,
        val_years=val_yr, test_years=test_yr,
    )
    splits = spatiotemporal_split(stations, years, groups,
                                  train_years=train_yr, val_years=val_yr, test_years=test_yr)
    tensors, stats = normalize_tensors(tensors, splits["train"])
    target_scale = stats["target_std_mm"]
    metadata = _get_metadata(tensors)
    train_stations = sorted([s for s, r in groups.items() if r == "train"])
    all_stations = sorted(set(str(s) for s in stations))

    base_out_dir = config.TUNING_DIR / args.study_name
    base_out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_station_tuning:
        n_total = len(all_stations) * len(args.loss_types) * args.per_station_trials
        print(f"Per-station tuning: {len(all_stations)} stations × "
              f"{len(args.loss_types)} loss types × {args.per_station_trials} trials "
              f"= {n_total} total trials")
        _run_per_station_tuning(
            tensors, meta, all_stations, stations, years,
            metadata, stats, target_scale, base_out_dir, args,
        )
        return

    print(f"Loss types to tune: {args.loss_types}  |  Stations per trial: {args.n_stations}")
    summary = {}
    for loss_type in args.loss_types:
        best_val, best_hp = _run_one_study(
            loss_type, tensors, meta, train_stations, stations, years,
            metadata, stats, target_scale, base_out_dir, args,
            tweedie_p=args.tweedie_p,
        )
        summary[loss_type] = {"best_val": best_val, **best_hp}

    print(f"\n{'='*60}")
    print("  TUNING SUMMARY (best val loss per loss type)")
    print(f"{'='*60}")
    for lt, info in summary.items():
        arch_str = str(info.get("hidden_sizes", "?"))
        print(f"  {lt:12s}  val={info['best_val']:.6f}  arch={arch_str}  "
              f"bs={info.get('bs')}  lr={info.get('lr', 0):.2e}")
    print(f"\nResults saved under {base_out_dir}/<loss_type>/")
    print("Pass --hp-dir to 08_train_site_mlp.py to use a specific loss type's HPs.")


def _save_top10(all_df: pd.DataFrame, out_dir, title_suffix: str = "") -> None:
    """Save top-10 trials table and HP distribution violin/bar plots."""
    if all_df.empty:
        return

    top10 = all_df.nsmallest(10, "value").reset_index(drop=True)
    top10.to_csv(out_dir / "top10_trials.csv", index=False)
    print(f"\n--- Top 10 Trials{title_suffix} ---")
    print(top10.to_string(index=False))

    hp_cols = [c for c in top10.columns if c not in ("trial", "value")]
    if not hp_cols:
        return

    n_cols = min(4, len(hp_cols))
    n_rows = (len(hp_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).ravel()
    colours = plt.cm.viridis_r(np.linspace(0.1, 0.9, len(top10)))

    for i, col in enumerate(hp_cols):
        ax = axes[i]
        vals = top10[col].dropna()
        try:
            numeric_vals = vals.astype(float).values
            ax.violinplot(numeric_vals, positions=[0], showmedians=True)
            jitter = np.random.RandomState(0).uniform(-0.05, 0.05, len(numeric_vals))
            for j, (v, c) in enumerate(zip(numeric_vals, colours)):
                ax.scatter(jitter[j], v, color=c, s=40, zorder=3)
            ax.set_xticks([])
        except (ValueError, TypeError):
            vc = vals.value_counts()
            ax.bar(range(len(vc)), vc.values, tick_label=vc.index.tolist(),
                   color="steelblue")
            ax.tick_params(axis="x", rotation=30)
        ax.set_title(col, fontsize=9)

    for j in range(len(hp_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"HP Distribution — Top 10 Trials{title_suffix}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "hp_distribution_top10.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved hp_distribution_top10.png")


def _save_tuning_visuals(study, out_dir) -> None:
    """Save Optuna HP importance, optimization history, and parallel coordinate plots."""
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )

        fig = plot_param_importances(study)
        fig.figure.savefig(str(out_dir / "hp_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        print("  Saved hp_importance.png")

        fig = plot_optimization_history(study)
        fig.figure.savefig(str(out_dir / "optimization_history.png"), dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        print("  Saved optimization_history.png")

        fig = plot_parallel_coordinate(study)
        fig.figure.savefig(str(out_dir / "parallel_coordinate.png"), dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        print("  Saved parallel_coordinate.png")

        fig = plot_slice(study)
        target_fig = fig.figure if hasattr(fig, "figure") else fig
        target_fig.savefig(str(out_dir / "slice_plots.png"), dpi=150, bbox_inches="tight")
        plt.close(target_fig)
        print("  Saved slice_plots.png")

    except Exception as e:
        print(f"  WARNING: Could not generate tuning visuals: {e}")


if __name__ == "__main__":
    main()
