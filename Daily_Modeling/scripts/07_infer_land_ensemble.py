"""Step 7: Run ensemble inference for a trained LAND run directory.

Loads all fold_*/model_seed*.pth checkpoints from a training run output folder and
runs inference on the spatiotemporally distinct test splits (test_temporal and
test_spatial).

Usage:
  python -m Daily_Modeling.scripts.07_infer_land_ensemble \
    --run-dir Daily_Modeling/output/results/land_bg_spatial_temporal
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, make_dataloaders, get_dataset_metadata
from Daily_Modeling.data_utils.splits import (
    assign_station_groups,
    compute_station_year_ranges,
    compute_year_boundaries,
    spatiotemporal_split,
)
from Daily_Modeling.models.land import create_land_model
from Daily_Modeling.utils.io_utils import load_json, load_model_state, save_json
from Daily_Modeling.utils.metrics import compute_metrics
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.utils.inference import predict


def _apply_saved_normalization(tensors: Dict[str, torch.Tensor], stats: dict) -> Tuple[Dict[str, torch.Tensor], float]:
    """Apply saved normalization parameters in-place; returns (tensors, target_scale)."""
    device = tensors["climate"].device

    cm = torch.tensor(stats["climate_mean"], device=device, dtype=tensors["climate"].dtype)
    cs = torch.tensor(stats["climate_std"], device=device, dtype=tensors["climate"].dtype)
    tensors["climate"] = (tensors["climate"] - cm[None, :, None, None]) / cs[None, :, None, None]

    for key in ("local_dem", "regional_dem"):
        m = torch.tensor(stats[f"{key}_mean"], device=device, dtype=tensors[key].dtype)
        s = torch.tensor(stats[f"{key}_std"], device=device, dtype=tensors[key].dtype)
        tensors[key] = (tensors[key] - m) / s

    target_scale = float(stats["target_std_mm"])
    return tensors, target_scale


def _discover_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = sorted(run_dir.glob("fold_*/model_seed*.pth"))
    if len(ckpts) == 0:
        # also support non-CV runs
        ckpts = sorted(run_dir.glob("model_seed*.pth"))
    return ckpts


def _save_ensemble_npz(path: Path, y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, stations: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        y_true=y_true,
        y_pred_mean=y_mean,
        y_pred_std=y_std,
        stations=stations,
    )


def _concat_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(a) for a in arrays if a is not None and len(a) > 0]
    if not valid:
        return np.array([])
    return np.concatenate(valid, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to trained run output dir (contains fold_*/model_seed*.pth, hyperparameters.json, normalization_stats.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for inference results (default: <run-dir>/inference)",
    )
    parser.add_argument(
        "--splits",
        default="both",
        choices=["temporal", "spatial", "both"],
        help="Which test split(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference dataloaders (default: 512)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Device: {device}")

    hp = load_json(run_dir / "hyperparameters.json")
    stats = load_json(run_dir / "normalization_stats.json")
    station_groups_payload = load_json(run_dir / "station_groups.json")
    groups = station_groups_payload.get("station_groups", station_groups_payload)

    tensors, meta = load_tensors_from_npz(device=device)
    stations = meta["stations"]
    years = meta["years"]

    train_yr, val_yr, test_yr = compute_year_boundaries(years)
    yr_ranges = compute_station_year_ranges(stations, years)

    # Rebuild splits using the saved station groups so we match the training run.
    # (This also ensures test sets are spatially/temporally distinct.)
    # If station_groups.json is missing/empty, fall back to deterministic assignment.
    if not groups:
        groups = assign_station_groups(
            sorted(set(str(s) for s in stations)),
            station_year_ranges=yr_ranges,
            val_years=val_yr,
            test_years=test_yr,
        )

    splits = spatiotemporal_split(
        stations,
        years,
        groups,
        train_years=train_yr,
        val_years=val_yr,
        test_years=test_yr,
    )

    tensors, target_scale = _apply_saved_normalization(tensors, stats)
    metadata = get_dataset_metadata(tensors)

    dem_crop = config.resolve_dem_crop(hp)
    if dem_crop is not None:
        lp = dem_crop["local_patch_size"]
        rp = dem_crop["regional_patch_size"]
        metadata["local_dem_shape"] = (lp, lp)
        metadata["regional_dem_shape"] = (rp, rp)
        print(f"DEM crop: local={lp}x{lp}@{dem_crop['local_km']}km  regional={rp}x{rp}@{dem_crop['regional_km']}km")

    # Build test loaders
    split_indices: Dict[str, np.ndarray] = {}
    if args.splits in ("temporal", "both"):
        split_indices["test_temporal"] = splits.get("test_temporal", np.array([], dtype=int))
    if args.splits in ("spatial", "both"):
        split_indices["test_spatial"] = splits.get("test_spatial", np.array([], dtype=int))

    if all(len(v) == 0 for v in split_indices.values()):
        raise RuntimeError(f"No test indices found for splits={args.splits}")

    loaders = make_dataloaders(
        tensors,
        split_indices,
        target_scale=target_scale,
        batch_size=args.batch_size,
        dem_crop_config=dem_crop,
    )

    ckpts = _discover_checkpoints(run_dir)
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoints found under {run_dir}")
    print(f"Found {len(ckpts)} checkpoints")

    output_head = hp.get("output_head", "softplus")

    split_outputs = {}
    # Predict per model per split
    for split_name, loader in loaders.items():
        model_preds_mm = []
        yt_mm_ref = None

        for ckpt_path in ckpts:
            model = create_land_model(hp, metadata).to(device)
            model = load_model_state(ckpt_path, model)
            yp, yt = predict(model, loader, device, output_head=output_head)
            yp_mm = yp * target_scale
            yt_mm = yt * target_scale

            model_preds_mm.append(yp_mm)
            if yt_mm_ref is None:
                yt_mm_ref = yt_mm

        preds_stack = np.stack(model_preds_mm, axis=0)  # (M, N)
        yp_mean = preds_stack.mean(axis=0)
        yp_std = preds_stack.std(axis=0)

        m = compute_metrics(yt_mm_ref, yp_mean)
        save_json(m, out_dir / f"metrics_{split_name}.json")
        print(
            f"{split_name}: RMSE={m['rmse']:.2f} mm  MAE={m['mae']:.2f} mm  R2={m['r2']:.4f}  "
            f"(models={len(ckpts)})"
        )

        idx = split_indices[split_name]
        _save_ensemble_npz(
            out_dir / f"predictions_{split_name}.npz",
            y_true=yt_mm_ref,
            y_mean=yp_mean,
            y_std=yp_std,
            stations=stations[idx] if len(idx) > 0 else np.array([]),
        )

        split_outputs[split_name] = {
            "y_true": yt_mm_ref,
            "y_pred_mean": yp_mean,
            "y_pred_std": yp_std,
            "stations": stations[idx] if len(idx) > 0 else np.array([]),
        }

    if {"test_temporal", "test_spatial"}.issubset(split_outputs.keys()):
        yt_all = _concat_arrays([
            split_outputs["test_temporal"]["y_true"],
            split_outputs["test_spatial"]["y_true"],
        ])
        yp_all = _concat_arrays([
            split_outputs["test_temporal"]["y_pred_mean"],
            split_outputs["test_spatial"]["y_pred_mean"],
        ])
        yp_std_all = _concat_arrays([
            split_outputs["test_temporal"]["y_pred_std"],
            split_outputs["test_spatial"]["y_pred_std"],
        ])
        stations_all = _concat_arrays([
            split_outputs["test_temporal"]["stations"],
            split_outputs["test_spatial"]["stations"],
        ])

        if len(yt_all) > 0:
            m_all = compute_metrics(yt_all, yp_all)
            save_json(m_all, out_dir / "metrics_test_all.json")
            print(
                f"test_all: RMSE={m_all['rmse']:.2f} mm  MAE={m_all['mae']:.2f} mm  R2={m_all['r2']:.4f}  "
                f"(n={len(yt_all)})"
            )
            _save_ensemble_npz(
                out_dir / "predictions_test_all.npz",
                y_true=yt_all,
                y_mean=yp_all,
                y_std=yp_std_all,
                stations=stations_all,
            )

    # Save a small manifest for reproducibility
    manifest = {
        "run_dir": str(run_dir),
        "n_models": int(len(ckpts)),
        "splits": args.splits,
        "output_head": output_head,
        "target_scale_mm": float(target_scale),
        "checkpoints": [str(p.relative_to(run_dir)) for p in ckpts],
    }
    save_json(manifest, out_dir / "inference_manifest.json")
    print(f"\nSaved inference outputs to: {out_dir}")


if __name__ == "__main__":
    main()
