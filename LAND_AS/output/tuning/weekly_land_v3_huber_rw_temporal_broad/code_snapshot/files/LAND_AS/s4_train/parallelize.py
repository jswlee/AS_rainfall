"""Parallel fold training for LAND_AS (worker pool behind s4_train/train.py).

Each fold is trained in a separate worker process. Workers load the dataset
once via a pool initializer and reuse it across tasks, so the feature arrays
are not pickled per fold. Folds whose checkpoints and histories already exist
are skipped, so a rerun resumes an interrupted run.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_BUNDLE = None  # per-process dataset cache, set by _init_worker


def _init_worker(freq="weekly"):
    """ProcessPoolExecutor initializer: load the dataset once per worker."""
    global _BUNDLE
    from LAND_AS.s2_dataset.data import load_data
    _BUNDLE = load_data(freq=freq)


def _train_fold(task, bundle=None):
    """Train every seed for one fold; returns (fold_number, held_out, scores)."""
    from LAND_AS import config
    from LAND_AS.s2_dataset.data import crop_from_hp, loaders, model_metadata, normalized_bundle
    from LAND_AS.s3_model.engine import device, fit, save_json
    from LAND_AS.s3_model.model import build_model

    fold_number, train_index, val_index, hp, seeds, epochs, patience, fold_output = task
    bundle = bundle if bundle is not None else _BUNDLE
    fold_output = Path(fold_output)
    held_out = sorted(set(bundle.metadata["stations"][val_index].astype(str)))[0]
    run_has_checkpoints = any(fold_output.parent.glob("fold_*/seed_*.pt"))
    run_has_norm_markers = any(
        fold_output.parent.glob("fold_*/seed_*_normalization.json")
    )
    legacy_normalization = run_has_checkpoints and not run_has_norm_markers
    fold_bundle = bundle if legacy_normalization else normalized_bundle(bundle, train_index)
    fold_loaders = loaders(
        fold_bundle, {"train": train_index, "val": val_index}, hp["batch_size"],
        crop=crop_from_hp(hp), balance_stations=hp.get("balanced_stations", False),
    )
    metadata = model_metadata(fold_bundle, hp.get("climate_patch"), hp.get("climate_lag_weeks"), hp.get("rain_lag_weeks"), dem_channels=hp.get("dem_channels"))
    target_device = device()
    print(f"Fold {fold_number}: held-out station = {held_out} ({len(val_index)} samples) on {target_device}")
    scores = []
    for seed_number in range(seeds):
        seed = config.SEED + seed_number
        history_path = fold_output / f"seed_{seed}_history.json"
        normalization_path = fold_output / f"seed_{seed}_normalization.json"
        if not legacy_normalization and not normalization_path.exists():
            save_json(fold_bundle.stats, normalization_path)
        if (fold_output / f"seed_{seed}.pt").exists() and history_path.exists():
            scores.append(json.loads(history_path.read_text())["best_val_mae_mm"])
            print(f"Fold {fold_number} seed {seed}: already trained -- skipping")
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(hp, metadata)
        fit_kwargs = {}
        if hp.get("model_type") == "bern_gamma":
            # pos_weight for the occurrence BCE: fold-local class ratio
            fold_target = fold_bundle.arrays["target"][train_index]
            n_wet = int((fold_target > 0).sum())
            fit_kwargs["dry_wet_ratio"] = (len(train_index) - n_wet) / max(n_wet, 1)
            fit_kwargs["lambda_bce"] = hp.get("lambda_bce", 1.0)
        history, score = fit(
            model, fold_loaders["train"], fold_loaders["val"], epochs, patience,
            hp["learning_rate"], hp["weight_decay"], fold_bundle.stats["target_scale"], target_device,
            rainfall_weight=hp.get("rainfall_weight", False),
            monitor=hp.get("monitor", "mae"), min_epochs=hp.get("min_epochs", 0),
            huber_delta=hp.get("huber_delta", 0.5), loss_type=hp.get("loss_type"),
            **fit_kwargs,
        )
        torch.save({"state_dict": model.state_dict(), "hyperparameters": hp}, fold_output / f"seed_{seed}.pt")
        save_json({**history, "held_out_station": held_out}, history_path)
        scores.append(history["best_val_mae_mm"])
    return fold_number, held_out, scores


def _fold_complete(fold_output, seeds):
    """True when every seed checkpoint and history file for the fold exists."""
    from LAND_AS import config
    return all(
        (fold_output / f"seed_{config.SEED + s}.pt").exists()
        and (fold_output / f"seed_{config.SEED + s}_history.json").exists()
        for s in range(seeds)
    )


def train_folds(bundle, folds, hp, seeds, epochs, patience, output, workers, trainer=None):
    """Train all folds; in parallel worker processes when ``workers`` > 1.

    ``trainer`` is an optional fold-training callable with the same contract
    as ``_train_fold``."""
    trainer = trainer or _train_fold
    tasks = []
    for fold_number, (train_index, val_index) in enumerate(folds):
        fold_output = output / f"fold_{fold_number}"
        fold_output.mkdir(parents=True, exist_ok=True)
        if _fold_complete(fold_output, seeds):
            held_out = sorted(set(bundle.metadata["stations"][val_index].astype(str)))[0]
            print(f"Fold {fold_number}: {held_out} already trained -- skipping")
            continue
        tasks.append((fold_number, train_index, val_index, hp, seeds, epochs, patience, str(fold_output)))

    if not tasks:
        print("All folds already complete.")
        return

    if workers <= 1:
        for task in tasks:
            fold_number, held_out, scores = trainer(task, bundle)
            print(f"Fold {fold_number} done ({held_out}): best val MAE {min(scores):.2f} mm")
        return

    failures = []
    freq = bundle.metadata.get("freq", "weekly")
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(freq,))
    try:
        futures = {pool.submit(trainer, task): task[0] for task in tasks}
        for future in as_completed(futures):
            fold_number = futures[future]
            try:
                _, held_out, scores = future.result()
                print(f"Fold {fold_number} done ({held_out}): best val MAE {min(scores):.2f} mm")
            except Exception as exc:
                failures.append(fold_number)
                print(f"Fold {fold_number} FAILED: {exc}")
    except KeyboardInterrupt:
        # Exit promptly on Ctrl+C; workers are daemonized and die with the
        # parent, and every completed fold is checkpointed for resume.
        print("Interrupted -- completed folds are kept; rerun to resume.")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    if failures:
        raise RuntimeError(f"Folds failed: {failures}")
