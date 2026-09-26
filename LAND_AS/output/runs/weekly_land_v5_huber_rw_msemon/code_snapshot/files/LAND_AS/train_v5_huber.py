import argparse
import json
import os

from LAND_AS import config
from LAND_AS.data import cv_folds, load_data
from LAND_AS.engine import device, save_json
from LAND_AS.parallelize import train_folds
from LAND_AS.provenance import TRAIN_SOURCE_FILES, snapshot_code


def main():
    parser = argparse.ArgumentParser(
        description="Train a Huber-output LAND using the exact weekly_land_v5 size, features, and LOSO protocol."
    )
    parser.add_argument(
        "--source-hyperparameters",
        default=str(config.RUNS_DIR / "weekly_land_v5" / "hyperparameters.json"),
        help="JSON containing the frozen v5 architecture/training configuration.",
    )
    parser.add_argument("--run", default="weekly_land_v5_huber")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--loss-type", choices=["huber", "huber_weighted"], default="huber")
    parser.add_argument("--monitor", choices=["mae", "mse"], default="mae")
    parser.add_argument("--balanced-stations", action="store_true",
                        help="sample training weeks with equal station representation")
    args = parser.parse_args()

    source_path = config.REPO_ROOT / args.source_hyperparameters
    hp = json.loads(source_path.read_text())
    hp.update({
        "model_type": "huber",
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "monitor": args.monitor,
        "balanced_stations": args.balanced_stations,
        "source_run": "weekly_land_v5",
    })

    bundle = load_data()
    folds = cv_folds(bundle, count=args.folds, mode="loso")
    output = config.RUNS_DIR / args.run
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(
        output / "code_snapshot",
        (*TRAIN_SOURCE_FILES, "LAND_AS/train_v5_huber.py"),
    )
    workers = max(1, min(args.workers, os.cpu_count() or 1, len(folds)))
    print(f"using device: {device()}")
    if workers > 1 and device().type == "cuda":
        print("warning: parallel workers share one GPU; consider --workers 1")
    print(f"v5-sized Huber LOSO folds: {len(folds)} ({workers} workers)")

    save_json(hp, output / "hyperparameters.json")
    save_json(bundle.stats, output / "normalization.json")
    save_json({"station_roles": bundle.metadata["roles"]}, output / "split.json")
    save_json({
        "experiment": "v5-sized scalar Huber",
        "source_hyperparameters": str(source_path),
        "changes_from_v5": {
            "model_type": "huber",
            "loss": args.loss_type,
            "huber_delta": args.huber_delta,
            "monitor": args.monitor,
            "balanced_stations": args.balanced_stations,
            "rainfall_weight": "retained in configuration but used only by the Gamma head",
        },
        "folds": len(folds),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": workers,
    }, output / "experiment.json")

    train_folds(bundle, folds, hp, args.seeds, args.epochs, args.patience, output, workers)
    print(f"models saved to {output}")


if __name__ == "__main__":
    main()
