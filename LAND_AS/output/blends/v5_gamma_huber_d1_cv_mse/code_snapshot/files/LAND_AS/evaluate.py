import argparse
import json

import numpy as np
import torch

from LAND_AS import config
from LAND_AS.data import crop_from_hp, load_data, loaders, model_metadata
from LAND_AS.engine import device, predict, save_json
from LAND_AS.metrics import extreme_metrics, regression_metrics
from LAND_AS.model import build_model
from LAND_AS.provenance import EVAL_SOURCE_FILES, snapshot_code


def main():
    parser = argparse.ArgumentParser(description="Evaluate a weekly LAND ensemble.")
    parser.add_argument("--run", default="weekly_land")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run_dir = config.RUNS_DIR / args.run
    hp = json.loads((run_dir / "hyperparameters.json").read_text())
    checkpoints = sorted(run_dir.glob("fold_*/seed_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    bundle = load_data()
    metadata = model_metadata(bundle, hp.get("climate_patch"), hp.get("climate_lag_weeks"), hp.get("rain_lag_weeks"))
    test_loaders = loaders(
        bundle,
        {name: index for name, index in bundle.splits.items() if name.startswith("test")},
        args.batch_size,
        shuffle_train=False,
        crop=crop_from_hp(hp),
    )
    output = run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", EVAL_SOURCE_FILES)
    target_device = device()

    for split_name, loader in test_loaders.items():
        predictions, observed = [], None
        for checkpoint in checkpoints:
            model = build_model(hp, metadata).to(target_device)
            model.load_state_dict(torch.load(checkpoint, map_location=target_device, weights_only=True)["state_dict"])
            current_observed, current_prediction = predict(model, loader, bundle.stats["target_scale"], target_device)
            observed = current_observed
            predictions.append(current_prediction)
        predictions = np.stack(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        metrics = regression_metrics(observed, mean)
        metrics.update(extreme_metrics(observed, mean))
        save_json(metrics, output / f"{split_name}_metrics.json")
        indices = bundle.splits[split_name]
        np.savez_compressed(
            output / f"{split_name}_predictions.npz",
            observed=observed,
            predicted=mean,
            uncertainty=std,
            stations=bundle.metadata["stations"][indices],
            years=bundle.metadata["years"][indices],
        )
    print(f"evaluation saved to {output}")


if __name__ == "__main__":
    main()
