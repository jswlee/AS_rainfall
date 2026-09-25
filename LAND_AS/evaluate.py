import argparse
import json

import numpy as np
import torch

from LAND_AS import config
from LAND_AS.data import load_data, loaders, model_metadata
from LAND_AS.engine import device, predict, save_json
from LAND_AS.metrics import regression_metrics
from LAND_AS.model import LAND


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
    metadata = model_metadata(bundle)
    test_loaders = loaders(
        bundle,
        {name: index for name, index in bundle.splits.items() if name.startswith("test")},
        args.batch_size,
        shuffle_train=False,
    )
    output = run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    target_device = device()

    for split_name, loader in test_loaders.items():
        predictions, observed = [], None
        for checkpoint in checkpoints:
            model = LAND(**metadata, **{key: hp[key] for key in ("climate_units", "dem_units", "month_units", "hidden_units", "dropout")}).to(target_device)
            model.load_state_dict(torch.load(checkpoint, map_location=target_device, weights_only=True)["state_dict"])
            current_observed, current_prediction = predict(model, loader, bundle.stats["target_scale"], target_device)
            observed = current_observed
            predictions.append(current_prediction)
        predictions = np.stack(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        save_json(regression_metrics(observed, mean), output / f"{split_name}_metrics.json")
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
