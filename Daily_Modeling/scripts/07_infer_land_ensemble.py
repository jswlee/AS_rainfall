"""Step 7: Run ensemble inference for a trained LAND run directory.

Thin CLI wrapper around ``Daily_Modeling.utils.inference.run_ensemble_inference_from_dir``.
All inference logic lives in that function; this script is kept only as a
convenient entry point for re-running inference on an already-trained run
without retraining.

Inference is also run automatically at the end of 06_train_land.py, so you
only need this script if you want to re-evaluate a previous run or use
different ``--splits`` / ``--batch-size`` options.

Usage:
  python -m Daily_Modeling.scripts.07_infer_land_ensemble \
    --run-dir Daily_Modeling/output/results/land_final
"""

import argparse
from pathlib import Path

from Daily_Modeling.utils.inference import run_ensemble_inference_from_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to trained run output dir (contains fold_*/model_seed*.pth, "
             "hyperparameters.json, normalization_stats.json)",
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
    parser.add_argument(
        "--wet-dry-threshold",
        type=float,
        default=1.0,
        help="Wet-day threshold in mm for wet/dry evaluation (default: 1.0)",
    )
    args = parser.parse_args()

    run_ensemble_inference_from_dir(
        run_dir=Path(args.run_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        splits=args.splits,
        batch_size=args.batch_size,
        wet_dry_threshold_mm=args.wet_dry_threshold,
    )


if __name__ == "__main__":
    main()
