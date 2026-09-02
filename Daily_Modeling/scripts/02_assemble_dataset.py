"""
Step 2: Assemble reanalysis patches, DEM patches, rainfall, and month
one-hot into a single NPZ ready for modelling.

Reads the intermediate NPZ files produced by step 01.

Usage:
    python -m Daily_Modeling.scripts.02_assemble_dataset
    python -m Daily_Modeling.scripts.02_assemble_dataset --freq weekly
"""

import argparse

from Daily_Modeling import config
from Daily_Modeling.data_utils.assemble_dataset import assemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freq", choices=["daily", "weekly"], default=config.FREQ,
        help="Temporal resolution of the assembled dataset.  'weekly' sums rainfall "
             "over ISO calendar weeks (Mon-Sun) and reduces each reanalysis channel "
             "to its within-week mean and std (default: %(default)s)",
    )
    parser.add_argument(
        "--min-days-per-week", type=int, default=config.WEEKLY_MIN_DAYS,
        help="Minimum daily records required to keep a week when --freq weekly "
             "(default: %(default)s)",
    )
    args = parser.parse_args()
    assemble(freq=args.freq, min_days_per_week=args.min_days_per_week)
    if args.freq != config.FREQ:
        print(f"\nNOTE: AS_RAINFALL_FREQ is '{config.FREQ}'.  Set it to '{args.freq}' "
              f"so the downstream scripts read this dataset:\n"
              f"  PowerShell:  $env:AS_RAINFALL_FREQ = \"{args.freq}\"\n"
              f"  bash:        export AS_RAINFALL_FREQ={args.freq}")


if __name__ == "__main__":
    main()
