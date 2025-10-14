#!/usr/bin/env python3
"""
Delete an Optuna study (and all its trials) by name from a database storage.

Usage examples:
  # Prompt for confirmation
  python -m Hyperparameter_Tuning.delete_study --study-name my_study --db-url postgresql+psycopg2://user:pass@host:5432/db

  # Non-interactive (CI-safe) deletion
  python -m Hyperparameter_Tuning.delete_study --study-name my_study --db-url postgresql+psycopg2://user:pass@host:5432/db --force

Notes:
- This is a destructive operation. It permanently removes the study and all trials from the storage.
- The DB URL must be a valid Optuna storage URL (e.g., sqlite:///optuna.db or postgresql+psycopg2://...)
"""

import argparse
import sys
from typing import Optional

import optuna


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt + " [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


essential_msg = (
    "This operation is irreversible. It will permanently remove the study and all related trials from the storage."
)


def resolve_study(storage_url: str, study_name: str) -> Optional[optuna.study.StudySummary]:
    """Return the StudySummary for a study name if it exists, else None."""
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    except Exception as e:
        print(f"Error connecting to storage: {e}", file=sys.stderr)
        return None
    for s in summaries:
        if s.study_name == study_name:
            return s
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete an Optuna study by name from storage")
    parser.add_argument("--study-name", required=True, help="Name of the study to delete")
    parser.add_argument(
        "--db-url",
        default="postgresql://postgres:mysecretpassword@localhost:5432/optuna_daily",
        help=(
            "Optuna storage URL, e.g. 'postgresql+psycopg2://user:pass@host:5432/db' or 'sqlite:///optuna.db'"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not prompt for confirmation (dangerous).",
    )

    args = parser.parse_args()

    study_name = args.study_name
    storage_url = args.db_url

    summary = resolve_study(storage_url, study_name)
    if summary is None:
        print(f"Study '{study_name}' not found or storage not accessible.")
        return 1

    print("Study found:")
    print(f"  Name:        {summary.study_name}")
    print(f"  Trials:      {summary.n_trials}")
    print(f"  Direction:   {getattr(summary.direction, 'name', 'n/a')}")
    if getattr(summary, 'best_trial', None) is not None:
        try:
            print(f"  Best Value:  {summary.best_trial.value}")
        except Exception:
            print("  Best Value:  n/a")
    else:
        print("  Best Value:  n/a")

    print()
    print(essential_msg)

    if not args.force:
        if not confirm(f"Are you sure you want to delete study '{study_name}' from {storage_url}?"):
            print("Aborted.")
            return 0

    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Failed to delete study: {e}", file=sys.stderr)
        return 2

    print(f"Study '{study_name}' has been deleted successfully from storage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
