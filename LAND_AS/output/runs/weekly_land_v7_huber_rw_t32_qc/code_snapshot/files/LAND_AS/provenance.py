import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from LAND_AS import config


TRAIN_SOURCE_FILES = (
    "LAND_AS/train.py",
    "LAND_AS/parallelize.py",
    "LAND_AS/model.py",
    "LAND_AS/data.py",
    "LAND_AS/engine.py",
    "LAND_AS/evaluate.py",
    "LAND_AS/metrics.py",
    "LAND_AS/tune.py",
    "LAND_AS/config.py",
    "LAND_AS/prepare.py",
    "LAND_AS/provenance.py",
    "LAND_AS/daily_modeling/config.py",
    "LAND_AS/daily_modeling/data_utils/assemble_dataset.py",
    "LAND_AS/daily_modeling/data_utils/build_features.py",
    "LAND_AS/daily_modeling/data_utils/load_raw.py",
)

EVAL_SOURCE_FILES = (
    "LAND_AS/evaluate.py",
    "LAND_AS/model.py",
    "LAND_AS/data.py",
    "LAND_AS/engine.py",
    "LAND_AS/metrics.py",
    "LAND_AS/config.py",
    "LAND_AS/provenance.py",
)

PACKAGE_NAMES = (
    "numpy", "pandas", "scikit-learn", "torch", "optuna", "xarray", "scipy",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command):
    try:
        result = subprocess.run(
            ["git", *command], cwd=config.REPO_ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions():
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot_code(destination, source_files, command=None):
    """Copy the exact source files used for an experiment into its output dir.

    The first snapshot is preserved on reruns/resumes so an interrupted
    experiment retains the code that produced its earlier checkpoints.
    """
    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)

    copied = []
    for relative in source_files:
        source = config.REPO_ROOT / relative
        if not source.exists():
            continue
        target = destination / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({
            "path": relative,
            "sha256": _sha256(source),
        })

    status = _git(["status", "--porcelain"])
    dataset_path = Path(config.DATASET_PATH)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or sys.argv,
        "cwd": str(Path.cwd()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(status) if status is not None else None,
            "status": status,
        },
        "dataset": {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "sha256": _sha256(dataset_path) if dataset_path.exists() else None,
            "size_bytes": dataset_path.stat().st_size if dataset_path.exists() else None,
        },
        "packages": _package_versions(),
        "source_files": copied,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if isinstance(manifest["command"], list):
        (destination / "command.txt").write_text(" ".join(manifest["command"]))
    else:
        (destination / "command.txt").write_text(str(manifest["command"]))
    return destination
