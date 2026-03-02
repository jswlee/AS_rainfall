# `scripts/run_daily_pipeline.ps1` — End-to-End Pipeline Runner

## Purpose
PowerShell script that executes the entire 9-step pipeline in sequence with logging. Each step is run as a Python module (`python -m Daily_Modeling.scripts.XX_...`), with stdout/stderr captured to timestamped log files. If any step fails, the pipeline halts immediately with an error message pointing to the relevant log.

## Relation to the Deep Downscaling Paper
The paper does not describe a pipeline runner, but reproducibility requires a single command that regenerates all results from raw data. This script ensures that feature building, assembly, EDA, tuning, training, and evaluation are run in the correct order with consistent parameters.

## Line-by-Line Walkthrough

**Lines 1–6: Script configuration**
```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
```
- `Stop` mode means any unhandled error terminates the script.
- `StrictMode` catches common PowerShell mistakes (undefined variables, etc.).

**Lines 8–12: Environment setup**
```powershell
# $env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONHASHSEED = "0"
```
- GPU selection is commented out (defaults to GPU 0).
- `PYTHONHASHSEED=0` improves reproducibility by making Python's hash function deterministic.

**Lines 14–17: Path setup**
```powershell
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logDir = Join-Path $repoRoot "Daily_Modeling\output\logs"
```
The script locates the repo root by going two levels up from its own location (`Daily_Modeling/scripts/`). Log files go to `Daily_Modeling/output/logs/`.

**Lines 22–41: `Run-Step` function**
```powershell
function Run-Step([string]$name, [string]$module, [string]$extraArgs = "") {
    $cmd = "python -m $module $extraArgs"
    cmd /c "$cmd 1> `"$logPath`" 2>&1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```
Runs a Python module with arguments, captures all output to a timestamped log file, and checks the exit code. The `cmd /c` wrapper is used for reliable stdout+stderr redirection on Windows.

**Lines 47–73: Pipeline steps**

| Step | Module | Arguments | Purpose |
|------|--------|-----------|---------|
| 01 | `01_build_features` | — | Extract reanalysis + DEM features from raw data |
| 02 | `02_assemble_dataset` | — | Combine features + rainfall into single NPZ |
| 03 | `03_eda` | — | Exploratory data analysis plots |
| 03b | `03b_inspect_dataset` | — | Comprehensive dataset audit |
| 04 | `04_tune_land` | `--n-trials 60` | Optuna HP tuning for LAND |
| 05 | `05_tune_site_mlp` | `--n-trials 50` | Optuna HP tuning for Site MLP |
| 06 | `06_train_land` | `--run-name land_final --hp-dir ...` | Train LAND with best HPs |
| 07 | `07_train_bernoulli_gamma` | `--run-name bernoulli_gamma_final` | Train GLM baseline |
| 08 | `08_train_site_mlp` | `--run-name site_mlp_final --hp-dir ...` | Train per-station MLPs |
| 09 | `09_evaluate_compare` | — | Cross-model comparison |

## Architecture Decisions
- **Sequential execution**: Each step depends on the previous (e.g. step 06 needs step 04's output). No parallelism is attempted.
- **Fail-fast**: If any step fails, the pipeline stops immediately rather than continuing with stale data.
- **Timestamped logs**: Each run gets a unique log file, so you can compare logs across pipeline runs.
- **Tuning trials**: 60 for LAND (13-dimensional search space) and 50 for MLP (11-dimensional). Increased from original 30 to give TPE more exploration budget.

## Areas of Improvement
- **Selective re-runs**: Currently the pipeline runs all steps. A `--start-from` flag would allow re-running from step 04 onward (e.g. after changing HP search ranges) without re-extracting features.
- **Parallel tuning**: Steps 04 and 05 are independent — they could run in parallel on different GPUs.
- **Progress monitoring**: The log-to-file approach hides real-time progress. Adding `Tee-Object` to both log and display stdout would help.
- **Container/environment reproducibility**: A `Dockerfile` or `conda environment.yml` would ensure the same Python/CUDA versions across machines.
- **The `Run-Step` verb**: PowerShell's script analyzer warns about the unapproved verb `Run`. Renaming to `Invoke-Step` would silence this, though it's cosmetic.
