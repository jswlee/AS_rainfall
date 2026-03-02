# run_daily_pipeline.ps1
# Runs Daily_Modeling pipeline steps 01→09 with logging.
# Run from repo root:  .\run_daily_pipeline.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Optional: choose GPU (uncomment if needed)
# $env:CUDA_VISIBLE_DEVICES = "0"

# Make output deterministic-ish
$env:PYTHONHASHSEED = "0"

# Script lives in Daily_Modeling/scripts/ — go two levels up for repo root
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logDir = Join-Path $repoRoot "Daily_Modeling\output\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Ensure we run from repo root so python -m works
Push-Location $repoRoot

function Run-Step([string]$name, [string]$module, [string]$extraArgs = "") {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $logPath = Join-Path $logDir ("{0}_{1}.log" -f $name, $ts)

    Write-Host ""
    Write-Host "============================================================"
    Write-Host ("RUN {0}: python -m {1} {2}" -f $name, $module, $extraArgs)
    Write-Host ("LOG: {0}" -f $logPath)
    Write-Host "============================================================"

    $cmd = "python -m $module $extraArgs"
    # Use cmd.exe to reliably redirect stdout+stderr to file
    cmd /c "$cmd 1> `"$logPath`" 2>&1"

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host ("ERROR: Step {0} failed. See log: {1}" -f $name, $logPath)
        exit $LASTEXITCODE
    }
}

# -------------------------
# Steps
# -------------------------

# Step 01: Build features (reanalysis + DEM)
Run-Step "01_build_features" "Daily_Modeling.scripts.01_build_features"

# Step 02: Assemble dataset
Run-Step "02_assemble_dataset" "Daily_Modeling.scripts.02_assemble_dataset"

# Step 03: EDA
Run-Step "03_eda" "Daily_Modeling.scripts.03_eda"

# Step 03b: Dataset inspection / audit visuals + normalization verification
Run-Step "03b_inspect_dataset" "Daily_Modeling.scripts.03b_inspect_dataset"

# Step 04/05: hyperparameter tuning
Run-Step "04_tune_land" "Daily_Modeling.scripts.04_tune_land" "--n-trials 60"
Run-Step "05_tune_site_mlp" "Daily_Modeling.scripts.05_tune_site_mlp" "--n-trials 50"

# Step 06: Train LAND (using tuned HP from step 04)
Run-Step "06_train_land" "Daily_Modeling.scripts.06_train_land" "--run-name land_final --hp-dir Daily_Modeling\output\tuning\land_daily"

# Step 07: Train Bernoulli-Gamma GLM
Run-Step "07_train_bernoulli_gamma" "Daily_Modeling.scripts.07_train_bernoulli_gamma" "--run-name bernoulli_gamma_final"

# Step 08: Train Site MLP (using tuned HP from step 05)
Run-Step "08_train_site_mlp" "Daily_Modeling.scripts.08_train_site_mlp" "--run-name site_mlp_final --hp-dir Daily_Modeling\output\tuning\site_mlp_daily"

# Step 09: Compare all models
Run-Step "09_evaluate_compare" "Daily_Modeling.scripts.09_evaluate_compare"

Pop-Location

Write-Host ""
Write-Host "DONE: Pipeline completed successfully."
Write-Host ("Logs: {0}" -f $logDir)