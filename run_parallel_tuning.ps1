# --- Configuration ---
# Set how many parallel processes you want to run.
$NumWorkers = 4
$TotalTrials = 50
$OutputDir = "output/tuning_run_1"
# ---------------------

echo "Starting $NumWorkers parallel workers for a total of $TotalTrials trials..."

# Loop to start each worker process in a new PowerShell window.
for ($i=0; $i -lt $NumWorkers; $i++) {
    $workerId = $i + 1
    echo "Starting Worker $workerId..."
    
    # Define the arguments for the python script
    $arguments = @(
        "-m", "Hyperparameter_Tuning.hp_tuning_simplified",
        "--output-dir", $OutputDir,
        "--n-trials", $TotalTrials,
        "--resume"
    )
    
    # Use Start-Process to launch the script in a new, independent window.
    Start-Process python -ArgumentList $arguments
}

echo "All workers have been launched in separate windows."