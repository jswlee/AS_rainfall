# This script will robustly drop and recreate the 'optuna' database.

Write-Host "Attempting to drop the 'optuna' database if it exists..." -ForegroundColor Yellow

# The FIX: We use 'bash -c "..."' to ensure the command is parsed correctly inside the container.
# We also add 'IF EXISTS' to prevent errors if the database is already gone.
docker exec -it optuna-db bash -c "psql -U postgres -c 'DROP DATABASE IF EXISTS optuna;'"

# Give the server a moment.
Start-Sleep -Seconds 1

Write-Host "Creating a fresh 'optuna' database..." -ForegroundColor Green

# Apply the same 'bash -c' fix here for consistency.
docker exec -it optuna-db bash -c "psql -U postgres -c 'CREATE DATABASE optuna;'"

Write-Host "Database reset complete." -ForegroundColor Cyan