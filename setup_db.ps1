# Stops the existing database container, ignoring errors if it doesn't exist.
docker stop optuna-db | Out-Null

# Removes the stopped container, ignoring errors.
docker rm optuna-db | Out-Null

# A small pause to ensure ports are freed up.
Start-Sleep -Seconds 2

# Start a fresh PostgreSQL server in the background.
docker run --name optuna-db -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -d postgres

# Wait for the database server to initialize.
echo "Waiting for the database server to start..."
Start-Sleep -Seconds 10

# Create the 'optuna' databases if missing.
echo "Creating the 'optuna' databases if missing..."

# Create optuna_monthly if missing
$existsMonthly = docker exec optuna-db psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='optuna_monthly'" 2>$null
if ([string]::IsNullOrWhiteSpace($existsMonthly)) {
    docker exec optuna-db psql -U postgres -c "CREATE DATABASE optuna_monthly;"
    Write-Host "Created database: optuna_monthly"
} else {
    Write-Host "Database already exists: optuna_monthly"
}

# Create optuna_daily if missing
$existsDaily = docker exec optuna-db psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='optuna_daily'" 2>$null
if ([string]::IsNullOrWhiteSpace($existsDaily)) {
    docker exec optuna-db psql -U postgres -c "CREATE DATABASE optuna_daily;"
    Write-Host "Created database: optuna_daily"
} else {
    Write-Host "Database already exists: optuna_daily"
}

echo "Database setup complete."