# Reset the 'optuna_daily' database.

$ErrorActionPreference = 'Stop'

Write-Host "Waiting for Postgres readiness..." -ForegroundColor Yellow
$maxWait = 20
$waited = 0
while ($waited -lt $maxWait) {
    $ready = docker exec optuna-db pg_isready -U postgres 2>$null
    if ($ready -match "accepting connections") {
        Write-Host "Postgres is ready." -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 2
    $waited += 2
}
if ($waited -ge $maxWait) {
    throw "Postgres not ready after $maxWait seconds."
}

Write-Host "Terminating connections to 'optuna_daily' (if any)..." -ForegroundColor Yellow
# Prevent new connects
docker exec optuna-db psql -U postgres -c "REVOKE CONNECT ON DATABASE optuna_daily FROM PUBLIC;" 2>$null
# Terminate current sessions (exclude our own psql)
docker exec optuna-db psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='optuna_daily' AND pid <> pg_backend_pid();" 2>$null

Write-Host "Dropping the 'optuna_daily' database if it exists..." -ForegroundColor Yellow
# Retry a few times in case backends take a moment to die
$maxTries = 5
for ($i=1; $i -le $maxTries; $i++) {
    $rc = docker exec optuna-db psql -U postgres -c "DROP DATABASE IF EXISTS optuna_daily;" 2>$null; $exit=$LASTEXITCODE
    if ($exit -eq 0) { break }
    Start-Sleep -Seconds 1
}

# Small delay to ensure catalog update
Start-Sleep -Seconds 1

Write-Host "Creating a new 'optuna_daily' database..." -ForegroundColor Yellow
docker exec optuna-db psql -U postgres -c "CREATE DATABASE optuna_daily;"

# Optionally re-allow PUBLIC connects (default posture)
docker exec optuna-db psql -U postgres -c "GRANT CONNECT ON DATABASE optuna_daily TO PUBLIC;" 2>$null

Write-Host "Verifying database exists..." -ForegroundColor Yellow
$exists = docker exec optuna-db psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='optuna_daily'" 2>$null
if ([string]::IsNullOrWhiteSpace($exists)) {
    throw "Verification failed: 'optuna_daily' was not created."
} else {
    Write-Host "'optuna_daily' is ready." -ForegroundColor Green
}