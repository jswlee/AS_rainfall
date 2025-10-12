#!/bin/bash

# Stops the existing database container, ignoring errors if it doesn't exist.
docker stop optuna-db > /dev/null 2>&1

# Removes the stopped container, ignoring errors.
docker rm optuna-db > /dev/null 2>&1

# A small pause to ensure ports are freed up.
sleep 2

# Start a fresh PostgreSQL server in the background.
docker run --name optuna-db -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -d postgres

# Wait for the database server to initialize.
echo "Waiting for the database server to start..."
sleep 10

# Create the 'optuna' databases if missing.
echo "Creating the 'optuna' databases if missing..."

# Create optuna_monthly if missing
exists_monthly=$(docker exec optuna-db psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='optuna_monthly'" 2>/dev/null)
if [ -z "$exists_monthly" ]; then
    docker exec optuna-db psql -U postgres -c "CREATE DATABASE optuna_monthly;"
    echo "Created database: optuna_monthly"
else
    echo "Database already exists: optuna_monthly"
fi

# Create optuna_daily if missing
exists_daily=$(docker exec optuna-db psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='optuna_daily'" 2>/dev/null)
if [ -z "$exists_daily" ]; then
    docker exec optuna-db psql -U postgres -c "CREATE DATABASE optuna_daily;"
    echo "Created database: optuna_daily"
else
    echo "Database already exists: optuna_daily"
fi

echo "Database setup complete."
