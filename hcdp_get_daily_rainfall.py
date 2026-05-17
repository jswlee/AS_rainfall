import requests
import dotenv
import os
import json
import time
from pathlib import Path

dotenv.load_dotenv()

# Base URL for the mesonet API
base_url = "https://api.hcdp.ikewai.org/mesonet/db/measurements"

# ALL rainfall variables to check (both sensors, all intervals)
RAINFALL_VARS = [
    # Daily totals (best for historical)
    "RF_1_Tot86400s", "RF_2_Tot86400s",
    # Hourly
    "RF_1_Tot3600s", "RF_2_Tot3600s",
    # 30-minute
    "RF_1_Tot1800s", "RF_2_Tot1800s",
    # 15-minute
    "RF_1_Tot900s", "RF_2_Tot900s",
    # 10-minute
    "RF_1_Tot600s", "RF_2_Tot600s",
    # 5-minute
    "RF_1_Tot300s", "RF_2_Tot300s",
    # 1-minute (highest resolution)
    "RF_1_Tot60s", "RF_2_Tot60s",
]

# Cache directory for raw station data
CACHE_DIR = Path("hcdp_station_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Final output file
OUTPUT_FILE = "hcdp_rainfall_all_vars.json"

# Load all station IDs from hcdp_stations.json
with open("hcdp_stations.json") as f:
    stations = json.load(f)

# Headers with authorization
headers = {
    "Authorization": f"Bearer {os.getenv('HCDP_API_KEY')}"
}

def fetch_station_variable(station_id, var_id, production="new", max_retries=3):
    """Fetch data for a single station and variable."""
    params = {
        "station_ids": station_id,
        "start_date": "1990-01-01T00:00:00-10:00",
        "end_date": "2024-12-31T23:59:59-10:00",
        "var_ids": var_id,
        "location": "hawaii",
        "limit": 1_000_000,  # Max allowed
        "join_metadata": "true",
        "row_mode": "json",
        "production": production,  # "new" or "legacy"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, list):
                    return data
                return []
            elif response.status_code in [502, 503, 504]:
                wait_time = (attempt + 1) * 2
                print(f"        Server error {response.status_code}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Variable likely doesn't exist for this station
                return []
        except requests.exceptions.Timeout:
            wait_time = (attempt + 1) * 3
            print(f"        Timeout, retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"        Error: {e}")
            return []
    
    return []

def get_cache_path(station_id):
    """Get cache file path for a station (stores all vars together)."""
    return CACHE_DIR / f"{station_id}_all_rf.json"

def load_cached_data(station_id):
    """Load cached data if it exists."""
    cache_file = get_cache_path(station_id)
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_data(station_id, data):
    """Save data to cache."""
    cache_file = get_cache_path(station_id)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

def fetch_station_all_vars(station):
    """Fetch all rainfall variables for a station."""
    station_id = station["station_id"]
    station_name = station.get("name", station_id)
    
    # Check cache first
    cached = load_cached_data(station_id)
    if cached is not None:
        total_records = sum(len(v.get("records", [])) for v in cached.get("variables", {}).values())
        print(f"  [{station_id}] {station_name}: Using cached data ({total_records} records)")
        return cached
    
    print(f"  [{station_id}] {station_name}: Checking all rainfall variables...")
    
    station_data = {
        "station_id": station_id,
        "station_name": station_name,
        "metadata": {
            "lat": station.get("lat"),
            "lng": station.get("lng"),
            "elevation": station.get("elevation"),
            "skn": station.get("skn"),
            "nws_id": station.get("nws_id"),
        },
        "variables": {}
    }
    
    found_any = False
    
    for var_id in RAINFALL_VARS:
        # Try both production methods
        all_records = []
        production_sources = []
        
        for production in ["new", "legacy"]:
            records = fetch_station_variable(station_id, var_id, production)
            
            if records:
                # Tag each record with its production source
                for r in records:
                    r["_production"] = production
                all_records.extend(records)
                production_sources.append(production)
        
        if all_records:
            found_any = True
            print(f"      {var_id}: {len(all_records):,} records (production: {', '.join(production_sources)})")
            
            # Store minimal record info (timestamp, value, flag, production)
            station_data["variables"][var_id] = {
                "record_count": len(all_records),
                "interval_seconds": all_records[0].get("interval_seconds") if all_records else None,
                "units": all_records[0].get("units") if all_records else None,
                "production": production_sources,
                "records": [
                    {
                        "timestamp": r["timestamp"],
                        "value": r["value"],
                        "flag": r.get("flag", 0),
                        "production": r["_production"]
                    }
                    for r in all_records
                ]
            }
        
        # Small delay between variables
        time.sleep(0.1)
    
    if found_any:
        save_cached_data(station_id, station_data)
        total = sum(len(v["records"]) for v in station_data["variables"].values())
        print(f"    -> Total: {total:,} records across {len(station_data['variables'])} variables")
    else:
        print(f"    -> No rainfall data available for this station")
    
    return station_data if found_any else None

# Main execution
print(f"Fetching ALL rainfall variables for {len(stations)} stations...")
print(f"Variables to check: {', '.join(RAINFALL_VARS)}")
print(f"Cache directory: {CACHE_DIR.absolute()}")
print()

all_station_data = []

for i, station in enumerate(stations, 1):
    station_data = fetch_station_all_vars(station)
    if station_data:
        all_station_data.append(station_data)
    
    # Progress indicator every 10 stations
    if i % 10 == 0 or i == len(stations):
        print(f"  ... Progress: {i}/{len(stations)} stations checked")
    
    # Small delay between stations
    if i < len(stations):
        time.sleep(0.3)

print()
print("=" * 60)
print(f"Fetch complete!")
print(f"  Stations with rainfall data: {len(all_station_data)}/{len(stations)}")

# Summary by variable
var_counts = {}
for sd in all_station_data:
    for var_id in sd.get("variables", {}):
        var_counts[var_id] = var_counts.get(var_id, 0) + 1

if var_counts:
    print(f"\n  Variable availability:")
    for var_id, count in sorted(var_counts.items()):
        print(f"    {var_id}: {count} stations")

# Save consolidated output
if all_station_data:
    output = {
        "metadata": {
            "fetch_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_stations": len(all_station_data),
            "variables_checked": RAINFALL_VARS,
        },
        "stations": all_station_data
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.1f} MB")
else:
    print("\n  No data fetched.")
