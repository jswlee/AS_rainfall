import requests
import dotenv
import os
import json

dotenv.load_dotenv()

# Base URL for the mesonet API
base_url = "https://api.hcdp.ikewai.org/mesonet/db/variables"


# Query parameters
params = {
    "limit": 1_000_000,
    "location": "hawaii"
}

# Headers with authorization
headers = {
    "Authorization": f"Bearer {os.getenv('HCDP_API_KEY')}"
}

# Make the GET request
response = requests.get(base_url, params=params, headers=headers)

# Check if request was successful
if response.status_code == 200:
    data = response.json()
    print("Success! Retrieved data:")
    print(f"Number of records: {len(data)}")
    # Print first record as example
    if len(data) > 0:
        print(f"\nFirst record: {data[0]}")
        print(f"\nLast record: {data[-1]}")
    output_file = "hcdp_stations.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
