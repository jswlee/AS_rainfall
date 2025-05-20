#!/usr/bin/env python3
"""
Visualize monthly rainfall data coverage for each station.
Creates a heatmap: stations vs. year-month, colored by data presence.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
MONTHLY_DIR = os.path.join(PROJECT_ROOT, '1_Process_Rainfall_Data', 'output', 'monthly_rainfall')

# Find all monthly files
monthly_files = sorted(glob.glob(os.path.join(MONTHLY_DIR, '*_monthly.csv')))

# Collect all year-months and station names
all_months = set()
station_names = []
data_dict = {}

for f in monthly_files:
    station = Path(f).stem.replace('_monthly','')
    station_names.append(station)
    df = pd.read_csv(f)
    # Try to handle both 'year_month' as period or string
    if 'year_month' in df.columns:
        months = df['year_month'].astype(str)
        values = df.iloc[:, 1]  # Assume second column is rainfall
        data_dict[station] = dict(zip(months, values))
        all_months.update(months)

all_months = sorted(list(all_months))

# Build presence/absence matrix
presence = np.zeros((len(station_names), len(all_months)), dtype=int)
for i, station in enumerate(station_names):
    for j, month in enumerate(all_months):
        val = data_dict.get(station, {}).get(month, np.nan)
        if not pd.isna(val):
            presence[i, j] = 1

# Plot heatmap
plt.figure(figsize=(min(20, 0.6*len(all_months)), max(6, 0.4*len(station_names))))
plt.imshow(presence, aspect='auto', cmap='Blues', interpolation='nearest')
plt.yticks(np.arange(len(station_names)), station_names)

# X-axis: show only every 5 years
# Parse years from all_months (format YYYY-MM)
years = [int(m.split('-')[0]) for m in all_months]
unique_years = sorted(set(years))
first_year = unique_years[0]
last_year = unique_years[-1]
step = 5
five_years = list(range(first_year, last_year+1, step))
# Find the first month index for each 5-year tick
xtick_indices = [next((i for i, m in enumerate(all_months) if int(m.split('-')[0]) == y and m.endswith('-01')), None) for y in five_years]
# Remove None (if a January is missing for a year)
xtick_indices = [i for i in xtick_indices if i is not None]
xtick_labels = [all_months[i][:7] for i in xtick_indices]
plt.xticks(xtick_indices, xtick_labels, rotation=45)

plt.xlabel('Year-Month')
plt.ylabel('Station')
plt.title('Monthly Rainfall Data Coverage by Station')
plt.tight_layout()

# Custom legend for data presence
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color=plt.cm.Blues(1.0), label='Data Present')
plt.legend(handles=[blue_patch], loc='upper right')

plt.savefig(os.path.join(PROJECT_ROOT, '1_Process_Rainfall_Data', 'figures', 'rainfall_monthly_coverage_heatmap.png'))
plt.show()

