from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_datapoint(date_str: str):
    # locate project root and data files
    project_root = Path(__file__).parent.parent
    feat_path = project_root / 'csv_data' / 'features.csv'
    targ_path = project_root / 'csv_data' / 'targets.csv'

    # load dataframes
    df_feat = pd.read_csv(feat_path, index_col='date')
    df_targ = pd.read_csv(targ_path, index_col='date')

    if date_str not in df_feat.index:
        raise ValueError(f"Date '{date_str}' not found in features.csv")

    # handle multiple entries per date by selecting the first
    rows = df_feat.loc[date_str]
    if isinstance(rows, pd.DataFrame):
        print(f"Multiple rows ({len(rows)}) for date '{date_str}', selecting first row for visualization.")
        feat = rows.iloc[0]
    else:
        feat = rows

    # split feature columns
    climate_cols = [c for c in df_feat.columns if c.startswith('climate_')]
    local_cols = sorted([c for c in df_feat.columns if c.startswith('local_dem_')], key=lambda x: int(x.split('_')[-1]))
    regional_cols = sorted([c for c in df_feat.columns if c.startswith('regional_dem_')], key=lambda x: int(x.split('_')[-1]))
    month_cols = sorted([c for c in df_feat.columns if c.startswith('month_')], key=lambda x: int(x.split('_')[-1]))

    # get rainfall (same for all entries)
    rain_vals = df_targ.loc[date_str, 'rainfall']
    if isinstance(rain_vals, (pd.Series, np.ndarray)):
        rainfall = rain_vals.iloc[0]
    else:
        rainfall = rain_vals

    # reshape patches
    local_patch = feat[local_cols].values.reshape((3, 3))
    regional_patch = feat[regional_cols].values.reshape((3, 3))
    climate_vals = feat[climate_cols].values
    month_vals = feat[month_cols].values

    # plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axes[0, 0].imshow(local_patch, cmap='viridis')
    axes[0, 0].set_title('Local DEM Patch')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(regional_patch, cmap='viridis')
    axes[0, 1].set_title('Regional DEM Patch')
    plt.colorbar(im1, ax=axes[0, 1])

    axes[0, 2].bar(range(1, 13), month_vals)
    axes[0, 2].set_title('Month One-Hot')
    axes[0, 2].set_xticks(range(1, 13))
    axes[0, 2].set_xticklabels(range(1, 13))

    fig.suptitle(f'Data for Single Grid Point for {date_str}', fontsize=16)

    # bottom row: grid-level heatmaps
    rows_feat_df = df_feat.loc[date_str]
    if isinstance(rows_feat_df, pd.Series):
        rows_feat_df = rows_feat_df.to_frame().T
    rows_targ_df = df_targ.loc[date_str]
    if isinstance(rows_targ_df, pd.Series):
        rows_targ_df = rows_targ_df.to_frame().T

    # reshape full grids
    rain_grid = rows_targ_df['rainfall'].values.reshape((5, 5))
    air_grid = rows_feat_df['climate_air_2m'].values.reshape((5, 5))
    tempdiff_grid = rows_feat_df['climate_air_temp_diff_1000_500'].values.reshape((5, 5))

    # highlight specific grid point (8th)
    idx = 7
    row_pt, col_pt = divmod(idx, 5)

    # plot bottom grids
    im2 = axes[1, 0].imshow(rain_grid, cmap='viridis')
    axes[1, 0].set_title('Rainfall Grid')
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(air_grid, cmap='viridis')
    axes[1, 1].set_title('Climate Air 2m Grid')
    plt.colorbar(im3, ax=axes[1, 1])

    im4 = axes[1, 2].imshow(tempdiff_grid, cmap='viridis')
    axes[1, 2].set_title('Climate Air Temp Diff 1000-500 Grid')
    plt.colorbar(im4, ax=axes[1, 2])

    # annotate point
    for ax_, grid_ in zip(axes[1], [rain_grid, air_grid, tempdiff_grid]):
        ax_.scatter(col_pt, row_pt, s=200, facecolors='none', edgecolors='red')
        val = grid_[row_pt, col_pt]
        ax_.text(col_pt, row_pt, f'{val:.2f}', color='white', ha='center', va='center', fontsize='small')

    # subtitle between rows
    fig.text(0.5, 0.45, 'Grid Point Value with Surrounding Grid Points', ha='center', fontsize=12)
    # adjust layout: increase vertical spacing between subplot rows
    fig.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=2.0)

    # save figure into this directory
    out_path = Path(__file__).parent / f'{date_str}.png'
    fig.savefig(out_path)
    print(f'Saved visualization to {out_path}')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a single data point from features.csv')
    parser.add_argument('--date', required=True, help='Date identifier (e.g. 000_1958-07)')
    args = parser.parse_args()
    visualize_datapoint(args.date)