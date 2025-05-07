from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def visualize_all_patches(date_str: str):
    project_root = Path(__file__).parent.parent
    feat_path = project_root / 'csv_data' / 'features.csv'

    # load features
    df_feat = pd.read_csv(feat_path, index_col='date')
    if date_str not in df_feat.index:
        raise ValueError(f"Date '{date_str}' not found in features.csv")

    rows = df_feat.loc[date_str]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T

    # define columns
    local_cols = sorted([c for c in df_feat.columns if c.startswith('local_dem_')], key=lambda x: int(x.split('_')[-1]))
    regional_cols = sorted([c for c in df_feat.columns if c.startswith('regional_dem_')], key=lambda x: int(x.split('_')[-1]))

    # prepare mosaics (5*3 x 5*3)
    dim = 3
    grid_n = 5
    local_mosaic = np.zeros((grid_n*dim, grid_n*dim))
    regional_mosaic = np.zeros((grid_n*dim, grid_n*dim))

    for idx, (_, row) in enumerate(rows.iterrows()):
        r, c = divmod(idx, grid_n)
        patch_l = row[local_cols].values.reshape((dim, dim))
        patch_r = row[regional_cols].values.reshape((dim, dim))
        local_mosaic[r*dim:(r+1)*dim, c*dim:(c+1)*dim] = patch_l
        regional_mosaic[r*dim:(r+1)*dim, c*dim:(c+1)*dim] = patch_r

    # plot mosaics side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(local_mosaic, cmap='viridis')
    axes[0].set_title('Local DEM Patches Mosaic')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(regional_mosaic, cmap='viridis')
    axes[1].set_title('Regional DEM Patches Mosaic')
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(f'DEM Patch Grid for {date_str}', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save
    out_path = Path(__file__).parent / f'{date_str}_patches.png'
    fig.savefig(out_path)
    print(f'Saved DEM patches mosaic to {out_path}')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 5x5 grid of 3x3 DEM patches')
    parser.add_argument('--date', required=True, help='Date identifier (e.g. 000_1958-07)')
    args = parser.parse_args()
    visualize_all_patches(args.date)