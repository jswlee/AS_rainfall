"""Stage 1 -- data preparation.

Reads raw_data/AS (station metadata, per-station daily rainfall CSVs, DEM,
reanalysis NetCDFs) and produces the cached artifacts under LAND_AS/data/:

    load_raw.py         station metadata + daily rainfall parsing (incl. QC)
    build_features.py   reanalysis/DEM patch extraction
    assemble_dataset.py daily -> weekly assembly into weekly_dataset.npz
    prepare.py          entry point orchestrating the three above
                        (python -m LAND_AS.s1_prepare.prepare)

The NPZ written here is consumed by s2_dataset.data.load_data().
"""