"""
Step 2: Assemble reanalysis patches, DEM patches, rainfall, and month
one-hot into a single NPZ ready for modelling.

Reads the intermediate NPZ files produced by step 01.

Usage:
    python -m Daily_Modeling.scripts.02_assemble_dataset
"""

from Daily_Modeling.data_utils.assemble_dataset import assemble


def main():
    assemble()


if __name__ == "__main__":
    main()
