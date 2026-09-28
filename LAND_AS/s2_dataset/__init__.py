"""Stage 2 -- dataset assembly at runtime.

data.py loads the weekly NPZ produced by s1_prepare and turns it into model
inputs: strict train/test splits, lagged rainfall/climate context, DEM patch
cropping, fold-local normalization, DataLoaders, and the LOSO/kfold/temporal
CV fold iterators used by s4_train and s5_evaluate.
"""