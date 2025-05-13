# Balanced hyperparameters with adjustments for better generalization

balanced_hyperparameters = {
    'na': 256,  # Reduced from 384
    'nb': 256,  # Reduced from 384
    'dropout_rate': 0.2,  # Reduced from 0.4
    'l2_reg': 1e-5,  # Increased regularization
    'learning_rate': 0.001,  # Increased from ~0.0002
    'weight_decay': 1e-6,  # Increased from ~1e-7
    'local_dem_units': 128,  # Slightly reduced
    'regional_dem_units': 128,  # Slightly reduced
    'month_units': 64,  # Kept the same
    'climate_units': 256,  # Reduced from 512
    'use_residual': True,  # Changed from False
    'activation': 'relu',  # Changed from elu
    'output_activation': 'softplus',  # Changed from relu for smoother non-zero outputs
}
