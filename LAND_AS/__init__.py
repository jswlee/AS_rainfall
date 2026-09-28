"""Weekly LAND rainfall downscaling for American Samoa.

The package is organized in pipeline order:

    s1_prepare/   raw data -> feature caches -> weekly dataset NPZ
                    (python -m LAND_AS.s1_prepare.prepare)
    s2_dataset/   NPZ -> train/test splits, lag features, loaders, CV folds
    s3_model/     LAND architectures (Gamma / Huber heads), fit engine, metrics
    s4_train/     Optuna tuning + LOSO ensemble training entry points
                    (python -m LAND_AS.s4_train.tune / .train)
    s5_evaluate/  run evaluation + climatology/tabular baselines
                    (python -m LAND_AS.s5_evaluate.evaluate / .baselines.evaluate)

``config.py`` and ``provenance.py`` at this level are shared by every stage.
"""
