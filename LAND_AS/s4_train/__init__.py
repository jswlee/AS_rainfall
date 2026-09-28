"""Stage 4 -- hyperparameter tuning and ensemble training.

tune.py         Optuna study entry point; evaluates HP trials over CV folds
                from s2_dataset and writes output/tuning/<study>/
                (python -m LAND_AS.s4_train.tune)
train.py        trains a LOSO ensemble for a chosen study trial via
                parallelize.py and writes output/runs/<run>/
                (python -m LAND_AS.s4_train.train)
parallelize.py  per-fold/per-seed worker pool used by train.py; resumes
                partially completed runs by skipping existing checkpoints
"""