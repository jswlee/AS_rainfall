"""Stage 3 -- model definitions and training engine.

model.py    LAND network architectures and heads (Gamma / scalar Huber) plus
            the loss functions; instances are built from s2_dataset metadata
engine.py   fit/predict loop, device selection, JSON helpers -- used by
            s4_train (tune.py, parallelize.py) and s5_evaluate (evaluate.py)
metrics.py  regression and extreme-event metrics shared by evaluation and
            baselines
"""