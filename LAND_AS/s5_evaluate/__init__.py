"""Stage 5 -- evaluation and baselines.

evaluate.py          scores a trained run's LOSO ensemble on the held-out
                     test set; writes output/runs/<run>/evaluation/
                     (python -m LAND_AS.s5_evaluate.evaluate --run NAME)
baselines/models.py  climatology / persistence / tabular baseline definitions
baselines/evaluate.py  fits baselines on the same splits and aligns saved run
                     predictions for comparison
                     (python -m LAND_AS.s5_evaluate.baselines.evaluate)
"""