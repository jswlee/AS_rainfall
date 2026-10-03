# Test metrics by station

Each table is stations x models. Rainfall units are mm/week.

## mae

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 50.20 | 48.85 | 64.58 | 34.70 | 39.37 | 35.58 | 37.15 | 38.96 | 37.94 | 39.21 | 38.94 | 38.84 | 38.89 | 35.58 | 34.80 |
| aunuu_UH | 46.24 | 48.76 | 55.33 | 24.40 | 29.42 | 24.38 | 27.02 | 26.26 | 29.11 | 26.04 | 31.74 | 31.91 | 29.74 | 27.03 | 26.48 |
| poloa_UH | 48.16 | 46.95 | 61.08 | 35.61 | 36.60 | 33.75 | 33.35 | 35.72 | 35.37 | 35.71 | 35.96 | 36.21 | 36.08 | 34.46 | 33.74 |
| vaipito_UH | 51.85 | 50.21 | 68.99 | 37.02 | 37.18 | 33.57 | 34.31 | 35.34 | 36.42 | 35.30 | 35.46 | 35.72 | 35.87 | 36.05 | 35.52 |

## rmse

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 72.46 | 71.04 | 90.20 | 49.08 | 69.43 | 51.22 | 51.95 | 54.09 | 52.82 | 54.16 | 53.91 | 53.53 | 53.21 | 52.78 | 52.16 |
| aunuu_UH | 54.95 | 56.81 | 76.84 | 35.23 | 43.37 | 37.39 | 35.64 | 34.40 | 38.07 | 34.16 | 38.28 | 39.34 | 37.13 | 39.29 | 39.41 |
| poloa_UH | 63.04 | 62.32 | 84.79 | 49.10 | 56.23 | 50.03 | 48.41 | 49.23 | 49.16 | 49.30 | 49.46 | 49.30 | 49.04 | 49.61 | 48.99 |
| vaipito_UH | 69.55 | 68.55 | 92.39 | 53.54 | 57.05 | 47.54 | 46.65 | 48.42 | 50.14 | 48.29 | 48.95 | 48.38 | 49.00 | 52.69 | 51.86 |

## bias

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -13.01 | -13.50 | -0.49 | -2.78 | -8.40 | 1.07 | 1.33 | 5.10 | 2.41 | 5.91 | 3.29 | 5.98 | 4.77 | -7.22 | -4.12 |
| aunuu_UH | 15.36 | 21.52 | 2.34 | -9.48 | -17.26 | -11.13 | 6.68 | 3.22 | 5.75 | 3.78 | 13.98 | 13.88 | 12.28 | -6.17 | -6.51 |
| poloa_UH | 0.75 | 0.61 | 0.30 | -6.53 | -23.14 | -16.56 | -6.44 | 2.50 | -0.38 | 3.12 | 2.32 | 4.34 | 3.92 | -5.82 | -6.32 |
| vaipito_UH | -11.74 | -12.06 | -0.74 | -21.27 | -10.97 | -8.38 | 0.48 | -0.72 | -2.66 | -0.32 | -1.75 | 0.56 | -1.29 | -14.89 | -13.59 |

## r2

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -0.033 | 0.007 | -0.601 | 0.526 | 0.051 | 0.484 | 0.469 | 0.424 | 0.451 | 0.423 | 0.428 | 0.436 | 0.443 | 0.452 | 0.465 |
| aunuu_UH | -0.085 | -0.160 | -1.121 | 0.554 | 0.324 | 0.498 | 0.543 | 0.575 | 0.479 | 0.581 | 0.473 | 0.444 | 0.505 | 0.445 | 0.442 |
| poloa_UH | -0.000 | 0.023 | -0.809 | 0.393 | 0.204 | 0.370 | 0.410 | 0.390 | 0.392 | 0.388 | 0.384 | 0.388 | 0.395 | 0.381 | 0.396 |
| vaipito_UH | -0.029 | 0.000 | -0.816 | 0.390 | 0.308 | 0.519 | 0.537 | 0.501 | 0.465 | 0.504 | 0.490 | 0.502 | 0.489 | 0.409 | 0.428 |

## spearman_r

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH |  | 0.250 | 0.178 | 0.717 | 0.728 | 0.701 | 0.688 | 0.680 | 0.669 | 0.681 | 0.682 | 0.677 | 0.674 | 0.716 | 0.719 |
| aunuu_UH |  | 0.113 | -0.125 | 0.753 | 0.795 | 0.774 | 0.751 | 0.754 | 0.724 | 0.752 | 0.738 | 0.733 | 0.760 | 0.775 | 0.769 |
| poloa_UH |  | 0.165 | 0.092 | 0.644 | 0.689 | 0.667 | 0.643 | 0.622 | 0.606 | 0.621 | 0.612 | 0.617 | 0.622 | 0.654 | 0.669 |
| vaipito_UH |  | 0.196 | 0.100 | 0.727 | 0.750 | 0.751 | 0.727 | 0.722 | 0.700 | 0.720 | 0.711 | 0.718 | 0.702 | 0.717 | 0.723 |

## obs_std

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 |
| aunuu_UH | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 |
| poloa_UH | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 |
| vaipito_UH | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 |

## pred_std

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_dem025_recent_bal | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_recent_notbal | weekly_land_v3_huber_rw_recent_broad | weekly_land_v3_huber_rw_recent_broad_notbal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 0.00 | 13.07 | 70.91 | 44.52 | 70.49 | 48.00 | 48.82 | 50.22 | 47.63 | 50.83 | 51.29 | 50.16 | 50.70 | 48.88 | 50.82 |
| aunuu_UH | 0.00 | 9.46 | 52.04 | 38.63 | 20.06 | 33.02 | 34.21 | 30.51 | 30.09 | 31.43 | 31.71 | 32.76 | 32.91 | 32.95 | 33.23 |
| poloa_UH | 0.00 | 12.90 | 62.93 | 44.90 | 35.44 | 42.61 | 39.56 | 43.51 | 38.93 | 44.14 | 40.74 | 42.03 | 40.87 | 38.22 | 37.81 |
| vaipito_UH | 0.00 | 12.85 | 67.13 | 44.33 | 55.52 | 47.15 | 51.07 | 49.56 | 45.20 | 50.03 | 50.99 | 47.97 | 50.73 | 43.28 | 42.54 |
