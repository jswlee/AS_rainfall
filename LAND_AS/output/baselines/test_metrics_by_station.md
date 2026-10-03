# Test metrics by station

Each table is stations x models. Rainfall units are mm/week.

## mae

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 50.20 | 48.85 | 64.58 | 36.98 | 39.44 | 34.64 | 38.96 | 37.94 | 39.21 | 38.94 | 38.84 | 35.58 |
| aunuu_UH | 46.24 | 48.76 | 55.33 | 24.10 | 29.74 | 25.31 | 26.26 | 29.11 | 26.04 | 31.74 | 31.91 | 27.03 |
| poloa_UH | 48.16 | 46.95 | 61.08 | 38.81 | 36.79 | 34.26 | 35.72 | 35.37 | 35.71 | 35.96 | 36.21 | 34.46 |
| vaipito_UH | 51.85 | 50.21 | 68.99 | 35.66 | 37.17 | 32.41 | 35.34 | 36.42 | 35.30 | 35.46 | 35.72 | 36.05 |

## rmse

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 72.46 | 71.04 | 90.20 | 49.58 | 68.59 | 52.92 | 54.09 | 52.82 | 54.16 | 53.91 | 53.53 | 52.78 |
| aunuu_UH | 54.95 | 56.81 | 76.84 | 34.10 | 44.00 | 37.80 | 34.40 | 38.07 | 34.16 | 38.28 | 39.34 | 39.29 |
| poloa_UH | 63.04 | 62.32 | 84.79 | 50.00 | 56.59 | 50.78 | 49.23 | 49.16 | 49.30 | 49.46 | 49.30 | 49.61 |
| vaipito_UH | 69.55 | 68.55 | 92.39 | 50.89 | 57.09 | 46.34 | 48.42 | 50.14 | 48.29 | 48.95 | 48.38 | 52.69 |

## bias

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -13.01 | -13.50 | -0.49 | 7.27 | -11.36 | -5.34 | 5.10 | 2.41 | 5.91 | 3.29 | 5.98 | -7.22 |
| aunuu_UH | 15.36 | 21.52 | 2.34 | -4.84 | -18.24 | -10.73 | 3.22 | 5.75 | 3.78 | 13.98 | 13.88 | -6.17 |
| poloa_UH | 0.75 | 0.61 | 0.30 | 11.14 | -23.96 | -19.30 | 2.50 | -0.38 | 3.12 | 2.32 | 4.34 | -5.82 |
| vaipito_UH | -11.74 | -12.06 | -0.74 | -13.50 | -9.26 | -10.50 | -0.72 | -2.66 | -0.32 | -1.75 | 0.56 | -14.89 |

## r2

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -0.033 | 0.007 | -0.601 | 0.516 | 0.074 | 0.449 | 0.424 | 0.451 | 0.423 | 0.428 | 0.436 | 0.452 |
| aunuu_UH | -0.085 | -0.160 | -1.121 | 0.582 | 0.304 | 0.487 | 0.575 | 0.479 | 0.581 | 0.473 | 0.444 | 0.445 |
| poloa_UH | -0.000 | 0.023 | -0.809 | 0.371 | 0.194 | 0.351 | 0.390 | 0.392 | 0.388 | 0.384 | 0.388 | 0.381 |
| vaipito_UH | -0.029 | 0.000 | -0.816 | 0.449 | 0.306 | 0.543 | 0.501 | 0.465 | 0.504 | 0.490 | 0.502 | 0.409 |

## spearman_r

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH |  | 0.250 | 0.178 | 0.719 | 0.728 | 0.697 | 0.680 | 0.669 | 0.681 | 0.682 | 0.677 | 0.716 |
| aunuu_UH |  | 0.113 | -0.125 | 0.756 | 0.795 | 0.758 | 0.754 | 0.724 | 0.752 | 0.738 | 0.733 | 0.775 |
| poloa_UH |  | 0.165 | 0.092 | 0.643 | 0.689 | 0.654 | 0.622 | 0.606 | 0.621 | 0.612 | 0.617 | 0.654 |
| vaipito_UH |  | 0.196 | 0.100 | 0.725 | 0.750 | 0.765 | 0.722 | 0.700 | 0.720 | 0.711 | 0.718 | 0.717 |

## obs_std

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 |
| aunuu_UH | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 |
| poloa_UH | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 |
| vaipito_UH | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 |

## pred_std

| station | pooled_mean | month_climatology | persistence | ols | tweedie_glm | gbm | weekly_land_v3_huber_loso_bal | weekly_land_v3_huber_loso_bal_2 | weekly_land_v3_huber_recent_bal | weekly_land_v3_huber_recent_bal_2 | weekly_land_v3_huber_recent_bal_lowlr | weekly_land_v3_huber_rw_recent_broad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 0.00 | 13.07 | 70.91 | 44.94 | 67.72 | 43.97 | 50.22 | 47.63 | 50.83 | 51.29 | 50.16 | 48.88 |
| aunuu_UH | 0.00 | 9.46 | 52.04 | 39.22 | 19.53 | 33.37 | 30.51 | 30.09 | 31.43 | 31.71 | 32.76 | 32.95 |
| poloa_UH | 0.00 | 12.90 | 62.93 | 45.60 | 34.82 | 40.32 | 43.51 | 38.93 | 44.14 | 40.74 | 42.03 | 38.22 |
| vaipito_UH | 0.00 | 12.85 | 67.13 | 44.84 | 56.88 | 47.67 | 49.56 | 45.20 | 50.03 | 50.99 | 47.97 | 43.28 |
