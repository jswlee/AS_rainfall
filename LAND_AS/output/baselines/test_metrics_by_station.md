# Test metrics by station

Each table is stations x models. Rainfall units are mm/week.

## mae

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 51.45 | 49.97 | 64.70 | 35.63 | 38.26 | 36.38 |
| afono_UH | 51.43 | 50.07 | 67.10 | 32.62 | 34.02 | 33.79 |
| aunuu_UH | 50.53 | 53.52 | 55.66 | 24.25 | 27.78 | 27.40 |
| poloa_UH | 50.60 | 49.61 | 61.08 | 37.59 | 35.31 | 32.79 |
| vaipito_UH | 52.89 | 51.37 | 68.99 | 35.43 | 36.33 | 32.83 |

## rmse

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 71.44 | 69.86 | 90.27 | 48.87 | 64.61 | 53.33 |
| afono_UH | 71.58 | 69.66 | 98.03 | 47.80 | 52.69 | 50.77 |
| aunuu_UH | 57.80 | 60.64 | 77.08 | 33.79 | 39.20 | 35.50 |
| poloa_UH | 63.68 | 62.94 | 84.80 | 49.77 | 53.46 | 47.57 |
| vaipito_UH | 68.64 | 67.62 | 92.40 | 49.42 | 54.68 | 46.16 |

## bias

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | -4.74 | -5.26 | -0.30 | 2.94 | -8.35 | -0.70 |
| afono_UH | 0.69 | 1.31 | 0.13 | -6.96 | -8.82 | -3.95 |
| aunuu_UH | 23.63 | 29.95 | 2.67 | -1.53 | -8.48 | 4.46 |
| poloa_UH | 9.01 | 8.84 | 0.35 | 4.65 | -17.71 | -6.31 |
| vaipito_UH | -3.47 | -3.83 | -0.68 | -6.98 | -11.23 | -8.48 |

## r2

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | -0.004 | 0.039 | -0.604 | 0.530 | 0.178 | 0.440 |
| afono_UH | -0.000 | 0.053 | -0.876 | 0.554 | 0.458 | 0.497 |
| aunuu_UH | -0.201 | -0.321 | -1.135 | 0.590 | 0.448 | 0.547 |
| poloa_UH | -0.020 | 0.003 | -0.809 | 0.377 | 0.281 | 0.431 |
| vaipito_UH | -0.003 | 0.027 | -0.817 | 0.480 | 0.364 | 0.547 |

## spearman_r

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH |  | 0.253 | 0.176 | 0.719 | 0.731 | 0.677 |
| afono_UH |  | 0.282 | 0.077 | 0.722 | 0.745 | 0.713 |
| aunuu_UH |  | 0.096 | -0.131 | 0.760 | 0.783 | 0.713 |
| poloa_UH |  | 0.166 | 0.092 | 0.635 | 0.685 | 0.658 |
| vaipito_UH |  | 0.195 | 0.100 | 0.728 | 0.760 | 0.767 |

## obs_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 |
| afono_UH | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 |
| aunuu_UH | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 |
| poloa_UH | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 |
| vaipito_UH | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 |

## pred_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 0.00 | 13.43 | 70.89 | 47.88 | 63.24 | 45.91 |
| afono_UH | 0.00 | 12.94 | 71.45 | 51.43 | 66.39 | 43.76 |
| aunuu_UH | 0.00 | 9.30 | 52.15 | 41.23 | 23.36 | 36.62 |
| poloa_UH | 0.00 | 13.25 | 62.93 | 48.32 | 36.37 | 41.91 |
| vaipito_UH | 0.00 | 13.20 | 67.12 | 47.69 | 50.20 | 45.71 |
