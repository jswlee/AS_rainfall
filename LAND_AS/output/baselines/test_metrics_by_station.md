# Test metrics by station

Each table is stations x models. Rainfall units are mm/week.

## mae

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 50.36 | 48.97 | 64.60 | 35.68 | 39.18 | 36.04 | 38.13 | 37.90 | 38.06 | 38.86 | 39.68 | 40.02 |
| afono_UH | 49.94 | 48.54 | 67.00 | 32.38 | 34.78 | 32.91 | 36.21 | 36.04 | 34.23 | 35.17 | 34.83 | 33.99 |
| aunuu_UH | 46.90 | 49.47 | 55.39 | 23.92 | 30.09 | 23.75 | 25.29 | 25.28 | 24.41 | 25.52 | 26.85 | 26.69 |
| poloa_UH | 48.53 | 47.32 | 61.08 | 36.71 | 36.98 | 34.60 | 36.10 | 36.06 | 34.14 | 33.97 | 34.74 | 34.95 |
| vaipito_UH | 51.97 | 50.34 | 68.99 | 35.24 | 37.06 | 33.76 | 36.52 | 36.49 | 34.27 | 34.69 | 35.10 | 35.11 |

## rmse

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 72.22 | 70.78 | 90.21 | 48.74 | 67.65 | 53.94 | 52.10 | 51.92 | 52.36 | 53.59 | 55.28 | 55.35 |
| afono_UH | 71.84 | 69.93 | 97.98 | 47.51 | 55.71 | 51.19 | 54.91 | 54.91 | 53.35 | 53.73 | 52.83 | 51.63 |
| aunuu_UH | 55.35 | 57.26 | 76.87 | 34.04 | 44.51 | 36.53 | 33.08 | 33.21 | 34.30 | 36.27 | 38.03 | 37.56 |
| poloa_UH | 63.08 | 62.35 | 84.79 | 48.94 | 56.86 | 52.06 | 49.88 | 49.80 | 48.61 | 49.26 | 49.83 | 49.33 |
| vaipito_UH | 69.33 | 68.33 | 92.39 | 49.25 | 56.66 | 47.47 | 49.71 | 49.63 | 47.16 | 47.63 | 48.37 | 48.58 |

## bias

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -11.61 | -12.19 | -0.46 | 3.72 | -10.74 | 1.69 | 4.45 | 3.80 | 4.53 | 6.64 | 3.90 | 4.78 |
| afono_UH | -6.18 | -5.51 | 0.02 | -5.10 | -8.58 | -9.44 | -0.70 | -1.05 | -5.94 | -2.67 | -5.05 | -4.67 |
| aunuu_UH | 16.76 | 22.93 | 2.40 | -7.02 | -19.08 | -9.38 | 2.99 | 3.01 | 0.33 | -1.49 | -0.29 | -0.12 |
| poloa_UH | 2.14 | 1.98 | 0.31 | 2.59 | -24.79 | -16.87 | 5.90 | 6.28 | -4.88 | -8.60 | -8.94 | -5.53 |
| vaipito_UH | -10.34 | -10.70 | -0.73 | -6.97 | -7.81 | -8.23 | 0.98 | 1.03 | -1.64 | 1.92 | -1.17 | -0.76 |

## r2

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -0.027 | 0.014 | -0.602 | 0.532 | 0.099 | 0.427 | 0.466 | 0.469 | 0.460 | 0.435 | 0.399 | 0.397 |
| afono_UH | -0.007 | 0.045 | -0.874 | 0.559 | 0.394 | 0.489 | 0.411 | 0.411 | 0.444 | 0.436 | 0.455 | 0.480 |
| aunuu_UH | -0.101 | -0.178 | -1.123 | 0.584 | 0.288 | 0.520 | 0.607 | 0.604 | 0.577 | 0.527 | 0.480 | 0.493 |
| poloa_UH | -0.001 | 0.022 | -0.809 | 0.397 | 0.186 | 0.318 | 0.374 | 0.376 | 0.406 | 0.389 | 0.375 | 0.388 |
| vaipito_UH | -0.023 | 0.007 | -0.816 | 0.484 | 0.317 | 0.520 | 0.474 | 0.476 | 0.527 | 0.517 | 0.502 | 0.498 |

## spearman_r

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH |  | 0.250 | 0.178 | 0.724 | 0.729 | 0.679 | 0.679 | 0.678 | 0.683 | 0.674 | 0.632 | 0.626 |
| afono_UH |  | 0.275 | 0.079 | 0.726 | 0.744 | 0.703 | 0.686 | 0.688 | 0.692 | 0.690 | 0.689 | 0.708 |
| aunuu_UH |  | 0.113 | -0.130 | 0.754 | 0.794 | 0.769 | 0.756 | 0.753 | 0.763 | 0.769 | 0.755 | 0.757 |
| poloa_UH |  | 0.165 | 0.092 | 0.640 | 0.688 | 0.648 | 0.628 | 0.631 | 0.644 | 0.641 | 0.636 | 0.627 |
| vaipito_UH |  | 0.196 | 0.100 | 0.727 | 0.751 | 0.740 | 0.698 | 0.699 | 0.719 | 0.714 | 0.702 | 0.701 |

## obs_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 |
| afono_UH | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 | 71.57 |
| aunuu_UH | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 |
| poloa_UH | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 |
| vaipito_UH | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 |

## pred_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v1_gamma | weekly_land_v1_gamma_OLD | weekly_land_v1_huber | weekly_land_v1_huber_bs2048 | weekly_land_v1_huber_bs2048_OLD | weekly_land_v1_huber_OLD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 0.00 | 13.18 | 70.90 | 45.85 | 67.05 | 47.68 | 52.68 | 52.71 | 51.10 | 54.01 | 54.29 | 54.28 |
| afono_UH | 0.00 | 12.72 | 71.45 | 49.18 | 73.32 | 44.03 | 50.23 | 50.49 | 45.36 | 46.64 | 46.18 | 47.53 |
| aunuu_UH | 0.00 | 9.49 | 52.06 | 39.77 | 19.01 | 37.22 | 38.75 | 38.80 | 35.32 | 32.26 | 36.92 | 38.11 |
| poloa_UH | 0.00 | 13.03 | 62.93 | 46.13 | 33.93 | 41.27 | 50.20 | 50.58 | 40.07 | 37.20 | 37.11 | 39.75 |
| vaipito_UH | 0.00 | 12.97 | 67.13 | 45.58 | 57.32 | 47.72 | 54.85 | 54.91 | 50.93 | 53.70 | 54.46 | 52.97 |
