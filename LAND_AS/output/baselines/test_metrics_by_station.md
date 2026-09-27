# Test metrics by station

Each table is stations x models. Rainfall units are mm/week.

## mae

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 50.85 | 49.42 | 64.65 | 37.00 | 38.43 | 35.09 | 37.27 | 37.30 | 38.29 | 38.37 | 38.53 | 37.72 | 37.91 | 37.67 | 37.72 | 38.88 | 38.44 | 36.77 | 37.25 | 37.18 | 37.19 | 37.06 | 37.11 |
| afono_UH | 51.46 | 49.82 | 64.13 | 34.08 | 34.07 | 34.25 | 36.50 | 35.26 | 36.19 | 36.44 | 36.47 | 35.80 | 36.00 | 35.75 | 35.41 | 35.16 | 36.29 | 35.42 | 36.21 | 36.30 | 36.30 | 36.21 | 36.22 |
| aunuu_UH | 48.66 | 51.49 | 55.52 | 24.10 | 28.14 | 27.45 | 26.91 | 26.82 | 26.73 | 26.97 | 26.95 | 26.73 | 26.93 | 26.49 | 26.59 | 25.87 | 25.98 | 26.35 | 26.76 | 26.81 | 26.79 | 26.81 | 26.82 |
| poloa_UH | 49.49 | 48.39 | 61.08 | 39.19 | 35.40 | 34.33 | 35.91 | 34.23 | 36.12 | 36.32 | 36.26 | 35.68 | 35.93 | 34.19 | 34.97 | 35.64 | 35.59 | 34.62 | 35.79 | 35.81 | 35.78 | 35.71 | 35.76 |
| vaipito_UH | 52.36 | 50.75 | 68.99 | 36.11 | 36.60 | 34.93 | 36.43 | 35.09 | 35.30 | 35.61 | 35.59 | 35.11 | 35.22 | 36.15 | 35.80 | 35.44 | 35.38 | 35.16 | 35.84 | 35.96 | 36.00 | 35.93 | 35.92 |

## rmse

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 71.76 | 70.22 | 90.24 | 49.56 | 65.02 | 52.60 | 51.56 | 52.64 | 52.92 | 53.12 | 53.15 | 52.52 | 52.52 | 52.14 | 51.73 | 53.31 | 52.51 | 51.80 | 51.64 | 51.68 | 51.65 | 51.51 | 51.52 |
| afono_UH | 71.54 | 69.38 | 95.64 | 48.34 | 51.99 | 51.07 | 54.22 | 53.94 | 53.59 | 54.19 | 54.21 | 53.75 | 53.43 | 54.98 | 54.25 | 50.99 | 52.93 | 53.69 | 53.77 | 53.97 | 53.98 | 53.92 | 53.81 |
| aunuu_UH | 56.45 | 58.98 | 76.97 | 34.26 | 40.12 | 37.93 | 36.56 | 38.94 | 37.87 | 37.71 | 37.75 | 38.07 | 37.76 | 35.27 | 35.66 | 34.89 | 36.19 | 37.80 | 36.71 | 36.60 | 36.57 | 36.68 | 36.61 |
| poloa_UH | 63.28 | 62.51 | 84.79 | 50.54 | 53.91 | 50.94 | 49.69 | 48.62 | 49.60 | 49.31 | 48.98 | 48.93 | 49.02 | 49.65 | 49.08 | 48.67 | 49.45 | 48.55 | 49.40 | 49.33 | 49.27 | 49.30 | 49.29 |
| vaipito_UH | 68.91 | 67.89 | 92.40 | 51.46 | 55.17 | 49.14 | 49.28 | 49.20 | 48.38 | 48.48 | 48.61 | 48.27 | 48.13 | 48.88 | 49.30 | 48.06 | 48.20 | 48.75 | 48.66 | 48.75 | 48.82 | 48.78 | 48.71 |

## bias

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -8.27 | -8.75 | -0.38 | 7.42 | -10.45 | -3.64 | 2.93 | -4.60 | 4.19 | 2.07 | 1.34 | 0.94 | 3.05 | 2.43 | 1.58 | 7.84 | 3.27 | -2.10 | 3.32 | 2.70 | 2.55 | 2.47 | 2.96 |
| afono_UH | 0.78 | 0.91 | 0.25 | 3.10 | -7.57 | -1.12 | -3.25 | -9.38 | -1.62 | -2.58 | -2.48 | -3.61 | -2.03 | -8.41 | -6.35 | 0.31 | -0.01 | -7.35 | -2.74 | -3.07 | -3.06 | -3.33 | -2.94 |
| aunuu_UH | 20.10 | 26.27 | 2.53 | -5.51 | -10.56 | 3.40 | -1.55 | -10.12 | -1.44 | -0.28 | -0.49 | -2.02 | -0.60 | -1.22 | -3.85 | 0.43 | -2.44 | -7.27 | -1.52 | -1.21 | -1.29 | -1.66 | -1.31 |
| poloa_UH | 5.49 | 5.32 | 0.33 | 11.17 | -18.59 | -12.59 | 3.82 | -4.56 | 4.29 | 4.51 | 5.20 | 2.90 | 3.95 | -9.61 | -3.61 | -0.79 | -1.71 | -1.78 | 3.97 | 4.01 | 4.16 | 3.61 | 3.86 |
| vaipito_UH | -7.00 | -7.34 | -0.71 | -15.42 | -12.40 | -11.47 | -1.36 | -7.46 | 0.06 | 0.19 | -0.78 | -1.55 | -0.22 | -0.21 | -1.91 | 1.65 | -0.86 | -5.44 | -0.92 | -0.94 | -1.22 | -1.40 | -1.07 |

## r2

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | -0.013 | 0.030 | -0.603 | 0.517 | 0.168 | 0.456 | 0.477 | 0.455 | 0.449 | 0.445 | 0.444 | 0.457 | 0.457 | 0.465 | 0.473 | 0.441 | 0.457 | 0.472 | 0.475 | 0.474 | 0.475 | 0.478 | 0.478 |
| afono_UH | -0.000 | 0.060 | -0.787 | 0.543 | 0.472 | 0.490 | 0.426 | 0.432 | 0.439 | 0.426 | 0.426 | 0.435 | 0.442 | 0.409 | 0.425 | 0.492 | 0.453 | 0.437 | 0.435 | 0.431 | 0.431 | 0.432 | 0.434 |
| aunuu_UH | -0.145 | -0.250 | -1.129 | 0.578 | 0.422 | 0.483 | 0.520 | 0.455 | 0.485 | 0.489 | 0.488 | 0.479 | 0.488 | 0.553 | 0.543 | 0.562 | 0.529 | 0.487 | 0.516 | 0.519 | 0.519 | 0.516 | 0.518 |
| poloa_UH | -0.008 | 0.017 | -0.809 | 0.357 | 0.269 | 0.347 | 0.379 | 0.405 | 0.381 | 0.388 | 0.396 | 0.398 | 0.395 | 0.380 | 0.394 | 0.404 | 0.385 | 0.407 | 0.386 | 0.388 | 0.389 | 0.388 | 0.389 |
| vaipito_UH | -0.010 | 0.019 | -0.816 | 0.436 | 0.352 | 0.486 | 0.483 | 0.485 | 0.502 | 0.500 | 0.497 | 0.504 | 0.507 | 0.492 | 0.483 | 0.508 | 0.506 | 0.494 | 0.496 | 0.494 | 0.493 | 0.494 | 0.495 |

## spearman_r

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH |  | 0.253 | 0.177 | 0.717 | 0.732 | 0.689 | 0.682 | 0.655 | 0.666 | 0.659 | 0.658 | 0.672 | 0.669 | 0.677 | 0.698 | 0.667 | 0.673 | 0.671 | 0.682 | 0.680 | 0.680 | 0.682 | 0.682 |
| afono_UH |  | 0.297 | 0.178 | 0.703 | 0.735 | 0.663 | 0.647 | 0.655 | 0.648 | 0.638 | 0.636 | 0.649 | 0.651 | 0.652 | 0.670 | 0.669 | 0.656 | 0.657 | 0.650 | 0.647 | 0.647 | 0.650 | 0.650 |
| aunuu_UH |  | 0.096 | -0.132 | 0.751 | 0.781 | 0.726 | 0.689 | 0.672 | 0.711 | 0.714 | 0.733 | 0.713 | 0.717 | 0.710 | 0.713 | 0.742 | 0.740 | 0.684 | 0.706 | 0.704 | 0.704 | 0.700 | 0.697 |
| poloa_UH |  | 0.166 | 0.092 | 0.639 | 0.688 | 0.624 | 0.608 | 0.617 | 0.610 | 0.604 | 0.610 | 0.613 | 0.612 | 0.624 | 0.615 | 0.609 | 0.607 | 0.617 | 0.611 | 0.609 | 0.610 | 0.611 | 0.611 |
| vaipito_UH |  | 0.195 | 0.100 | 0.723 | 0.757 | 0.725 | 0.690 | 0.700 | 0.706 | 0.689 | 0.689 | 0.702 | 0.704 | 0.695 | 0.698 | 0.713 | 0.705 | 0.698 | 0.696 | 0.692 | 0.693 | 0.695 | 0.695 |

## obs_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 | 71.28 |
| afono_UH | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 | 71.54 |
| aunuu_UH | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 | 52.75 |
| poloa_UH | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 | 63.04 |
| vaipito_UH | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 | 68.56 |

## pred_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm | weekly_land_v5 | weekly_land_v5_huber | weekly_land_v5_huber_d025 | weekly_land_v5_huber_d1 | weekly_land_v5_huber_d2 | weekly_land_v5_huber_rw | weekly_land_v5_huber_rw_msemon | weekly_land_v6_gamma_kfold_t18 | weekly_land_v6_gamma_temporal_mse | weekly_land_v6_huber_rw_t36 | weekly_land_v6_huber_rw_temporal_mse | v5_gamma_huber_cv_mse | v5_gamma_huber_d025_cv_mse | v5_gamma_huber_d1_cv_mse | v5_gamma_huber_d2_cv_mse | v5_gamma_huber_rw_cv_mse | v5_gamma_huber_rw_msemon_cv_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aasu_UH | 0.00 | 13.18 | 70.89 | 46.43 | 62.33 | 48.61 | 49.90 | 52.59 | 49.25 | 46.73 | 46.10 | 47.60 | 48.34 | 48.34 | 48.64 | 54.43 | 48.24 | 51.34 | 49.35 | 48.75 | 48.70 | 49.11 | 49.23 |
| afono_UH | 0.00 | 12.73 | 71.35 | 49.98 | 63.41 | 46.68 | 46.02 | 46.90 | 43.44 | 40.82 | 40.84 | 42.15 | 42.49 | 41.65 | 43.55 | 45.73 | 44.41 | 46.30 | 44.94 | 44.39 | 44.54 | 44.92 | 44.91 |
| aunuu_UH | 0.00 | 9.19 | 52.10 | 40.35 | 22.51 | 44.83 | 34.27 | 32.28 | 30.92 | 29.28 | 28.52 | 29.51 | 29.82 | 31.99 | 31.51 | 32.00 | 30.88 | 32.78 | 32.97 | 32.69 | 32.65 | 32.97 | 32.93 |
| poloa_UH | 0.00 | 12.99 | 62.93 | 47.06 | 35.93 | 44.88 | 46.11 | 44.07 | 43.12 | 41.78 | 42.05 | 42.13 | 42.28 | 38.52 | 40.92 | 41.35 | 40.82 | 44.47 | 44.92 | 44.70 | 44.89 | 44.99 | 44.93 |
| vaipito_UH | 0.00 | 12.93 | 67.12 | 46.24 | 49.97 | 48.53 | 49.67 | 52.07 | 50.42 | 49.37 | 49.28 | 49.90 | 50.00 | 48.60 | 47.87 | 52.70 | 49.29 | 50.91 | 49.58 | 49.29 | 49.29 | 49.46 | 49.47 |
