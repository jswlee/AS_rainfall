# Test metrics by station

Each table is stations x models. Rainfall units are mm/day.

## mae

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 12.23 | 12.04 | 14.15 | 9.87 | 9.95 | 9.33 |
| afono_UH | 12.28 | 12.21 | 13.29 | 9.13 | 9.82 | 8.76 |
| aunuu_UH | 10.94 | 11.41 | 10.11 | 6.50 | 6.57 | 6.66 |
| poloa_UH | 11.56 | 11.49 | 12.38 | 8.83 | 8.25 | 8.17 |
| vaipito_UH | 12.39 | 12.30 | 13.81 | 9.69 | 9.42 | 9.06 |

## rmse

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 20.75 | 20.65 | 25.69 | 16.81 | 39.39 | 16.87 |
| afono_UH | 20.46 | 20.35 | 24.37 | 16.26 | 43.45 | 16.33 |
| aunuu_UH | 15.69 | 15.82 | 19.55 | 12.65 | 13.91 | 13.08 |
| poloa_UH | 18.87 | 18.83 | 24.11 | 16.22 | 17.62 | 16.42 |
| vaipito_UH | 20.15 | 20.09 | 24.98 | 16.62 | 18.22 | 16.21 |

## bias

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | -1.65 | -1.77 | -0.00 | 0.53 | -1.84 | -0.58 |
| afono_UH | -0.81 | -0.74 | 0.02 | -0.44 | -1.34 | -1.49 |
| aunuu_UH | 2.37 | 3.19 | 0.02 | -1.05 | -3.33 | -1.68 |
| poloa_UH | 0.32 | 0.30 | 0.01 | -0.15 | -4.30 | -2.45 |
| vaipito_UH | -1.49 | -1.54 | 0.00 | -0.83 | -2.11 | -1.53 |

## r2

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | -0.006 | 0.003 | -0.542 | 0.340 | -2.624 | 0.335 |
| afono_UH | -0.002 | 0.009 | -0.421 | 0.368 | -3.517 | 0.362 |
| aunuu_UH | -0.023 | -0.040 | -0.589 | 0.335 | 0.196 | 0.289 |
| poloa_UH | -0.000 | 0.004 | -0.633 | 0.261 | 0.128 | 0.243 |
| vaipito_UH | -0.005 | -0.000 | -0.546 | 0.316 | 0.178 | 0.349 |

## spearman_r

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH |  | 0.128 | 0.313 | 0.625 | 0.629 | 0.641 |
| afono_UH |  | 0.132 | 0.304 | 0.620 | 0.641 | 0.631 |
| aunuu_UH |  | 0.102 | 0.299 | 0.585 | 0.618 | 0.563 |
| poloa_UH |  | 0.099 | 0.336 | 0.562 | 0.581 | 0.557 |
| vaipito_UH |  | 0.119 | 0.334 | 0.599 | 0.632 | 0.624 |

## obs_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 20.69 | 20.69 | 20.69 | 20.69 | 20.69 | 20.69 |
| afono_UH | 20.45 | 20.45 | 20.45 | 20.45 | 20.45 | 20.45 |
| aunuu_UH | 15.51 | 15.51 | 15.51 | 15.51 | 15.51 | 15.51 |
| poloa_UH | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 |
| vaipito_UH | 20.09 | 20.09 | 20.09 | 20.09 | 20.09 | 20.09 |

## pred_std

| station | pooled_mean | month_climatology | persistence | ridge | tweedie_glm | gbm |
|---|---|---|---|---|---|---|
| aasu_UH | 0.00 | 1.90 | 20.68 | 9.98 | 39.24 | 11.60 |
| afono_UH | 0.00 | 1.84 | 20.44 | 10.72 | 44.14 | 10.73 |
| aunuu_UH | 0.00 | 1.47 | 15.51 | 8.49 | 4.25 | 7.40 |
| poloa_UH | 0.00 | 1.88 | 18.87 | 9.82 | 7.64 | 9.74 |
| vaipito_UH | 0.00 | 1.88 | 20.09 | 9.92 | 13.50 | 12.27 |
