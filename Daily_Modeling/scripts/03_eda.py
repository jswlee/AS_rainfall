"""
Step 3: Exploratory data analysis on the assembled dataset.

Produces plots for rainfall distributions, seasonality, station coverage,
reanalysis variable summaries, DEM statistics, spatial maps, correlation
analysis, temporal autocorrelation, exceedance curves, dry/wet spell
distributions, and model architecture diagrams — all broken down by
the spatio-temporal train/val/test splits.

Usage:
    python -m Daily_Modeling.scripts.03_eda
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.data_utils.load_raw import load_station_metadata
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
    compute_year_boundaries,
)
from Daily_Modeling.utils.visualization import (
    plot_rainfall_histograms,
    plot_monthly_seasonality,
    plot_station_sample_counts,
    plot_per_station_histograms,
    plot_station_map,
    plot_reanalysis_rainfall_correlation,
    plot_temporal_autocorrelation,
    plot_rainfall_exceedance,
    plot_dry_wet_spells,
    plot_rainfall_by_station_boxplot,
    plot_annual_rainfall_trends,
    plot_model_architecture,
)


def _plot_rainfall_histograms_detailed(rain_mm, split_indices, save_path):
    """Plot detailed rainfall histograms for all splits with log scale and zero counts."""
    import matplotlib.pyplot as plt
    
    n_splits = len(split_indices)
    fig, axes = plt.subplots(2, n_splits, figsize=(4 * n_splits, 8))
    if n_splits == 1:
        axes = axes.reshape(2, 1)
    
    for i, (name, idx) in enumerate(split_indices.items()):
        y = rain_mm[idx]
        n_zero = np.sum(y == 0)
        n_wet = np.sum(y > 0)
        pct_zero = 100 * n_zero / len(y) if len(y) > 0 else 0
        
        # Top row: full histogram with zeros
        ax = axes[0, i]
        ax.hist(y, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f"{name}\nn={len(y):,}  zeros={pct_zero:.1f}%")
        ax.set_xlabel("Rainfall (mm)")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        
        # Bottom row: wet days only (y > 0) with fitted distributions overlay
        ax = axes[1, i]
        y_wet = y[y > 0]
        if len(y_wet) > 10:
            ax.hist(y_wet, bins=50, density=True, edgecolor='black', alpha=0.7, label="Observed")
            ax.set_title(f"{name} (wet days only)\nn={len(y_wet):,}")
            ax.set_xlabel("Rainfall (mm)")
            ax.set_ylabel("Density")
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "Too few wet days", ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _fit_rainfall_distributions(rain_mm, split_indices, out_dir):
    """Fit conditional (wet-day) distributions per split and report goodness-of-fit.

    This fits distributions to Y|Y>0. For any-day likelihood comparisons (including
    zeros), use ``_fit_rainfall_hurdle_models`` which adds a point-mass at zero.
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    results = []
    
    for name, idx in split_indices.items():
        y = rain_mm[idx]
        y_wet = y[y > 0]  # Only fit to wet days
        
        if len(y_wet) < 50:
            continue
        
        n_zero = np.sum(y == 0)
        pct_zero = 100 * n_zero / len(y)
        
        row = {
            "split": name,
            "n_total": len(y),
            "n_wet": len(y_wet),
            "pct_zero": pct_zero,
            "mean_wet": np.mean(y_wet),
            "std_wet": np.std(y_wet),
            "median_wet": np.median(y_wet),
            "max_wet": np.max(y_wet),
        }
        
        # Fit distributions
        distributions = {
            "exponential": stats.expon,
            "gamma": stats.gamma,
            "lognormal": stats.lognorm,
            "weibull": stats.weibull_min,
        }
        
        best_aic = float("inf")
        best_dist = None
        
        for dist_name, dist in distributions.items():
            try:
                params = dist.fit(y_wet)
                # Log-likelihood
                ll = np.sum(dist.logpdf(y_wet, *params))
                k = len(params)  # number of parameters
                n = len(y_wet)
                aic = 2 * k - 2 * ll
                bic = k * np.log(n) - 2 * ll
                
                # KS test
                ks_stat, ks_pval = stats.kstest(y_wet, dist.cdf, args=params)
                
                row[f"{dist_name}_aic"] = aic
                row[f"{dist_name}_bic"] = bic
                row[f"{dist_name}_ks_stat"] = ks_stat
                row[f"{dist_name}_ks_pval"] = ks_pval
                row[f"{dist_name}_params"] = str(params)
                
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist_name
            except Exception as e:
                row[f"{dist_name}_error"] = str(e)
        
        row["best_dist_aic"] = best_dist
        results.append(row)
        
        # Plot fitted distributions for this split
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(y_wet, bins=50, density=True, alpha=0.5, label="Observed", edgecolor="black")
        
        # Overlay fitted PDFs
        x = np.linspace(0.01, np.percentile(y_wet, 99), 200)
        colors = {"exponential": "red", "gamma": "green", "lognormal": "blue", "weibull": "orange"}
        
        for dist_name, dist in distributions.items():
            try:
                params = dist.fit(y_wet)
                pdf = dist.pdf(x, *params)
                aic = row.get(f"{dist_name}_aic", float("inf"))
                label = f"{dist_name} (AIC={aic:.0f})"
                ax.plot(x, pdf, color=colors[dist_name], linewidth=2, label=label)
            except:
                pass
        
        ax.set_xlabel("Rainfall (mm)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution Fitting: {name} (wet days, n={len(y_wet):,})\nBest fit: {best_dist}")
        ax.legend()
        ax.set_xlim(0, np.percentile(y_wet, 99))
        
        plt.tight_layout()
        plt.savefig(out_dir / f"distribution_fit_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # Save summary table
    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_dir / "distribution_fitting_summary_wetday.csv", index=False)
        
        # Print summary
        print("\n=== Distribution Fitting Summary ===")
        for r in results:
            print(f"\n{r['split']}:")
            print(f"  n={r['n_total']:,}  wet={r['n_wet']:,}  zeros={r['pct_zero']:.1f}%")
            print(f"  Best fit (AIC): {r.get('best_dist_aic', 'N/A')}")
            for dist in ["exponential", "gamma", "lognormal", "weibull"]:
                aic = r.get(f"{dist}_aic")
                ks = r.get(f"{dist}_ks_stat")
                if aic is not None:
                    print(f"    {dist:12s}: AIC={aic:,.0f}  KS={ks:.4f}")


def _fit_rainfall_hurdle_models(rain_mm, split_indices, out_dir):
    """Fit any-day hurdle models: P(Y=0)=pi0, P(Y>0)=(1-pi0)*F+(y).

    We estimate:
      - pi0 as empirical zero rate (equivalently MLE for Bernoulli)
      - F+ parameters by MLE on wet days only

    Then we compute the *full-data* log-likelihood (including zeros) and report
    AIC/BIC for the any-day model. This matches the structure of a Bernoulli +
    continuous head (e.g. Bernoulli-Gamma, Bernoulli-Lognormal, etc.).
    """
    from scipy import stats

    distributions = {
        "exponential": stats.expon,
        "gamma": stats.gamma,
        "lognormal": stats.lognorm,
        "weibull": stats.weibull_min,
    }

    rows = []
    for split_name, idx in split_indices.items():
        y = rain_mm[idx]
        if len(y) == 0:
            continue

        is_zero = (y == 0)
        y_wet = y[~is_zero]

        n = int(len(y))
        n0 = int(is_zero.sum())
        n_wet = int(len(y_wet))
        pi0 = float(n0 / n) if n > 0 else 0.0

        # Skip if too few wet samples to fit anything
        if n_wet < 20:
            continue

        # Bernoulli log-likelihood for zeros/non-zeros
        # MLE is pi0; compute loglik with clamping for stability
        eps = 1e-12
        pi0_c = min(max(pi0, eps), 1.0 - eps)
        ll_bern = n0 * np.log(pi0_c) + (n - n0) * np.log(1.0 - pi0_c)

        base = {
            "split": split_name,
            "n_total": n,
            "n_zero": n0,
            "n_wet": n_wet,
            "pct_zero": 100.0 * pi0,
            "pi0_mle": pi0,
        }

        best_aic = float("inf")
        best_model = None

        for dist_name, dist in distributions.items():
            row = dict(base)
            try:
                params = dist.fit(y_wet)
                # Full-data log-likelihood: Bernoulli part + sum log f+(y_wet)
                ll_pos = float(np.sum(dist.logpdf(y_wet, *params)))
                ll = float(ll_bern + ll_pos)

                # Parameter count: 1 for pi0 + len(params) for positive dist
                k = 1 + len(params)
                aic = 2 * k - 2 * ll
                bic = k * np.log(n) - 2 * ll

                row.update({
                    "model": f"hurdle_{dist_name}",
                    "ll": ll,
                    "k": k,
                    "aic": aic,
                    "bic": bic,
                    "pos_params": str(params),
                })
                rows.append(row)

                if aic < best_aic:
                    best_aic = aic
                    best_model = dist_name
            except Exception as e:
                row.update({
                    "model": f"hurdle_{dist_name}",
                    "error": str(e),
                })
                rows.append(row)

        # Record best model for this split
        if best_model is not None:
            rows.append({
                **base,
                "model": "best",
                "best_model_by_aic": f"hurdle_{best_model}",
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "distribution_fitting_summary_anyday.csv", index=False)


def _tweedie_unit_deviance(y: np.ndarray, mu: float, p: float, eps: float = 1e-12) -> np.ndarray:
    """Tweedie unit deviance for 1<p<2.

    Uses the standard deviance form:
      d(y,mu) = 2 * ( y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) )

    Handles y=0 safely.
    """
    if not (1.0 < p < 2.0):
        raise ValueError("Tweedie p must be in (1,2)")
    y = np.asarray(y, dtype=np.float64)
    mu = float(max(mu, eps))

    term1 = np.where(
        y > 0,
        np.power(np.maximum(y, eps), 2.0 - p) / ((1.0 - p) * (2.0 - p)),
        0.0,
    )
    term2 = y * np.power(mu, 1.0 - p) / (1.0 - p)
    term3 = np.power(mu, 2.0 - p) / (2.0 - p)
    d = 2.0 * (term1 - term2 + term3)
    # Numerical safety
    d = np.where(np.isfinite(d), d, np.nan)
    return d


def _tweedie_p_sweep_anyday(rain_mm, split_indices, out_dir, p_grid=None):
    """Diagnose which Tweedie power p best matches any-day rainfall.

    We are NOT fitting a full Tweedie MLE here (that would require estimating dispersion
    and a proper likelihood). Instead we compare splits via average unit deviance,
    using mu = mean(y) for the split.

    Output:
      - tweedie_p_sweep.csv
      - tweedie_p_sweep.png
    """
    import matplotlib.pyplot as plt

    if p_grid is None:
        p_grid = np.round(np.linspace(1.1, 1.9, 17), 2)

    rows = []
    best = {}
    for split_name, idx in split_indices.items():
        y = np.asarray(rain_mm[idx], dtype=np.float64)
        if y.size == 0:
            continue
        mu = float(np.mean(y))
        if not np.isfinite(mu) or mu <= 0:
            continue

        best_p = None
        best_dev = float("inf")
        for p in p_grid:
            d = _tweedie_unit_deviance(y, mu=mu, p=float(p))
            dev = float(np.nanmean(d))
            rows.append({
                "split": split_name,
                "p": float(p),
                "mu": mu,
                "mean_unit_deviance": dev,
            })
            if np.isfinite(dev) and dev < best_dev:
                best_dev = dev
                best_p = float(p)

        if best_p is not None:
            best[split_name] = {"best_p": best_p, "best_mean_unit_deviance": best_dev, "mu": mu}

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tweedie_p_sweep.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for split_name in sorted(set(df["split"])):
        sdf = df[df["split"] == split_name].sort_values("p")
        ax.plot(sdf["p"], sdf["mean_unit_deviance"], marker="o", linewidth=2, label=str(split_name))

    ax.set_xlabel("Tweedie p")
    ax.set_ylabel("Mean unit deviance (lower is better)")
    ax.set_title("Tweedie p sweep (any-day; mu = split mean)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tweedie_p_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Tweedie p sweep (any-day) ===")
    for split_name, info in best.items():
        print(
            f"{split_name}: best_p={info['best_p']:.2f}  "
            f"mean_unit_deviance={info['best_mean_unit_deviance']:.4f}  mu={info['mu']:.3f}"
        )



def main():
    out = config.EDA_DIR
    out.mkdir(parents=True, exist_ok=True)

    tensors, meta = load_tensors_from_npz(device=select_device())

    stations = meta["stations"]
    years = meta["years"]
    months = meta["months"]
    rain_mm = tensors["targets"].cpu().numpy()
    climate = tensors["climate"].cpu().numpy()

    unique_stations = sorted(set(str(s) for s in stations))
    print(f"Unique stations: {len(unique_stations)}")

    # Splits (match tuning/training)
    train_yr, val_yr, test_yr = compute_year_boundaries(years)
    yr_ranges = compute_station_year_ranges(stations, years)
    groups = assign_station_groups(
        unique_stations,
        station_year_ranges=yr_ranges,
        val_years=val_yr,
        test_years=test_yr,
    )
    splits = spatiotemporal_split(
        stations,
        years,
        groups,
        train_years=train_yr,
        val_years=val_yr,
        test_years=test_yr,
    )

    # ---- Rainfall histograms (per split) ----
    core_splits = {k: splits[k] for k in ("train", "val", "test") if k in splits}
    plot_rainfall_histograms(rain_mm, core_splits, out / "rainfall_histograms.png")
    print("Saved rainfall histograms")

    # ---- Detailed rainfall histograms for all 5 splits ----
    all_splits = {k: splits[k] for k in splits if len(splits[k]) > 0}
    _plot_rainfall_histograms_detailed(rain_mm, all_splits, out / "rainfall_histograms_all_splits.png")
    print("Saved rainfall_histograms_all_splits.png")

    # ---- Distribution fitting analysis ----
    _fit_rainfall_distributions(rain_mm, all_splits, out)
    _fit_rainfall_hurdle_models(rain_mm, all_splits, out)
    _tweedie_p_sweep_anyday(rain_mm, all_splits, out)
    print("Saved distribution fitting analysis")

    # ---- Monthly seasonality ----
    plot_monthly_seasonality(rain_mm, months, core_splits, out / "monthly_seasonality.png")
    print("Saved monthly seasonality")

    # ---- Station sample counts ----
    plot_station_sample_counts(stations, core_splits, out / "station_sample_counts.png")
    print("Saved station sample counts")

    # ---- Per-station histograms ----
    plot_per_station_histograms(rain_mm, stations, out / "station_histograms")
    print("Saved per-station histograms")

    # ---- Rainfall summary table ----
    rows = []
    for name, idx in core_splits.items():
        y = rain_mm[idx]
        if len(y) == 0:
            rows.append({"split": name, "n": 0})
            continue
        rows.append({
            "split": name, "n": len(y),
            "mean": np.mean(y), "std": np.std(y),
            "min": np.min(y), "p50": np.median(y),
            "p90": np.quantile(y, 0.9), "p95": np.quantile(y, 0.95),
            "p99": np.quantile(y, 0.99), "max": np.max(y),
            "pct_zero": 100.0 * np.mean(y == 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "rainfall_summary.csv", index=False)
    print("Saved rainfall_summary.csv")
    print(df.to_string(index=False))

    # ---- Reanalysis variable means per split ----
    variables = meta.get("variables", np.array([]))
    var_names = list(variables) if len(variables) > 0 else []
    if climate.ndim == 4 and len(variables) > 0:
        re_mean = np.nanmean(climate, axis=(2, 3))  # (N, C)
        rows = []
        for name, idx in core_splits.items():
            for ci, v in enumerate(variables):
                vals = re_mean[idx, ci]
                rows.append({
                    "split": name, "variable": str(v),
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                })
        re_df = pd.DataFrame(rows)
        re_df.to_csv(out / "reanalysis_summary.csv", index=False)
        print("Saved reanalysis_summary.csv")

        # Train vs test shift
        train_m = re_df[re_df["split"] == "train"].set_index("variable")["mean"]
        test_m = re_df[re_df["split"] == "test"].set_index("variable")["mean"]
        delta = (test_m - train_m).sort_values(key=lambda s: np.abs(s), ascending=False)
        print("\nReanalysis variable shift (test - train mean):")
        print(delta.to_string())

    # ---- Station group summary ----
    print("\nStation assignments:")
    for role in ("train", "val", "test"):
        sts = [s for s, r in groups.items() if r == role]
        print(f"  {role}: {len(sts)} stations - {', '.join(sts)}")

    # ==================================================================
    # NEW: Enhanced visualizations
    # ==================================================================

    # ---- Spatial map of station locations coloured by role ----
    try:
        station_meta = load_station_metadata()
        plot_station_map(station_meta, station_groups=groups,
                         save_path=out / "station_map.png")
        print("Saved station_map.png")
    except Exception as e:
        print(f"WARNING: Could not generate station map: {e}")

    # ---- Reanalysis-rainfall correlation ----
    if climate.ndim == 4 and len(var_names) > 0:
        plot_reanalysis_rainfall_correlation(
            climate, rain_mm, var_names,
            save_path=out / "reanalysis_rainfall_correlation.png",
        )
        print("Saved reanalysis_rainfall_correlation.png")

    # ---- Temporal autocorrelation ----
    plot_temporal_autocorrelation(
        rain_mm, stations, max_lag=14,
        save_path=out / "temporal_autocorrelation.png",
    )
    print("Saved temporal_autocorrelation.png")

    # ---- Rainfall exceedance curve ----
    plot_rainfall_exceedance(
        rain_mm, split_indices=core_splits,
        save_path=out / "rainfall_exceedance.png",
    )
    print("Saved rainfall_exceedance.png")

    # ---- Dry/wet spell distributions ----
    plot_dry_wet_spells(
        rain_mm, stations, threshold_mm=1.0,
        save_path=out / "dry_wet_spells.png",
    )
    print("Saved dry_wet_spells.png")

    # ---- Rainfall box plots by station ----
    plot_rainfall_by_station_boxplot(
        rain_mm, stations,
        save_path=out / "rainfall_boxplot_by_station.png",
    )
    print("Saved rainfall_boxplot_by_station.png")

    # ---- Annual rainfall trends ----
    plot_annual_rainfall_trends(
        rain_mm, years, stations,
        save_path=out / "annual_rainfall_trends.png",
    )
    print("Saved annual_rainfall_trends.png")

    # ---- Model architecture diagrams (dynamic from actual model classes) ----
    try:
        from Daily_Modeling.models.land import create_land_model
        from Daily_Modeling.models.site_mlp import SiteMLP, compute_input_size

        # LAND model with default HP
        land_metadata = {
            "climate_shape": tuple(tensors["climate"].shape[1:]),
            "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
            "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
            "num_month_features": int(tensors["temporal"].shape[1]),
            "num_climate_vars": int(tensors["climate"].shape[1]),
        }
        land_model = create_land_model(config.LAND_DEFAULT_HP, land_metadata)

        # Build dummy input for torchviz computation graph
        bs = 2
        land_input = {
            "climate": torch.randn(bs, *land_metadata["climate_shape"]),
            "local_dem": torch.randn(bs, *land_metadata["local_dem_shape"]),
            "regional_dem": torch.randn(bs, *land_metadata["regional_dem_shape"]),
            "temporal": torch.randn(bs, land_metadata["num_month_features"]),
        }
        plot_model_architecture(
            land_model, model_name="LAND Model (default HP)",
            input_data=land_input,
            save_path=out / "architecture_land.png",
        )
        print("Saved architecture_land.png")

        # Site MLP with default HP
        input_size = compute_input_size(
            land_metadata["climate_shape"],
            land_metadata["local_dem_shape"],
            land_metadata["regional_dem_shape"],
            land_metadata["num_month_features"],
        )
        mlp_model = SiteMLP(input_size, config.MLP_DEFAULT_HP["hidden_sizes"],
                            config.MLP_DEFAULT_HP["dropout_rate"])
        mlp_input = torch.randn(bs, input_size)
        plot_model_architecture(
            mlp_model, model_name="Site MLP (default HP)",
            input_data=mlp_input,
            save_path=out / "architecture_site_mlp.png",
        )
        print("Saved architecture_site_mlp.png")
    except Exception as e:
        print(f"WARNING: Could not generate architecture diagrams: {e}")

    print(f"\nAll EDA outputs saved to {out}")


if __name__ == "__main__":
    main()
