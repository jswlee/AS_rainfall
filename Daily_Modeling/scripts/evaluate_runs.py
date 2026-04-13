"""
evaluate_runs.py

Scans Daily_Modeling/output/results and produces a comparative evaluation
across all training runs. Outputs are saved to Daily_Modeling/output/evaluation/.

Produced files:
  run_summary.txt             - Text summary with hyperparams and all metrics
  cv_metrics_comparison.png   - CV val R², RMSE, MAE horizontal bar chart
  test_metrics_comparison.png - Test set metrics (all / spatial / temporal)
  cv_vs_test_comparison.png   - CV val vs test side-by-side bars
  fold_distributions.png      - Per-fold metric box plots
  hyperparameter_effects.png  - LR / batch-size / parity scatter plots
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "evaluation"

FOLD_METRICS = ["r2", "rmse", "mae", "mbe", "spearman_r", "csi"]
METRIC_LABELS = {
    "r2": "R²",
    "rmse": "RMSE (mm)",
    "mae": "MAE (mm)",
    "mbe": "MBE (mm)",
    "spearman_r": "Spearman r",
    "csi": "CSI",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _nan(v):
    try:
        return float("nan") if v is None else float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_run(run_dir: Path) -> dict:
    """Load all metrics and hyperparameters for a single run directory."""
    run = {"name": run_dir.name}

    # Hyperparameters
    hp_path = run_dir / "hyperparameters.json"
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
        run["loss_type"] = hp.get("loss_type", "unknown")
        run["output_head"] = hp.get("output_head", "unknown")
        run["learning_rate"] = _nan(hp.get("learning_rate"))
        run["batch_size"] = int(hp.get("batch_size", 0))
        run["dropout_rate"] = _nan(hp.get("dropout_rate"))
        run["use_batch_norm"] = bool(hp.get("use_batch_norm", False))

    # CV strategy and monitor inferred from directory name
    run["cv_strategy"] = "spatial" if "cv3spatial" in run_dir.name else "both"
    m = re.search(r"_(mse|mae)_", run_dir.name)
    run["monitor"] = m.group(1) if m else "unknown"

    # Completed folds from cv_summary
    cv_path = run_dir / "cv_summary.json"
    if cv_path.exists():
        with open(cv_path) as f:
            cv = json.load(f)
        run["cv_n_folds"] = cv.get("n_completed_folds", 0)
    else:
        run["cv_n_folds"] = 0

    # Per-fold metrics
    fold_metrics = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        m_path = fold_dir / "metrics_cv_val.json"
        if m_path.exists():
            with open(m_path) as f:
                fm = json.load(f)
            fm["fold"] = fold_dir.name
            fold_metrics.append(fm)
    run["fold_metrics"] = fold_metrics

    # Aggregate fold metrics (mean ± std)
    if fold_metrics:
        for key in FOLD_METRICS:
            vals = [_nan(fm.get(key)) for fm in fold_metrics]
            vals = [v for v in vals if not np.isnan(v)]
            run[f"cv_{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            run[f"cv_{key}_std"] = float(np.std(vals)) if vals else float("nan")

    # Inference / test set metrics
    inf_dir = run_dir / "inference"
    for split in ["all", "spatial", "temporal"]:
        m_path = inf_dir / f"metrics_test_{split}.json"
        if m_path.exists():
            with open(m_path) as f:
                tm = json.load(f)
            for k, v in tm.items():
                run[f"test_{split}_{k}"] = _nan(v)

    return run


# ---------------------------------------------------------------------------
# Short display label
# ---------------------------------------------------------------------------

def short_label(name: str) -> str:
    """Compact display label derived from the run directory name."""
    label = name.replace("land_daily_", "")
    label = label.replace("bernoulli_gamma", "bg")
    label = label.replace("_mse_cv3both_n100", "")
    label = label.replace("_mse_cv3spatial_n100", "_spatial")
    label = label.replace("_lowlrlowbatch_pat20", "_llb20")
    label = label.replace("_superlowlr", "_slr")
    label = label.replace("_lowlr", "_llr")
    return label


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def write_text_summary(runs: list, output_dir: Path):
    """Write a detailed text summary of all runs to file and stdout."""
    SEP = "=" * 100
    THIN = "-" * 100
    lines = [SEP, "DAILY MODELING RUN COMPARISON SUMMARY", SEP]

    for run in runs:
        lines += [f"\n{THIN}", f"Run: {run['name']}", THIN]

        lines.append("  Hyperparameters:")
        lines.append(f"    Loss Type:      {run.get('loss_type', 'N/A')}")
        lines.append(f"    Output Head:    {run.get('output_head', 'N/A')}")
        lines.append(f"    CV Strategy:    {run.get('cv_strategy', 'N/A')}")
        lines.append(f"    Monitor:        {run.get('monitor', 'N/A')}")
        lr = run.get("learning_rate", float("nan"))
        lines.append(f"    Learning Rate:  {lr:.3e}" if not np.isnan(lr) else "    Learning Rate:  N/A")
        lines.append(f"    Batch Size:     {run.get('batch_size', 'N/A')}")
        dr = run.get("dropout_rate", float("nan"))
        lines.append(f"    Dropout Rate:   {dr:.2f}" if not np.isnan(dr) else "    Dropout Rate:   N/A")
        lines.append(f"    Batch Norm:     {run.get('use_batch_norm', False)}")
        lines.append(f"    Completed Folds: {run.get('cv_n_folds', '?')}")

        lines.append("\n  Cross-Validation (val set, mean ± std across folds):")
        for key in FOLD_METRICS:
            m_val = run.get(f"cv_{key}_mean", float("nan"))
            s_val = run.get(f"cv_{key}_std", float("nan"))
            label = METRIC_LABELS.get(key, key)
            if not np.isnan(m_val):
                lines.append(f"    {label:<14}  {m_val:.4f} ± {s_val:.4f}")

        if run.get("fold_metrics"):
            lines.append("\n  Per-Fold Detail:")
            for fm in run["fold_metrics"]:
                parts = [
                    f"R²={fm.get('r2', float('nan')):.4f}",
                    f"RMSE={fm.get('rmse', float('nan')):.4f}",
                    f"MAE={fm.get('mae', float('nan')):.4f}",
                    f"Spearman r={fm.get('spearman_r', float('nan')):.4f}",
                    f"CSI={fm.get('csi', float('nan')):.4f}",
                ]
                lines.append(f"    {fm['fold']}: {', '.join(parts)}")

        lines.append("\n  Test Set Metrics:")
        for split in ["all", "spatial", "temporal"]:
            prefix = f"test_{split}_"
            if f"{prefix}r2" in run:
                parts = [
                    f"R²={run.get(f'{prefix}r2', float('nan')):.4f}",
                    f"RMSE={run.get(f'{prefix}rmse', float('nan')):.4f}",
                    f"MAE={run.get(f'{prefix}mae', float('nan')):.4f}",
                    f"MBE={run.get(f'{prefix}mbe', float('nan')):.4f}",
                    f"Spearman r={run.get(f'{prefix}spearman_r', float('nan')):.4f}",
                ]
                lines.append(f"    [{split:8s}] {', '.join(parts)}")

    # Rankings table
    lines += [f"\n{SEP}", "RANKINGS (by CV R², descending):"]
    sorted_runs = sorted(runs, key=lambda r: r.get("cv_r2_mean", float("-inf")), reverse=True)
    header = f"  {'Rank':<5} {'Run':<65} {'CV R²':>8} {'CV RMSE':>10} {'CV MAE':>10} {'Test R²':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for i, r in enumerate(sorted_runs, 1):
        lines.append(
            f"  {i:<5} {r['name']:<65} "
            f"{r.get('cv_r2_mean', float('nan')):>8.4f} "
            f"{r.get('cv_rmse_mean', float('nan')):>10.4f} "
            f"{r.get('cv_mae_mean', float('nan')):>10.4f} "
            f"{r.get('test_all_r2', float('nan')):>8.4f}"
        )

    text = "\n".join(lines)
    print(text)
    out_path = output_dir / "run_summary.txt"
    out_path.write_text(text)
    print(f"\nText summary saved to: {out_path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)


def _barh_annotated(ax, y, values, errors, color, label=None, height=0.6, capsize=4):
    """Horizontal bar chart with value labels; errors may contain NaN."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    # Replace NaN errors with 0 for xerr (no visible error bar)
    xerr = np.where(np.isnan(errors), 0.0, errors)
    bars = ax.barh(
        y, values, xerr=xerr,
        color=color, alpha=0.82, capsize=capsize, height=height, label=label,
        error_kw={"linewidth": 1.2},
    )
    for bar, val, err in zip(bars, values, errors):
        if np.isnan(val):
            continue
        pad = (err if not np.isnan(err) else 0.0)
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=7.5, clip_on=False,
        )


# ---------------------------------------------------------------------------
# Plot 1 – CV metrics comparison
# ---------------------------------------------------------------------------

def plot_cv_metrics(runs: list, output_dir: Path):
    """Horizontal bar charts of CV val R², RMSE, MAE across runs."""
    sorted_runs = sorted(runs, key=lambda r: r.get("cv_r2_mean", 0), reverse=True)
    labels = [short_label(r["name"]) for r in sorted_runs]
    y = np.arange(len(sorted_runs))
    height = max(5, len(sorted_runs) * 0.7)

    fig, axes = plt.subplots(1, 3, figsize=(18, height))
    spec = [
        ("cv_r2_mean",   "cv_r2_std",   "R²",        "steelblue"),
        ("cv_rmse_mean", "cv_rmse_std", "RMSE (mm)", "coral"),
        ("cv_mae_mean",  "cv_mae_std",  "MAE (mm)",  "mediumseagreen"),
    ]
    for ax, (mk, sk, xlabel, color) in zip(axes, spec):
        means = np.array([r.get(mk, float("nan")) for r in sorted_runs])
        stds  = np.array([r.get(sk, float("nan")) for r in sorted_runs])
        _barh_annotated(ax, y, means, stds, color)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"CV Val {xlabel}\n(mean ± std across folds)", fontsize=10)
        _style_ax(ax)

    plt.suptitle("Cross-Validation Metrics Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "cv_metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 – Test set metrics by holdout type
# ---------------------------------------------------------------------------

def plot_test_metrics(runs: list, output_dir: Path):
    """Grouped horizontal bar charts of test set metrics (all/spatial/temporal)."""
    sorted_runs = sorted(runs, key=lambda r: r.get("cv_r2_mean", 0), reverse=True)
    labels = [short_label(r["name"]) for r in sorted_runs]
    y = np.arange(len(sorted_runs))
    height = max(5, len(sorted_runs) * 0.95)

    splits = ["all", "spatial", "temporal"]
    colors = {"all": "steelblue", "spatial": "darkorange", "temporal": "mediumpurple"}
    metrics = [("r2", "R²"), ("rmse", "RMSE (mm)"), ("mae", "MAE (mm)")]
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, height))
    for ax, (metric, mlabel) in zip(axes, metrics):
        for i, split in enumerate(splits):
            vals = np.array([r.get(f"test_{split}_{metric}", float("nan")) for r in sorted_runs])
            offset = (i - 1) * width
            ax.barh(y + offset, vals, height=width, color=colors[split], alpha=0.82, label=split.capitalize())
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(mlabel, fontsize=10)
        ax.set_title(f"Test {mlabel}\n(by holdout type)", fontsize=10)
        _style_ax(ax)
        if metric == "r2":
            ax.legend(fontsize=8)

    plt.suptitle("Test Set Metrics by Holdout Type", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "test_metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 – CV val vs test side-by-side
# ---------------------------------------------------------------------------

def plot_cv_vs_test(runs: list, output_dir: Path):
    """Side-by-side horizontal bars: CV val vs test (all) for R², RMSE, MAE."""
    sorted_runs = sorted(runs, key=lambda r: r.get("cv_r2_mean", 0), reverse=True)
    labels = [short_label(r["name"]) for r in sorted_runs]
    y = np.arange(len(sorted_runs))
    height = max(5, len(sorted_runs) * 0.9)
    width = 0.35

    spec = [
        ("cv_r2_mean",   "cv_r2_std",   "test_all_r2",   "R²"),
        ("cv_rmse_mean", "cv_rmse_std", "test_all_rmse", "RMSE (mm)"),
        ("cv_mae_mean",  "cv_mae_std",  "test_all_mae",  "MAE (mm)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, height))
    for ax, (cv_k, cv_s, test_k, xlabel) in zip(axes, spec):
        cv_vals   = np.array([r.get(cv_k, float("nan")) for r in sorted_runs])
        cv_stds   = np.array([r.get(cv_s, float("nan")) for r in sorted_runs])
        test_vals = np.array([r.get(test_k, float("nan")) for r in sorted_runs])

        xerr = np.where(np.isnan(cv_stds), 0.0, cv_stds)
        ax.barh(y + width / 2, cv_vals, xerr=xerr, height=width,
                color="steelblue", alpha=0.82, label="CV val",
                capsize=3, error_kw={"linewidth": 1.2})
        ax.barh(y - width / 2, test_vals, height=width,
                color="tomato", alpha=0.82, label="Test (all)")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"{xlabel}: CV val vs Test", fontsize=10)
        _style_ax(ax)
        ax.legend(fontsize=8)

    plt.suptitle("CV Validation vs Test Set Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "cv_vs_test_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 – Per-fold metric distributions
# ---------------------------------------------------------------------------

def plot_fold_distributions(runs: list, output_dir: Path):
    """Box plots of per-fold metric distributions per run."""
    valid = [r for r in runs if r.get("fold_metrics")]
    if not valid:
        return
    sorted_runs = sorted(valid, key=lambda r: r.get("cv_r2_mean", 0), reverse=True)
    labels = [short_label(r["name"]) for r in sorted_runs]
    height = max(4, len(sorted_runs) * 0.8)

    fold_keys = [("r2", "R²"), ("rmse", "RMSE (mm)"), ("mae", "MAE (mm)"), ("spearman_r", "Spearman r")]

    fig, axes = plt.subplots(1, len(fold_keys), figsize=(5 * len(fold_keys), height))
    for ax, (key, xlabel) in zip(axes, fold_keys):
        data = []
        for run in sorted_runs:
            vals = [_nan(fm.get(key)) for fm in run["fold_metrics"]]
            data.append([v for v in vals if not np.isnan(v)])

        bp = ax.boxplot(
            data, vert=False, patch_artist=True, tick_labels=labels,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.65)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"Fold Distribution: {xlabel}", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        _style_ax(ax)

    plt.suptitle("Per-Fold Metric Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "fold_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 – Hyperparameter effects
# ---------------------------------------------------------------------------

def plot_hyperparameter_effects(runs: list, output_dir: Path):
    """Scatter plots: LR vs CV R², batch size vs CV R², and CV R² vs test R²."""
    loss_types = sorted({r.get("loss_type", "unknown") for r in runs})
    cv_strats  = sorted({r.get("cv_strategy", "both") for r in runs})
    palette = ["steelblue", "coral", "mediumseagreen", "gold"]
    colors  = dict(zip(loss_types, palette))
    markers = dict(zip(cv_strats, ["o", "s", "^"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for run in runs:
        lt     = run.get("loss_type", "unknown")
        cs     = run.get("cv_strategy", "both")
        color  = colors.get(lt, "gray")
        marker = markers.get(cs, "o")
        cv_r2  = run.get("cv_r2_mean", float("nan"))
        lbl    = short_label(run["name"])

        # Learning rate vs CV R²
        lr = run.get("learning_rate", float("nan"))
        if not (np.isnan(lr) or np.isnan(cv_r2)):
            axes[0].scatter(lr, cv_r2, color=color, marker=marker, s=90, alpha=0.85, zorder=3)
            axes[0].annotate(lbl, (lr, cv_r2), fontsize=6.5, ha="left", va="bottom",
                             xytext=(3, 3), textcoords="offset points")

        # Batch size vs CV R²
        bs = float(run.get("batch_size", float("nan")))
        if not (np.isnan(bs) or np.isnan(cv_r2)):
            axes[1].scatter(bs, cv_r2, color=color, marker=marker, s=90, alpha=0.85, zorder=3)
            axes[1].annotate(lbl, (bs, cv_r2), fontsize=6.5, ha="left", va="bottom",
                             xytext=(3, 3), textcoords="offset points")

        # CV R² vs Test R² parity
        test_r2 = run.get("test_all_r2", float("nan"))
        if not (np.isnan(cv_r2) or np.isnan(test_r2)):
            axes[2].scatter(cv_r2, test_r2, color=color, marker=marker, s=90, alpha=0.85, zorder=3)
            axes[2].annotate(lbl, (cv_r2, test_r2), fontsize=6.5, ha="left", va="bottom",
                             xytext=(3, 3), textcoords="offset points")

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning Rate (log scale)", fontsize=10)
    axes[0].set_ylabel("CV Val R²", fontsize=10)
    axes[0].set_title("Learning Rate vs CV R²", fontsize=11)

    axes[1].set_xlabel("Batch Size", fontsize=10)
    axes[1].set_ylabel("CV Val R²", fontsize=10)
    axes[1].set_title("Batch Size vs CV R²", fontsize=11)

    # Parity line
    all_r2 = [r.get("cv_r2_mean", float("nan")) for r in runs] + \
             [r.get("test_all_r2", float("nan")) for r in runs]
    finite = [v for v in all_r2 if not np.isnan(v)]
    if finite:
        lo, hi = min(finite) - 0.05, max(finite) + 0.05
        axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="parity")
    axes[2].set_xlabel("CV Val R²", fontsize=10)
    axes[2].set_ylabel("Test R² (all)", fontsize=10)
    axes[2].set_title("CV R² vs Test R² (parity)", fontsize=11)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[lt],
               markersize=8, label=f"loss={lt}")
        for lt in loss_types
    ] + [
        Line2D([0], [0], marker=markers[cs], color="gray",
               markersize=8, label=f"cv={cs}")
        for cs in cv_strats
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Hyperparameter Effects on Performance", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = output_dir / "hyperparameter_effects.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if run_dir.is_dir():
            runs.append(load_run(run_dir))

    if not runs:
        print(f"No run directories found in {RESULTS_DIR}")
        return

    print(f"Found {len(runs)} runs in {RESULTS_DIR}\n")

    write_text_summary(runs, OUTPUT_DIR)
    plot_cv_metrics(runs, OUTPUT_DIR)
    plot_test_metrics(runs, OUTPUT_DIR)
    plot_cv_vs_test(runs, OUTPUT_DIR)
    plot_fold_distributions(runs, OUTPUT_DIR)
    plot_hyperparameter_effects(runs, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
