"""
Visualization helpers used by the LAND pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap


def _save_and_close(fig, path, dpi: int = 150, message: Optional[str] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    if message:
        print(message)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optuna tuning plots
# ---------------------------------------------------------------------------
def save_optuna_visualizations(study, out_dir: Path) -> None:
    """Save common Optuna matplotlib visualizations."""
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )

        figures = {
            "hp_importance.png": plot_param_importances(study),
            "optimization_history.png": plot_optimization_history(study),
            "slice_plots.png": plot_slice(study),
        }

        for filename, fig in figures.items():
            if isinstance(fig, np.ndarray):
                target_fig = fig.ravel()[0].figure if fig.size else None
            elif hasattr(fig, "figure"):
                target_fig = fig.figure
            else:
                target_fig = fig
            if target_fig is None:
                continue
            _save_and_close(target_fig, Path(out_dir) / filename)
    except Exception as e:
        print(f"  WARNING: Could not generate tuning visuals: {e}")


# ---------------------------------------------------------------------------
# Data-split and training plots
# ---------------------------------------------------------------------------
def plot_split_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    station_groups: Dict[str, str],
    train_years: Tuple[int, int],
    val_years: Tuple[int, int],
    test_years: Tuple[int, int],
    save_path: Optional[Path] = None,
    title: str = "Spatiotemporal Split",
):
    unique_stations = sorted(set(str(s) for s in stations))
    yr_int = years.astype(int)
    unique_years = sorted(set(yr_int))
    s2i = {s: i for i, s in enumerate(unique_stations)}
    y2j = {y: j for j, y in enumerate(unique_years)}

    grid = np.zeros((len(unique_stations), len(unique_years)), dtype=int)
    for k in range(len(stations)):
        si = s2i[str(stations[k])]
        yj = y2j[int(yr_int[k])]
        role = station_groups.get(str(stations[k]), "train")
        yr_val = int(yr_int[k])

        in_train_yr = train_years[0] <= yr_val <= train_years[1]
        in_val_yr = val_years[0] <= yr_val <= val_years[1]
        in_test_yr = test_years[0] <= yr_val <= test_years[1]

        if role == "train" and in_train_yr:
            grid[si, yj] = 1
        elif role == "val" and in_val_yr:
            grid[si, yj] = 2
        elif role == "test" and in_test_yr:
            grid[si, yj] = 3
        elif role == "train" and in_val_yr:
            grid[si, yj] = 4
        elif role == "train" and in_test_yr:
            grid[si, yj] = 5
        elif grid[si, yj] == 0:
            grid[si, yj] = 6

    colours = ["white", "#4c72b0", "#55a868", "#c44e52", "#b5cf6b", "#f4a460", "#cccccc"]
    labels = ["No data", "Train", "Val spatial", "Test spatial", "Val temporal", "Test temporal", "Unused"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22), max(5, len(unique_stations) * 0.35)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(unique_stations)))
    ax.set_yticklabels(unique_stations, fontsize=7)
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    ax.set_title(title)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=True)

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path, message=f"  Split heatmap saved to {save_path}")
    return fig


def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Prediction / evaluation plots
# ---------------------------------------------------------------------------
def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Observed",
    units: str = "mm",
    save_path: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if save_path:
            _save_and_close(fig, save_path)
        return fig

    ax.scatter(yt, yp, s=4, alpha=0.3, rasterized=True)
    lo = min(yt.min(), yp.min(), 0)
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
    ax.set_xlabel(f"Observed ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_model_architecture(
    model,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
):
    """Save a simple block-diagram summary of the model layers."""
    import torch.nn as nn

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    for name, m in model.named_modules():
        if name == "" or list(m.children()):
            continue
        t = type(m).__name__
        shape = ""
        if isinstance(m, nn.Linear):
            shape = f"{m.in_features}→{m.out_features}"
        elif isinstance(m, nn.Conv2d):
            shape = f"{m.in_channels}→{m.out_channels} k={m.kernel_size[0]}"
        rows.append((name, t, shape))

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.45)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, len(rows) + 1)
    ax.axis("off")
    ax.text(5, len(rows) + 0.5, f"{model_name}  ({total:,} trainable params)",
            ha="center", fontsize=12, fontweight="bold")
    for i, (name, t, shape) in enumerate(rows):
        y = len(rows) - i - 0.5
        ax.add_patch(plt.Rectangle((1, y - 0.35), 8, 0.7, facecolor="steelblue",
                                    alpha=0.15, edgecolor="steelblue"))
        ax.text(1.2, y, f"{t}  {name}", fontsize=8, va="center")
        if shape:
            ax.text(8.8, y, shape, fontsize=7, ha="right", va="center", color="gray")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_wetdry_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_mm: float = 1.0,
    title: str = "Wet/Dry Evaluation",
    save_path: Optional[Path] = None,
    period_noun: str = "day",
) -> "plt.Figure":
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    obs_wet = yt >= threshold_mm
    pred_wet = yp >= threshold_mm

    tp = int(np.sum(obs_wet & pred_wet))
    fn = int(np.sum(obs_wet & ~pred_wet))
    fp = int(np.sum(~obs_wet & pred_wet))
    tn = int(np.sum(~obs_wet & ~pred_wet))
    n = len(yt)

    def _safe(num, denom):
        return float(num / denom) if denom > 0 else float("nan")

    pod = _safe(tp, tp + fn)
    far = _safe(fp, tp + fp)
    freq_bias = _safe(tp + fp, tp + fn)
    csi = _safe(tp, tp + fp + fn)
    tc = (tp + fp) * (tp + fn) / n if n > 0 else 0.0
    ets = _safe(tp - tc, tp + fp + fn - tc)
    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = _safe(hss_num, hss_denom)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"{title}\n(wet-{period_noun} threshold = {threshold_mm:.1f} mm)", fontsize=13, fontweight="bold")

    ax_cm = axes[0, 0]
    cm = np.array([[tp, fp], [fn, tn]], dtype=float)
    labels = np.array([
        [f"Hit\n{tp:,}", f"False\nAlarm\n{fp:,}"],
        [f"Miss\n{fn:,}", f"Correct\nNeg\n{tn:,}"],
    ])
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Obs Wet", "Obs Dry"], fontsize=10)
    ax_cm.set_yticklabels(["Pred Wet", "Pred Dry"], fontsize=10)
    ax_cm.set_title("Contingency Table", fontsize=11)
    for r in range(2):
        for c in range(2):
            ax_cm.text(c, r, labels[r, c], ha="center", va="center",
                       fontsize=9, color="white" if cm[r, c] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_sc = axes[0, 1]
    wet_mask = obs_wet
    if wet_mask.sum() >= 2:
        yt_w, yp_w = yt[wet_mask], yp[wet_mask]
        hi = max(float(yt_w.max()), float(yp_w.max())) * 1.05
        ax_sc.scatter(yt_w, yp_w, s=4, alpha=0.25, color="steelblue", rasterized=True)
        ax_sc.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
        res = yp_w - yt_w
        r2 = float(1.0 - np.sum(res ** 2) / np.sum((yt_w - yt_w.mean()) ** 2)) if yt_w.std() > 0 else float("nan")
        ax_sc.set_title(
            f"Wet-{period_noun} scatter (n={int(wet_mask.sum()):,})\n"
            f"MAE={float(np.mean(np.abs(res))):.2f} mm  R²={r2:.3f}",
            fontsize=10,
        )
    else:
        ax_sc.set_title(f"Wet-{period_noun} scatter (insufficient data)")
    ax_sc.set_xlabel("Observed (mm)", fontsize=9)
    ax_sc.set_ylabel("Predicted (mm)", fontsize=9)
    ax_sc.set_xlim(left=0)
    ax_sc.set_ylim(bottom=0)
    ax_sc.grid(alpha=0.3)

    ax_kd = axes[1, 0]
    if wet_mask.sum() >= 5:
        yt_w = yt[wet_mask]
        yp_w_obs = yp[wet_mask]
        q99 = float(np.nanpercentile(yt_w, 99))
        bins = np.linspace(threshold_mm, q99, 40)
        ax_kd.hist(yt_w, bins=bins, alpha=0.55, color="steelblue", density=True,
                   label=f"Observed wet {period_noun}s (n={int(wet_mask.sum()):,})")
        ax_kd.hist(yp_w_obs, bins=bins, alpha=0.55, color="coral", density=True,
                   label=f"Predicted | obs wet (n={int(wet_mask.sum()):,})")
        ax_kd.set_title(f"Wet-{period_noun} amount distribution (≥{threshold_mm:.1f} mm)", fontsize=10)
        ax_kd.legend(fontsize=8)
    else:
        ax_kd.set_title(f"Wet-{period_noun} distribution (insufficient data)")
    ax_kd.set_xlabel("Rainfall (mm)", fontsize=9)
    ax_kd.set_ylabel("Density", fontsize=9)
    ax_kd.grid(alpha=0.3)

    ax_bar = axes[1, 1]
    skill_scores = {
        "POD": pod,
        "1-FAR": (1.0 - far) if np.isfinite(far) else float("nan"),
        "CSI": csi,
        "ETS": ets,
        "HSS": hss,
    }
    names = list(skill_scores.keys())
    vals = [v if np.isfinite(v) else 0.0 for v in skill_scores.values()]
    colours_bar = ["steelblue" if v >= 0 else "tomato" for v in vals]
    bars = ax_bar.bar(names, vals, color=colours_bar, alpha=0.8, edgecolor="white")
    ax_bar.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_bar.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_bar.set_ylim(-0.1, 1.15)
    ax_bar.set_title(
        f"Classification skill scores\n"
        f"Freq Bias={freq_bias:.3f}  n_total={n:,}",
        fontsize=10,
    )
    ax_bar.set_ylabel("Score", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, skill_scores.values()):
        if np.isfinite(val):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig
