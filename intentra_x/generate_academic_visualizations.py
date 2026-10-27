"""Generate publication-style figures for the INTENTRA-X presentation.

This script intentionally uses a restrained Matplotlib style: white background,
standard axes, compact titles, limited colors, and no decorative dashboard UI.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "ppt_visualizations_academic"
sys.path.insert(0, str(HERE))

import generate_ppt_visualizations as base  # noqa: E402
import train_airsim_attention as taa  # noqa: E402


BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B2"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"


def configure_style() -> None:
    installed = {font.name for font in fm.fontManager.ttflist}
    serif_candidates = ["Times New Roman", "Microsoft YaHei", "SimSun", "DejaVu Serif"]
    selected = [name for name in serif_candidates if name in installed]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": selected or ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "axes.titlesize": 14,
            "axes.titleweight": "normal",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.frameon": True,
            "legend.framealpha": 0.90,
            "legend.edgecolor": LIGHT_GRAY,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig, stem: str, tight: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Architecture of the Feature-Attention LSTM Risk Model", pad=16)

    xs = [0.025, 0.225, 0.425, 0.625, 0.825]
    width, height, y = 0.15, 0.24, 0.49
    titles = ["Input sequence", "Feature attention", "Weighted sequence", "Two-layer LSTM", "Risk prediction"]
    bodies = [
        r"$X\in\mathbb{R}^{20\times9}$",
        r"$a_t=\sigma(\mathrm{MLP}(x_t))$",
        r"$x'_t=a_t\odot x_t$",
        r"$h_T=\mathrm{LSTM}(x'_{1:T})$",
        r"$p=\sigma(\mathrm{MLP}(h_T))$",
    ]
    fills = ["#F2F2F2", "#EAF2F8", "#F2F2F2", "#F2F2F2", "#FDEDEC"]
    edges = [GRAY, BLUE, GRAY, GRAY, RED]
    for x, title, body, fill, edge in zip(xs, titles, bodies, fills, edges):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + 0.16, title, ha="center", va="center", fontsize=10.5)
        ax.text(x + width / 2, y + 0.075, body, ha="center", va="center", fontsize=11)
    for i in range(4):
        ax.add_patch(
            FancyArrowPatch(
                (xs[i] + width + 0.006, y + height / 2),
                (xs[i + 1] - 0.006, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=13,
                color="black",
                linewidth=1.0,
            )
        )

    ax.text(
        0.5,
        0.33,
        "The attention gate independently weights each input feature at every time step; "
        "the LSTM then encodes temporal dependencies.",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.22,
        "Input features: wind speed, drone speed, roll, pitch, altitude, obstacle distance, "
        "obstacle detection, and wind-direction components.",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, "01_model_architecture_academic")


def generate_spatial_risk() -> None:
    flight = base.read_sample_flight()
    x_grid = np.linspace(-40, 75, 180)
    y_grid = np.linspace(-35, 75, 180)
    altitude = float(np.median(flight["altitude"]))
    z = base.compute_spatial_risk(x_grid, y_grid, altitude)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    levels = np.linspace(float(z.min()), float(z.max()), 12)
    contour = ax.contourf(x_grid, y_grid, z, levels=levels, cmap="YlOrRd", alpha=0.85)
    ax.contour(x_grid, y_grid, z, levels=6, colors="k", linewidths=0.35, alpha=0.35)

    ax.add_patch(Rectangle((-30, -30), 60, 60, fill=False, edgecolor=GRAY, lw=1.1, linestyle="--", label="Building zone"))
    ax.add_patch(Circle((20, 30), 10, fill=False, edgecolor=GREEN, lw=1.4, label="Tree zone"))
    ax.add_patch(Wedge((10, 10), 58, 0, 90, fill=False, edgecolor=BLUE, lw=1.1, linestyle=":", label="Sensor FOV"))
    ax.scatter([10], [10], marker="^", s=55, facecolor="black", edgecolor="white", linewidth=0.6, zorder=5, label="Sensor")

    x, y, risk = flight["pos_x"], flight["pos_y"], flight["risk_score"]
    ax.plot(x, y, color="black", lw=1.6, zorder=5, label="Flight trajectory")
    points = ax.scatter(x, y, c=risk, cmap="YlOrRd", vmin=0, vmax=1, s=30, edgecolor="black", linewidth=0.35, zorder=6)
    ax.scatter([x[0]], [y[0]], marker="o", s=55, facecolor=GREEN, edgecolor="black", linewidth=0.6, zorder=7, label="Start")
    ax.scatter([x[-1]], [y[-1]], marker="X", s=65, facecolor=RED, edgecolor="black", linewidth=0.6, zorder=7, label="End")

    ax.set_title(f"Spatial Risk Field and UAV Trajectory (Altitude ≈ {altitude:.0f} m)")
    ax.set_xlabel("East position, x (m)")
    ax.set_ylabel("North position, y (m)")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(y_grid.min(), y_grid.max())
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    cbar = fig.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label("Environmental risk")
    save_figure(fig, "02_spatial_risk_trajectory_academic")


def select_timeline_interval(data):
    pred, label = data["pred"], data["label"]
    rows = data["flights"]
    timestamps = np.asarray([rows[fid][idx]["timestamp"] for fid, idx in data["meta"]])
    time_sec = timestamps - timestamps[0]
    median_dt = float(np.median(np.diff(time_sec)))
    window = min(len(pred), max(300, int(round(55 / median_dt))))
    best_start, best_score = 0, -np.inf
    for start in range(0, len(pred) - window + 1, max(1, window // 25)):
        section = slice(start, start + window)
        transitions = np.abs(np.diff(label[section])).sum()
        score = 2.5 * transitions + 4.0 * np.std(pred[section]) + label[section].mean()
        if score > best_score:
            best_start, best_score = start, score
    section = slice(best_start, best_start + window)
    return time_sec[section] - time_sec[best_start], section, median_dt


def generate_timeline(data) -> None:
    threshold = base.parse_threshold()
    time_sec, section, median_dt = select_timeline_interval(data)
    pred = data["pred"][section]
    label = data["label"][section]
    importance = data["importance"][section].T

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12},
    )
    for start, end in base.contiguous_spans(label > 0.5):
        ax1.axvspan(time_sec[start], time_sec[min(end - 1, len(time_sec) - 1)] + median_dt, color=RED, alpha=0.12, lw=0)
    ax1.plot(time_sec, pred, color=BLUE, label="Predicted risk")
    ax1.axhline(threshold, color=ORANGE, linestyle="--", linewidth=1.4, label=f"Decision threshold = {threshold:.2f}")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Risk probability")
    ax1.set_title("Risk Prediction and Feature-Attention Dynamics")
    ax1.grid(alpha=0.55)
    ax1.legend(loc="upper right")
    ax1.text(0.01, 0.90, "Shaded intervals: observed collision-risk windows", transform=ax1.transAxes, fontsize=8.5, color=RED)

    image = ax2.imshow(
        importance,
        origin="upper",
        aspect="auto",
        extent=[time_sec[0], time_sec[-1], len(taa.FEATURE_NAMES) - 0.5, -0.5],
        cmap="Blues",
        interpolation="nearest",
    )
    ax2.set_yticks(np.arange(len(taa.FEATURE_NAMES)))
    ax2.set_yticklabels(taa.FEATURE_NAMES)
    ax2.set_xlabel("Relative time (s)")
    ax2.set_ylabel("Input feature")
    cbar = fig.colorbar(image, ax=ax2, pad=0.015, fraction=0.025)
    cbar.set_label("Attention weight")
    save_figure(fig, "03_risk_attention_timeline_academic")


def generate_timeline_compact(data, duration_seconds: float = 26.0) -> None:
    """Render a shorter, PPT-friendly risk timeline with fewer visible events."""
    threshold = base.parse_threshold()
    full_time, full_section, median_dt = select_timeline_interval(data)
    keep = full_time <= duration_seconds
    time_sec = full_time[keep]
    source_indices = np.arange(full_section.start, full_section.stop)[keep]
    pred = data["pred"][source_indices]
    label = data["label"][source_indices]

    fig, ax = plt.subplots(figsize=(7.6, 4.25))
    for start, end in base.contiguous_spans(label > 0.5):
        ax.axvspan(
            time_sec[start],
            time_sec[min(end - 1, len(time_sec) - 1)] + median_dt,
            color=RED,
            alpha=0.12,
            lw=0,
        )
    ax.plot(time_sec, pred, color=BLUE, label="Predicted risk")
    ax.axhline(
        threshold,
        color=ORANGE,
        linestyle="--",
        linewidth=1.4,
        label=f"Decision threshold = {threshold:.2f}",
    )
    ax.set_xlim(0, duration_seconds)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, duration_seconds + 0.1, 5))
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel("Risk probability")
    ax.set_title("Risk Prediction and Feature-Attention Dynamics")
    ax.grid(alpha=0.55)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.90,
        "Shaded intervals: observed collision-risk windows",
        transform=ax.transAxes,
        fontsize=8.2,
        color=RED,
    )
    save_figure(fig, "03_risk_attention_timeline_compact")


def generate_importance(data) -> None:
    importance = data["importance"]
    mean = importance.mean(axis=0)
    std = importance.std(axis=0)
    order = np.argsort(mean)
    labels = np.asarray(taa.FEATURE_NAMES)[order]

    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    positions = np.arange(len(order))
    ax.barh(positions, mean[order], xerr=std[order], color=BLUE, alpha=0.88, edgecolor="black", linewidth=0.45, capsize=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(0.75, float((mean + std).max() + 0.05)))
    ax.set_xlabel("Mean attention weight")
    ax.set_title("Global Feature-Attention Importance on the Test Set")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    for y, value in zip(positions, mean[order]):
        ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=8.5)
    save_figure(fig, "04_attention_feature_importance_academic")


def generate_performance(data) -> None:
    pred, label = data["pred"], data["label"]
    threshold = base.parse_threshold()
    cm = confusion_matrix(label.astype(int), (pred >= threshold).astype(int))
    roc_auc = roc_auc_score(label, pred)
    pr_auc = average_precision_score(label, pred)
    fpr, tpr, _ = roc_curve(label, pred)
    precision, recall, _ = precision_recall_curve(label, pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    ax = axes[0]
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            ax.text(j, i, f"{value}", ha="center", va="center", fontsize=13, color="white" if value > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Safe", "Risk"])
    ax.set_yticks([0, 1], ["Safe", "Risk"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(rf"(a) Confusion Matrix ($\tau$={threshold:.2f})")

    ax = axes[1]
    ax.plot(fpr, tpr, color=BLUE, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(b) ROC Curve")
    ax.grid(alpha=0.55)
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.plot(recall, precision, color=GREEN, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(label.mean(), color=GRAY, linestyle="--", linewidth=1, label=f"Prevalence = {label.mean():.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(c) Precision–Recall Curve")
    ax.grid(alpha=0.55)
    ax.legend(loc="lower left")
    save_figure(fig, "05_model_performance_academic")


def generate_weight_comparison() -> None:
    rows = base.parse_weight_comparison()
    weights = np.asarray([row[0] for row in rows])
    pr_auc = np.asarray([row[1] for row in rows])
    f1 = np.asarray([row[2] for row in rows])
    thresholds = np.asarray([row[3] for row in rows])
    x = np.arange(len(weights))

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    width = 0.30
    bars1 = ax.bar(x - width / 2, pr_auc, width, color=BLUE, edgecolor="black", linewidth=0.4, label="PR-AUC")
    bars2 = ax.bar(x + width / 2, f1, width, color=GREEN, edgecolor="black", linewidth=0.4, label="F1-score")
    ax2 = ax.twinx()
    ax2.plot(x, thresholds, color=ORANGE, marker="o", linewidth=1.5, label="Optimal threshold")
    ax.set_xticks(x, [f"{weight:g}" for weight in weights])
    ax.set_ylim(0, 0.75)
    ax2.set_ylim(0, 0.75)
    ax.set_xlabel("Positive-class weight")
    ax.set_ylabel("Validation score")
    ax2.set_ylabel("Optimal decision threshold")
    ax.set_title("Validation Performance under Different Positive-Class Weights")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower center", ncol=3)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "06_class_weight_comparison_academic")


def counterfactual_curve(data, feature_index, values, detected_rule=None):
    threshold = base.parse_threshold()
    indices = np.flatnonzero((data["label"] < 0.5) & (data["pred"] < threshold))
    if len(indices) > 420:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, size=420, replace=False))
    raw = data["test_raw"][indices]
    means, lower, upper = [], [], []
    for value in values:
        perturbed = raw.copy()
        perturbed[:, :, feature_index] = value
        if detected_rule is not None:
            perturbed[:, :, 6] = detected_rule(value)
        prediction = base.predict_raw(data["model"], perturbed, data["mean"], data["std"])
        mean = float(prediction.mean())
        ci = 1.96 * float(prediction.std(ddof=1)) / math.sqrt(len(prediction))
        means.append(mean)
        lower.append(mean - ci)
        upper.append(mean + ci)
    return np.asarray(means), np.asarray(lower), np.asarray(upper), len(indices)


def generate_counterfactual(data) -> None:
    wind_values = np.asarray([0, 3, 6, 9, 12, 15, 19], dtype=float)
    distance_values = np.asarray([5, 10, 20, 40, 60, 80, 100], dtype=float)
    wind_mean, wind_lo, wind_hi, n = counterfactual_curve(data, 0, wind_values)
    dist_mean, dist_lo, dist_hi, _ = counterfactual_curve(data, 5, distance_values, detected_rule=lambda value: float(value < 100))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    ax.plot(wind_values, wind_mean, color=BLUE, marker="o", label="Mean predicted risk")
    ax.fill_between(wind_values, wind_lo, wind_hi, color=BLUE, alpha=0.18, label="95% CI")
    ax.set_xlabel("Counterfactual wind speed (m/s)")
    ax.set_ylabel("Predicted risk")
    ax.set_title("(a) Wind-Speed Sensitivity")
    ax.set_ylim(0, max(0.45, float(wind_hi.max() + 0.05)))
    ax.grid(alpha=0.55)
    ax.legend()

    ax = axes[1]
    ax.plot(distance_values, dist_mean, color=RED, marker="s", label="Mean predicted risk")
    ax.fill_between(distance_values, dist_lo, dist_hi, color=RED, alpha=0.16, label="95% CI")
    ax.set_xlabel("Counterfactual obstacle distance (m)")
    ax.set_ylabel("Predicted risk")
    ax.set_title("(b) Obstacle-Distance Sensitivity")
    ax.set_ylim(0, max(0.45, float(dist_hi.max() + 0.05)))
    ax.grid(alpha=0.55)
    ax.legend()
    fig.suptitle(f"Counterfactual Sensitivity Analysis on Initially Safe Test Windows (n={n})", fontsize=14, y=1.02)
    save_figure(fig, "07_counterfactual_sensitivity_academic")


def parse_training_history(weight: float):
    text = base.REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"Training model with positive class_weight={re.escape(str(weight))}(.*?)(?=Training model with positive class_weight=|Class weight comparison summary:)",
        re.S,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Training block for class_weight={weight} was not found")
    epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+train_loss=([\d.]+)\s+\|\s+val_loss=([\d.]+)")
    rows = [(int(ep), float(train), float(val)) for ep, train, val in epoch_pattern.findall(match.group(1))]
    return np.asarray(rows, dtype=float)


def generate_training_curve(weight: float, stem: str) -> None:
    history = parse_training_history(weight)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(history[:, 0], history[:, 1], color=BLUE, label="train_loss")
    ax.plot(history[:, 0], history[:, 2], color=ORANGE, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE")
    ax.set_title(f"AirSim Attention-LSTM Training Curve (class_weight={weight:g})")
    ax.legend(loc="upper left")
    ax.grid(False)
    save_figure(fig, stem)


def main() -> None:
    configure_style()
    data = base.load_model_data()
    generate_architecture()
    generate_spatial_risk()
    generate_timeline(data)
    generate_importance(data)
    generate_performance(data)
    generate_weight_comparison()
    generate_counterfactual(data)
    generate_training_curve(1.0, "08_training_curve_selected_weight_1p0")
    generate_training_curve(2.32, "09_training_curve_weight_2p32")
    print(f"Generated 9 academic figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
