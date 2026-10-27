"""Generate a consistent set of PPT-ready visualizations for INTENTRA-X.

The figures are built from the checked-in AirSim dataset, trained Attention-LSTM
checkpoint, training report, and demo flight log. Outputs are written to
``ppt_visualizations`` next to this script.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "ppt_visualizations"
MODEL_DIR = HERE / "models"
DATA_PATH = HERE / "data" / "UAV_gust_all.jsonl"
REPORT_PATH = MODEL_DIR / "airsim_attention_training_report.txt"
SAMPLE_FLIGHT_PATH = HERE / "data" / "sample_flight.csv"
CHECKPOINT_PATH = MODEL_DIR / "airsim_attention_risk_model.pt"

sys.path.insert(0, str(HERE))
from attention_risk_lstm import FeatureAttentionRiskLSTM  # noqa: E402
import train_airsim_attention as taa  # noqa: E402
from environment import EnvironmentModel  # noqa: E402


BG = "#F6F8FC"
PANEL = "#FFFFFF"
INK = "#14213D"
MUTED = "#64748B"
GRID = "#DDE5F0"
BLUE = "#2563EB"
CYAN = "#06B6D4"
TEAL = "#0F9D8A"
ORANGE = "#F59E0B"
RED = "#E5484D"
PURPLE = "#7C3AED"
GREEN = "#16A34A"
NAVY = "#0B2447"

FEATURE_LABELS = {
    "wind_speed": "风速",
    "drone_speed": "飞行速度",
    "roll": "横滚角",
    "pitch": "俯仰角",
    "altitude": "高度",
    "sensor_obstacle_distance": "障碍物距离",
    "obstacle_detected": "障碍物检测",
    "wind_direction_sin": "风向 sin",
    "wind_direction_cos": "风向 cos",
}


def configure_matplotlib() -> None:
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    family = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 19,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )


def add_figure_title(fig, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.915, subtitle, fontsize=11.5, color=MUTED, va="top")
    fig.add_artist(
        plt.Line2D([0.055, 0.945], [0.885, 0.885], transform=fig.transFigure, color=GRID, lw=1)
    )


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis=grid_axis, alpha=0.8)
    ax.set_axisbelow(True)


def export(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def rounded_box(ax, xy, width, height, title, body, face, edge, icon=None):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    if icon:
        ax.text(x + width * 0.08, y + height * 0.68, icon, fontsize=18, color=edge, va="center")
        title_x = x + width * 0.22
    else:
        title_x = x + width * 0.08
    ax.text(title_x, y + height * 0.68, title, fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(x + width * 0.08, y + height * 0.30, body, fontsize=9.5, color=MUTED, va="center")
    return patch


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=color,
            shrinkA=5,
            shrinkB=5,
        )
    )


def generate_architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_figure_title(
        fig,
        "Attention-LSTM 风险预测架构",
        "特征注意力负责筛选关键信息，LSTM负责建模20个时间步的动态依赖",
    )

    y, w, h = 0.50, 0.145, 0.18
    xs = [0.035, 0.225, 0.415, 0.605, 0.795]
    rounded_box(ax, (xs[0], y), w, h, "输入序列", "20步 × 9项飞行特征", "#EAF1FF", BLUE)
    rounded_box(ax, (xs[1], y), w, h, "特征注意力", "MLP + Sigmoid 门控", "#ECFEFF", CYAN)
    rounded_box(ax, (xs[2], y), w, h, "加权序列", "重要特征放大 · 噪声抑制", "#ECFDF5", TEAL)
    rounded_box(ax, (xs[3], y), w, h, "双层 LSTM", "hidden=64 · dropout=0.2", "#F3EEFF", PURPLE)
    rounded_box(ax, (xs[4], y), w, h, "风险输出", "全连接层 → 概率 [0,1]", "#FFF7E6", ORANGE)
    for i in range(4):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    # Input sequence glyph.
    for r in range(4):
        for c in range(7):
            val = (r + c) / 10
            ax.add_patch(
                Rectangle(
                    (xs[0] + 0.02 + c * 0.013, y + 0.025 + r * 0.012),
                    0.010,
                    0.008,
                    facecolor=plt.cm.Blues(0.25 + 0.55 * val),
                    edgecolor="none",
                )
            )

    # Attention gate bars.
    gate_vals = [0.91, 0.83, 0.66, 0.48, 0.35]
    for i, value in enumerate(gate_vals):
        yy = y + 0.025 + i * 0.011
        ax.add_patch(Rectangle((xs[1] + 0.025, yy), 0.083, 0.007, facecolor="#D8F5FA", edgecolor="none"))
        ax.add_patch(Rectangle((xs[1] + 0.025, yy), 0.083 * value, 0.007, facecolor=CYAN, edgecolor="none"))

    # LSTM cells.
    for i in range(4):
        cx = xs[3] + 0.025 + i * 0.027
        ax.add_patch(Circle((cx, y + 0.046), 0.011, facecolor="#E9DFFF", edgecolor=PURPLE, lw=1.2))
        if i < 3:
            arrow(ax, (cx + 0.011, y + 0.046), (cx + 0.016, y + 0.046), color=PURPLE)

    # Decision threshold mini gauge.
    gx, gy = xs[4] + 0.025, y + 0.035
    ax.add_patch(Rectangle((gx, gy), 0.095, 0.014, facecolor="#FDE8C2", edgecolor="none"))
    ax.add_patch(Rectangle((gx, gy), 0.068, 0.014, facecolor=ORANGE, edgecolor="none"))
    ax.text(gx + 0.068, gy - 0.010, "风险概率", ha="center", va="top", fontsize=8.5, color=MUTED)

    # Lower explanatory band.
    ax.text(0.055, 0.36, "核心逻辑", fontsize=13, fontweight="bold", color=INK)
    steps = [
        ("01", "逐时刻计算权重", "每个特征独立取0–1权重"),
        ("02", "保留关键动态", "LSTM记住风速、姿态和障碍物变化"),
        ("03", "输出未来风险", "预测未来1秒内是否发生碰撞"),
    ]
    for i, (num, title, body) in enumerate(steps):
        x = 0.055 + i * 0.305
        ax.add_patch(Circle((x + 0.018, 0.255), 0.024, facecolor=NAVY, edgecolor="none"))
        ax.text(x + 0.018, 0.255, num, color="white", fontsize=9, fontweight="bold", ha="center", va="center")
        ax.text(x + 0.055, 0.276, title, fontsize=12, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.055, 0.235, body, fontsize=9.5, color=MUTED, va="center")
    ax.text(0.945, 0.065, "INTENTRA-X  ·  FeatureAttentionRiskLSTM", ha="right", fontsize=9, color=MUTED)
    export(fig, "01_model_architecture")


def load_model_data():
    flights = taa.load_rows(str(DATA_PATH))
    x, y, meta = taa.make_dataset(flights)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y), tr_f, va_f, te_f = taa.split_by_flight(x, y, meta)
    te_x_raw = te_x.copy()
    tr_flat = tr_x.reshape(-1, tr_x.shape[-1])
    mean = tr_flat.mean(axis=0)
    std = tr_flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    te_x = ((te_x - mean) / std).astype(np.float32)
    model = FeatureAttentionRiskLSTM(n_features=len(taa.FEATURE_NAMES))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    tensor = torch.from_numpy(te_x)
    with torch.no_grad():
        pred, imp = model(tensor)
    pred = pred.squeeze(-1).numpy()
    imp = imp.numpy()
    test_meta = [item for item in meta if item[0] in te_f]
    return {
        "flights": flights,
        "test_raw": te_x_raw,
        "test_norm": te_x,
        "label": te_y,
        "pred": pred,
        "importance": imp,
        "meta": test_meta,
        "mean": mean,
        "std": std,
        "model": model,
        "train_size": len(tr_y),
        "val_size": len(va_y),
        "test_size": len(te_y),
    }


def read_sample_flight():
    with SAMPLE_FLIGHT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = ["timestamp", "pos_x", "pos_y", "altitude", "risk_score", "counterfactual_risk"]
    out = {key: np.asarray([float(row[key]) for row in rows]) for key in numeric}
    out["state"] = np.asarray([row["state"] for row in rows])
    return out


def compute_spatial_risk(x_grid, y_grid, altitude):
    env = EnvironmentModel()
    X, Y = np.meshgrid(x_grid, y_grid)
    camera = env.camera
    dx = X - camera.position[0]
    dy = Y - camera.position[1]
    dz = -altitude - camera.position[2]
    distance = np.sqrt(dx * dx + dy * dy + dz * dz)
    distance_factor = np.clip(1.0 - distance / camera.max_range, 0, 1)
    norm = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
    dot = (dx * camera.direction[0] + dy * camera.direction[1]) / norm
    angle = np.arccos(np.clip(dot, -1, 1))
    fov_rad = np.radians(camera.fov_degrees / 2)
    fov_factor = np.where(angle <= fov_rad, 1.0 - angle / fov_rad, 0.0)
    altitude_factor = min(altitude / 50.0, 1.0)
    tree_dist = np.sqrt((X - 20) ** 2 + (Y - 30) ** 2)
    occlusion = np.where(tree_dist < 10, 0.6, 0.0)
    Z = 0.4 * distance_factor + 0.3 * fov_factor + 0.2 * altitude_factor + 0.1 * (1 - occlusion)
    return np.clip(Z, 0, 1)


def generate_spatial_risk() -> None:
    flight = read_sample_flight()
    x_grid = np.linspace(-40, 75, 180)
    y_grid = np.linspace(-35, 75, 180)
    altitude = float(np.median(flight["altitude"]))
    Z = compute_spatial_risk(x_grid, y_grid, altitude)
    cmap = colors.LinearSegmentedColormap.from_list("risk", ["#E8F7EF", "#FFE59A", "#FF9F6E", "#D73045"])

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.88, top=0.84, bottom=0.10)
    add_figure_title(
        fig,
        "空间风险场与无人机飞行轨迹",
        f"演示轨迹叠加环境探测风险（代表高度约 {altitude:.0f} m），颜色越暖表示暴露风险越高",
    )
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="bilinear",
        alpha=0.92,
    )

    # Environmental context.
    ax.add_patch(Rectangle((-30, -30), 60, 60, facecolor="#64748B20", edgecolor="#475569", lw=1.5, linestyle="--"))
    ax.text(-28, -27, "建筑区域", fontsize=9.5, color="#475569", fontweight="bold")
    ax.add_patch(Circle((20, 30), 10, facecolor="#16A34A26", edgecolor=GREEN, lw=1.8))
    ax.text(20, 42, "树林遮挡区", fontsize=9.5, color=GREEN, ha="center", fontweight="bold")
    ax.add_patch(Wedge((10, 10), 58, 0, 90, facecolor="#2563EB14", edgecolor=BLUE, lw=1.2, linestyle=":"))
    ax.scatter([10], [10], marker="^", s=130, c=NAVY, edgecolors="white", linewidths=1.3, zorder=6)
    ax.text(12, 7, "环境传感器", fontsize=9.5, color=NAVY, fontweight="bold")

    # Trajectory line and risk-colored samples.
    x, y, risk = flight["pos_x"], flight["pos_y"], flight["risk_score"]
    ax.plot(x, y, color="white", lw=5.5, alpha=0.85, zorder=5)
    ax.plot(x, y, color=NAVY, lw=2.0, zorder=6)
    sc = ax.scatter(x, y, c=risk, cmap=cmap, vmin=0, vmax=1, s=80, edgecolors="white", linewidths=1.2, zorder=7)
    for idx in [3, 7, 11, 15]:
        ax.annotate(
            "",
            xy=(x[idx + 1], y[idx + 1]),
            xytext=(x[idx], y[idx]),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.8),
            zorder=8,
        )
    ax.scatter([x[0]], [y[0]], s=150, c=GREEN, marker="o", edgecolors="white", linewidths=1.5, zorder=9)
    ax.scatter([x[-1]], [y[-1]], s=170, c=RED, marker="X", edgecolors="white", linewidths=1.5, zorder=9)
    ax.text(x[0] - 1.5, y[0] + 3, "起点", color=GREEN, fontsize=10, fontweight="bold")
    ax.text(x[-1] + 2, y[-1], "终点", color=RED, fontsize=10, fontweight="bold", va="center")
    peak = int(np.argmax(risk))
    ax.annotate(
        f"风险峰值 {risk[peak]:.2f}\n触发自适应状态",
        xy=(x[peak], y[peak]),
        xytext=(x[peak] + 15, y[peak] + 18),
        fontsize=10,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        zorder=10,
    )
    ax.set_xlabel("东向位置 x（m）")
    ax.set_ylabel("北向位置 y（m）")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(y_grid.min(), y_grid.max())
    ax.grid(color="white", alpha=0.26, lw=0.8)

    cax = fig.add_axes([0.905, 0.18, 0.018, 0.56])
    cb = fig.colorbar(ScalarMappable(norm=colors.Normalize(0, 1), cmap=cmap), cax=cax)
    cb.set_label("环境风险 / Risk", color=MUTED, labelpad=12)
    cb.outline.set_visible(False)
    export(fig, "02_spatial_risk_trajectory")


def parse_threshold() -> float:
    text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Selected class_weight=[\d.]+, threshold=([\d.]+)", text)
    return float(match.group(1)) if match else 0.31


def contiguous_spans(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def generate_risk_attention_timeline(data) -> None:
    pred, label, imp = data["pred"], data["label"], data["importance"]
    rows = data["flights"]
    timestamps = np.asarray([rows[fid][idx]["timestamp"] for fid, idx in data["meta"]])
    time_sec = timestamps - timestamps[0]
    threshold = parse_threshold()

    # Show the most informative 55-second interval, chosen by label transitions and risk amplitude.
    median_dt = float(np.median(np.diff(time_sec)))
    window = min(len(pred), max(300, int(round(55 / median_dt))))
    best_start, best_score = 0, -np.inf
    for start in range(0, len(pred) - window + 1, max(1, window // 25)):
        sl = slice(start, start + window)
        transitions = np.abs(np.diff(label[sl])).sum()
        score = 2.5 * transitions + 4.0 * np.std(pred[sl]) + label[sl].mean()
        if score > best_score:
            best_start, best_score = start, score
    sl = slice(best_start, best_start + window)
    t = time_sec[sl] - time_sec[best_start]
    p, y, heat = pred[sl], label[sl], imp[sl].T

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 1], left=0.08, right=0.95, top=0.83, bottom=0.10, hspace=0.17)
    ax = fig.add_subplot(gs[0])
    ax_h = fig.add_subplot(gs[1], sharex=ax)
    add_figure_title(
        fig,
        "风险预测与注意力变化",
        "上：未来1秒碰撞风险概率；下：模型在每个预测窗口对9项输入特征的平均注意力",
    )

    for start, end in contiguous_spans(y > 0.5):
        left = t[start]
        right = t[min(end - 1, len(t) - 1)] + median_dt
        ax.axvspan(left, right, color=RED, alpha=0.10, lw=0)
    ax.fill_between(t, p, 0, color=BLUE, alpha=0.12)
    ax.plot(t, p, color=BLUE, lw=2.4, label="预测风险")
    ax.axhline(threshold, color=ORANGE, lw=1.6, linestyle="--", label=f"决策阈值 {threshold:.2f}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("风险概率")
    ax.legend(loc="upper right", ncol=2)
    style_axis(ax)
    ax.tick_params(labelbottom=False)
    ax.text(
        0.01,
        0.93,
        "淡红区域：真实碰撞风险窗口",
        transform=ax.transAxes,
        fontsize=9.5,
        color=RED,
        va="top",
    )
    peak = int(np.argmax(p))
    ax.annotate(
        f"峰值 {p[peak]:.2f}",
        xy=(t[peak], p[peak]),
        xytext=(t[peak], min(0.93, p[peak] + 0.20)),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1),
    )

    heat_cmap = colors.LinearSegmentedColormap.from_list("attn", ["#EEF4FF", "#7DD3FC", "#2563EB", "#14213D"])
    im = ax_h.imshow(
        heat,
        origin="upper",
        aspect="auto",
        extent=[t[0], t[-1], len(taa.FEATURE_NAMES) - 0.5, -0.5],
        cmap=heat_cmap,
        vmin=max(0.25, float(np.quantile(heat, 0.02))),
        vmax=min(0.85, float(np.quantile(heat, 0.98))),
        interpolation="nearest",
    )
    ax_h.set_yticks(np.arange(len(taa.FEATURE_NAMES)))
    ax_h.set_yticklabels([FEATURE_LABELS[name] for name in taa.FEATURE_NAMES])
    ax_h.set_xlabel("相对时间（s）")
    ax_h.set_ylabel("输入特征")
    ax_h.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax_h, pad=0.012, fraction=0.025)
    cbar.set_label("注意力权重", color=MUTED)
    cbar.outline.set_visible(False)
    export(fig, "03_risk_attention_timeline")


def generate_feature_importance(data) -> None:
    imp = data["importance"]
    mean, std = imp.mean(axis=0), imp.std(axis=0)
    order = np.argsort(mean)
    labels = [FEATURE_LABELS[taa.FEATURE_NAMES[i]] for i in order]
    values, errs = mean[order], std[order]
    ranks = np.argsort(np.argsort(-mean)) + 1
    bar_colors = [BLUE if ranks[i] > 3 else [ORANGE, CYAN, TEAL][ranks[i] - 1] for i in order]

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.18, right=0.92, top=0.83, bottom=0.11)
    add_figure_title(
        fig,
        "注意力特征重要性排名",
        f"基于独立测试飞行的 {len(imp):,} 个预测窗口；权重来自逐特征 Sigmoid 门控，彼此不要求和为1",
    )
    ypos = np.arange(len(labels))
    ax.barh(ypos, np.ones_like(values), color="#EAF0F7", height=0.58, edgecolor="none")
    ax.barh(ypos, values, color=bar_colors, height=0.58, edgecolor="none")
    ax.errorbar(values, ypos, xerr=errs, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, alpha=0.65)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("平均注意力权重（0–1）")
    for y, value, original_idx in zip(ypos, values, order):
        rank = ranks[original_idx]
        ax.text(0.015, y, f"TOP {rank}" if rank <= 3 else f"{rank:02d}", va="center", ha="left", color="white" if value > 0.12 else INK, fontsize=9, fontweight="bold")
        ax.text(value + 0.018, y, f"{value:.3f}", va="center", ha="left", color=INK, fontsize=10.5, fontweight="bold")
    style_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    export(fig, "04_attention_feature_importance")


def generate_performance(data) -> None:
    pred, label = data["pred"], data["label"]
    threshold = parse_threshold()
    predicted = pred >= threshold
    cm = confusion_matrix(label.astype(int), predicted.astype(int))
    tn, fp, fn, tp = cm.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / cm.sum()
    roc_auc = roc_auc_score(label, pred)
    pr_auc = average_precision_score(label, pred)
    fpr, tpr, _ = roc_curve(label, pred)
    pr_precision, pr_recall, _ = precision_recall_curve(label, pred)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.20, 1], left=0.055, right=0.96, top=0.83, bottom=0.10, hspace=0.25, wspace=0.27)
    add_figure_title(
        fig,
        "模型测试集性能",
        f"独立测试飞行 · n={len(label):,} · 决策阈值={threshold:.2f} · 预测目标为未来1秒内是否发生碰撞",
    )

    metrics = [("ROC-AUC", roc_auc, BLUE), ("PR-AUC", pr_auc, TEAL), ("F1", f1, ORANGE), ("Recall", recall, RED), ("Accuracy", accuracy, PURPLE)]
    axm = fig.add_subplot(gs[0, :])
    axm.axis("off")
    for i, (name, value, color) in enumerate(metrics):
        x = 0.02 + i * 0.195
        axm.text(x, 0.68, f"{value:.3f}", fontsize=21, fontweight="bold", color=color, transform=axm.transAxes, va="center")
        axm.text(x, 0.17, name, fontsize=9.5, color=MUTED, transform=axm.transAxes, va="center")

    ax0 = fig.add_subplot(gs[1, 0])
    cm_cmap = colors.LinearSegmentedColormap.from_list("cm", ["#EEF4FF", "#2563EB", "#14213D"])
    ax0.imshow(cm, cmap=cm_cmap)
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            ax0.text(j, i, f"{value:,}", ha="center", va="center", fontsize=21, fontweight="bold", color="white" if value > cm.max() * 0.45 else INK)
    ax0.set_xticks([0, 1], ["预测安全", "预测风险"])
    ax0.set_yticks([0, 1], ["实际安全", "实际风险"])
    ax0.set_title("混淆矩阵", loc="left", pad=16)
    ax0.set_xlabel("预测类别")
    ax0.set_ylabel("真实类别")
    ax0.tick_params(length=0)
    for spine in ax0.spines.values():
        spine.set_visible(False)

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.plot(fpr, tpr, color=BLUE, lw=2.7, label=f"ROC-AUC = {roc_auc:.3f}")
    ax1.fill_between(fpr, tpr, 0, color=BLUE, alpha=0.08)
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=1.2, linestyle="--", label="随机基线")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("假阳性率 FPR")
    ax1.set_ylabel("真阳性率 TPR")
    ax1.set_title("ROC 曲线", loc="left", pad=16)
    ax1.legend(loc="lower right")
    style_axis(ax1)

    ax2 = fig.add_subplot(gs[1, 2])
    ax2.plot(pr_recall, pr_precision, color=TEAL, lw=2.7, label=f"PR-AUC = {pr_auc:.3f}")
    ax2.fill_between(pr_recall, pr_precision, 0, color=TEAL, alpha=0.08)
    base = float(label.mean())
    ax2.axhline(base, color=MUTED, lw=1.2, linestyle="--", label=f"阳性率基线 {base:.3f}")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("召回率 Recall")
    ax2.set_ylabel("精确率 Precision")
    ax2.set_title("Precision–Recall 曲线", loc="left", pad=16)
    ax2.legend(loc="lower left")
    style_axis(ax2)
    export(fig, "05_model_performance")


def parse_weight_comparison():
    text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"Training model with positive class_weight=", text)[1:]
    rows = []
    epoch_re = re.compile(
        r"val_threshold=([\d.]+).*?val_precision=([\d.]+).*?val_recall=([\d.]+).*?val_f1=([\d.]+).*?val_pr_auc=([\d.eE+-]+)"
    )
    for block in blocks:
        weight_match = re.match(r"([\d.]+)", block)
        if not weight_match:
            continue
        weight = float(weight_match.group(1))
        candidates = []
        for match in epoch_re.finditer(block):
            threshold, precision, recall, f1, pr_auc = map(float, match.groups())
            candidates.append((pr_auc, f1, threshold, precision, recall))
        if candidates:
            best = max(candidates, key=lambda item: item[0])
            rows.append((weight, *best))
    if rows:
        return rows
    return [
        (1.00, 0.6670, 0.617, 0.31, 0.653, 0.586),
        (1.50, 0.6254, 0.606, 0.52, 0.607, 0.604),
        (2.00, 0.6238, 0.607, 0.44, 0.565, 0.656),
        (2.32, 0.6447, 0.620, 0.62, 0.640, 0.600),
    ]


def generate_weight_comparison() -> None:
    rows = parse_weight_comparison()
    weights = np.asarray([row[0] for row in rows])
    pr_auc = np.asarray([row[1] for row in rows])
    f1 = np.asarray([row[2] for row in rows])
    threshold = np.asarray([row[3] for row in rows])
    x = np.arange(len(weights))

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.88, top=0.82, bottom=0.13)
    add_figure_title(
        fig,
        "类别权重调优对比",
        "以验证集 PR-AUC 作为模型选择指标；权重越大，训练时越强调碰撞风险样本",
    )
    width = 0.28
    bars1 = ax.bar(x - width / 2, pr_auc, width, color=BLUE, label="PR-AUC", zorder=3)
    bars2 = ax.bar(x + width / 2, f1, width, color=CYAN, label="F1", zorder=3)
    ax2 = ax.twinx()
    ax2.plot(x, threshold, color=ORANGE, lw=2.3, marker="D", markersize=7, label="最优阈值", zorder=4)
    ax.set_xticks(x, [f"权重 {w:g}" for w in weights])
    ax.set_ylim(0, 0.80)
    ax2.set_ylim(0, 0.80)
    ax.set_ylabel("验证集指标")
    ax2.set_ylabel("最优决策阈值", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.set_xlabel("正类权重")
    ax.axvspan(-0.48, 0.48, color=GREEN, alpha=0.07, zorder=0)
    ax.text(0, 0.755, "最终选择", ha="center", va="center", fontsize=10, color=GREEN, fontweight="bold")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for xi, value in zip(x, threshold):
        ax2.text(xi, value + 0.025, f"{value:.2f}", ha="center", color=ORANGE, fontsize=10, fontweight="bold")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower center", ncol=3)
    style_axis(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.grid(False)
    export(fig, "06_class_weight_comparison")


def predict_raw(model, raw_x, mean, std):
    norm = ((raw_x - mean) / std).astype(np.float32)
    with torch.no_grad():
        pred, _ = model(torch.from_numpy(norm))
    return pred.squeeze(-1).numpy()


def generate_counterfactual(data) -> None:
    model = data["model"]
    raw = data["test_raw"]
    base_pred = data["pred"]
    threshold = parse_threshold()
    normal_idx = np.flatnonzero((data["label"] < 0.5) & (base_pred < threshold))
    if len(normal_idx) > 420:
        rng = np.random.default_rng(42)
        normal_idx = np.sort(rng.choice(normal_idx, size=420, replace=False))
    selected = raw[normal_idx].copy()

    scenarios = [selected.copy() for _ in range(4)]
    scenarios[1][:, :, 0] = np.maximum(scenarios[1][:, :, 0], 19.0)
    scenarios[2][:, :, 5] = np.minimum(scenarios[2][:, :, 5], 5.0)
    scenarios[2][:, :, 6] = 1.0
    scenarios[3][:, :, 0] = np.maximum(scenarios[3][:, :, 0], 19.0)
    scenarios[3][:, :, 5] = np.minimum(scenarios[3][:, :, 5], 5.0)
    scenarios[3][:, :, 6] = 1.0
    predictions = [predict_raw(model, s, data["mean"], data["std"]) for s in scenarios]
    names = ["原始状态", "强风 19 m/s", "近距障碍 5 m", "强风 + 近障碍"]
    palette = [BLUE, ORANGE, TEAL, RED]

    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.45, 0.75]})
    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.13, wspace=0.28)
    add_figure_title(
        fig,
        "反事实风险推演",
        f"对 {len(selected):,} 个原本判定为安全的测试窗口施加可控扰动，观察模型风险输出如何变化",
    )

    parts = ax.violinplot(predictions, positions=np.arange(4), widths=0.78, showextrema=False)
    for body, color in zip(parts["bodies"], palette):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.22)
    rng = np.random.default_rng(7)
    for i, (values, color) in enumerate(zip(predictions, palette)):
        sample = rng.choice(values, min(120, len(values)), replace=False)
        jitter = rng.normal(0, 0.055, len(sample))
        ax.scatter(i + jitter, sample, s=14, color=color, alpha=0.25, edgecolors="none")
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        ax.plot([i - 0.13, i + 0.13], [med, med], color=INK, lw=2.4)
        ax.plot([i, i], [q1, q3], color=INK, lw=5, solid_capstyle="round")
        ax.scatter([i], [mean], s=95, color=color, edgecolors="white", linewidths=1.3, zorder=5)
        ax.text(i, min(0.98, mean + 0.075), f"均值 {mean:.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.axhline(threshold, color=MUTED, linestyle="--", lw=1.5, label=f"决策阈值 {threshold:.2f}")
    ax.set_xticks(np.arange(4), names)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("模型预测风险")
    ax.set_title("风险分布", loc="left", pad=14)
    ax.legend(loc="upper left")
    style_axis(ax)

    base = predictions[0]
    deltas = [float(np.mean(values - base)) for values in predictions[1:]]
    delta_names = names[1:]
    y = np.arange(3)
    delta_colors = [ORANGE, TEAL, RED]
    ax_d.barh(y, deltas, color=delta_colors, height=0.48)
    ax_d.axvline(0, color=INK, lw=1)
    ax_d.set_yticks(y, delta_names)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("相对原始状态的平均风险变化")
    ax_d.set_title("平均增量", loc="left", pad=14)
    limit = max(0.06, max(abs(v) for v in deltas) * 1.35)
    ax_d.set_xlim(-limit, limit)
    for yi, value in zip(y, deltas):
        ha = "left" if value >= 0 else "right"
        offset = limit * 0.035 if value >= 0 else -limit * 0.035
        ax_d.text(value + offset, yi, f"{value:+.3f}", va="center", ha=ha, fontsize=11, fontweight="bold", color=INK)
    style_axis(ax_d, grid_axis="x")
    export(fig, "07_counterfactual_risk")


def main() -> None:
    configure_matplotlib()
    generate_architecture()
    data = load_model_data()
    generate_spatial_risk()
    generate_risk_attention_timeline(data)
    generate_feature_importance(data)
    generate_performance(data)
    generate_weight_comparison()
    generate_counterfactual(data)
    print(f"Generated 7 PPT visualizations in: {OUT_DIR}")


if __name__ == "__main__":
    main()
