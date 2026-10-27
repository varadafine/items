"""
Legacy 9-feature Attention-LSTM trainer on the original gust-flight data.

For the controlled 12-feature experiment on manual sessions, use
train_air_risk_12_compare.py. This legacy script intentionally retains its
original checkpoint names and label definition for reproducibility.

The label is derived from simulation outcomes rather than hand-written
feature thresholds: a sample is risky if a ground or obstacle collision
occurs within the future prediction horizon.
"""
import io
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attention_risk_lstm import FeatureAttentionRiskLSTM

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    average_precision_score = None
    roc_auc_score = None


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "UAV_gust_all.jsonl")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
HORIZON_SECONDS = 1.0
BATCH_SIZE = 64
EPOCHS = 120
LR = 1e-3
EARLY_STOP_PATIENCE = 15
RNG_SEED = 42
CLASS_WEIGHTS_TO_TRY = [1.0, 1.5, 2.0, 2.32]
EARLY_STOP_METRIC = "val_pr_auc"

FEATURE_NAMES = [
    "wind_speed",
    "drone_speed",
    "roll",
    "pitch",
    "altitude",
    "sensor_obstacle_distance",
    "obstacle_detected",
    "wind_direction_sin",
    "wind_direction_cos",
]

REPORT = io.StringIO()


def log(message=""):
    print(message)
    REPORT.write(str(message) + "\n")


def load_rows(path):
    flights = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            flights[row["flight_id"]].append(row)

    for rows in flights.values():
        rows.sort(key=lambda r: r["timestamp"])
    return dict(flights)


def row_to_features(row):
    """把一行 AirSim 日志转换成模型输入特征。

    注意：这里不使用 nearest_obstacle_distance 作为训练特征，因为它可能来自
    碰撞/场景后处理信息，正式实验中容易造成标签泄露。模型只使用传感器测距
    sensor_obstacle_distance，并增加 obstacle_detected 表示传感器是否检测到近距障碍。
    """
    position = row.get("position") or [0.0, 0.0, 0.0]
    velocity = row.get("velocity") or [0.0, 0.0, 0.0]
    attitude = row.get("attitude") or {}

    speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
    altitude = max(0.0, -float(position[2]))
    sensor_distance = row.get("sensor_obstacle_distance", 100.0) or 100.0
    obstacle_detected = 1.0 if sensor_distance < 100.0 else 0.0

    return np.asarray([
        row.get("wind_speed", 0.0) or 0.0,
        speed,
        attitude.get("roll", 0.0) or 0.0,
        attitude.get("pitch", 0.0) or 0.0,
        altitude,
        sensor_distance,
        obstacle_detected,
        row.get("wind_direction_sin", 0.0) or 0.0,
        row.get("wind_direction_cos", 1.0) or 1.0,
    ], dtype=np.float32)


def row_is_collision(row):
    return bool(row.get("ground_collision") or row.get("obstacle_collision"))


def make_dataset(flights):
    """使用滑动窗口构造 LSTM 训练样本。

    X：过去 SEQ_LEN 个采样点的飞行状态序列。
    y：从窗口末尾开始，未来 HORIZON_SECONDS 秒内是否发生仿真碰撞。
       ground_collision 或 obstacle_collision 任意一个为 True，即认为未来有风险。
    """
    all_x, all_y, meta = [], [], []

    for flight_id, rows in flights.items():
        features = np.stack([row_to_features(row) for row in rows]).astype(np.float32)
        collision = np.asarray([row_is_collision(row) for row in rows], dtype=bool)
        timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)

        for end_idx in range(SEQ_LEN - 1, len(rows)):
            horizon_end = np.searchsorted(
                timestamps,
                timestamps[end_idx] + HORIZON_SECONDS,
                side="right",
            )
            future_collision = collision[end_idx:horizon_end].any()
            all_x.append(features[end_idx - SEQ_LEN + 1:end_idx + 1])
            all_y.append(1.0 if future_collision else 0.0)
            meta.append((flight_id, end_idx))

    x = np.stack(all_x).astype(np.float32)
    y = np.asarray(all_y, dtype=np.float32)
    return x, y, meta


def normalize(train_x, val_x, test_x):
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
        mean,
        std,
    )


def split_by_flight(x, y, meta):
    """按 flight_id 划分训练/验证/测试集，避免相邻时间窗口泄露到验证集。"""
    flight_ids = sorted({flight_id for flight_id, _ in meta})
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(flight_ids)

    train_flights = set(flight_ids[:3])
    val_flights = {flight_ids[3]}
    test_flights = {flight_ids[4]}

    idx = np.arange(len(meta))
    tr = np.asarray([i for i in idx if meta[i][0] in train_flights])
    va = np.asarray([i for i in idx if meta[i][0] in val_flights])
    te = np.asarray([i for i in idx if meta[i][0] in test_flights])

    return (x[tr], y[tr]), (x[va], y[va]), (x[te], y[te]), train_flights, val_flights, test_flights


def make_loader(x, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_count = 0.0, 0
    preds, labels, importances = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred, imp = model(xb)
            pred = pred.squeeze(-1)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
            preds.append(pred.cpu().numpy())
            labels.append(yb.cpu().numpy())
            importances.append(imp.cpu().numpy())

    p = np.concatenate(preds)
    y = np.concatenate(labels)
    imp = np.concatenate(importances)
    acc = ((p >= 0.5) == (y >= 0.5)).mean()
    return total_loss / total_count, float(acc), p, y, imp


def binary_metrics(pred, label, threshold=0.5):
    """计算二分类风险预测指标，方便判断模型是否只偏向某一类。"""
    pred_label = pred >= threshold
    true_label = label >= 0.5

    tp = int(np.logical_and(pred_label, true_label).sum())
    fp = int(np.logical_and(pred_label, ~true_label).sum())
    tn = int(np.logical_and(~pred_label, ~true_label).sum())
    fn = int(np.logical_and(~pred_label, true_label).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if roc_auc_score is not None and len(np.unique(label)) == 2:
        roc_auc = float(roc_auc_score(label, pred))
    else:
        roc_auc = None

    if average_precision_score is not None and len(np.unique(label)) == 2:
        pr_auc = float(average_precision_score(label, pred))
    else:
        pr_auc = None

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def tune_threshold_on_val(pred, label):
    """在验证集上搜索最佳分类阈值，不再固定使用 0.5。

    这里用 F1 作为阈值选择目标，因为 F1 同时考虑 Precision 和 Recall，
    比单纯 accuracy 更适合碰撞风险这类正负样本不均衡问题。
    """
    best_threshold = 0.5
    best_metrics = binary_metrics(pred, label, threshold=best_threshold)
    best_f1 = best_metrics["f1"]

    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = binary_metrics(pred, label, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = metrics["f1"]

    return best_threshold, best_metrics


def early_stop_score(metrics):
    """返回早停使用的验证集指标，优先使用 PR-AUC，缺失时退回 F1。"""
    if EARLY_STOP_METRIC == "val_pr_auc" and metrics["pr_auc"] is not None:
        return metrics["pr_auc"], "val_pr_auc"
    return metrics["f1"], "val_f1"


def format_weight(weight):
    return str(weight).replace(".", "p")


def train_model(train_loader, val_loader, class_weight):
    """训练一个指定正样本权重的模型，并用验证集指标早停。"""
    torch.manual_seed(RNG_SEED)
    model = FeatureAttentionRiskLSTM(n_features=len(FEATURE_NAMES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss(reduction="none")
    pos_weight = torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)
    best_score, no_improve = -float("inf"), 0
    best_threshold, best_metric_name = 0.5, "val_f1"
    train_history, val_loss_history, val_score_history = [], [], []
    ckpt_path = os.path.join(
        MODEL_DIR,
        f"airsim_attention_risk_model_w{format_weight(class_weight)}.pt",
    )

    def weighted_bce(pred, target):
        loss = criterion(pred, target)
        weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
        return (loss * weights).mean()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred, _ = model(xb)
            loss = weighted_bce(pred.squeeze(-1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

        train_loss = total_loss / total_count
        val_loss, _, val_pred, val_label, _ = evaluate(model, val_loader, weighted_bce)
        val_threshold, val_metrics = tune_threshold_on_val(val_pred, val_label)
        val_score, metric_name = early_stop_score(val_metrics)
        train_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_score_history.append(val_score)
        log(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_threshold={val_threshold:.2f} | "
            f"val_precision={val_metrics['precision']:.3f} | "
            f"val_recall={val_metrics['recall']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f} | "
            f"val_pr_auc={val_metrics['pr_auc'] if val_metrics['pr_auc'] is not None else 'N/A'}"
        )

        if val_score > best_score + 1e-4:
            best_score = val_score
            best_threshold = val_threshold
            best_metric_name = metric_name
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                log(
                    f"Early stopping at epoch {epoch}; "
                    f"best_{best_metric_name}={best_score:.4f}, "
                    f"best_threshold={best_threshold:.2f}"
                )
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_history, label="train_loss")
    plt.plot(val_loss_history, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("weighted BCE")
    plt.title(f"AirSim Attention-LSTM Training Curve (class_weight={class_weight})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            MODEL_DIR,
            f"airsim_attention_training_curve_w{format_weight(class_weight)}.png",
        ),
        dpi=120,
    )
    plt.close()
    return {
        "model": model,
        "criterion": weighted_bce,
        "class_weight": class_weight,
        "best_score": best_score,
        "best_metric_name": best_metric_name,
        "best_threshold": best_threshold,
        "ckpt_path": ckpt_path,
    }


def save_feature_importance(importances):
    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)
    order = np.argsort(-mean_imp)

    log("")
    log("Feature importance ranking on test set:")
    for rank, idx in enumerate(order, start=1):
        log(f"  {rank:02d}. {FEATURE_NAMES[idx]:28s} {mean_imp[idx]:.4f} +/- {std_imp[idx]:.4f}")

    plt.figure(figsize=(9, 4.8))
    xs = np.arange(len(FEATURE_NAMES))
    plt.bar(xs, mean_imp[order], yerr=std_imp[order], capsize=5, color="#2f6f9f")
    plt.xticks(xs, [FEATURE_NAMES[i] for i in order], rotation=25, ha="right")
    plt.ylabel("attention weight")
    plt.title("AirSim Gust Risk Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "airsim_attention_feature_importance.png"), dpi=120)
    plt.close()
    return mean_imp, std_imp, order


def log_run_config():
    """在报告开头输出完整超参配置清单，使报告自包含、可复现。

    所有取值均引用脚本内的全局常量与模型默认结构，改超参时报告会自动同步。
    """
    probe = FeatureAttentionRiskLSTM(n_features=len(FEATURE_NAMES))
    lstm = probe.lstm

    log("=" * 72)
    log("RUN CONFIGURATION (超参配置清单)")
    log("=" * 72)
    log("[数据参数]")
    log(f"  data_path                = {DATA_PATH}")
    log(f"  sequence_length (SEQ_LEN)= {SEQ_LEN}")
    log(f"  horizon_seconds          = {HORIZON_SECONDS}")
    log(f"  n_features               = {len(FEATURE_NAMES)}")
    log(f"  feature_names            = {FEATURE_NAMES}")
    log("[训练超参]")
    log(f"  batch_size               = {BATCH_SIZE}")
    log(f"  max_epochs               = {EPOCHS}")
    log(f"  learning_rate            = {LR}")
    log(f"  early_stop_patience      = {EARLY_STOP_PATIENCE}")
    log(f"  early_stop_metric        = {EARLY_STOP_METRIC} (fallback: val_f1)")
    log(f"  class_weights_to_try     = {CLASS_WEIGHTS_TO_TRY}")
    log(f"  optimizer                = Adam")
    log(f"  loss                     = weighted BCE (positive class up-weighted)")
    log(f"  threshold_selection      = search 0.05~0.95 by F1 on val set")
    log(f"  rng_seed                 = {RNG_SEED}")
    log(f"  device                   = {DEVICE}")
    log("[模型结构] FeatureAttentionRiskLSTM")
    log(f"  attention_type           = per-feature sigmoid gate (bottleneck MLP)")
    log(f"  lstm_hidden_size         = {lstm.hidden_size}")
    log(f"  lstm_num_layers          = {lstm.num_layers}")
    log(f"  lstm_dropout             = {lstm.dropout}")
    log(f"  regression_head          = Linear(hidden,32)->ReLU->Dropout(0.3)->Linear(32,1)->Sigmoid")
    log(f"  normalization            = per-feature z-score (mean/std from train set)")
    log("=" * 72)
    log("")


def main():
    log_run_config()
    log(f"Device: {DEVICE}")
    log(f"Data: {DATA_PATH}")
    flights = load_rows(DATA_PATH)
    x, y, meta = make_dataset(flights)
    log(f"Flights: {len(flights)}")
    log(f"Samples: {len(x)} | positive risk windows: {int(y.sum())} ({y.mean() * 100:.2f}%)")
    log(f"Sequence length: {SEQ_LEN} | horizon: {HORIZON_SECONDS:.1f}s")

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y), tr_f, va_f, te_f = split_by_flight(x, y, meta)
    tr_x, va_x, te_x, mean, std = normalize(tr_x, va_x, te_x)
    log(f"Train flights: {sorted(tr_f)}")
    log(f"Val flights: {sorted(va_f)}")
    log(f"Test flights: {sorted(te_f)}")
    log(f"Train/Val/Test: {len(tr_x)}/{len(va_x)}/{len(te_x)}")
    log(f"Train positive rate: {tr_y.mean() * 100:.2f}%")

    train_loader = make_loader(tr_x, tr_y, shuffle=True)
    val_loader = make_loader(va_x, va_y, shuffle=False)
    test_loader = make_loader(te_x, te_y, shuffle=False)

    log("")
    log(f"Class weights to compare: {CLASS_WEIGHTS_TO_TRY}")
    log(f"Early stopping metric: {EARLY_STOP_METRIC} (fallback: val_f1)")

    results = []
    for class_weight in CLASS_WEIGHTS_TO_TRY:
        log("")
        log("=" * 72)
        log(f"Training model with positive class_weight={class_weight}")
        log("=" * 72)
        result = train_model(train_loader, val_loader, class_weight)
        results.append(result)
        log(
            f"Finished class_weight={class_weight}: "
            f"best_{result['best_metric_name']}={result['best_score']:.4f}, "
            f"best_threshold={result['best_threshold']:.2f}"
        )

    best_result = max(results, key=lambda item: item["best_score"])
    model = best_result["model"]
    criterion = best_result["criterion"]
    best_threshold = best_result["best_threshold"]
    log("")
    log("Class weight comparison summary:")
    for result in results:
        log(
            f"  class_weight={result['class_weight']:.2f} | "
            f"best_{result['best_metric_name']}={result['best_score']:.4f} | "
            f"best_threshold={result['best_threshold']:.2f}"
        )
    log(
        f"Selected class_weight={best_result['class_weight']:.2f}, "
        f"threshold={best_threshold:.2f}"
    )
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "airsim_attention_risk_model.pt"),
    )

    test_loss, test_acc, pred, label, imp = evaluate(model, test_loader, criterion)
    metrics = binary_metrics(pred, label, threshold=best_threshold)
    log("")
    log(f"Test weighted BCE={test_loss:.4f} | Test accuracy@0.5={test_acc:.3f}")
    log(f"Test predicted risk mean={pred.mean():.4f} | label mean={label.mean():.4f}")
    log("")
    log(f"Test binary classification metrics at threshold={best_threshold:.2f}:")
    log(f"  TP={metrics['TP']} | FP={metrics['FP']} | TN={metrics['TN']} | FN={metrics['FN']}")
    log(f"  Precision={metrics['precision']:.4f}")
    log(f"  Recall={metrics['recall']:.4f}")
    log(f"  F1={metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        log(f"  ROC-AUC={metrics['roc_auc']:.4f}")
    else:
        log("  ROC-AUC=N/A (requires scikit-learn and both classes in test labels)")
    if metrics["pr_auc"] is not None:
        log(f"  PR-AUC={metrics['pr_auc']:.4f}")
    else:
        log("  PR-AUC=N/A (requires scikit-learn and both classes in test labels)")

    save_feature_importance(imp)

    report_path = os.path.join(MODEL_DIR, "airsim_attention_training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(REPORT.getvalue())
    log("")
    log(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
